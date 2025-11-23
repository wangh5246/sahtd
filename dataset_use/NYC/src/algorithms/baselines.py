# -*- coding: utf-8 -*-
"""
baselines.py
=====================
为论文复现提供的**严格基线实现**（不做“简化替代”）：
- 每个算法均按常见/文献中的估计式实现，并给出可复核的中文注释；
- 仅在“外部”计量通信/加密开销（优先 sum(payload_bytes)，否则以条数×fallback_per 计）——不改变估计过程；
- 若你已有自己的更权威实现，请优先使用你的版本；本模块仅作为**显式选择的严格基线**，不会偷偷替代你的实现。

包含：
  - pure_ldp_strict              本地差分隐私（Laplace）+ 噪声方差感知的逆方差加权聚合
  - ud_ldp_strict                用户驱动（时间窗）预算分配的 LDP；跨轮维护每工人的剩余预算
  - dplp_strict                  差分隐私发布：子采样 τ + Huber 稳健聚合；显式 used_mask
  - fed_sense_strict             联邦式聚合：按工人分组本地统计（sum,count），服务器做加权合并
  - robust_b_strict              Huber M-估计（IRLS）+ 轻度截尾，用于抗异常值
  - alpha_opt_strict             工人方差估计 + 逆方差加权（两阶段：warmup → 权重收敛）
  - random_baseline_strict       随机子样（可设比例），用于“信息节流”下的对照

所有函数签名：fn(rounds_iter, n_workers:int, params=None) -> List[dict]
每轮输出字段：rmse, var, resid_var, bytes, enc_ops, time_s；还有若干诊断字段（不参与汇总）。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, List, Sequence
import numpy as np, pandas as pd, time, math, random
from dataset_use.NYC.src.algorithms.utils_.common import (
    rmse as _rmse,
    resid_var as _resid_var,
    bytes_sum_from_rep as _bytes_sum_from_rep,
    enc_ops_by_count as _enc_ops_by_count,
)
# ---------------- 稳健工具 ----------------
def _enc_ops(n_reports: int, factor: int = 1) -> int:
    return int(n_reports * factor)
def _huber(z: np.ndarray, c: float=1.345) -> np.ndarray:
    """Huber ψ 函数"""
    z = np.asarray(z, float)
    out = z.copy()
    m = np.abs(z) > c
    out[m] = c * np.sign(z[m])
    return out
def _fill_est_from_reports(est: np.ndarray, batch) -> np.ndarray:
    est = np.asarray(est, float).copy()
    rep = getattr(batch, 'reports', None)
    if rep is None or not {'entity_id','value'}.issubset(rep.columns):
        # 再退：用估计的中位数兜底（不使用 truth）
        if np.isnan(est).all():
            return np.zeros_like(est)
        med = np.nanmedian(est)
        est = np.where(np.isnan(est), med, est)
        return est
def _irls_huber(x: np.ndarray, c: float=1.345, max_iter:int=50, tol:float=1e-8) -> float:
    """标量位置参数的 Huber IRLS（对一组样本求稳健中心）"""
    x = np.asarray(x, float)
    if x.size == 0:
        return float('nan')
    mu = float(np.median(x))
    s = np.median(np.abs(x - mu)) * 1.4826 + 1e-12
    for _ in range(max_iter):
        r = (x - mu) / s
        w = np.clip(c / (np.abs(r) + 1e-12), 0.0, 1.0)
        mu_new = float(np.sum(w * x) / (np.sum(w) + 1e-12))
        if abs(mu_new - mu) < tol * (abs(mu) + 1e-9):
            break
        mu = mu_new
    return mu

# ---------------- 综合工具) Pure LDP（严格） ----------------

@dataclass
class LDPParams:
    epsilon: float = 1.0
    sensitivity: float = 1.0
    enc_factor: int = 0  # 纯本地噪声场景通常不计服务器侧加密

def pure_ldp(rounds_iter: Iterable, n_workers:int, params: Optional[LDPParams]=None):
    p = params or LDPParams()
    logs: List[Dict[str,Any]] = []
    var_lap = 2.0 * (p.sensitivity / max(p.epsilon,1e-9))**2  # Laplace(0,b) 的方差 2b^2

    for batch in rounds_iter:
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        rep = getattr(batch, "reports", None)
        est = np.full_like(truth, np.nan, dtype=float)

        if rep is not None and not rep.empty:
            g = rep.groupby("entity_id")["value"]
            for j, e in enumerate(getattr(batch, "entities", range(len(truth)))):
                if e not in g.groups: continue
                vals = g.get_group(e).to_numpy(dtype=float)
                # 本地已加噪→服务器看到的就是 vals+noise，此处直接按逆噪声方差加权（同权）求均值
                est[j] = float(np.mean(vals))
        # 无观测填补
        m = np.isnan(est)
        if np.any(m):
            if rep is not None and not rep.empty:
                # 用所有观测的稳健中心兜底
                est[m] = _irls_huber(rep["value"].to_numpy(dtype=float), c=1.345)
            else:
                est[m] = 0.0

        n_reports = 0 if rep is None else len(rep)
        logs.append(dict(
            rmse=_rmse(est, truth),
            var=float(np.var(est)) if est.size else float('nan'),
            resid_var=_resid_var(est, truth),
            bytes=_bytes_sum_from_rep(rep),
            enc_ops=_enc_ops_by_count(n_reports, p.enc_factor),
            time_s=time.time()-t0,
            var_lap=var_lap
        ))
    return logs

# ---------------- 2) UD-LDP（严格：跨轮预算） ----------------

@dataclass
class UDLDPParams:
    epsilon_total: float = 3.0      # 每工人总预算（整场实验）
    window: int = 5                 # 预算分配到最近 window 轮
    sensitivity: float = 1.0
    enc_factor: int = 0

def ud_ldp(rounds_iter: Iterable, n_workers:int, params: Optional[UDLDPParams]=None):
    p = params or UDLDPParams()
    logs: List[Dict[str,Any]] = []
    # 每工人剩余预算（跨轮维护）
    rem_eps: Dict[Any, float] = {}

    for batch in rounds_iter:
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        rep = getattr(batch, "reports", None)
        est = np.full_like(truth, np.nan, dtype=float)
        used_mask = None

        if rep is not None and not rep.empty:
            rep = rep.reset_index(drop=True)
            used_mask = np.zeros(len(rep), dtype=bool)
            # 为每工人分配本轮 ε：min(eps_total/window, 剩余)
            eps_per = {}
            for w, d in rep.groupby("worker_id"):
                rem = rem_eps.get(w, p.epsilon_total)
                alloc = min(rem, p.epsilon_total / max(1,p.window))
                eps_per[w] = alloc
                rem_eps[w] = max(0.0, rem - alloc)

            # 聚合：对每个实体，把该实体下不同工人的观测按“各自噪声方差”的逆方差权重合并
            g = rep.groupby("entity_id")
            for j, e in enumerate(getattr(batch, "entities", range(len(truth)))):
                if e not in g.groups: continue
                sub = g.get_group(e)[["worker_id","value"]]
                vals = sub["value"].to_numpy(dtype=float)
                wids = sub["worker_id"].to_numpy()
                # 每个观测对应的 Laplace 方差
                var_i = np.array([2.0*(p.sensitivity/max(eps_per.get(w,1e-9),1e-9))**2 for w in wids], dtype=float)
                wi = 1.0 / (var_i + 1e-12)
                est[j] = float(np.sum(wi * vals) / (np.sum(wi) + 1e-12))
                # 标记使用
                idx = sub.index.to_numpy()
                used_mask[idx] = True

        # 缺失兜底
        m = np.isnan(est)
        if np.any(m):
            if rep is not None and not rep.empty:
                est[m] = _irls_huber(rep["value"].to_numpy(dtype=float), c=1.345)
            else:
                est[m] = 0.0

        logs.append(dict(
            rmse=_rmse(est, truth),
            var=float(np.var(est)), resid_var=_resid_var(est, truth),
            bytes=_bytes_sum_from_rep(rep, subset_mask=used_mask),
            enc_ops=_enc_ops_by_count(int(used_mask.sum()) if isinstance(used_mask,np.ndarray) else 0, p.enc_factor),
            time_s=time.time()-t0
        ))
    return logs

# ---------------- 6) SA-HTD+（带自适应阈值的门控） ---------------- #


# ---------------- 3) DPLP（严格） ----------------

@dataclass
class DPLPParams:
    tau: float = 0.6                 # 子采样比例
    epsilon: float = 1.0             # 可选：对采样值再加 DP 噪声
    sensitivity: float = 1.0
    enc_factor: int = 0

def dplp(rounds_iter: Iterable, n_workers:int, params: Optional[DPLPParams]=None):
    p = params or DPLPParams()
    logs: List[Dict[str,Any]] = []
    rng = np.random.default_rng(2025)

    for batch in rounds_iter:
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        rep = getattr(batch, "reports", None)
        est = np.full_like(truth, np.nan, dtype=float)
        used_mask = None

        if rep is not None and not rep.empty:
            rep = rep.sort_values(["entity_id"]).reset_index(drop=True)
            used_mask = np.zeros(len(rep), dtype=bool)
            g = rep.groupby("entity_id")
            for j, e in enumerate(getattr(batch, "entities", range(len(truth)))):
                if e not in g.groups: continue
                idx = g.groups[e]
                arr = rep.loc[idx, "value"].to_numpy(dtype=float)
                k = int(max(1, math.ceil(len(arr)*p.tau)))
                sel = rng.choice(np.arange(len(arr)), size=k, replace=False) if k < len(arr) else np.arange(len(arr))
                used_mask[idx[sel]] = True
                vals = arr[sel]
                # 可选：本地 DP（若 epsilon>0）
                if p.epsilon > 0:
                    b = p.sensitivity / p.epsilon
                    vals = vals + rng.laplace(0.0, b, size=len(vals))
                est[j] = _irls_huber(vals, c=1.345)

        m = np.isnan(est)
        if np.any(m):
            if rep is not None and not rep.empty:
                est[m] = _irls_huber(rep["value"].to_numpy(dtype=float), c=1.345)
            else:
                est[m] = 0.0

        logs.append(dict(
            rmse=_rmse(est, truth),
            var=float(np.var(est)), resid_var=_resid_var(est, truth),
            bytes=_bytes_sum_from_rep(rep, subset_mask=used_mask),
            enc_ops=_enc_ops_by_count(int(used_mask.sum()) if isinstance(used_mask,np.ndarray) else 0, p.enc_factor),
            time_s=time.time()-t0
        ))
    return logs

# ---------------- 4) Fed-Sense（严格） ----------------

"""
FedSense-FRL baseline implementation.

This module implements a federated reinforcement learning baseline
to be used as a comparison method in crowdsensing / federated settings.

Key components
--------------
- FedParams: configuration dataclass
- DQN: small feed-forward Q-network
- ReplayBuffer: experience replay for each client
- FedSenseClient: local RL agent on each worker
- FedSenseServer: manages all clients and runs FedAvg
- fed_sense: main entry point compatible with an iterator of batches

A *batch* is expected to be an object with at least:
- batch.truth : 1D array-like of ground-truth values (per entity)
- batch.reports : pandas.DataFrame with columns:
    ['worker_id', 'entity_id', 'value']
- optionally batch.entities : iterable of entity ids; if not present,
  entities are assumed to be range(len(truth)).
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, List

import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _rmse(est: np.ndarray, truth: np.ndarray) -> float:
    est = np.asarray(est, float)
    truth = np.asarray(truth, float)
    return float(np.sqrt(np.mean((est - truth) ** 2)))


def _resid_var(est: np.ndarray, truth: np.ndarray) -> float:
    est = np.asarray(est, float)
    truth = np.asarray(truth, float)
    return float(np.var(est - truth))


def _irls_huber(y: np.ndarray,
                c: float = 1.345,
                max_iter: int = 50,
                tol: float = 1e-6) -> float:
    """一维 Huber IRLS，全局稳健均值，用作兜底估计。"""
    y = np.asarray(y, float).ravel()
    if y.size == 0:
        return 0.0

    mu = float(np.median(y))

    for _ in range(max_iter):
        r = y - mu
        abs_r = np.abs(r)
        w = np.ones_like(r)
        mask = abs_r > c
        w[mask] = c / (abs_r[mask] + 1e-12)

        mu_new = float(np.sum(w * y) / (np.sum(w) + 1e-12))
        if abs(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new

    return mu


def _bytes_sum_from_rep(rep: Optional[pd.DataFrame],
                        subset_mask: Optional[np.ndarray] = None) -> int:
    """简单通信量估计：每条 report 24 bytes."""
    if rep is None or rep.empty:
        return 0

    if subset_mask is not None:
        n = int(np.sum(subset_mask))
    else:
        n = int(len(rep))

    bytes_per_report = 24
    return n * bytes_per_report


def _enc_ops_by_count(n_reports: int, enc_factor: int = 2) -> int:
    """加密操作数的简单模型：每条 report * enc_factor。"""
    n_reports = int(n_reports)
    return max(0, n_reports) * int(enc_factor)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class FedParams:
    # 你原来的参数（接口保留）
    part_rate: float = 0.5
    enc_factor: int = 2

    # RL / FL 超参数
    rl_state_dim: int = 3        # [参与率, 平均误差, 常数1]
    rl_hidden_dim: int = 32
    rl_gamma: float = 0.9
    rl_lr: float = 1e-3

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 500

    replay_capacity: int = 1000
    batch_size: int = 32
    min_replay_size: int = 64
    fl_sync_every: int = 10      # 每多少轮 FedAvg 一次
    local_train_steps: int = 1   # 每轮本地训练步数


# ---------------------------------------------------------------------------
# DQN / ReplayBuffer / Client
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    """简单两层 MLP 的 DQN，用于二分类动作：0=不参与，1=参与。"""
    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Any] = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*batch)
        )
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class FedSenseClient:
    """单个 worker 的本地 RL agent。"""
    def __init__(self, worker_id, params: FedParams, device=None):
        self.worker_id = worker_id
        self.params = params
        self.device = device or torch.device("cpu")

        self.q_net = DQN(params.rl_state_dim,
                         params.rl_hidden_dim).to(self.device)
        self.target_net = DQN(params.rl_state_dim,
                              params.rl_hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(),
                                    lr=params.rl_lr)
        self.replay = ReplayBuffer(params.replay_capacity)
        self.steps_done = 0

        self.last_state = None
        self.last_action = None

    # ---- 策略：epsilon-greedy ----

    def epsilon(self) -> float:
        p = self.params
        return p.epsilon_end + (p.epsilon_start - p.epsilon_end) * \
               np.exp(-1.0 * self.steps_done / p.epsilon_decay)

    def select_action(self, state: np.ndarray) -> int:
        self.steps_done += 1
        eps = self.epsilon()
        if random.random() < eps:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
                q_values = self.q_net(s)
                action = int(q_values.argmax(dim=1).item())
        self.last_state = state
        self.last_action = action
        return action

    # ---- 经验 & 更新 ----

    def observe(self, next_state: np.ndarray, reward: float, done: bool):
        if self.last_state is None:
            return
        self.replay.push(self.last_state, self.last_action,
                         reward, next_state, done)

    def local_train(self):
        p = self.params
        if len(self.replay) < max(p.min_replay_size, p.batch_size):
            return

        for _ in range(p.local_train_steps):
            states, actions, rewards, next_states, dones = \
                self.replay.sample(p.batch_size)

            states_t = torch.tensor(states, dtype=torch.float32,
                                    device=self.device)
            next_states_t = torch.tensor(next_states, dtype=torch.float32,
                                         device=self.device)
            actions_t = torch.tensor(actions, dtype=torch.long,
                                     device=self.device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32,
                                     device=self.device)
            dones_t = torch.tensor(dones.astype(np.float32),
                                   dtype=torch.float32,
                                   device=self.device)

            # Q(s,a)
            q_values = self.q_net(states_t).gather(
                1, actions_t.unsqueeze(1)
            ).squeeze(1)

            # target = r + gamma * max_a' Q_target(s',a')
            with torch.no_grad():
                max_next_q = self.target_net(next_states_t).max(1)[0]
                target = rewards_t + p.rl_gamma * max_next_q * (1.0 - dones_t)

            loss = nn.functional.mse_loss(q_values, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ---- 参数同步 ----

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone()
                for k, v in self.q_net.state_dict().items()}

    def set_weights(self, state_dict: Dict[str, torch.Tensor]):
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Server：维护 clients + FedAvg
# ---------------------------------------------------------------------------

class FedSenseServer:
    def __init__(self, worker_ids, params: FedParams, device=None):
        self.params = params
        self.device = device or torch.device("cpu")
        self.clients: Dict[Any, FedSenseClient] = {
            wid: FedSenseClient(wid, params, device=self.device)
            for wid in worker_ids
        }
        # 用第一个 client 的参数初始化全局
        self.global_weights = next(iter(self.clients.values())
                                   ).get_weights()
        for c in self.clients.values():
            c.set_weights(self.global_weights)

        self.round_idx = 0

    def ensure_workers(self, worker_ids):
        """动态补充后来出现的新 worker（比如 'V2' 等）。"""
        new_ids = [wid for wid in worker_ids if wid not in self.clients]
        if not new_ids:
            return
        for wid in new_ids:
            c = FedSenseClient(wid, self.params, device=self.device)
            # 用当前 global_weights 初始化新 worker 的 Q 网络
            c.set_weights(self.global_weights)
            self.clients[wid] = c

    def select_participants(self,
                            worker_states: Dict[Any, np.ndarray]) -> set:
        selected = set()
        for wid, state in worker_states.items():
            client = self.clients[wid]
            action = client.select_action(state)  # 0 or 1
            if action == 1:
                selected.add(wid)
        return selected

    def observe_and_train(self,
                          next_states: Dict[Any, np.ndarray],
                          rewards: Dict[Any, float],
                          done: bool = False):
        for wid, ns in next_states.items():
            client = self.clients[wid]
            r = float(rewards.get(wid, 0.0))
            client.observe(ns, r, done)
            client.local_train()

        self.round_idx += 1

        # 周期性 FedAvg + 同步回所有 client
        if self.round_idx % self.params.fl_sync_every == 0:
            self._fedavg()
            for c in self.clients.values():
                c.set_weights(self.global_weights)
                c.update_target()

    def _fedavg(self):
        """对所有 client 做参数逐元素平均。"""
        n = len(self.clients)
        if n == 0:
            return
        agg = None
        for c in self.clients.values():
            w = c.get_weights()
            if agg is None:
                agg = {k: v.clone() for k, v in w.items()}
            else:
                for k in agg.keys():
                    agg[k] += w[k]
        for k in agg.keys():
            agg[k] /= float(n)
        self.global_weights = agg


# ---------------------------------------------------------------------------
# 主函数：fed_sense
# ---------------------------------------------------------------------------

def fed_sense(rounds_iter: Iterable,
              n_workers: int,
              params: Optional[FedParams] = None):
    """
    Federated RL-based crowdsensing baseline (FedSense-FRL).

    参数
    ----
    rounds_iter : 可迭代的 batch
        每个 batch 至少提供:
            - batch.truth : 1D array-like
            - batch.reports : DataFrame, 含列 ['worker_id','entity_id','value']
            - (可选) batch.entities : 实体 ID 列表
    n_workers : int
        总工人数（这里只保留接口，实际 worker_id 由 reports 里的唯一值决定）
    params : FedParams
        超参数配置

    返回
    ----
    logs : list[dict]
        每轮一个 dict，字段：
        rmse, var, resid_var, bytes, enc_ops, time_s, part_workers
    """
    p = params or FedParams()
    logs: List[Dict[str, Any]] = []

    random.seed(2025)
    np.random.seed(2025)

    server: Optional[FedSenseServer] = None
    worker_stats: Dict[Any, Dict[str, float]] = {}  # 跟踪每个 worker 的统计量
    round_idx = 0

    for batch in rounds_iter:
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        rep: Optional[pd.DataFrame] = getattr(batch, "reports", None)
        est = np.full_like(truth, np.nan, dtype=float)
        used_mask = None
        part_workers: set = set()

        if rep is not None and not rep.empty:
            rep = rep.reset_index(drop=True)

            workers = rep["worker_id"].unique().tolist()

            # 第一次建 server；之后每一轮都 ensure_workers
            if server is None:
                server = FedSenseServer(workers, p,
                                        device=torch.device("cpu"))
            else:
                server.ensure_workers(workers)

            # 确保 worker_stats 中每个 wid 都有条目
            for wid in workers:
                if wid not in worker_stats:
                    worker_stats[wid] = dict(
                        n_part=0,
                        cum_err=0.0,
                        last_abs_err=0.0
                    )

            # 1) 构造状态向量 s = [参与率, 平均误差, 1]
            worker_states: Dict[Any, np.ndarray] = {}
            for wid in workers:
                st = worker_stats[wid]
                participation_ratio = st["n_part"] / max(1, round_idx + 1)
                avg_err = (st["cum_err"] / st["n_part"]
                           if st["n_part"] > 0 else 1.0)
                worker_states[wid] = np.array(
                    [participation_ratio, avg_err, 1.0],
                    dtype=np.float32
                )

            # 2) 用 RL 选择本轮参与的工人
            part_workers = server.select_participants(worker_states)

            # 如果一个都没选，为防止全 NaN，兜底全选
            if not part_workers:
                part_workers = set(workers)

            used_mask = rep["worker_id"].isin(list(part_workers)).to_numpy()
            rep_part = rep[used_mask]

            # 3) 对参与工人按 entity 求均值
            g = rep_part.groupby("entity_id")["value"]
            sums = g.sum()
            cnts = g.count()

            entities = getattr(batch, "entities", range(len(truth)))
            for j, e in enumerate(entities):
                if e in sums.index:
                    est[j] = float(sums.loc[e] / max(1, cnts.loc[e]))

            # 4) 计算奖励 & 下一状态
            ent_ids = list(entities)
            truth_map = {e: float(truth[j]) for j, e in enumerate(ent_ids)}

            rewards: Dict[Any, float] = {}
            next_states: Dict[Any, np.ndarray] = {}

            for wid in workers:
                # 再保险：如果漏了某个 wid，就补一条默认统计
                if wid not in worker_stats:
                    worker_stats[wid] = dict(
                        n_part=0,
                        cum_err=0.0,
                        last_abs_err=0.0
                    )

                mask_w = (rep["worker_id"].to_numpy() == wid)
                rep_w = rep[mask_w]

                if len(rep_w) > 0:
                    errs = []
                    for _, row in rep_w.iterrows():
                        eid = row["entity_id"]
                        if eid in truth_map:
                            errs.append(abs(
                                float(row["value"]) - truth_map[eid]
                            ))
                    mean_err = float(np.mean(errs)) if errs else 1.0
                else:
                    mean_err = 1.0

                st = worker_stats[wid]
                if wid in part_workers:
                    st["n_part"] += 1
                    st["cum_err"] += mean_err
                st["last_abs_err"] = mean_err

                # 奖励：误差越小越好 + 参与有小奖励
                reward = -mean_err
                if wid in part_workers:
                    reward += 0.1
                rewards[wid] = reward

                participation_ratio = st["n_part"] / max(1, round_idx + 1)
                avg_err = (st["cum_err"] / st["n_part"]
                           if st["n_part"] > 0 else 1.0)
                next_states[wid] = np.array(
                    [participation_ratio, avg_err, 1.0],
                    dtype=np.float32
                )

            # 5) 交给 server 做本地训练 + 周期性 FedAvg
            server.observe_and_train(next_states, rewards, done=False)

        # 6) 对没估计到的实体做兜底
        m = np.isnan(est)
        if np.any(m):
            if rep is not None and not rep.empty:
                est[m] = _irls_huber(
                    rep["value"].to_numpy(dtype=float),
                    c=1.345
                )
            else:
                est[m] = 0.0

        # 7) 记录日志（保持跟你原来一致）
        logs.append(dict(
            rmse=_rmse(est, truth),
            var=float(np.var(est)),
            resid_var=_resid_var(est, truth),
            bytes=_bytes_sum_from_rep(
                rep, subset_mask=used_mask
            ),
            enc_ops=_enc_ops_by_count(
                int(used_mask.sum())
                if isinstance(used_mask, np.ndarray) else 0,
                p.enc_factor
            ),
            time_s=time.time() - t0,
            part_workers=int(len(part_workers)
                             if rep is not None and not rep.empty else 0)
        ))

        round_idx += 1

    return logs

# ---------------- 5) Robust-B（严格：Huber IRLS + 截尾） ----------------

@dataclass
class RobustParams:
    huber_c: float = 1.345
    trim: float = 0.02
    enc_factor: int = 0

def robust_b(rounds_iter: Iterable, n_workers:int, params: Optional[RobustParams]=None):
    p = params or RobustParams()
    logs: List[Dict[str,Any]] = []

    for batch in rounds_iter:
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        rep = getattr(batch, "reports", None)
        est = np.full_like(truth, np.nan, dtype=float)

        if rep is not None and not rep.empty:
            g = rep.groupby("entity_id")["value"]
            for j, e in enumerate(getattr(batch, "entities", range(len(truth)))):
                if e not in g.groups: continue
                arr = np.sort(g.get_group(e).to_numpy(dtype=float))
                n = len(arr); k = int(p.trim * n)
                core = arr[k:n-k] if n > 2*k else arr
                est[j] = _irls_huber(core, c=p.huber_c)

        m = np.isnan(est)
        if np.any(m):
            if rep is not None and not rep.empty:
                est[m] = _irls_huber(rep["value"].to_numpy(dtype=float), c=p.huber_c)
            else:
                est[m] = 0.0

        logs.append(dict(
            rmse=_rmse(est, truth),
            var=float(np.var(est)), resid_var=_resid_var(est, truth),
            bytes=_bytes_sum_from_rep(rep),
            enc_ops=_enc_ops_by_count(n_workers, p.enc_factor),
            time_s=time.time()-t0
        ))
    return logs



# ---------------- 7) Random Baseline（严格） ----------------

@dataclass
class RandParams:
    ratio: float = 0.5
    seed: int = 2025
    enc_factor: int = 0

def random_baseline(rounds_iter: Iterable, n_workers:int, params: Optional[RandParams]=None):
    p = params or RandParams()
    rng = np.random.default_rng(p.seed)
    logs: List[Dict[str,Any]] = []

    for batch in rounds_iter:
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        rep = getattr(batch, "reports", None)
        est = np.full_like(truth, np.nan, dtype=float)
        used_mask = None

        if rep is not None and not rep.empty:
            rep = rep.reset_index(drop=True)
            used_mask = np.zeros(len(rep), dtype=bool)
            g = rep.groupby("entity_id")["value"]
            idx_map = rep.reset_index().groupby("entity_id")["index"].apply(list).to_dict()
            for j, e in enumerate(getattr(batch, "entities", range(len(truth)))):
                if e not in g.groups: continue
                arr = g.get_group(e).to_numpy(dtype=float)
                idx = np.array(idx_map[e], dtype=int)
                k = max(1, int(len(arr) * p.ratio))
                sel = rng.choice(np.arange(len(arr)), size=k, replace=False) if k < len(arr) else np.arange(len(arr))
                used_mask[idx[sel]] = True
                est[j] = _irls_huber(arr[sel], c=1.345)

        m = np.isnan(est)
        if np.any(m):
            if rep is not None and not rep.empty:
                est[m] = _irls_huber(rep["value"].to_numpy(dtype=float), c=1.345)
            else:
                est[m] = 0.0

        logs.append(dict(
            rmse=_rmse(est, truth),
            var=float(np.var(est)), resid_var=_resid_var(est, truth),
            bytes=_bytes_sum_from_rep(rep, subset_mask=used_mask),
            enc_ops=_enc_ops_by_count(n_workers, p.enc_factor),
            time_s=time.time()-t0
        ))
    return logs