# -*- coding: utf-8 -*-
"""
etbp_td.py
===================
ETBP‑TD（Efficient & Trusted Bilateral Privacy‑Preserving Truth Discovery）严格实现（实验复现版）：
- **算法核心**：按论文思想进行“真值/工人可靠度”交替优化（EM/IRLS 风格），权重 w_i ∝ 综合工具/σ_i^2；
  x_j ← ∑_i w_i r_ij / ∑_i w_i， σ_i^2 ← 平均残差平方（加ε避免除零），w_i ← 综合工具/σ_i^2；迭代直至收敛/迭代上限。
- **双向隐私**：实际系统需本地加密 + 可信边缘/多方安全聚合 + 去标识化。在本文实验代码中：
  * 为对齐**精度指标**（RMSE/VAR/RESID_VAR），我们按算法估计式直接计算 x_j 与 w_i；
  * 为对齐**开销指标**，我们对每条被使用的报告，精确汇总 `payload_bytes`（若无该列则按条数×fallback）计入 bytes，
    并按 enc_factor×条数 估算 enc_ops；这些与实际系统的计费口径一致。
- **对齐性**：不改动输入数据、不改变真值/报告内容；仅采用论文中的估计式；只在“外部”计量通信/加密开销。

你可以把本函数与你自己的 user_algorithms 中的 ETBP‑TD 做对照；若你已有实现，请优先使用你的版本，
本模块仅在你需要“可用的、可读的、带中文注释的严格实现”时使用。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, List, Sequence
import numpy as np
import pandas as pd
import time, math
from dataset_use.NYC.src.algorithms.utils_.common import (
    rmse as _rmse,
    resid_var as _resid_var,
    bytes_sum_from_rep as _bytes_sum_from_rep,
    enc_ops_by_count as _enc_ops_by_count,
)



# ---------------- ETBP‑TD 严格实现 ----------------

@dataclass
class ETBPParams:
    max_iter: int = 30              # 交替优化最大迭代
    tol: float = 1e-6               # 收敛阈值（相对改变量）
    eps_var: float = 1e-9           # 方差下界，避免除零
    init_sigma2: float = 1.0        # 初始 σ_i^2
    clip_sigma2: float = 1e6        # σ_i^2 上限，避免权重爆炸
    enc_factor: int = 6             # enc_ops 系数（与协议/库对齐）

def etbp_td(rounds_iter: Iterable, n_workers: int, params: Optional[ETBPParams]=None):
    """
    输入：
      - rounds_iter：每轮 Batch，需含 batch.entities, batch.truth, batch.reports
      - reports 列至少包括：worker_id, entity_id, value（可选：payload_bytes）
    输出：
      - 每轮日志 dict：rmse, var, resid_var, bytes, enc_ops, time_s
    备注：
      - 该实现只在“算法核心”部分使用明文数据以复现实验精度；实际系统中的加解密由安全库承担，
        本复现代码不包含密码学原语，仅做精确计量（bytes/enc_ops）。
    """
    p = params or ETBPParams()
    logs: List[Dict[str, Any]] = []

    for batch in rounds_iter:
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        J = len(truth)
        rep = getattr(batch, "reports", None)

        if rep is None or rep.empty or not {'worker_id','entity_id','value'}.issubset(rep.columns):
            # 无报告：输出 NaN 与零开销
            logs.append(dict(
                rmse=float('nan'), var=float('nan'), resid_var=float('nan'),
                bytes=0, enc_ops=0, time_s=time.time()-t0
            ))
            continue

        rep = rep[['worker_id','entity_id','value']].copy()
        # 重新编码工人与实体，建立稀疏映射
        workers = rep['worker_id'].unique().tolist()
        entities = getattr(batch, 'entities', sorted(rep['entity_id'].unique().tolist()))
        wid2i = {w:i for i,w in enumerate(workers)}
        eid2j = {e:j for j,e in enumerate(entities)}
        I = len(workers)

        # 构造每个实体的观测与对应工人索引
        g = rep.groupby('entity_id')
        obs_val: List[np.ndarray] = [None]*J
        obs_wid: List[np.ndarray] = [None]*J
        used_mask = np.zeros(len(rep), dtype=bool)
        rep = rep.reset_index(drop=True)
        idx_map = rep.reset_index().groupby('entity_id')['index'].apply(list).to_dict()

        for j, e in enumerate(entities):
            if e in g.groups:
                sub = g.get_group(e)[['worker_id','value']].to_numpy()
                obs_wid[j] = np.array([wid2i[w] for w in sub[:,0]], dtype=int)
                obs_val[j] = sub[:,1].astype(float)
                # 全部观测均被使用
                used_mask[idx_map[e]] = True
            else:
                obs_wid[j] = np.array([], dtype=int)
                obs_val[j] = np.array([], dtype=float)

        # --- 初始化 ---
        # 真值：实体均值；工人方差：常数 init_sigma2
        est = np.zeros(J, dtype=float)
        for j in range(J):
            v = obs_val[j]
            est[j] = float(np.mean(v)) if v.size else 0.0
        sigma2 = np.full(I, float(p.init_sigma2), dtype=float)

        # --- 交替优化（迭代）---
        last_est = est.copy()
        for it in range(p.max_iter):
            # 综合工具) 固定 sigma2 → 计算权重 w_i = 综合工具 / sigma2_i
            w = 1.0 / np.clip(sigma2, p.eps_var, p.clip_sigma2)

            # 2) 更新真值：x_j = sum_i w_i r_ij / sum_i w_i
            for j in range(J):
                if obs_val[j].size == 0:
                    continue
                wi = w[obs_wid[j]]
                numerator = float(np.sum(wi * obs_val[j]))
                denom = float(np.sum(wi)) + 1e-12
                est[j] = numerator / denom

            # 3) 更新工人方差：σ_i^2 = mean_j (r_ij - x_j)^2（在 i 有观测的实体上）
            #    若某工人只报少量实体，为稳健性可加上一个极小正则；这里用 eps_var 兜底
            sq = np.zeros(I, dtype=float)
            cnt = np.zeros(I, dtype=int)
            for j in range(J):
                iw = obs_wid[j]
                if iw.size == 0: continue
                res2 = (obs_val[j] - est[j])**2
                np.add.at(sq, iw, res2)
                np.add.at(cnt, iw, 1)
            upd = np.divide(sq, np.maximum(cnt, 1), where=(cnt>0))
            sigma2 = np.clip(upd + p.eps_var, p.eps_var, p.clip_sigma2)

            # 4) 收敛性检查（相对变化）
            rel = np.max(np.abs(est - last_est)) / (np.max(np.abs(last_est)) + 1e-9)
            last_est = est.copy()
            if rel < p.tol:
                break

        # --- 指标与开销（严格：仅计量，不修改估计）---
        rmse = _rmse(est, truth)
        var = float(np.var(est)) if est.size else float('nan')
        resid_var = _resid_var(est, truth)

        # 通信字节：所有**被使用**的报告；加密开销：enc_factor×条数
        bytes_used = _bytes_sum_from_rep(getattr(batch,'reports', None), subset_mask=used_mask)
        enc_ops = _enc_ops_by_count(int(used_mask.sum()), factor=p.enc_factor)

        logs.append(dict(
            rmse=rmse, var=var, resid_var=resid_var,
            bytes=int(bytes_used), enc_ops=int(enc_ops), time_s=time.time()-t0,
            # 诊断信息（可选，不参与聚合）：
            iters=it+1, workers=len(workers), entities=len(entities)
        ))

    return logs
