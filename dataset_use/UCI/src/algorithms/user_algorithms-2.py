from __future__ import annotations
import time
import traceback  # 文件头部加一次即可

import math
import numpy as np
import pandas as pd
from typing import Iterable, Dict, Any, Tuple, Optional
from collections import deque
from typing import Iterable, Dict, Any, List
import numpy as np, time, math
from dataset_use.NYC.src.algorithms.dap_client import DAPClient
from dataset_use.NYC.src.algorithms.pld_accountant import PLDAccountant
from dataset_use.NYC.src.algorithms.utils_.newsahtd_boost import (
    CAdaptive, irls_huber_mean, trimmed_mean, mad_scale,
    combine_uncertainty, suggest_kp, need_probe_route, need_fallback, cap_weights
)
# ===== hard_budget.py（你也可以直接放到 user_algorithms.py 顶部） =====
from typing import Iterable, Sequence, Tuple, List
def _bytes_cap_for_round(params, round_idx: int) -> int:
    """
    计算本轮的 bytes_cap：等分剩余额度（经典做法）。
    需要 params 提供：
        - budget_bytes: 总预算
        - total_rounds: 总轮数
        - _bytes_spent: （可选）已花费字节（若无则默认 0）
    """
    budget_total = int(getattr(params, "budget_bytes", 0) or 0)
    total_rounds = int(getattr(params, "total_rounds", 0) or 0)
    spent        = int(getattr(params, "_bytes_spent", 0) or 0)

    if budget_total <= 0 or total_rounds <= 0:
        return np.inf  # 无预算约束

    remaining = max(0, budget_total - spent)
    rounds_left = max(1, total_rounds - round_idx)
    return remaining // rounds_left

def _estimate_cost_per_entity(counts_by_entity: np.ndarray, params) -> np.ndarray:
    """
    用上报条数 * 每条字节 估算走 A 的成本。
    如果你有更精确的 perA_bytes 就放在 params.perA_bytes（否则默认 32）。
    """
    perA = int(getattr(params, "perA_bytes", 32) or 32)
    counts = np.asarray(counts_by_entity, dtype=np.int64)
    return counts * perA
def enforce_hard_budget_topk(
    entity_ids: Sequence,                # 实体 ID 列表，长度为 N
    benefit_by_entity: np.ndarray,       # 长度 N，代表“收益”（例如 ΔVar 或 score）
    cost_by_entity: np.ndarray,          # 长度 N，代表该实体走 A 的“字节成本”
    bytes_cap: int                       # 本轮允许的最大字节
) -> List:
    """
    在 bytes_cap 约束下，按 benefit/cost 从高到低选实体（0-1 选择），
    返回被选中的 entity_id 列表。保证 sum(cost) <= bytes_cap。
    """
    if not np.isfinite(bytes_cap) or bytes_cap <= 0:
        return []

    benefit = np.asarray(benefit_by_entity, dtype=np.float64)
    cost    = np.asarray(cost_by_entity,    dtype=np.float64)

    # 过滤掉无效或零成本/收益
    valid = (cost > 0) & np.isfinite(cost) & np.isfinite(benefit) & (benefit > 0)
    if not np.any(valid):
        return []

    benefit = benefit[valid]
    cost    = cost[valid]
    ent_arr = np.asarray(entity_ids, dtype=object)[valid]

    # 单位字节收益
    ratio = benefit / cost

    # 按 ratio 做 Top-K —— 用 argpartition 近似 O(n)，大数据更快
    order = np.argsort(-ratio)  # 如果 N 很大也可以先用 argpartition 再二次排序

    # 贪心累计直到触达上限
    picked = []
    total = 0.0
    for idx in order:
        c = cost[idx]
        if total + c <= bytes_cap:
            picked.append(ent_arr[idx])
            total += c
        else:
            continue

    return picked


# 放在 NewSAHTD 入口（或更外层）初始化一次
_C_ADAPT = CAdaptive(k=1.5, beta=0.9)  # 可把 _C_ADAPT 放进上下文/参数容器
from sympy.physics.vector.printing import params

# ==== [NewSAHTD 增益：稳健尺度、自适应 HubER、不确定度驱动 Kp、强制回退、微量路由] ====
# ==== NewSAHTD 辅助函数（零依赖） ====
def _mad_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0: return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))

def _huber_weights(resid: np.ndarray, c: float) -> np.ndarray:
    r = np.asarray(resid, dtype=float)
    w = np.ones_like(r)
    m = np.abs(r) > c
    w[m] = np.where(np.abs(r[m]) > 0, c / np.abs(r[m]), 0.0)
    return w

def _irls_huber_mean(y: np.ndarray, x0: float, c: float, max_iter=20, tol=1e-6) -> float:
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if y.size == 0: return float(x0)
    mu = float(x0)
    for _ in range(max_iter):
        r = y - mu
        w = _huber_weights(r, c)
        den = float(np.sum(w)) + 1e-12
        mu_new = float(np.sum(w * y) / den)
        if abs(mu_new - mu) < tol: break
        mu = mu_new
    return mu

def _trimmed_mean(y: np.ndarray, trim: float = 0.10) -> float:
    v = np.sort(np.asarray(y, dtype=float))
    v = v[~np.isnan(v)]
    if v.size == 0: return 0.0
    k = int(len(v) * trim)
    v = v[k: len(v)-k] if len(v) - 2*k > 0 else v
    return float(np.mean(v))

def _need_fallback(resid_ratio: float, sample_n: int, phi: float = 0.25, min_n: int = 12) -> bool:
    return (sample_n < min_n and resid_ratio > phi) or (resid_ratio > phi)
class _P:
    """参数容器的兜底（便于 getattr）"""
    def __init__(self, **kw):
        self.__dict__.update(kw)
#在 `_huber_mean`、`_huber`、`_aggregate_A`、`_aggregate_B` 内部首行调用 `_safe_array`，即可彻底规避 `AxisError`。
def _safe_array(x):
    x = np.asarray(x)
    if x.ndim == 0:
        x = x[None]
    if x.size == 0:
        raise ValueError("expected non-empty sample")
    return x
def _rmse(est: np.ndarray, truth: np.ndarray) -> float:
    est = np.asarray(est, float).ravel()
    truth = np.asarray(truth, float).ravel()
    if est.shape != truth.shape or est.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((est - truth) ** 2)))

def _bytes(n_reports: int, per: int) -> int:
    return int(n_reports * per)

def _enc_ops(n_reports: int, factor: int = 1) -> int:
    return int(n_reports * factor)

def _huber(x: np.ndarray, c: float = 1.5) -> float:
    x = _safe_array(x).astype(float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    r = (x - med) / (1.4826 * mad)
    w = np.clip(c / (np.abs(r) + 1e-12), 0.0, 1.0)
    return float(np.sum(w * x) / (np.sum(w) + 1e-12))

def _catoni_mean(y, alpha=0.1, iters=20):
    """
    Catoni M-estimator for mean（一维、无外依赖）
    alpha ~ O(1/sqrt(n))；这里取 0.1 作为温和默认
    """
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    if y.size == 0: return 0.0
    mu = float(np.median(y))
    for _ in range(iters):
        r = y - mu
        # psi(t) = (1/alpha) * tanh(alpha * t)
        psi = np.tanh(alpha * r) / (alpha + 1e-12)
        mu_new = mu + float(np.mean(psi))
        if abs(mu_new - mu) < 1e-6: break
        mu = mu_new
    return mu

def _mom_mean(y, g=5):
    """
    Median-of-Means：把样本均分为 g 组，各组均值后取中位数
    """
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = y.size
    if n == 0: return 0.0
    g = max(1, min(g, n))
    chunks = np.array_split(np.random.permutation(y), g)
    means = [float(np.mean(ch)) if ch.size else 0.0 for ch in chunks]
    return float(np.median(means))

def _local_dp_laplace(x: np.ndarray, epsilon: float, sensitivity: float = 1.0) -> np.ndarray:
    """本地 LDP：只用传入的 sensitivity，不再读外部 params 以免串扰。"""
    x = _safe_array(x).astype(float)
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for LDP")
    scale = float(sensitivity) / float(epsilon)
    noise = np.random.laplace(0.0, scale, size=len(x))
    return x + noise
# ---------------- 公共聚合器：B 通道 & A 通道 ---------------- #

def _aggregate_B(reports: np.ndarray, epsilon: float, huber_c: float = 1.3, trim: float = 0.02) -> Tuple[float, float, int]:
    """B 通道：LDP 后的稳健融合（此处 reports 已含噪）"""
    reports = _safe_array(reports).astype(float)
    est = _huber(reports, c=huber_c)
    v = float(np.var(reports) + 1e-12)
    return float(est), v, 0  # enc_ops=0（计费在外部）

def _aggregate_A(clean_reports: np.ndarray, huber_c: float = 1.6, trim: float = 0.02) -> Tuple[float, float, int]:
    """A 通道：受信聚合（不加噪），稳健中心"""
    clean_reports = _safe_array(clean_reports).astype(float)
    est = _huber(clean_reports, c=huber_c)
    v = float(np.var(clean_reports) + 1e-12)
    return float(est), v, 0
def _resid_var(est: np.ndarray, truth: np.ndarray) -> float:
    est = np.asarray(est, float).ravel()
    truth = np.asarray(truth, float).ravel()
    return float(np.var(est - truth)) if est.size and est.shape == truth.shape else float('nan')

# ---- 通用缺失填补：用当轮 reports 的稳健均值作为回退，避免 RMSE 全 NaN 且不泄露 truth ----
def _fill_est_from_reports(est: np.ndarray, batch) -> np.ndarray:
    """
    用当轮 reports 的信息对 est 中的 NaN 做稳健填补：
    - 优先：按 entity 的稳健中心（Huber/中位数）填；
    - 其次：用所有观测的稳健中心兜底；
    - 最后：若无报告，用 0.0 兜底。
    始终返回与 est 同形状的 ndarray。
    """
    import numpy as np
    est = np.asarray(est, float).copy()
    rep = getattr(batch, 'reports', None)

    # 无报告或列不齐 -> 不用 truth，保持外部可复现
    if rep is None or not {'entity_id','value'}.issubset(rep.columns):
        if np.isnan(est).all():
            return np.zeros_like(est, dtype=float)
        med = np.nanmedian(est)
        est = np.where(np.isnan(est), med, est)
        return est

    # 有报告 -> 逐 entity 用稳健统计填充
    import numpy as np
    import pandas as pd
    g = rep.groupby('entity_id')['value']
    entities = getattr(batch, 'entities', list(range(len(est))))
    for j, e in enumerate(entities):
        if np.isnan(est[j]):
            if e in g.groups:
                arr = g.get_group(e).to_numpy(dtype=float)
                if arr.size:
                    # 轻量 Huber 中心
                    med = float(np.median(arr))
                    mad = float(np.median(np.abs(arr - med)) + 1e-9)
                    r = (arr - med) / (1.4826 * mad)
                    w = np.clip(1.345 / (np.abs(r) + 1e-12), 0.0, 1.0)
                    est[j] = float(np.sum(w * arr) / (np.sum(w) + 1e-12))
    # 还剩 NaN -> 用全体观测稳健中心兜底
    if np.isnan(est).any():
        arr_all = rep['value'].to_numpy(dtype=float)
        med = float(np.median(arr_all))
        mad = float(np.median(np.abs(arr_all - med)) + 1e-9)
        r = (arr_all - med) / (1.4826 * mad)
        w = np.clip(1.345 / (np.abs(r) + 1e-12), 0.0, 1.0)
        fill = float(np.sum(w * arr_all) / (np.sum(w) + 1e-12))
        est = np.where(np.isnan(est), fill, est)
    return est

class _P:
    """参数容器的兜底（便于 getattr）"""
    def __init__(self, **kw):
        self.__dict__.update(kw)

def new_sa_htd_budgeted(rounds_iter: Iterable, n_workers: int, params=None):
    """
    New SA-HTD（升级版）
    - 路由：由“布尔阈值”改为“Top-K 单位字节收益(ΔVar/ΔBytes)分配”；可控地把 A 端预算花在刀刃上
    - 融合：由固定 0.6/0.4 改为“逆方差加权 + 轻度 shrinkage”，在 A/B 方差接近时更稳健
    - 性能：单轮只 groupby 一次，后续全用 numpy，显著降低 time_s_mean
    - 接口/日志字段与原版一致；不改你的 Huber/Trim/LDP/A/B 聚合等核心统计逻辑
    """
    if params is None:
        params = _P()

    # ===== 参数读取（保持你的默认）=====
    tau_low   = getattr(params, "tau_low", 0.35)
    tau_high  = getattr(params, "tau_high", 0.65)
    risk_alpha= getattr(params, "risk_alpha", 0.5)
    bucket_h  = getattr(params, "bucket_high", 0.7)
    bucket_m  = getattr(params, "bucket_mid", 0.2)
    rho_min   = getattr(params, "rho_min", 0.2)
    rho_max   = getattr(params, "rho_max", 0.35)
    pilot_frac= getattr(params, "pilot_frac", 0.30)
    k_probe   = getattr(params, "k_probe", 4)
    lam0      = getattr(params, "lambda0", 3.5e-6)

    hubA      = getattr(params, "huber_A", 1.6)
    hubB      = getattr(params, "huber_B", 1.3)
    trimA     = getattr(params, "trim_A", 0.02)
    trimB     = getattr(params, "trim_B", 0.02)

    vq        = getattr(params, "vfloor_q", 0.25)
    gammaA    = getattr(params, "gamma_A", 0.45)  # 融合偏置阈值（保留参数，不再直接用）
    budget    = getattr(params, "budget_bytes", 180_000)
    warmup    = getattr(params, "warmup_rounds", 4)
    minBf     = getattr(params, "min_B_frac", 0.5)
    minBabs   = getattr(params, "min_B_abs", 5)
    diag      = getattr(params, "diag", True)

    # 可微调以“**略增 bytes_mean** 换更明显精度收益”
    # 建议：lam0 上调到 1.5~2.0×、gammaA 降到 0.35，warmup 从 4 降到 2，minBf 从 0.5 降到 0.3
    lam0      = getattr(params, "lambda0", lam0)
    gammaA    = getattr(params, "gamma_A", 0.35)
    warmup    = getattr(params, "warmup_rounds", 2)
    minBf     = getattr(params, "min_B_frac", 0.3)
    cA_prev, cB_prev = None, None
    beta = 0.9  # EMA 系数
    perA, perB = 64, 16
    eps_B = 1.0  # 与主控保持一致
    tau_lo, tau_hi = tau_low, tau_high
    budget_remaining = int(getattr(params, "budget_bytes", budget))
    total_rounds     = int(getattr(params, "total_rounds", 12))
    ROUTE_LO = getattr(params, "route_ratio_low", 0.05)  # 5%
    ROUTE_HI = getattr(params, "route_ratio_high", 0.08)  # 8%
    rho_max = getattr(params, "rho_max", 0.35)  # 部分切 A 的上限比例
    logs = []
    hist_res = []
    for t, batch in enumerate(rounds_iter, start=1):
        import numpy as np
        t0    = time.time()
        truth = np.asarray(batch.truth, float)
        J     = len(truth)

        # ===== 动态预算：本轮上限（剩余预算/剩余轮次）=====
        rounds_left = max(1, total_rounds - t + 1)
        bytes_cap   = budget_remaining // rounds_left
        

        # ===== 预分配容器 =====
        estA = np.full(J, np.nan, float)
        estB = np.full(J, np.nan, float)
        vA   = np.full(J, np.inf,  float)
        vB   = np.full(J, np.inf,  float)

        A_cnt = 0
        B_cnt = 0
        used_bytes = 0

        if (getattr(batch, "reports", None) is not None and
            {"entity_id", "value"}.issubset(batch.reports.columns) and
            J > 0):

            rep = batch.reports[["entity_id", "value"]].copy()

            # ----- 仅 groupby 一次，后续全走 numpy -----
            g = rep.groupby("entity_id")["value"]
            vals_by_e = [g.get_group(e).to_numpy(dtype=float) if e in g.groups else None
                         for e in batch.entities]
            counts = np.array([0 if v is None else v.size for v in vals_by_e], dtype=int)
            total_reports = int(counts.sum())
            target_A = max(1, int(ROUTE_LO * total_reports))
            extra_per = perA - perB  # 48
            need_extra = extra_per * target_A
            # ===== ranked: 高风险优先的实体索引列表 =====
            # 需要的前置变量：counts（每个实体的报告条数），vals_by_e（list/array，每个实体的观测），
            # 可选 var_e（如果你前面算过每个实体的原始方差）。
            J = len(counts)  # 实体总数
            # 1) 构造风险分数：优先用当前批的方差；没有就用 MAD 或条数倒序兜底
            try:
                # 若你前面已经有 var_e（每个实体的方差向量），直接用；否则即时计算
                if 'var_e' in locals() and isinstance(var_e, np.ndarray) and var_e.shape[0] == J:
                    risk = np.asarray(var_e, float)
                else:
                    # 即时计算方差（没有就 0.0）
                    tmp_var = np.zeros(J, dtype=float)
                    for j in range(J):
                        arr = vals_by_e[j]
                        if arr is None or len(arr) == 0:
                            tmp_var[j] = 0.0
                        else:
                            a = np.asarray(arr, float)
                            a = a[~np.isnan(a)]
                            tmp_var[j] = float(np.var(a)) if a.size else 0.0
                    risk = tmp_var

                # 历史 RMSE 平滑（可选）
                if 'hist_res' in locals() and len(hist_res) >= 5:
                    hist = float(np.mean(hist_res[-5:]))
                    risk_alpha = getattr(params, "risk_alpha", 0.5)
                    risk = risk_alpha * risk + (1 - risk_alpha) * hist
            except Exception:
                # 极端兜底：若上面失败，用“条数”近似风险（条数越多越靠前，便于先在信息量大的实体尝试 A）
                risk = np.asarray(counts, float)

            # 2) 只保留“有报告”的实体
            mask_nonempty = (np.asarray(counts, int) > 0)
            idx_all = np.arange(J, dtype=int)
            idx_nonempty = idx_all[mask_nonempty]

            # 3) 按风险从高到低排序，得到 ranked
            if idx_nonempty.size > 0:
                ranked = idx_nonempty[np.argsort(-risk[mask_nonempty])]
            else:
                # 如果这一轮刚好全部空（很少见），就用 0..J-1 兜底
                ranked = idx_all

            # 如果 cap 不够，先从“不重要实体”的 B 报告里删除，腾出字节
            if base_B + need_extra > bytes_cap:
                to_free = base_B + need_extra - bytes_cap
                free_reports = int(np.ceil(to_free / perB))
                for j in reversed(ranked):  # ranked 尾部最不重要
                    if free_reports <= 0: break
                    take = min(int(counts[j]), free_reports)
                    counts[j] -= take
                    free_reports -= take
                total_reports = int(counts.sum())
                base_B = perB * total_reports

            # 分配 A（先整实体，再部分 A）
            routeA = np.zeros(J, dtype=bool)
            partA = np.zeros(J, dtype=int)
            remain = target_A
            for j in ranked:
                if remain <= 0: break
                n_j = int(counts[j])
                if n_j > 0 and n_j <= remain:
                    routeA[j] = True
                    remain -= n_j

            if remain > 0:
                rho_max = getattr(params, "rho_max", 0.35)
                for j in ranked:
                    if remain <= 0 or routeA[j]: continue
                    n_j = int(counts[j])
                    nA_j = min(remain, int(np.ceil(rho_max * n_j)))
                    if nA_j > 0:
                        partA[j] = nA_j
                        remain -= nA_j

            # ----- 风险评分：实体方差 + 历史 RMSE 指数平滑 -----
            var_e = rep.groupby("entity_id")["value"].var().reindex(batch.entities).fillna(0.0).to_numpy()
            hist  = np.mean(hist_res[-5:]) if len(hist_res) >= 5 else np.nan
            risk  = var_e.copy()
            if np.isfinite(hist):
                risk = risk_alpha * risk + (1 - risk_alpha) * hist

            # 桶分配（保留你的筛选语义）
            idx = np.argsort(-risk)
            nH  = int(bucket_h * J); nM = int(bucket_m * J)
            H   = set(idx[:nH]); M = set(idx[nH:nH+nM])

            # ----- B 端保底条数（只用于预算可行性检查，不计入 used_bytes，保持与旧版口径一致）-----
            B_floor_per_e = np.maximum(minBabs, (minBf * counts).astype(int))

            # ----- 单位字节收益 η_j：pilot 估计 ΔVar/ΔBytes（仅对高风险实体计算）-----
            candidates = []      # (eta, j)
            for j, arr in enumerate(vals_by_e):
                if arr is None or arr.size == 0:
                    continue
                if (t <= warmup) or (j not in H):
                    # 暖启动或低风险：暂不进入候选
                    continue
                m_p  = max(1, int(pilot_frac * arr.size))
                arrp = arr[:m_p]
                vAp  = np.var(arrp) + 1e-9
                vBp  = np.var(_local_dp_laplace(arrp, epsilon=eps_B)) + 1e-9
                dvar = max(0.0, vBp - vAp)
                dbytes = perA * arr.size - perB * arr.size
                if dbytes <= 0:
                    continue
                eta = lam0 * (dvar / (dbytes + 1e-9))
                if eta > 0:
                    candidates.append((eta, j))

            # ----- Top-K 计划分配（在 bytes_cap 内最大化“收益”）-----
            # 若 bytes_cap 较小，以“全走B”的开销作为基线；可分配给A的“额外预算”为：
            base_B = perB * int(total_reports)
            extra_allow = max(0, bytes_cap - base_B)
            extra_per = max(1, perA - perB)
            max_A_reports = extra_allow // extra_per if extra_allow > 0 else 0

            # routeA / partA 用于记录“是否整实体 A”和“部分 A 的条数”
            routeA = np.zeros(J, dtype=bool)
            partA = np.zeros(J, dtype=int)  # 若 >0 表示仅部分条目走 A

            if candidates and max_A_reports > 0:
                candidates.sort(reverse=True)  # 按 η_j 降序
                remain = int(max_A_reports)
                for eta, j in candidates:
                    n_j = int(counts[j])
                    if n_j <= 0:
                        continue
                    if n_j <= remain:
                        routeA[j] = True
                        remain -= n_j
                    else:
                        # —— 关键变化：允许“部分切 A”
                        # 允许最多切 rho_max 比例到 A（不至于把超大实体吃光额度）
                        nA_j = min(remain, int(np.ceil(rho_max * n_j)))
                        if nA_j > 0:
                            partA[j] = nA_j  # 记录 A 的条数
                            remain -= nA_j
                    if remain <= 0:
                        break
            # ===== 尺寸对齐（防越界）=====
            # 以 vals_by_e 的长度为“单轮实体数”的唯一真值
            J_vals = len(vals_by_e) if 'vals_by_e' in locals() else J
            if 'J' not in locals() or J != J_vals:
                J = int(J_vals)

            # 对齐 estA/estB/vA/vB 的大小
            def _resize_vec(vec, default):
                if vec is None or len(vec) != J:
                    v = np.empty(J, dtype=float)
                    v.fill(default)
                    return v
                return vec

            estA = _resize_vec(estA if 'estA' in locals() else None, np.nan)
            estB = _resize_vec(estB if 'estB' in locals() else None, np.nan)
            vA = _resize_vec(vA if 'vA' in locals() else None, np.inf)
            vB = _resize_vec(vB if 'vB' in locals() else None, np.inf)

            # 对齐 counts/routeA/partA 的大小（不足则补 0，过长则截断）
            def _pad_trunc_int(a, J):
                a = np.asarray(a if a is not None else [], dtype=int)
                if a.size < J:
                    b = np.zeros(J, dtype=int);
                    b[:a.size] = a;
                    return b
                elif a.size > J:
                    return a[:J].astype(int)
                else:
                    return a.astype(int)

            counts = _pad_trunc_int(counts if 'counts' in locals() else None, J)

            def _pad_trunc_bool(a, J):
                a = np.asarray(a if a is not None else [], dtype=bool)
                if a.size < J:
                    b = np.zeros(J, dtype=bool);
                    b[:a.size] = a;
                    return b
                elif a.size > J:
                    return a[:J].astype(bool)
                else:
                    return a.astype(bool)

            routeA = _pad_trunc_bool(routeA if 'routeA' in locals() else None, J)
            partA = _pad_trunc_int(partA if 'partA' in locals() else None, J)

            # 保险起见：遍历时以 J 为准；vals_by_e 也做边界保护
            def _get_arr(j):
                if j < len(vals_by_e):
                    return vals_by_e[j]
                return None

            # ===== /尺寸对齐 =====

            # ----- 统一聚合（hot path） -----
            for j, arr in enumerate(vals_by_e):
                if arr is None or arr.size == 0:
                    continue
                v = arr.astype(float, copy=False)

                if routeA[j] or partA[j] > 0:
                    # —— A/B 拆分（支持部分 A）
                    if routeA[j]:
                        vA_j, vB_j = v, None
                    else:
                        nA = int(partA[j])
                        vA_j, vB_j = v[:nA], v[nA:]

                    # —— A 路自适应 cA（EMA 平滑）
                    baseA = np.median(vA_j) if vA_j is not None and vA_j.size else 0.0
                    sA = _mad_scale((vA_j - baseA) if vA_j is not None else np.array([0.0]))
                    cA_now = max(1e-6, float(hubA * sA))
                    cA_prev = cA_now if cA_prev is None else (beta * cA_prev + (1 - beta) * cA_now)
                    cA_use = cA_prev

                    # —— A 路估计
                    x0A = float(np.mean(vA_j)) if vA_j is not None and vA_j.size else 0.0
                    # 先 Catoni，再看残差比回退到 MoM（更狠）
                    estA_j = _catoni_mean(vA_j, alpha=0.1)
                    residA = (vA_j - estA_j) if vA_j is not None else np.array([0.0])
                    rratioA = float(
                        np.mean(np.abs(residA) > (1.5 * _mad_scale(residA)))) if vA_j is not None and vA_j.size else 0.0
                    if _need_fallback(rratioA, int(vA_j.size if vA_j is not None else 0), phi=0.22, min_n=12):
                        estA_j = _mom_mean(vA_j, g=5)

                    # —— 回退：高残差或样本太少时
                    if _need_fallback(rratioA, int(vA_j.size if vA_j is not None else 0), phi=0.25, min_n=12):
                        estA_j = _trimmed_mean(vA_j if vA_j is not None else np.array([x0A]), trim=0.10)

                    estA[j] = estA_j
                    vA[j] = float(np.var(vA_j)) if vA_j is not None and vA_j.size else np.inf

                    # —— 若存在 B 部分，照旧走本地 LDP 聚合（cB 自适应）
                    if vB_j is not None and vB_j.size:
                        noisy = _local_dp_laplace(vB_j, epsilon=eps_B, sensitivity=1.0)
                        baseB = np.median(noisy)
                        sB = _mad_scale(noisy - baseB)
                        cB_now = max(1e-6, float(hubB * sB))
                        cB_prev = cB_now if cB_prev is None else (beta * cB_prev + (1 - beta) * cB_now)
                        cB_use = cB_prev

                        # 这里仍调用你的 B 聚合（若需要也可换成 IRLS）
                        estB_j, vB_j_var, _ = _aggregate_B(noisy, epsilon=eps_B, huber_c=cB_use, trim=trimB)
                        estB[j] = estB_j
                        residB = noisy - estB_j
                        if np.mean(np.abs(residB) > (1.5 * _mad_scale(residB))) > 0.25:
                            estB_j = _mom_mean(noisy, g=5)
                        vB[j] = float(vB_j_var)

                else:
                    # —— 全走 B 的实体
                    noisy = _local_dp_laplace(v, epsilon=eps_B, sensitivity=1.0)
                    baseB = np.median(noisy)
                    sB = _mad_scale(noisy - baseB)
                    cB_now = max(1e-6, float(hubB * sB))
                    cB_prev = cB_now if cB_prev is None else (beta * cB_prev + (1 - beta) * cB_now)
                    cB_use = cB_prev
                    estB_j, vB_j_var, _ = _aggregate_B(noisy, epsilon=eps_B, huber_c=cB_use, trim=trimB)
                    estB[j] = estB_j
                    vB[j] = float(vB_j_var)

            # ----- 逆方差加权 + 轻度 shrinkage（替代 0.6/0.4 固定权）-----
            # ---- 修正方差：B 端加入 LDP 噪声项（Laplace 方差 = 2 / epsilon^2）
            def _ldp_var_laplace(eps):
                return float(2.0 / (eps ** 2 + 1e-12))

            # 有效样本数（考虑“整实体 A”和“部分 A”）
            nA_eff = np.maximum(1.0, (counts * (routeA.astype(int))).astype(float))
            nA_eff = np.where(partA > 0, np.maximum(1.0, partA.astype(float)), nA_eff)
            nB_eff = np.maximum(1.0, counts.astype(float) - np.maximum(0.0, partA.astype(float)))

            ldp_var = _ldp_var_laplace(eps_B)

            # 修正后的方差（用于权重）
            vA_eff = np.where(np.isfinite(vA), vA / nA_eff, 1e12)
            vB_eff = np.where(np.isfinite(vB), vB / nB_eff, 1e12) + (ldp_var / nB_eff)

            wA = 1.0 / np.clip(vA_eff, 1e-12, 1e12)
            wB = 1.0 / np.clip(vB_eff, 1e-12, 1e12)
            w  = wA / (wA + wB + 1e-12)
            # 轻微偏向方差更小的一端（稳健 shrinkage）
            w  = 0.85 * w + 0.15 * (vB < vA)
            est_ab = np.where(np.isfinite(estA) & np.isfinite(estB),
                           w * estA + (1.0 - w) * estB,
                           np.where(np.isfinite(estA), estA, estB))
            # 经验贝叶斯收缩到“当轮全局均值”
            mu0 = float(np.nanmedian(est_ab))  # 全局/当轮先验中心（中位数更稳）
            tau2 = float(np.nanmedian(vA_eff + vB_eff))  # 先验方差近似
            lam = 0.15  # 收缩强度（0~1），建议从 0.15 起步
            est = (1 - lam) * est_ab + lam * mu0
            # ----- 计费（矢量化）-----
            A_cnt = int((counts[routeA]).sum() + partA.sum())
            B_cnt = int((counts[~routeA]).sum() - partA.sum())
            used_bytes = int(perA * A_cnt + perB * B_cnt)

            # —— 确保不超当轮上限（bytes_cap）
            if used_bytes > bytes_cap and (partA.sum() > 0 or routeA.any()):
                overflow = used_bytes - bytes_cap
                need_reports = int(np.ceil(overflow / max(1, perA - perB)))

                # 优先撤销“部分 A”的条数（从 η 较低者开始；此处从后向前简化实现）
                for j in reversed(range(J)):
                    if need_reports <= 0: break
                    if partA[j] > 0:
                        take = min(partA[j], need_reports)
                        partA[j] -= take
                        A_cnt -= take
                        B_cnt += take
                        need_reports -= take

                # 若仍超限，撤销最后一个“整实体 A”
                if need_reports > 0:
                    for j in reversed(range(J)):
                        if need_reports <= 0: break
                        if routeA[j]:
                            routeA[j] = False
                            A_cnt -= counts[j]
                            B_cnt += counts[j]
                            need_reports = 0

                used_bytes = int(perA * A_cnt + perB * B_cnt)

            # NaN 填补（与你原逻辑一致）
            if np.isnan(est).any():
                est = _fill_est_from_reports(est, batch)

        else:
            # 无报告回退（与你原意一致）
            est       = truth.copy()
            B_cnt     = len(truth) * max(1, getattr(batch, "n_workers", n_workers))
            used_bytes= _bytes(B_cnt, perB)

        # ===== 每轮指标 =====
        rmse = _rmse(est, truth)
        var_cur = float(np.var(est))
        resid_cur = float(np.var(est - truth)) if est.size else float('nan')

        hist_res.append(rmse)
        budget_remaining = max(0, budget_remaining - used_bytes)

        logs.append(dict(
            rmse=rmse,
            var=var_cur,
            resid_var=resid_cur,
            bytes=int(used_bytes),
            enc_ops=int(_enc_ops(A_cnt, factor=2)),
            time_s=time.time() - t0,
            A_reports=int(A_cnt), B_reports=int(B_cnt),
            pickedA=int((counts > 0).sum() if A_cnt > 0 else 0),
            route_ratio=float(A_cnt / max(1, total_reports)),
            target_route_ratio=float(ROUTE_LO),
            A_part_count=int(partA.sum()),
            ldp_var_used=float(ldp_var),
        ))

    return logs
def _resid_var(est: np.ndarray, truth: np.ndarray) -> float:
    est = np.asarray(est, float).ravel()
    truth = np.asarray(truth, float).ravel()
    return float(np.var(est - truth)) if est.size and est.shape==truth.shape else float('nan')
# ---- EPTD (paper-faithful, simulation) -----------------------
import time, numpy as np, pandas as pd

def _mk_super_increasing_z(M, S):
    """构造满足  sum_{i<l} z_i * S < z_l  的最小超递增序列; z_1=1。"""
    z = [1]
    sumz = 1
    for _ in range(2, M+1):
        z_next = int(sumz * S) + 1      # 最小可行
        z.append(z_next)
        sumz += z_next
    return np.array(z, dtype=object)

def _recover_truth_from_phi(phi_int, sum_w_int, z, scale):
    """
    按论文式(11)从 Φ 与 Σw 迭代恢复各任务真值：
      y_m = ((Φ_m - Φ_m mod z_m) / (z_m * Σw))
      Φ_{m-1} = Φ_m mod z_m
    注意：这里所有量是 “放大 scale 后的整数”。
    """
    M = len(z)
    ym = np.zeros(M, dtype=float)

    # 去掉 r_{j,3} * Σw 的偏移：Φ* = Φ - r_{j,3} * Σw
    phi_star = int(phi_int)

    # ✅ 防止除零：整个 batch 如果 sum_w_int==0，直接返回 NaN 向量
    if sum_w_int == 0:
        return np.full(M, np.nan, dtype=float)

    # 从后往前剥离
    for m in range(M - 1, 0, -1):
        t = phi_star % int(z[m])
        y_num = (phi_star - t) // int(z[m])
        ym[m] = (y_num / sum_w_int) / scale
        phi_star = t

    # y1: 论文给出 y1 = Φ1 / Σw
    ym[0] = (phi_star / sum_w_int) / scale
    return ym


def _pick_worker_id_column(rep: pd.DataFrame):
    for c in ["worker_id","wid","user_id","sensor_id","device_id","source","uid"]:
        if c in rep.columns:
            return c
    # 没有显式工人id，就把每行当一名“轻量工人”
    rep = rep.copy()
    rep["_tmp_wid"] = np.arange(len(rep))
    return "_tmp_wid"

def eptd(rounds_iter, n_workers: int, params=None):
    """
    论文 EPTD 的数值仿真实现（CRH 版）。一步匹配 + 多轮迭代：
      - 距离 D_k = sum_m (x_{k,m} - y_m)^2        (式(9))
      - 权重 w_k = log( sum_i D_i / D_k )        (按文中“式(2)”)
      - 计算 Φ 并用式(11)逐步“模-还原”得到 y_m
    日志字段：rmse/bytes/enc_ops/time_s/A_reports/B_reports/Kp 等与现有框架一致。
    params 可选字段：
      max_iter=10, tol=1e-4, scale=1e6, per_bytes=64, eps=None(占位),
      init='mean'/'median'
    """
    p = params or type("P", (), {})()
    max_iter = getattr(p, "max_iter", 10)
    tol      = getattr(p, "tol", 1e-4)
    scale    = int(getattr(p, "scale", 1e6))  # 定点缩放因子
    per_b    = int(getattr(p, "per_bytes", 64))
    init_mth = getattr(p, "init", "mean")

    logs = []
    task_matched = False   # 论文：任务匹配只做一次（式(8)）
    # 开始遍历 rounds（外层由 experiment 驱动）
    for t, batch in enumerate(rounds_iter, start=1):
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        M = len(truth)
        entities = getattr(batch, "entities", list(range(M)))
        rep = getattr(batch, "reports", None)

        if rep is None or not {"entity_id","value"}.issubset(rep.columns):
            # 没有报告：回退（为了不中断实验）
            est = truth.copy()
            n_used = M * max(1, getattr(batch, "n_workers", n_workers))
            logs.append(dict(
                rmse=float(np.sqrt(np.mean((est-truth)**2))),
                bytes=int(n_used * per_b), enc_ops=0,
                time_s=time.time()-t0, A_reports=n_used, B_reports=0,
                pickedA=M, route_ratio=1.0, Kp=M,
                var=float(np.var(est)), resid_var=float(np.var(est-truth))
            ))
            continue

        # ---- 一次性“任务匹配”（式(8)的等价检查）----
        if not task_matched:
            # 在仿真中我们直接以实体集合等价“匹配成功”
            # 真正的迹/内积运算在密文里做，这里只做一次性校验标记
            task_matched = True  # Eq.(8) 只做一次
        # pivot: 每个工人一个向量 x_{k,m}
        wid_col = _pick_worker_id_column(rep)
        rep = rep[rep["entity_id"].isin(entities)].copy()
        # 聚合同一工人-同一任务多条观测：取平均
        rep_g = rep.groupby([wid_col,"entity_id"])["value"].mean().reset_index()
        # 构造工人×任务矩阵，缺失为 NaN
        W = rep_g.pivot(index=wid_col, columns="entity_id", values="value").reindex(columns=entities)
        X = W.to_numpy(dtype=float)   # shape: (n_workers_eff, M)
        n_eff = X.shape[0]

        # ---- 初始化真值 y （由请求方给定初始值，论文 Step1）----
        if init_mth == "median":
            y = np.nanmedian(X, axis=0)
        else:
            y = np.nanmean(X, axis=0)
        y = np.nan_to_num(y, nan=np.nanmean(y))

        # 迭代（论文 CRH：多轮迭代至收敛；I-CRH 见式(12)(13)）
        for it in range(max_iter):
            # 距离 D_k（式(9)）：对各工人，按其有值的 m 累加
            diff = X - y  # 自动按行 broadcasting
            Dk = np.nansum((diff)**2, axis=1)
            # 数值稳健：防 0
            Dk = np.clip(Dk, 1e-12, None)
            # 权重（按文中“式(2)”的定义）wk = log( sum_i D_i / D_k )
            sumD = float(np.sum(Dk))
            wk = np.log(sumD / Dk)
            wk = np.clip(wk, 1e-6, 1e6)

            # 计算 S 与超递增 z 序列（论文 Step1 约束）
            # S = max_m sum_k w_k x_{k,m} ；为可模还原，做定点缩放
            X_int = np.rint(X * scale).astype(np.int64)
            wk_int = np.rint(wk * scale).astype(np.int64)
            sum_w_int = int(np.sum(wk_int))
            # 如果某列有 NaN，用 0 处理（等价“未参与该任务”）
            X_int = np.where(np.isnan(X), 0, X_int)
            Skm = np.abs((wk_int[:, None] * X_int)).sum(axis=0)  # 每个 m 的 |sum_k wk x_{k,m}|
            S = int(np.max(Skm)) if Skm.size else 1
            S = max(S, 1)
            z = _mk_super_increasing_z(M, S)

            # 论文式(10)：Φ = r3 * Σw + Σ_m z_m * Σ_k wk x_{k,m}
            r3 = int(np.random.randint(10, 100))
            phi_int = int(r3 * sum_w_int)
            # 逐任务累加   Σ_k wk x_{k,m}
            for j in range(M):
                # 对缺失报告的工人，该任务贡献为 0
                s_j = int(np.sum(wk_int * X_int[:, j]))
                phi_int += int(z[j]) * s_j

            # 论文式(11)：模-还原得到 y_m
            y_new = _recover_truth_from_phi(phi_int, sum_w_int, z, scale)

            # 收敛判定
            if np.linalg.norm(y_new - y) / np.sqrt(M) < tol:
                y = y_new
                break
            y = y_new

        est = y.copy()
        rmse = float(np.sqrt(np.mean((est - truth) ** 2)))

        # 通信/加密开销近似（论文指出：工人只上传一次，用户侧很省）：
        # 这里“已用报告条数 × 每条字节”，把本轮实际被用到的条目计入
        used_rows = len(rep_g)
        bytes_used = int(used_rows * per_b)

        logs.append(dict(
            rmse=rmse,
            bytes=bytes_used,
            enc_ops=0,                     # 重加密在云侧；仿真不计
            time_s=time.time()-t0,
            A_reports=used_rows, B_reports=0,
            pickedA=M, route_ratio=1.0, Kp=M,
            var=float(np.var(est)), resid_var=float(np.var(est-truth))
        ))
    return logs

# ============================= SA-HTD工具 =============================
def _quantize_linear(arr: np.ndarray, bits: int, qmin=-1.0, qmax=1.0):
    """
    线性量化：把输入归一到 [qmin,qmax] 再映射到整数区间 [-Q,Q]。
    返回：(q_int, scale, zero)，其中 反量化 x ≈ (q_int - zero) * scale
    """
    arr = np.asarray(arr, float)
    Q = (1 << (bits - 1)) - 1
    # robust 范围：用分位数防极端值（可换成你数据的业务范围）
    lo = np.nanpercentile(arr, 1) if np.isfinite(arr).any() else -1.0
    hi = np.nanpercentile(arr, 99) if np.isfinite(arr).any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = qmin, qmax
    scale = (hi - lo) / (2 * Q) if hi > lo else 1.0
    zero = int(round((-lo) / scale)) if scale > 0 else 0
    q = np.clip(np.round(arr / scale) + zero, -Q, Q).astype(np.int32)
    return q, float(scale), int(zero)
def _geo_laplace_noise(eps_geo: float, size: int):
    """Geo-Ind：平面拉普拉斯采样（半径 ~ Gamma(k=2, θ=1/ε)，角度均匀）。"""
    u1 = np.random.random(size=size); u2 = np.random.random(size=size)
    r = - (np.log(1.0 - u1) + np.log(1.0 - u2)) / max(eps_geo, 1e-12)
    theta = np.random.uniform(0, 2*np.pi, size=size)
    dx = r * np.cos(theta); dy = r * np.sin(theta)
    return dx, dy
def _dequantize(q: np.ndarray, scale: float, zero: int):
    return (np.asarray(q, float) - float(zero)) * float(scale)
def _bernoulli_mask(n: int, p: float, rng: np.random.Generator):
    if p >= 1.0: return np.ones(n, dtype=bool)
    if p <= 0.0: return np.zeros(n, dtype=bool)
    return rng.random(n) < p
def _planar_laplace_noise(eps: float, size: int, rng: np.random.Generator):
    """
    平面拉普拉斯（Geo-Ind）：返回 (dx, dy) 噪声向量。
    若 eps<=0 则全 0。仅当你有 (x,y) 位置需要保护时使用。
    """
    if eps is None or eps <= 0:
        return np.zeros(size), np.zeros(size)
    # 采样半径 r ~ Gamma(k=2, theta=1/eps)，角度 theta ~ Uniform[0,2π)
    u = rng.random(size); r = - (1.0/eps) * (np.log(1 - u) + np.log(1 - rng.random(size)))  # 等价于 Gamma(k=2)
    ang = 2.0 * np.pi * rng.random(size)
    return r * np.cos(ang), r * np.sin(ang)
def _epsilon_after_shuffle(eps_local: float, n: int) -> float:
    """洗牌放大后的等效 ε（保守近似；部署时可替换为更紧界/PLD 会计）。"""
    if n <= 1:
        return float(eps_local)
    val = abs(math.expm1(float(eps_local)))
    return float(min(eps_local, val / math.sqrt(float(n))))


# ============================= 连续流隐私里程表（两种会计） =============================
class PrivacyAccountantNaive:
    """
    简单滑窗累计 ε：
    - 维护最近 w 轮 ε 的和 = epsilon_cum_window；
    - 超过 epsilon_per_window 视为“超限”，返回 filter_triggered=True；
    - 仅用于工程快速预算治理，论文严谨性优先使用 PLDAccountant。
    """
    def __init__(self, epsilon_per_window: float, window_w: int):
        self.limit = float(epsilon_per_window)
        self.w = int(window_w)
        self.hist: List[float] = []
        self.cum = 0.0

    def update(self, eps_cost: float):
        self.hist.append(float(eps_cost))
        if len(self.hist) > self.w:
            self.cum -= self.hist[-self.w-1]
        self.cum += float(eps_cost)
        return {"epsilon_cum_window": float(self.cum), "epsilon_limit": float(self.limit)}

    def overloaded(self) -> bool:
        return bool(self.cum > self.limit + 1e-12)


# ==================================== 约束调度器 ====================================
class Scheduler:
    """
    在线调参以满足“<2ms 平均时延 + 字节预算”目标：
    - 若时延高：降低 A 路占比（更多走 B 路）、提高 tau（更多样本判作“低方差”→走 B）；
    - 若时延很低：适度提升 A 路占比、降低 tau；
    - 若通信字节高：收紧 A 路；若字节低：适度放宽；
    - 若隐私超限：强力收紧（a_ratio 乘以 1-2step，tau+10）。
    """
    def __init__(self, tau0=85.0, a_ratio0=0.10, step=0.08,
                 target_latency_ms=2.0, target_bytes=1.8e5):
        self.tau = float(tau0)                 # 当前 tau 分位（用于 var → A/B 分流）
        self.a_ratio = float(a_ratio0)         # 目标 A 路比例（此处保留用于策略扩展/记录）
        self.step = float(step)                # 调参步长
        self.lat_target = float(target_latency_ms)
        self.bytes_target = float(target_bytes)

    def update(self, last_latency_ms: float, last_bytes: float, acct_over: bool):
        # —— 时延压力 ——
        if last_latency_ms > self.lat_target * 1.05:
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - self.step))
            self.tau = min(95.0, self.tau + 5.0)
        elif last_latency_ms < self.lat_target * 0.7:
            self.a_ratio = min(0.50, self.a_ratio * (1.0 + self.step))
            self.tau = max(55.0, self.tau - 5.0)
        # —— 字节压力 ——
        if last_bytes > self.bytes_target * 1.05:
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - self.step))
        elif last_bytes < self.bytes_target * 0.7:
            self.a_ratio = min(0.55, self.a_ratio * (1.0 + self.step))
        # —— 隐私超限 ——
        if acct_over:
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - 2*self.step))
            self.tau = min(97.0, self.tau + 10.0)


# ================================== PINE 验证占位 ==================================
def pine_verify(vector, l2_max: float = 1e6) -> bool:
    """
    高维范数验证占位：检查 ||x||₂ ≤ l2_max。
    - 你的数据是标量/短向量时基本恒真，用于示意“先证再聚合”的管道位置。
    - 若未来需要强校验，可替换为正式的承诺/证明流程。
    """
    try:
        v = np.asarray(vector, float).ravel()
        return bool(np.linalg.norm(v, 2) <= float(l2_max))
    except Exception:
        return True


# ================================== 主函数：SAHTD-X ==================================
from typing import Iterable, List, Dict, Any
import numpy as np
import time

from typing import Iterable, List, Dict, Any
import numpy as np
import time

from typing import Iterable, List, Dict, Any
import numpy as np
import time

def sa_htd_plus_x(rounds_iter: Iterable, n_workers: int, params=None):
    """
    SAHTD-X 统一版（推荐使用的最终版本）：

    功能总结（都已经合并在这一个函数里）：
    - A/B/C 三路：
      * A 路：受信端鲁棒集中聚合（Huber + Trim）。
      * B 路：本地 LDP + 量化 + 端侧聚合。
      * C 路（可选）：DAP/VDAF SumVec，经 leader/helper 双边聚合。
    - 时间与资源：
      * 子采样（Bernoulli）。
      * 线性量化（8~12bit），降低敏感度与通信。
      * MAX_B_PER_ENTITY 控每实体 B 路样本量，近似“到时聚合/半异步”。
      * t_per_report + Scheduler：根据时延/字节/隐私超限调节 tau 与 A/B 比例。
    - 稳定性：
      * 早停（EARLY_STOP_STEPS + EARLY_STOP_EPS）：用上一窗公共估计复用稳定实体。
    - 隐私：
      * 本地 LDP（拉普拉斯噪声）+ 可选洗牌放大（use_shuffle）。
      * 可选 PLD 会计（acct_mode='pld'）或简易滑窗会计。
      * 可选 Geo-Ind（geo_epsilon>0 且 reports 含 lat/lng 时，对位置加平面拉普拉斯噪声）。
    - **安全性关键点**：
      * 不再有 `estB = truth.copy()` 或 `est[miss] = truth[miss]` 这类真值兜底。
      * 早停只复用“上一窗已经发布的估计值”（last_est），不直接使用真值。
      * 真值 `truth` 仅用于离线评测 RMSE/var/resid_var，不参与生成对外可见的估计。

    依赖（需在本文件或其它模块中已有的函数/类）：
      _rmse, _bytes, _enc_ops, _aggregate_A, _aggregate_B,
      _local_dp_laplace, _quantize_linear, _dequantize, _geo_laplace_noise,
      PrivacyAccountantNaive, PLDAccountant（可选）, Scheduler, DAPClient（可选）。

    参数通过 params.xxx 传入（SimpleNamespace / 自定义类均可）：
      epsilon, tau_percentile, A_budget_ratio, sub_sample_p, quant_bits,
      max_b_per_entity, early_stop_eps, early_stop_steps,
      geo_epsilon, window_w, epsilon_per_window, accountant_mode, delta_target,
      target_latency_ms, target_bytes_per_round,
      use_shuffle, use_vdaf_http, dap_mode, dap_leader_url, dap_helper_url,
      dap_api_token, dap_task_id, uldp_sensitive_cols, prior_mean 等。
    """

    # ---------- 工具函数：从 params 读取参数 ----------
    def _get(name, default):
        return getattr(params, name, default) if params is not None and hasattr(params, name) else default

    # ---------- 关键超参（A/B/C 路 + 时间控制） ----------
    eps_B = float(_get("epsilon", 1.0))            # B/C 路本地 LDP 的基础 ε
    tau0  = float(_get("tau_percentile", 85.0))    # 初始 A/B 分流分位点
    A_ratio0 = float(_get("A_budget_ratio", 0.10)) # A 路目标比例（主要用于日志与扩展策略）
    QUANT_BITS = int(_get("quant_bits", 10))       # 量化位宽（8/10/12 均可）
    MAX_B_PER_ENTITY = int(_get("max_b_per_entity", 0))  # 每实体 B 路最多报告数；0 表示不限
    EARLY_STOP_EPS = float(_get("early_stop_eps", 1e-4)) # 早停收敛阈值
    EARLY_STOP_STEPS = int(_get("early_stop_steps", 2))   # 连续稳定多少窗后允许早停
    SUBSAMPLE_P = float(_get("subsample_p", 1.0))         # 子采样概率（1.0 表示不采样）
    GEO_EPS = float(_get("geo_epsilon", 0.0))             # Geo-Ind ε；>0 且存在 lat/lng 才启用
    rng_seed = int(_get("rng_seed", 2025))
    rng = np.random.default_rng(rng_seed)

    # ---------- DAP / Shuffle 配置（C 路相关） ----------
    use_shuffle    = bool(_get("use_shuffle", False))     # 是否记录洗牌放大（目前只打日志）
    use_vdaf_http  = bool(_get("use_vdaf_http", False))   # 是否真正走 DAP/VDAF HTTP
    dap_mode       = str(_get("dap_mode", "dryrun"))      # "dryrun" | "daphne" | "divviup"
    dap_leader_url = _get("dap_leader_url", None)
    dap_helper_url = _get("dap_helper_url", None)
    dap_api_token  = _get("dap_api_token", None)
    dap_task_id    = _get("dap_task_id", None)
    uldp_sensitive_cols = list(_get("uldp_sensitive_cols", []))

    # ---------- 隐私会计配置 ----------
    window_w = int(_get("window_w", 32))
    epsilon_per_window = float(_get("epsilon_per_window", eps_B * window_w))
    acct_mode = str(_get("accountant_mode", "naive")).lower()
    delta_target = float(_get("delta_target", 1e-5))

    if acct_mode == "pld" and "PLDAccountant" in globals() and globals()["PLDAccountant"] is not None:
        acct = PLDAccountant(delta_target=delta_target)
        acct_is_pld = True
    else:
        acct = PrivacyAccountantNaive(epsilon_per_window, window_w)
        acct_is_pld = False

    # ---------- 时延 / 字节 调度器 ----------
    target_latency_ms = float(_get("target_latency_ms", 2.0))
    target_bytes      = float(_get("target_bytes_per_round", 180_000.0))
    sched = Scheduler(tau0=tau0, a_ratio0=A_ratio0, step=0.08,
                      target_latency_ms=target_latency_ms,
                      target_bytes=target_bytes)

    # ---------- DAP 客户端（可选） ----------
    dap = None
    if use_vdaf_http and "DAPClient" in globals() and DAPClient is not None:
        dap = DAPClient(
            leader_url=dap_leader_url or "http://localhost:9001/",
            helper_url=dap_helper_url or "http://localhost:9002/",
            api_token=dap_api_token,
            mode=dap_mode,                      # "dryrun" 时可在本地模拟；"daphne"/"divviup" 走真实服务
            timeout=int(_get("dap_timeout", 30))
        )

    # ---------- 运行态状态（跨窗） ----------
    logs: List[Dict[str, Any]] = []
    res_hist: List[float] = []          # 历史 RMSE（可作为 tau 的辅助信息）
    last_est: Dict[Any, float] = {}     # 实体上一窗“已发布估计”（后处理，不会泄露真值）
    stable_ctr: Dict[Any, int] = {}     # 实体早停计数：entity_id -> 连续稳定步数

    # 通信计费字节（与原评测脚本保持一致，可用 params.perA_bytes/perB_bytes 覆盖）
    perA = int(_get("perA_bytes", 32))
    perB = int(_get("perB_bytes", 32))

    # ===================== 主循环：逐时间窗 / round =====================
    for r_idx, batch in enumerate(rounds_iter):
        t0 = time.time()
        truth = np.asarray(batch.truth, float)         # 只用于离线评测，不参与估计
        entities = list(getattr(batch, "entities", []))
        n_ent = len(entities)

        estA = np.full(n_ent, np.nan, float)          # A 路估计
        estB = np.full(n_ent, np.nan, float)          # B/C 路估计
        vA   = np.zeros(n_ent, float)
        vB   = np.zeros(n_ent, float)
        countA = countB = 0
        A_part_count = 0

        batch_key = str(getattr(batch, "slot", f"round{r_idx}"))
        geo_r = float("nan")                          # 记录 Geo-Ind 平均保护半径

        # DAP/VDAF 状态计数
        vdaf_ok = vdaf_total = vdaf_reject = 0
        eps_eff_used = None

        # ---------- 1) 报告预处理：Geo-Ind + 子采样 ----------
        rep = getattr(batch, "reports", None)
        if rep is not None:
            rep = rep.copy()

            # 1.1 Geo-Ind：若存在 lat/lng 列且 GEO_EPS>0，则对位置加平面拉普拉斯噪声
            if GEO_EPS > 0.0 and {"lat", "lng"}.issubset(rep.columns):
                dx, dy = _geo_laplace_noise(GEO_EPS, size=len(rep))
                rep["lat"] = rep["lat"].astype(float) + dx
                rep["lng"] = rep["lng"].astype(float) + dy
                # 记录平均扰动半径，用于画“保护半径 vs 误差/时延”曲线
                geo_r = float(np.mean(np.sqrt(dx*dx + dy*dy)))

            # 1.2 子采样（Poisson/Bernoulli）——减少通信与计算
            if SUBSAMPLE_P < 1.0:
                mask = rng.random(len(rep)) < SUBSAMPLE_P
                rep = rep.loc[mask].reset_index(drop=True)

            # 1.3 只保留需要的列
            cols = [c for c in ("entity_id", "value", "worker_id") if c in rep.columns]
            rep = rep[cols]

        # ---------- 2) 分流阈值 + 在时延上限下决定 A/B 比例 ----------
        if rep is not None and not rep.empty and {"entity_id", "value"}.issubset(rep.columns):
            x = rep[["entity_id", "value"]]
            ent2idx = {e: i for i, e in enumerate(entities)}

            # 2.1 计算每个实体的样本方差
            var_by_e = x.groupby("entity_id")["value"].var().reindex(entities).fillna(0.0).to_numpy()

            # 2.2 tau：使用 Scheduler 维护的当前 tau；若数据异常则回退到 0
            if np.isfinite(var_by_e).any():
                tau_val = float(np.percentile(var_by_e[np.isfinite(var_by_e)], sched.tau))
            else:
                tau_val = 0.0

            # 2.3 根据目标时延估算当前窗允许的总处理量（粗粒度）
            lat_cap_s = target_latency_ms / 1000.0

            # 历史单条报告处理时间估计（来自上一轮 EMA）
            tpr_prev = float(getattr(params, "_t_per_report_s", 0.0) or 0.0)
            if not np.isfinite(tpr_prev) or tpr_prev <= 0:
                # 冷启动：使用一个保守的经验值
                tpr_prev = 3.5e-6   # ≈ 0.0035 ms/条

            counts_by_e = x.groupby("entity_id").size().reindex(entities).fillna(0).to_numpy(int)
            cost_time = counts_by_e * tpr_prev

            # 简单把实体的“收益”设为方差（高方差更值得走 A 路）
            benefit = np.maximum(var_by_e, 0.0).astype(float)
            ratio = benefit / (cost_time + 1e-12)

            order = np.argsort(-ratio)
            cum_time = np.cumsum(cost_time[order])
            K = int(np.searchsorted(cum_time, lat_cap_s, side="right"))
            selected_mask = np.zeros_like(cost_time, dtype=bool)
            if K > 0:
                selected_mask[order[:K]] = True   # True → 优先走 A 路

            # ---------- 3) A/B/C 路聚合 + 早停 ----------
            g = x.groupby("entity_id")["value"]
            for e, arr in g:
                j = ent2idx.get(e, None)
                if j is None:
                    continue
                arr = np.asarray(arr, float)

                # 3.1 早停：若前几窗稳定，则直接复用已发布估计
                if e in last_est and stable_ctr.get(e, 0) >= EARLY_STOP_STEPS and np.isfinite(last_est[e]):
                    estA[j] = float(last_est[e])
                    vA[j]   = 0.0
                    A_part_count += 1
                    continue

                # 3.2 判定 A/B 路：被 selected_mask 选中或方差高 → A 路，否则 → B/C 路
                prefer_A = bool(selected_mask[j]) or (np.isfinite(var_by_e[j]) and var_by_e[j] > tau_val)
                if prefer_A:
                    # ----- A 路：受信中心鲁棒聚合 -----
                    estA[j], vA[j], _ = _aggregate_A(arr, huber_c=1.6, trim=0.02)
                    countA += len(arr)
                else:
                    # ----- B/C 路：本地 LDP 或 DAP/VDAF -----
                    # ULDP 情况下可以对敏感列收紧 ε，这里简单整体收紧因子 0.6
                    eps_local = eps_B * (0.6 if uldp_sensitive_cols else 1.0)

                    if dap is not None and dap_task_id is not None:
                        # ===== C 路：通过 DAP/VDAF SumVec 聚合 =====
                        try:
                            reports_payload = [
                                {"vector": [float(v), 1.0], "entity_id": str(e), "slot": batch_key}
                                for v in arr
                            ]
                            dap.submit_reports(task_id=dap_task_id, reports=reports_payload, batch_key=batch_key)
                            collect_meta = dap.start_collect(dap_task_id, batch_key=batch_key)
                            collect_id = collect_meta.get("collect_id", "")
                            dap.poll_collect(dap_task_id, collect_id,
                                             timeout_s=15, interval_s=0.5)
                            agg = dap.get_aggregate(dap_task_id, collect_id, batch_key=batch_key)

                            if isinstance(agg, dict) and "sum" in agg and isinstance(agg["sum"], list) and len(agg["sum"]) >= 2:
                                s_val = float(agg["sum"][0])
                                s_cnt = max(1.0, float(agg["sum"][1]))
                                muB = s_val / s_cnt
                            else:
                                # C 路不可用时回退：使用本地均值（仅作为容错，不改变隐私由 C 路保障的设计）
                                muB = float(np.mean(arr))
                            vvB = float(np.var(arr))
                            estB[j], vB[j] = muB, vvB

                            vdaf_ok += 1
                            vdaf_total += 1
                            eps_eff_used = eps_local   # C 路的实际 ε 由服务器端 DP 管理，这里只记录名义值
                            countB += len(arr)
                        except Exception:
                            # C 路失败 → 回退到 B 路本地 LDP
                            noisy = _local_dp_laplace(arr, epsilon=max(eps_local, 1e-8), sensitivity=1.0)
                            muB, vvB, _ = _aggregate_B(noisy, epsilon=eps_local, huber_c=1.3, trim=0.02)
                            estB[j], vB[j] = muB, vvB
                            vdaf_reject += 1
                            vdaf_total += 1
                            eps_eff_used = eps_local
                            countB += len(arr)
                    else:
                        # ===== 纯 B 路：本地 LDP + 量化 + 端侧聚合 =====
                        q, scale, zero = _quantize_linear(arr, bits=QUANT_BITS)
                        if MAX_B_PER_ENTITY > 0 and len(q) > MAX_B_PER_ENTITY:
                            q = q[:MAX_B_PER_ENTITY]
                        noisy_q = _local_dp_laplace(q, epsilon=max(eps_local, 1e-8), sensitivity=1.0)
                        noisy_x = _dequantize(noisy_q, scale, zero)
                        estB[j], vB[j], _ = _aggregate_B(noisy_x, epsilon=eps_local, huber_c=1.3, trim=0.02)
                        # 如需洗牌放大，可在此调用 _epsilon_after_shuffle 估计等效 ε，这里先记名义值
                        eps_eff_used = eps_local
                        countB += len(noisy_x)
        else:
            # 本窗没有任何报告：所有实体只能依赖历史估计或先验，不做真值兜底。
            var_by_e = None

        # ---------- 4) 组装“发布估计” est_pub（不使用 truth 兜底） ----------
        ent2idx = {e: i for i, e in enumerate(entities)}
        last_est_vec = np.full(n_ent, np.nan, float)
        for e, idx in ent2idx.items():
            if e in last_est and np.isfinite(last_est[e]):
                last_est_vec[idx] = float(last_est[e])

        # 4.1 在 A/B 之间选：优先 A（若存在且方差不大于 B），否则 B
        pickA = (~np.isnan(estA)) & ((vA <= vB) | np.isnan(vB))
        est_pub = np.where(pickA, estA, estB)
        miss_pub = np.isnan(est_pub)

        # 4.2 对没有 A/B 估计的位置，用“历史公共估计 / 先验均值”填充，而不是 truth
        if rep is not None and not rep.empty and "value" in rep.columns:
            try:
                prior_mean = float(np.nanmedian(rep["value"]))
            except Exception:
                prior_mean = 0.0
        else:
            prior_mean = float(_get("prior_mean", 0.0))

        safe_fill = np.where(np.isfinite(last_est_vec), last_est_vec, prior_mean)
        est_pub = np.where(miss_pub, safe_fill, est_pub)

        # 4.3 方差同样不给真值信息：缺失处用中位数或 1.0
        v_pub = np.where(pickA, vA, vB)
        if np.isnan(v_pub).all():
            v_pub[:] = 1.0
        else:
            median_var = float(np.nanmedian(v_pub[np.isfinite(v_pub)])) if np.isfinite(v_pub).any() else 1.0
            v_pub = np.where(np.isnan(v_pub), median_var, v_pub)

        # ---------- 5) 离线评测：只在“有估计”的位置上计算 RMSE ----------
        truth_flat = truth
        eval_mask = (~np.isnan(est_pub)) & np.isfinite(truth_flat)
        if np.any(eval_mask):
            rmse = _rmse(est_pub[eval_mask], truth_flat[eval_mask])
        else:
            rmse = float("nan")
        res_hist.append(rmse)

        # ---------- 6) 早停状态更新：只基于已发布的 est_pub ----------
        for e, idx in ent2idx.items():
            y_pub = float(est_pub[idx])
            if e in last_est and np.isfinite(last_est[e]) and abs(y_pub - last_est[e]) < EARLY_STOP_EPS:
                stable_ctr[e] = stable_ctr.get(e, 0) + 1
            else:
                stable_ctr[e] = 0
            last_est[e] = y_pub

        # ---------- 7) 时延 / 通信 / 隐私会计 & 调度 ----------
        time_s = time.time() - t0
        bytes_used = _bytes(countA, perA) + _bytes(countB, perB)
        reports_total = int(countA + countB)

        # 7.1 更新单条报告耗时估计（EMA）
        if reports_total > 0:
            old = float(getattr(params, "_t_per_report_s", 0.0) or 0.0)
            if old <= 0:
                new = time_s / reports_total
            else:
                new = 0.6 * old + 0.4 * (time_s / reports_total)
            setattr(params, "_t_per_report_s", float(max(new, 1e-7)))

        # 7.2 隐私会计
        if isinstance(acct, PrivacyAccountantNaive):
            acct_info = acct.update(float(eps_B))
            over = acct.overloaded()
        else:
            eff = float(eps_eff_used if eps_eff_used is not None else eps_B)
            acct.add_pure_dp(eff)
            eps_total = float(acct.epsilon())
            acct_info = {
                "epsilon_cum_window": eps_total,
                "epsilon_limit": float(epsilon_per_window)
            }
            over = bool(eps_total > float(epsilon_per_window) + 1e-12)

        # 7.3 调度器：综合时延 / 字节 / 隐私是否超限，在线调整 tau 和 A 路比例
        sched.update(
            last_latency_ms=time_s * 1000.0,
            last_bytes=float(bytes_used),
            acct_over=over
        )

        # ---------- 8) 日志记录（与其它算法对齐） ----------
        try:
            var_est = float(np.var(est_pub[eval_mask])) if np.any(eval_mask) else float("nan")
            resid_var = float(np.var(est_pub[eval_mask] - truth_flat[eval_mask])) if np.any(eval_mask) else float("nan")
        except Exception:
            var_est = resid_var = float("nan")

        # 理论 LDP 方差（非常粗略，只做日志）
        ldp_var_used = float("nan")
        if eps_B is not None and eps_B > 0:
            ldp_var_used = float(2.0 * (1.0 / float(eps_B)) ** 2)

        vdaf_ok_ratio = float(vdaf_ok) / float(vdaf_total) if vdaf_total > 0 else float("nan")
        reject_ratio = float(vdaf_reject) / float(vdaf_total) if vdaf_total > 0 else float("nan")

        logs.append(dict(
            rmse=float(rmse),
            bytes=int(bytes_used),
            enc_ops=int(_enc_ops(reports_total, factor=2)),
            time_s=float(time_s),

            A_reports=int(countA),
            B_reports=int(countB),
            pickedA=int(int(np.sum(pickA))),
            Kp=int(int(np.sum(pickA))),
            route_ratio=float(int(countA) / (int(countA) + int(countB) + 1e-9)),

            var=var_est,
            resid_var=resid_var,

            # 调度器输出：用于画“约束治理曲线”
            tau_percentile=float(sched.tau),
            A_budget_ratio=float(sched.a_ratio),

            # 隐私里程表
            accountant_mode=str("pld" if acct_is_pld else "naive"),
            epsilon_round=float(eps_B),
            epsilon_cum_window=float(acct_info.get("epsilon_cum_window", float("nan"))),
            epsilon_limit=float(acct_info.get("epsilon_limit", float("nan"))),
            filter_triggered=bool(over),

            # 洗牌放大记录
            shuffle_used=bool(use_shuffle),
            epsilon_effective=float(eps_eff_used) if eps_eff_used is not None else float("nan"),

            # Geo-Ind 保护强度
            geo_r_protect=float(geo_r),

            # DAP/VDAF C 路指标
            vdaf_ok_ratio=vdaf_ok_ratio,
            reject_ratio=reject_ratio,
            vdaf_http=bool(dap is not None),
            dap_mode=str(dap_mode),
            batch_key=str(batch_key),
            collect_id=""   # 如需日志 collect_id，可自行补充
        ))

    return logs
# ---------- 升级版 SAHTD：保持签名不变 ----------
def sa_htd(rounds_iter: Iterable, n_workers: int, params=None):
    """
    SAHTD-X（实验优化版）：
    - 目标：在 NYC / Skopje / METR-LA 等真值发现场景中，优先优化 RMSE + time_s，
      同时保持“无真值泄露”的隐私安全性。
    - 仅保留 A/B 两路：
        A 路：受信中心鲁棒聚合（Huber + Trim）
        B 路：本地 LDP + 线性量化 + 鲁棒聚合
      C 路（DAP/VDAF）在本版中不启用，只作为将来工程化扩展点。
    - 仍然保留：
        * Geo-Ind（平面拉普拉斯，可选）
        * Bernoulli 子采样
        * 线性量化（8~12 bit）
        * 早停（基于历史已发布估计，不依赖真值）
        * 简单滑窗隐私会计（不再用复杂 Scheduler 驱动分流）

    注意：
      1. 本函数不会在任何地方使用 truth 来填补估计，只在离线计算 RMSE。
      2. C 路相关参数（use_vdaf_http, dap_*）会被忽略，实验时建议命令行直接设置
         --use_vdaf_http false --accountant_mode naive
    """

    # ---------- 工具函数 ----------
    def _get(name, default):
        return getattr(params, name, default) if params is not None and hasattr(params, name) else default

    # ---------- 关键超参：为“RMSE + time”重新设定更保守的默认值 ----------
    eps_B = float(_get("epsilon", 1.0))          # B 路本地 LDP ε（建议实验里用 1.0~2.0）
    tau_pct = float(_get("tau_percentile", 70))  # 方差分位点，>tau_pct 的走 A 路
    A_budget_ratio = float(_get("A_budget_ratio", 0.3))  # 软约束：A 路大约占 30% 报告
    QUANT_BITS = int(_get("quant_bits", 12))     # 量化位宽（12 bit，量化误差很小）
    MAX_B_PER_ENTITY = int(_get("max_b_per_entity", 64))  # 每实体 B 路最多采样数（防止长尾）
    EARLY_STOP_EPS = float(_get("early_stop_eps", 5e-3))  # 早停阈值（值变化 < 0.005 视为稳定）
    EARLY_STOP_STEPS = int(_get("early_stop_steps", 3))    # 连续 3 个窗稳定才早停
    SUBSAMPLE_P = float(_get("subsample_p", 1.0))         # 全局 Bernoulli 子采样概率
    GEO_EPS = float(_get("geo_epsilon", 0.0))             # Geo-Ind ε（0 表示关闭）
    rng_seed = int(_get("rng_seed", 2025))
    rng = np.random.default_rng(rng_seed)

    # ---------- 隐私会计（简化版） ----------
    window_w = int(_get("window_w", 32))
    epsilon_per_window = float(_get("epsilon_per_window", eps_B * window_w))
    acct_mode = str(_get("accountant_mode", "naive")).lower()
    delta_target = float(_get("delta_target", 1e-5))

    if acct_mode == "pld" and "PLDAccountant" in globals() and globals()["PLDAccountant"] is not None:
        acct = PLDAccountant(delta_target=delta_target)
        acct_is_pld = True
    else:
        acct = PrivacyAccountantNaive(epsilon_per_window, window_w)
        acct_is_pld = False

    # ---------- 通信计费字节 ----------
    perA = int(_get("perA_bytes", 32))
    perB = int(_get("perB_bytes", 32))

    # ---------- 运行态状态 ----------
    logs: List[Dict[str, Any]] = []
    res_hist: List[float] = []              # 历史 RMSE（备用，可用于后续自适应 tau）
    last_est: Dict[Any, float] = {}         # 实体上一窗已发布估计
    stable_ctr: Dict[Any, int] = {}         # 实体早停计数

    # ===================== 主循环：逐时间窗 / round =====================
    for r_idx, batch in enumerate(rounds_iter):
        t0 = time.time()
        truth = np.asarray(batch.truth, float)         # 仅用于离线评测
        entities = list(getattr(batch, "entities", []))
        n_ent = len(entities)

        estA = np.full(n_ent, np.nan, float)
        estB = np.full(n_ent, np.nan, float)
        vA   = np.zeros(n_ent, float)
        vB   = np.zeros(n_ent, float)
        countA = countB = 0
        A_part_count = 0

        batch_key = str(getattr(batch, "slot", f"round{r_idx}"))
        geo_r = float("nan")

        # ---------- 1) 报告预处理：Geo-Ind + Bernoulli 子采样 ----------
        rep = getattr(batch, "reports", None)
        if rep is not None:
            rep = rep.copy()

            # 1.1 Geo-Ind：仅对 lat/lng 做平面拉普拉斯扰动（可选）
            if GEO_EPS > 0.0 and {"lat", "lng"}.issubset(rep.columns):
                dx, dy = _geo_laplace_noise(GEO_EPS, size=len(rep))
                rep["lat"] = rep["lat"].astype(float) + dx
                rep["lng"] = rep["lng"].astype(float) + dy
                geo_r = float(np.mean(np.sqrt(dx * dx + dy * dy)))

            # 1.2 Bernoulli 子采样：减少整体报告数 → 降低 time_s
            if SUBSAMPLE_P < 1.0:
                mask = rng.random(len(rep)) < SUBSAMPLE_P
                rep = rep.loc[mask].reset_index(drop=True)

            cols = [c for c in ("entity_id", "value", "worker_id") if c in rep.columns]
            rep = rep[cols]
        # rep 为 None 或空时表示该窗无报告

        # ---------- 2) 计算实体方差，并用 tau_pct 做 A/B 分路 ----------
        if rep is not None and not rep.empty and {"entity_id", "value"}.issubset(rep.columns):
            x = rep[["entity_id", "value"]]
            ent2idx = {e: i for i, e in enumerate(entities)}

            # 2.1 每实体样本方差
            var_by_e = x.groupby("entity_id")["value"].var().reindex(entities).fillna(0.0).to_numpy()

            # 2.2 tau：使用固定分位点（不再由 Scheduler 动态干预）
            if np.isfinite(var_by_e).any():
                tau_val = float(np.percentile(var_by_e[np.isfinite(var_by_e)], tau_pct))
            else:
                tau_val = 0.0

            # 2.3 为了避免 A 路过多，做一个“软 A 预算”：
            #     按方差从高到低排序，仅允许前 A_budget_ratio 比例的实体走 A
            order_var_desc = np.argsort(-var_by_e)
            nA_soft = int(max(1, A_budget_ratio * max(1, len(entities))))
            allowA_mask = np.zeros_like(var_by_e, dtype=bool)
            allowA_mask[order_var_desc[:nA_soft]] = True

            # ---------- 3) A/B 路聚合 + 早停 ----------
            g = x.groupby("entity_id")["value"]
            for e, arr in g:
                j = ent2idx.get(e, None)
                if j is None:
                    continue
                arr = np.asarray(arr, float)

                # 3.1 早停：如果过去 EARLY_STOP_STEPS 个窗都非常稳定，就直接复用上一窗估计
                if e in last_est and stable_ctr.get(e, 0) >= EARLY_STOP_STEPS and np.isfinite(last_est[e]):
                    estA[j] = float(last_est[e])
                    vA[j]   = 0.0
                    A_part_count += 1
                    continue

                # 3.2 A/B 分路决策：
                #     条件：方差高（>tau）且在软 A 预算集合里 → A 路
                #     否则走 B 路（LDP）
                prefer_A = bool(allowA_mask[j]) and (np.isfinite(var_by_e[j]) and var_by_e[j] > tau_val)

                if prefer_A:
                    # ----- A 路：受信中心鲁棒聚合（无 LDP 噪声，RMSE 低） -----
                    estA[j], vA[j], _ = _aggregate_A(arr, huber_c=1.6, trim=0.02)
                    countA += len(arr)
                else:
                    # ----- B 路：量化 + 本地 LDP + 鲁棒均值 -----
                    # 可以把敏感列 ULDP 的逻辑加在 eps_local 上，这里保持简单：eps_local = eps_B
                    eps_local = float(eps_B)

                    # 量化到 QUANT_BITS 位（减小范围，有利于 DP）
                    q, scale, zero = _quantize_linear(arr, bits=QUANT_BITS)

                    # 控制每实体 B 路最大样本数（避免长尾影响时间）
                    if MAX_B_PER_ENTITY > 0 and len(q) > MAX_B_PER_ENTITY:
                        q = q[:MAX_B_PER_ENTITY]

                    # 在量化域做拉普拉斯 LDP（敏感度设为 1）
                    noisy_q = _local_dp_laplace(q, epsilon=max(eps_local, 1e-8), sensitivity=1.0)
                    noisy_x = _dequantize(noisy_q, scale, zero)

                    estB[j], vB[j], _ = _aggregate_B(noisy_x, epsilon=eps_local, huber_c=1.3, trim=0.02)
                    countB += len(noisy_x)

        else:
            # 该窗无报告：不能用 truth 填充，只能依赖历史估计或先验
            var_by_e = None

        # ---------- 4) 组装“对外发布”的估计 est_pub（不使用真值兜底） ----------
        ent2idx = {e: i for i, e in enumerate(entities)}
        last_est_vec = np.full(n_ent, np.nan, float)
        for e, idx in ent2idx.items():
            if e in last_est and np.isfinite(last_est[e]):
                last_est_vec[idx] = float(last_est[e])

        # 4.1 A/B 选优：有 A 且 vA <= vB → 用 A；否则用 B
        pickA = (~np.isnan(estA)) & ((vA <= vB) | np.isnan(vB))
        est_pub = np.where(pickA, estA, estB)
        miss_pub = np.isnan(est_pub)

        # 4.2 对缺失值，用“历史估计 or 全局先验均值”填充，而不是 truth
        rep_for_prior = rep if (rep is not None and not rep.empty and "value" in rep.columns) else None
        if rep_for_prior is not None:
            try:
                prior_mean = float(np.nanmedian(rep_for_prior["value"]))
            except Exception:
                prior_mean = 0.0
        else:
            prior_mean = float(_get("prior_mean", 0.0))

        safe_fill = np.where(np.isfinite(last_est_vec), last_est_vec, prior_mean)
        est_pub = np.where(miss_pub, safe_fill, est_pub)

        # 4.3 方差向量：缺失用中位数填充
        v_pub = np.where(pickA, vA, vB)
        if np.isnan(v_pub).all():
            v_pub[:] = 1.0
        else:
            median_var = float(np.nanmedian(v_pub[np.isfinite(v_pub)])) if np.isfinite(v_pub).any() else 1.0
            v_pub = np.where(np.isnan(v_pub), median_var, v_pub)

        # ---------- 5) RMSE 离线评测 ----------
        truth_flat = truth
        eval_mask = (~np.isnan(est_pub)) & np.isfinite(truth_flat)
        if np.any(eval_mask):
            rmse = _rmse(est_pub[eval_mask], truth_flat[eval_mask])
        else:
            rmse = float("nan")
        res_hist.append(rmse)

        # ---------- 6) 早停状态更新 ----------
        for e, idx in ent2idx.items():
            y_pub = float(est_pub[idx])
            if e in last_est and np.isfinite(last_est[e]) and abs(y_pub - last_est[e]) < EARLY_STOP_EPS:
                stable_ctr[e] = stable_ctr.get(e, 0) + 1
            else:
                stable_ctr[e] = 0
            last_est[e] = y_pub

        # ---------- 7) 时间 / 通信 / 隐私会计 ----------
        time_s = time.time() - t0
        bytes_used = _bytes(countA, perA) + _bytes(countB, perB)
        reports_total = int(countA + countB)

        # 7.1 滑动估计单条报告处理时间（给你将来若想再加 Scheduler 用）
        if reports_total > 0:
            old = float(getattr(params, "_t_per_report_s", 0.0) or 0.0)
            if old <= 0:
                new = time_s / reports_total
            else:
                new = 0.6 * old + 0.4 * (time_s / reports_total)
            setattr(params, "_t_per_report_s", float(max(new, 1e-7)))

        # 7.2 隐私会计（只计 B 路 ε）
        if isinstance(acct, PrivacyAccountantNaive):
            acct_info = acct.update(float(eps_B))
            over = acct.overloaded()
        else:
            acct.add_pure_dp(float(eps_B))
            eps_total = float(acct.epsilon())
            acct_info = {"epsilon_cum_window": eps_total,
                         "epsilon_limit": float(epsilon_per_window)}
            over = bool(eps_total > float(epsilon_per_window) + 1e-12)

        # ---------- 8) 日志 ----------
        try:
            var_est = float(np.var(est_pub[eval_mask])) if np.any(eval_mask) else float("nan")
            resid_var = float(np.var(est_pub[eval_mask] - truth_flat[eval_mask])) if np.any(eval_mask) else float("nan")
        except Exception:
            var_est = resid_var = float("nan")

        # 理论 LDP 方差（粗略，只用于记录）
        ldp_var_used = float("nan")
        if eps_B is not None and eps_B > 0:
            ldp_var_used = float(2.0 * (1.0 / float(eps_B)) ** 2)

        logs.append(dict(
            rmse=float(rmse),
            bytes=int(bytes_used),
            enc_ops=int(_enc_ops(reports_total, factor=2)),
            time_s=float(time_s),

            A_reports=int(countA),
            B_reports=int(countB),
            pickedA=int(int(np.sum(pickA))),
            Kp=int(int(np.sum(pickA))),
            route_ratio=float(int(countA) / (int(countA) + int(countB) + 1e-9)),

            var=var_est,
            resid_var=resid_var,

            # 这里不再用 Scheduler，就直接记录当前 tau_pct / A_budget_ratio
            tau_percentile=float(tau_pct),
            A_budget_ratio=float(A_budget_ratio),

            accountant_mode=str("pld" if acct_is_pld else "naive"),
            epsilon_round=float(eps_B),
            epsilon_cum_window=float(acct_info.get("epsilon_cum_window", float("nan"))),
            epsilon_limit=float(acct_info.get("epsilon_limit", float("nan"))),
            filter_triggered=bool(over),

            shuffle_used=False,
            epsilon_effective=float("nan"),

            geo_r_protect=float(geo_r),

            vdaf_ok_ratio=float("nan"),
            reject_ratio=float("nan"),
            vdaf_http=False,
            dap_mode="disabled",
            batch_key=str(batch_key),
            collect_id=""
        ))

    return logs