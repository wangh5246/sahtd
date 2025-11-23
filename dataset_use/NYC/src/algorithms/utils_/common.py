from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, List, Sequence
import numpy as np
import pandas as pd
import time, math

def rmse(est, truth):
    est = np.asarray(est, float).ravel(); truth = np.asarray(truth, float).ravel()
    return float(np.sqrt(np.mean((est - truth) ** 2))) if est.size and est.shape == truth.shape else float('nan')

def huber(x, c=1.345):
    x = np.asarray(x, float)
    if x.size == 0: return float('nan')
    med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-9
    r = (x - med) / (1.4826 * mad); w = np.clip(c / (np.abs(r) + 1e-12), 0.0, 1.0)
    return float(np.sum(w * x) / (np.sum(w) + 1e-12))

def ldp_laplace(x, epsilon: float, sens: float = 1.0):
    if epsilon <= 0: raise ValueError("epsilon must be > 0")
    return np.asarray(x, float) + np.random.laplace(0.0, sens / float(epsilon), size=len(x))
# ---------------- 公用度量（评测用途，不改变算法内部数据） ----------------


def resid_var(est: np.ndarray, truth: np.ndarray) -> float:
    est = np.asarray(est, float).ravel()
    truth = np.asarray(truth, float).ravel()
    return float(np.var(est - truth)) if est.size and est.shape==truth.shape else float('nan')

# ---------------- 通信/加密开销计量（只读） ----------------

def bytes_sum_from_rep(rep: Optional[pd.DataFrame],
                        subset_mask=None, subset_idx=None,
                        subset_workers: Optional[Sequence]=None,
                        subset_entities: Optional[Sequence]=None,
                        fallback_per: int = 16) -> int:
    """
    精确统计当轮**实际使用**的报告通信字节：
    - 优先 sum(payload_bytes)；否则 len(subset)×fallback_per；
    - 子集选择通过 used_mask / used_idx / used_worker_ids / used_entity_ids 之一给出。
    """
    if rep is None or rep.empty: return 0
    df = rep
    if subset_workers is not None:  df = df[df["worker_id"].isin(list(subset_workers))]
    if subset_entities is not None: df = df[df["entity_id"].isin(list(subset_entities))]
    if subset_idx is not None:      df = df.iloc[list(subset_idx)]
    elif subset_mask is not None:   df = df[subset_mask]
    if "payload_bytes" in df.columns: return int(df["payload_bytes"].sum())
    return int(len(df) * max(0, fallback_per))

def enc_ops_by_count(n_reports: int, factor: int) -> int:
    """按条数×系数估计加密/矩阵运算次数（计费口径）"""
    return int(max(0, n_reports) * max(0, factor))