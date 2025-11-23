# -*- coding: utf-8 -*-
"""
algorithms_bridge.py
=============================
严格桥接原则：
- 不对你的算法做**任何简化或替代实现**；
- 只从 user_algorithms.py 导入并调用你提供的方法（如：newsahtd / new_sa_htd_budgeted / eptd / 其它基线）；
- 若某方法在 user_algorithms.py 中不存在，则**跳过该方法**（给出提示），而不是用“我这边的简化版本”强行替代；
- 统一在**外部**补齐评测列（rmse/var/resid_var）与**通信字节**（payload_bytes 精确汇总）；若你已提供，则原样保留。

额外提供：etbp_td_strict（来自 etbp_td.py），供你需要 ETBP‑TD 时直接使用。
"""

from __future__ import annotations
from typing import Iterable, Optional, Dict, Any, List, Sequence, Callable, Tuple
import importlib, time, numpy as np, pandas as pd

# =============== 外部评测工具（不改你算法内部） ===============
def _purge_knobs_on_batch(batch):
    for k in ("A_budget_ratio", "tau_percentile"):
        if hasattr(batch, k):
            try: delattr(batch, k)
            except Exception: pass
    return batch

def _rmse(est, truth) -> float:
    est = np.asarray(est, float).ravel()
    truth = np.asarray(truth, float).ravel()
    return float(np.sqrt(np.mean((est - truth) ** 2))) if est.size and est.shape==truth.shape else float('nan')

def _resid_var(est, truth) -> float:
    est = np.asarray(est, float).ravel()
    truth = np.asarray(truth, float).ravel()
    return float(np.var(est - truth)) if est.size and est.shape==truth.shape else float('nan')

def _bytes_sum_from_rep(rep: Optional[pd.DataFrame], subset_mask=None, subset_idx=None,
                        subset_workers: Optional[Sequence]=None, subset_entities: Optional[Sequence]=None,
                        fallback_per: int = 16) -> int:
    if rep is None or (isinstance(rep, pd.DataFrame) and rep.empty): return 0
    df = rep
    if subset_workers is not None:  df = df[df["worker_id"].isin(list(subset_workers))]
    if subset_entities is not None: df = df[df["entity_id"].isin(list(subset_entities))]
    if subset_idx is not None:      df = df.iloc[list(subset_idx)]
    elif subset_mask is not None:   df = df[subset_mask]
    if "payload_bytes" in df.columns: return int(df["payload_bytes"].sum())
    return int(len(df) * max(0, fallback_per))

def _enc_ops_from_used(rep: Optional[pd.DataFrame], log: Dict[str,Any], factor_default=2) -> int:
    used_cnt = 0
    if log.get("used_mask") is not None:
        used_cnt = int(np.asarray(log["used_mask"]).sum())
    elif log.get("used_idx") is not None:
        used_cnt = int(len(log["used_idx"]))
    elif rep is not None and log.get("used_worker_ids") is not None:
        used_cnt = int(rep[rep["worker_id"].isin(log["used_worker_ids"])].shape[0])
    elif rep is not None and log.get("used_entity_ids") is not None:
        used_cnt = int(rep[rep["entity_id"].isin(log["used_entity_ids"])].shape[0])
    return int(used_cnt * factor_default)

# =============== 桥接入口（你的 NewSAHTD; 以及对比法） ===============

def _load_user_algorithms():
    try:
        # 相对导入：从当前包 dataset_use.NYC.src.algorithms 下面导入 user_algorithms
        from . import codex as UA
        import importlib
        return importlib.reload(UA)
    except Exception as e:
        raise RuntimeError(f"无法导入 user_algorithms.py：{e}")

def bridge_call(rounds_iter: Iterable, n_workers:int, func_names: Sequence[str], params=None) -> List[Dict[str,Any]]:
    """
    严格桥接：依次尝试从 user_algorithms 中寻找给定的函数名，找到即调用；找不到则抛错。
    - 不提供简化替代实现。
    - 返回 logs（list[dict]），后续仅做**外部评测补齐**。
    """
    UA = _load_user_algorithms()
    # 若 rounds_iter 是 _Spy，并且之前已经缓存过一些 batch，这里再统一清一次（双保险）
    if hasattr(rounds_iter, "batches"):
        for _b in list(getattr(rounds_iter, "batches") or []):
            _purge_knobs_on_batch(_b)

    core = None
    for name in func_names:
        if hasattr(UA, name):
            core = getattr(UA, name)
            break
    if core is None:
        raise RuntimeError(f"在 user_algorithms 中未找到任何一个入口：{func_names}")

    logs = core(rounds_iter, n_workers, params)

    # 只读补齐缺失指标/通信字节；若你已给出，对应键将保留
    enriched = []
    for i, log in enumerate(logs):
        # 评测列（若你提供 est，则补 rmse/var/resid_var；若已有则跳过）
        if ("rmse" not in log or "var" not in log or "resid_var" not in log) and ("est" in log):
            est = np.asarray(log["est"], float)
            truth = None
            if hasattr(rounds_iter, "batches") and i < len(rounds_iter.batches):
                truth = getattr(rounds_iter.batches[i], "truth", None)
            # 若 rounds_iter 是普通生成器，真值无法外取，这种情况只补通信字节，不补指标
            if truth is not None:
                if "rmse" not in log:      log["rmse"] = _rmse(est, truth)
                if "var" not in log:       log["var"] = float(np.var(est)) if est.size else float('nan')
                if "resid_var" not in log: log["resid_var"] = _resid_var(est, truth)

        # 通信字节（若缺失则按 used_* 或全量 payload_bytes 精确求和）；enc_ops 同理
        if hasattr(rounds_iter, "batches") and i < len(rounds_iter.batches):

            rep = getattr(rounds_iter.batches[i], "reports", None)
        else:
            rep = None
        if "bytes" not in log:
            log["bytes"] = _bytes_sum_from_rep(
                rep,
                subset_mask     = log.get("used_mask"),
                subset_idx      = log.get("used_idx"),
                subset_workers  = log.get("used_worker_ids"),
                subset_entities = log.get("used_entity_ids"),
            )
        if "enc_ops" not in log:
            log["enc_ops"] = _enc_ops_from_used(rep, log, factor_default=2)

        enriched.append(log)
    return enriched

# 便捷封装：你的 NewSAHTD
def sa_htd_paper_bridge(rounds_iter: Iterable, n_workers:int, params=None):
    # 兼容两种常见命名
    return bridge_call(_Spy(rounds_iter), n_workers, func_names=["sa_htd_paper"], params=params)

# 便捷封装：若你在 user_algorithms 中提供了 eptd/etbp_td 等，也可以直接桥接
def eptd_bridge(rounds_iter: Iterable, n_workers:int, params=None):
    return bridge_call(_Spy(rounds_iter), n_workers, func_names=["eptd", "etbp_td", "ETBP_TD"], params=params)

# 便捷封装：其它对比算法（名称示例：pure_ldp/ud_ldp/dplp/fed_sense/robust_b/alpha_opt/random_baseline 等）
def generic_bridge(rounds_iter: Iterable, n_workers:int, func_name: str, params=None):
    return bridge_call(_Spy(rounds_iter), n_workers, func_names=[func_name], params=params)
def sahtd_paper_bridge(rounds_iter: Iterable, n_workers:int, params=None):
    return bridge_call(_Spy(rounds_iter), n_workers, func_names=["sa_htd_paper"], params=params)
# 辅助：探针式迭代器（保存每轮 Batch，便于外部统计）
# 辅助：探针式迭代器（保存每轮 Batch，便于外部统计）
class _Spy:
    def __init__(self, it):
        self._it = iter(it)
        self.batches = []

    def __iter__(self):
        return self

    def __next__(self):
        # 在算法真正拿到 batch 之前，先做“入口清扫”
        b = next(self._it)
        _purge_knobs_on_batch(b)     # ★★★ 关键新增：清掉 A_budget_ratio / tau_percentile
        self.batches.append(b)
        return b

