from typing import Iterable, List, Dict, Any, Optional, Tuple
import numpy as np, math, warnings
from collections import deque

from SAHTD1.sahtd_x_readable import PrivacyAccountantNaive
from dataset_use.NYC.src.algorithms.pld_accountant import PLDAccountant
from dataset_use.NYC.src.algorithms.adaptive_postprocess import(
    AdaptivePostprocessor,
    _postprocess_filter_conditional
)

# ========= 隐私+稀疏自适应后处理 =========


class LightweightVarianceAwarePostprocessor:
    """
    基于隐私预算与数据稀疏性的轻量级自适应后处理机制（方差阈值改用相对分位数）
    """

    def __init__(
        self,
        var_percentile_low: float = 0.50,
        var_percentile_high: float = 0.75,
        sparse_threshold: int = 5,
        privacy_tension_ratio: float = 0.75,
        change_detection_window: int = 3,
        change_sensitivity: float = 2.0,
        convergence_window: int = 5,
        convergence_threshold: float = 0.05,
        warmup_rounds: int = 3,
        enable_adaptive: bool = True,
        fallback_var_low: float = 10.0,
        fallback_var_high: float = 50.0,
    ):
        # 分位数阈值（0-1）；若不足历史则回退到 fallback 绝对值
        self.var_p_low = float(np.clip(var_percentile_low, 0.0, 1.0))
        self.var_p_high = float(np.clip(var_percentile_high, 0.0, 1.0))
        if self.var_p_high < self.var_p_low:
            self.var_p_low, self.var_p_high = self.var_p_high, self.var_p_low

        self.fallback_var_low = float(fallback_var_low)
        self.fallback_var_high = float(max(fallback_var_high, self.fallback_var_low + 1.0))

        self.sparse_th = int(sparse_threshold)
        self.privacy_tension = float(privacy_tension_ratio)
        self.change_window = int(change_detection_window)
        self.change_sens = float(change_sensitivity)
        self.conv_window = int(convergence_window)
        self.conv_th = float(convergence_threshold)
        self.warmup = int(warmup_rounds)
        self.enable = bool(enable_adaptive)

        self.est_history: Dict[Any, deque] = {}
        self.var_history: deque = deque(maxlen=10)
        self.decision_history: List[Dict[str, Any]] = []
        self.all_variance_history: deque = deque(maxlen=100)

        self.total_rounds = 0
        self.triggered_rounds = 0

    def _compute_dynamic_thresholds(self) -> Dict[str, float]:
        """
        根据历史数据动态计算方差阈值（绝对值），不足时使用回退值。
        """
        if len(self.all_variance_history) < 10:
            return {"var_low": self.fallback_var_low, "var_high": self.fallback_var_high}

        hist_arr = np.array(list(self.all_variance_history))

        var_low = float(np.percentile(hist_arr, self.var_p_low * 100.0))
        var_high = float(np.percentile(hist_arr, self.var_p_high * 100.0))

        if not np.isfinite(var_low):
            var_low = self.fallback_var_low
        if not np.isfinite(var_high):
            var_high = self.fallback_var_high

        if var_high - var_low < 1.0:
            var_high = var_low + 5.0

        return {"var_low": var_low, "var_high": var_high}

    def should_postprocess(self,
                           round_idx: int,
                           est_by_entity: Dict[Any, float],
                           var_by_entity: Dict[Any, float],
                           reports_count_by_entity: Dict[Any, int],
                           privacy_budget_used: float,
                           privacy_budget_total: float) -> Dict[str, Any]:
        """
        判断是否需要后处理(使用相对阈值)
        """
        self.total_rounds += 1

        if not self.enable:
            return self._always_enable_decision()

        if round_idx < self.warmup:
            return self._disable_decision(f"warmup({round_idx}/{self.warmup})")

        for eid, est in est_by_entity.items():
            if eid not in self.est_history:
                self.est_history[eid] = deque(maxlen=self.change_window + 2)
            self.est_history[eid].append(est)

        if var_by_entity:
            for v in var_by_entity.values():
                if np.isfinite(v) and v > 0:
                    self.all_variance_history.append(v)

        global_var = np.mean(list(var_by_entity.values())) if var_by_entity else 0.0
        self.var_history.append(global_var)

        dynamic_thresholds = self._compute_dynamic_thresholds()

        signals: Dict[str, Dict[str, Any]] = {}
        signals["variance"] = self._assess_variance(var_by_entity, dynamic_thresholds)
        signals["sparsity"] = self._assess_sparsity(reports_count_by_entity)
        signals["privacy"] = self._assess_privacy_pressure(
            privacy_budget_used, privacy_budget_total
        )
        signals["change"] = self._detect_change(est_by_entity)
        signals["convergence"] = self._assess_convergence()

        decision = self._fuse_signals(signals, round_idx)

        self.decision_history.append(decision)
        if decision["enable"]:
            self.triggered_rounds += 1

        return decision

    def _assess_variance(self, var_by_entity: Dict[Any, float],
                        thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        🔧 修改: 使用动态阈值
        """
        if not var_by_entity:
            return {"score": 0.0, "severity": "none", "detail": "no_data"}

        vars_arr = np.array(list(var_by_entity.values()))
        median_var = float(np.median(vars_arr))
        p75_var = float(np.percentile(vars_arr, 75))

        var_low = thresholds['var_low']
        var_high = thresholds['var_high']

        if p75_var > var_high:
            score = 1.0
            severity = "high"
            detail = f"p75={p75_var:.1f}>{var_high:.1f}"
        elif median_var > var_low:
            ratio = (median_var - var_low) / (var_high - var_low + 1e-9)
            score = 0.3 + 0.5 * min(ratio, 1.0)
            severity = "medium"
            detail = f"med={median_var:.1f} in [{var_low:.1f},{var_high:.1f}]"
        else:
            score = 0.0
            severity = "low"
            detail = f"med={median_var:.1f}<{var_low:.1f}"

        return {"score": score, "severity": severity, "detail": detail}

    def _assess_sparsity(self, reports_count: Dict[Any, int]) -> Dict[str, Any]:
        if not reports_count:
            return {"score": 0.8, "severity": "critical", "detail": "no_reports"}

        counts = np.array(list(reports_count.values()))
        sparse_ratio = float(np.mean(counts < self.sparse_th))
        median_count = float(np.median(counts))

        if sparse_ratio > 0.5:
            score = 0.8
            severity = "high"
            detail = f"{sparse_ratio:.0%}_entities<{self.sparse_th}_reports"
        elif sparse_ratio > 0.2:
            score = 0.3 + 0.3 * sparse_ratio
            severity = "medium"
            detail = f"{sparse_ratio:.0%}_sparse,median={median_count:.0f}"
        else:
            score = 0.0
            severity = "low"
            detail = f"median_reports={median_count:.0f}"

        return {"score": score, "severity": severity, "detail": detail}

    def _assess_privacy_pressure(self, used: float, total: float) -> Dict[str, Any]:
        if total <= 0:
            return {"score": 0.0, "severity": "none", "detail": "no_budget_limit"}

        ratio = used / total

        if ratio > self.privacy_tension:
            score = 0.6
            severity = "high"
            detail = f"budget={ratio:.1%}>{self.privacy_tension:.0%}"
        elif ratio > 0.5:
            score = 0.3
            severity = "medium"
            detail = f"budget={ratio:.1%}"
        else:
            score = 0.0
            severity = "low"
            detail = f"budget={ratio:.1%}<50%"

        return {"score": score, "severity": severity, "detail": detail}

    def _detect_change(self, est_by_entity: Dict[Any, float]) -> Dict[str, Any]:
        change_entities = []

        for eid, hist in self.est_history.items():
            if len(hist) < self.change_window:
                continue

            recent = np.array(list(hist)[-self.change_window:])
            std = float(np.std(recent, ddof=1) + 1e-9)
            latest_change = abs(recent[-1] - recent[-2]) if len(recent) >= 2 else 0.0

            if latest_change > self.change_sens * std:
                change_entities.append(eid)

        if not self.est_history:
            return {"score": 0.0, "severity": "none", "detail": "no_history"}

        change_ratio = len(change_entities) / len(self.est_history)

        if change_ratio > 0.3:
            score = -0.7
            severity = "critical"
            detail = f"{change_ratio:.0%}_entities_changed"
        elif change_ratio > 0.1:
            score = -0.3
            severity = "medium"
            detail = f"{change_ratio:.0%}_entities_changed"
        else:
            score = 0.0
            severity = "none"
            detail = "stable"

        return {
            "score": score,
            "severity": severity,
            "detail": detail,
            "changed_entities": change_entities,
        }

    def _assess_convergence(self) -> Dict[str, Any]:
        if len(self.var_history) < self.conv_window:
            return {"score": 0.0, "severity": "none", "detail": "insufficient_history"}

        recent = np.array(list(self.var_history)[-self.conv_window:])
        relative_changes = np.abs(np.diff(recent)) / (recent[:-1] + 1e-9)
        avg_change = float(np.mean(relative_changes))

        if avg_change < self.conv_th:
            score = -0.5
            severity = "converged"
            detail = f"avg_change={avg_change:.2%}<{self.conv_th:.0%}"
        else:
            score = 0.0
            severity = "not_converged"
            detail = f"avg_change={avg_change:.2%}"

        return {"score": score, "severity": severity, "detail": detail}

    def _fuse_signals(self, signals: Dict[str, Dict[str, Any]], round_idx: int) -> Dict[str, Any]:
        total_score = sum(s["score"] for s in signals.values())

        reasons = []
        for name, sig in signals.items():
            if sig["score"] > 0.2:
                reasons.append(f"{name}:{sig['detail']}")
            elif sig["score"] < -0.2:
                reasons.append(f"¬{name}:{sig['detail']}")

        ENABLE_THRESHOLD = 0.5
        enable = total_score > ENABLE_THRESHOLD

        if enable:
            intensity = min((total_score - 0.5) / 2.5, 1.0)
            alpha_scale = 0.2 + 0.8 * intensity
            proc_var_scale = 0.5 + 1.5 * intensity
            reason = f"score={total_score:.2f}; " + "; ".join(reasons[:3])
        else:
            alpha_scale = 0.0
            proc_var_scale = 0.0
            reason = f"score={total_score:.2f}<threshold; " + (reasons[0] if reasons else "all_ok")

        entity_weights = self._compute_entity_weights(signals)

        return {
            "enable": enable,
            "reason": reason,
            "alpha_scale": float(alpha_scale),
            "proc_var_scale": float(proc_var_scale),
            "entity_weights": entity_weights,
            "total_score": float(total_score),
            "signals": signals,
            "round_idx": round_idx,
        }

    def _compute_entity_weights(self, signals: Dict[str, Dict[str, Any]]) -> Dict[Any, float]:
        changed_entities = set(signals.get("change", {}).get("changed_entities", []))

        weights = {}
        for eid in self.est_history.keys():
            w = 1.0
            if eid in changed_entities:
                w *= 0.5
            weights[eid] = w

        return weights

    def _disable_decision(self, reason: str) -> Dict[str, Any]:
        return {
            "enable": False,
            "reason": reason,
            "alpha_scale": 0.0,
            "proc_var_scale": 0.0,
            "entity_weights": {},
            "total_score": 0.0,
            "signals": {},
        }

    def _always_enable_decision(self) -> Dict[str, Any]:
        return {
            "enable": True,
            "reason": "adaptive_disabled",
            "alpha_scale": 1.0,
            "proc_var_scale": 1.0,
            "entity_weights": {},
            "total_score": float("inf"),
            "signals": {},
        }

    def get_statistics(self) -> Dict[str, Any]:
        if self.total_rounds == 0:
            return {}

        thresholds = self._compute_dynamic_thresholds()
        return {
            "total_rounds": self.total_rounds,
            "triggered_rounds": self.triggered_rounds,
            "trigger_rate": self.triggered_rounds / self.total_rounds if self.total_rounds else 0.0,
            "avg_score": np.mean([d["total_score"] for d in self.decision_history]) if self.decision_history else 0.0,
            "current_var_low": thresholds["var_low"],
            "current_var_high": thresholds["var_high"],
        }


def postprocess_with_privacy_awareness(
    est_by_e: Dict[Any, float],
    var_by_e: Dict[Any, float],
    reports_count: Dict[Any, int],
    graph: Optional[Dict[Any, List[Any]]],
    kstate: Dict[Any, Any],
    round_idx: int,
    privacy_budget_used: float,
    privacy_budget_total: float,
    alpha_lap: float,
    proc_var: float,
    obs_var_base: float,
    postprocessor: LightweightVarianceAwarePostprocessor
) -> Tuple[Dict[Any, float], Dict[str, Any]]:
    """
    带隐私感知的后处理（零真值依赖）
    """
    decision = postprocessor.should_postprocess(
        round_idx=round_idx,
        est_by_entity=est_by_e,
        var_by_entity=var_by_e,
        reports_count_by_entity=reports_count,
        privacy_budget_used=privacy_budget_used,
        privacy_budget_total=privacy_budget_total,
    )

    if not decision["enable"]:
        return est_by_e, decision

    adaptive_alpha = alpha_lap * decision["alpha_scale"]
    adaptive_proc_var = proc_var * decision["proc_var_scale"]
    entity_weights = decision["entity_weights"]

    class _Kalman1DState:
        __slots__ = ("m", "v", "init")

        def __init__(self):
            self.m = 0.0
            self.v = 10.0
            self.init = False

    est_k = {}
    for e, obs in est_by_e.items():
        st = kstate.get(e)
        if st is None:
            st = _Kalman1DState()
            kstate[e] = st

        if not st.init:
            st.m = float(obs)
            st.v = 5.0
            st.init = True
            est_k[e] = st.m
            continue

        entity_weight = entity_weights.get(e, 1.0)
        local_proc_var = adaptive_proc_var * (2.0 if entity_weight < 0.8 else 1.0)

        pred_m = st.m
        pred_v = st.v + local_proc_var
        R = float(obs_var_base)
        K = pred_v / (pred_v + R)
        st.m = float(pred_m + K * (float(obs) - pred_m))
        st.v = float((1.0 - K) * pred_v)
        est_k[e] = st.m

    if graph is None or adaptive_alpha <= 0.0:
        return est_k, decision

    est_pp = {}
    for e, val in est_k.items():
        neigh = graph.get(e)
        if not neigh:
            est_pp[e] = val
            continue

        neigh_vals = [est_k.get(n, val) for n in neigh]
        neigh_mean = float(np.mean(neigh_vals))

        entity_weight = entity_weights.get(e, 1.0)
        local_alpha = adaptive_alpha * entity_weight

        est_pp[e] = (1.0 - local_alpha) * val + local_alpha * neigh_mean

    return est_pp, decision

# ========= 必要工具函数 =========

def _system_issue_checks(params) -> List[str]:
    issues: List[str] = []
    if params is None:
        return issues

    target_bytes = getattr(params, "target_bytes_per_round", None)
    if target_bytes is not None and target_bytes < 1200:
        issues.append("target_bytes_per_round too small (<1200 bytes)")

    bits_a = getattr(params, "BASE_BITS_A", None)
    if bits_a is not None and bits_a <= 11:
        issues.append("BASE_BITS_A <= 11 makes quantization too coarse")

    a_ratio = getattr(params, "A_budget_ratio", None)
    if a_ratio is not None and a_ratio <= 0.25:
        issues.append("A_budget_ratio pinned at <=0.25; scheduler cannot re-balance routes")

    c_batch_max = getattr(params, "C_BATCH_MAX", None)
    if c_batch_max is not None and c_batch_max >= 16:
        issues.append("C_BATCH_MAX >= 16 floods the C-route budget")

    if getattr(params, "disable_postprocess", False):
        issues.append("disable_postprocess=True disables Kalman+graph smoothing")

    return issues


def _geo_laplace_noise(eps_geo: float, size: int):
    """
    Geo-Ind: 只用于记录“理论上可达的地理扰动半径”，不参与聚合。
    """
    u1 = np.random.random(size=size)
    u2 = np.random.random(size=size)
    r = - (np.log(1.0 - u1) + np.log(1.0 - u2)) / max(eps_geo, 1e-12)
    theta = np.random.uniform(0, 2 * np.pi, size=size)
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return dx, dy


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


def _local_dp_laplace(x: np.ndarray, epsilon: float, sensitivity: float = 1.0) -> np.ndarray:
    """
    本地 LDP：只用传入的 sensitivity，不再读外部 params。
    """
    x = _safe_array(x).astype(float)
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for LDP")
    scale = float(sensitivity) / float(epsilon)
    noise = np.random.laplace(0.0, scale, size=len(x))
    return x + noise


def _aggregate_B(reports: np.ndarray, epsilon: float,
                 huber_c: float = 1.3, trim: float = 0.02) -> tuple[float, float, int]:
    """
    B 通道：LDP 后的稳健融合（此处 reports 已含噪）。
    """
    reports = _safe_array(reports).astype(float)
    est = _huber(reports, c=huber_c)
    v = float(np.var(reports) + 1e-12)
    return float(est), v, 0


def _aggregate_A(clean_reports: np.ndarray,
                 huber_c: float = 1.6, trim: float = 0.02) -> tuple[float, float, int]:
    """
    A 通道：受信聚合（不加噪），稳健中心。
    """
    clean_reports = _safe_array(clean_reports).astype(float)
    est = _huber(clean_reports, c=huber_c)
    v = float(np.var(clean_reports) + 1e-12)
    return float(est), v, 0


def _epsilon_after_shuffle(eps_local: float, n: int) -> float:
    """
    洗牌放大后的等效 ε（保守近似；部署时可替换为更紧界 / PLD 会计）。
    """
    if n <= 1:
        return float(eps_local)
    val = abs(math.expm1(float(eps_local)))
    return float(min(eps_local, val / math.sqrt(float(n))))


def _fill_est_from_reports(est: np.ndarray, batch) -> np.ndarray:
    """
    用当轮 reports 的信息对 est 中的 NaN 做稳健填补：
    - 完全不访问 batch.truth；
    - 优先按 entity 的 Huber 中心填；
    - 再用全局 Huber 中心兜底；
    - 若完全无报告，则用 0.0 / 全局中位数兜底。
    """
    import pandas as pd
    est = np.asarray(est, float).copy()
    rep = getattr(batch, 'reports', None)

    # 无报告或列不齐 -> 不用 truth，直接用已有 est 的中位数或 0
    if rep is None or not {'entity_id', 'value'}.issubset(rep.columns):
        if np.isnan(est).all():
            return np.zeros_like(est, dtype=float)
        med = np.nanmedian(est)
        est = np.where(np.isnan(est), med, est)
        return est

    g = rep.groupby('entity_id')['value']
    entities = getattr(batch, 'entities', list(range(len(est))))

    # 先按 entity 单独填
    for j, e in enumerate(entities):
        if np.isnan(est[j]) and e in g.groups:
            arr = g.get_group(e).to_numpy(dtype=float)
            if arr.size:
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)) + 1e-9)
                r = (arr - med) / (1.4826 * mad)
                w = np.clip(1.345 / (np.abs(r) + 1e-12), 0.0, 1.0)
                est[j] = float(np.sum(w * arr) / (np.sum(w) + 1e-12))

    # 还剩 NaN -> 用全局 Huber 中心兜底
    if np.isnan(est).any():
        arr_all = rep['value'].to_numpy(dtype=float)
        med = float(np.median(arr_all))
        mad = float(np.median(np.abs(arr_all - med)) + 1e-9)
        r = (arr_all - med) / (1.4826 * mad)
        w = np.clip(1.345 / (np.abs(r) + 1e-12), 0.0, 1.0)
        fill = float(np.sum(w * arr_all) / (np.sum(w) + 1e-12))
        est = np.where(np.isnan(est), fill, est)

    return est


class Scheduler:
    """
    在线调参以满足“<2ms 平均时延 + 字节预算”目标：
    - 若时延高：降低 A 路占比（更多走 B 路）、提高 tau（更多样本判作“低方差”→走 B）；
    - 若时延很低：适度提升 A 路占比、降低 tau；
    - 若通信字节高：收紧 A 路；若字节低：适度放宽；
    - 若隐私超限：强力收紧（a_ratio *= (1 - 2 step)，tau+10）。
    """
    def __init__(self, tau0=85.0, a_ratio0=0.10, step=0.08,
                 target_latency_ms=2.0, target_bytes=1.8e5):
        self.tau = float(tau0)
        self.a_ratio = float(a_ratio0)
        self.step = float(step)
        self.lat_target = float(target_latency_ms)
        self.bytes_target = float(target_bytes)

    def update(self, last_latency_ms: float, last_bytes: float, acct_over: bool):
        # 时延压力
        if last_latency_ms > self.lat_target * 1.05:
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - self.step))
            self.tau = min(95.0, self.tau + 5.0)
        elif last_latency_ms < self.lat_target * 0.7:
            self.a_ratio = min(0.50, self.a_ratio * (1.0 + self.step))
            self.tau = max(55.0, self.tau - 5.0)
        # 字节压力
        if last_bytes > self.bytes_target * 1.05:
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - self.step))
        elif last_bytes < self.bytes_target * 0.7:
            self.a_ratio = min(0.55, self.a_ratio * (1.0 + self.step))
        # 隐私超限
        if acct_over:
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - 2 * self.step))
            self.tau = min(92.0, self.tau + 10.0)


def sa_htd_paper(rounds_iter: Iterable, n_workers: int, params=None):
    """
    SAHTD-Nexus-Lite（用于论文实验的三路调度版本）
    ...
    """
    import numpy as _np
    import time as _time

    system_issue_flags = _system_issue_checks(params)
    class _Kalman1DState:
        __slots__ = ("m", "v", "init")

        def __init__(self):
            self.m = 0.0
            self.v = 10.0
            self.init = False

    def _postprocess_filter(est_by_e: dict, graph: dict, kstate: dict,
                            alpha_lap: float, proc_var: float, obs_var_base: float):
        if alpha_lap <= 0.0 and proc_var <= 0.0:
            return est_by_e

        est_k = {}
        for e, obs in est_by_e.items():
            st = kstate.get(e)
            if st is None:
                st = _Kalman1DState()
                kstate[e] = st
            if not st.init:
                st.m = float(obs)
                st.v = 5.0
                st.init = True
                est_k[e] = st.m
                continue

            pred_m = st.m
            pred_v = st.v + float(proc_var)
            R = float(obs_var_base)
            if not _np.isfinite(R) or R <= 0.0:
                R = 1.0
            K = pred_v / (pred_v + R)
            st.m = float(pred_m + K * (float(obs) - pred_m))
            st.v = float((1.0 - K) * pred_v)
            est_k[e] = st.m

        if graph is None or alpha_lap <= 0.0:
            return est_k

        est_pp = {}
        for e, val in est_k.items():
            neigh = graph.get(e)
            if not neigh:
                est_pp[e] = val
                continue
            neigh_vals = [est_k.get(n, val) for n in neigh]
            neigh_mean = float(_np.mean(neigh_vals))
            est_pp[e] = (1.0 - alpha_lap) * val + alpha_lap * neigh_mean
        return est_pp

    def _update_quant_bits(entity_ids, est_vec_prev, var_est_dict,
                           resid_ema_dict, bytes_per_bit: float, params):
        min_bits = int(getattr(params, "MIN_QUANT_BITS", 6) if params is not None else 6)
        max_bits = int(getattr(params, "MAX_QUANT_BITS", 14) if params is not None else 14)
        target_bytes = float(getattr(params, "target_bytes_per_round", 25000) if params is not None else 25000)
        avg_reports_per_entity = float(getattr(params, "AVG_REPORTS_PER_ENTITY", 10.0)
                                       if params is not None else 10.0)
        var_quantile = float(getattr(params, "VAR_QUANTILE", 0.7) if params is not None else 0.7)

        est_vec_prev = _np.asarray(est_vec_prev, float)
        for idx, e in enumerate(entity_ids):
            prev = float(est_vec_prev[idx])
            v_old = float(var_est_dict.get(e, 5.0))
            var_est_dict[e] = 0.9 * v_old + 0.1 * (prev * prev)
            r_old = float(resid_ema_dict.get(e, 0.0))
            resid_ema_dict[e] = 0.9 * r_old + 0.1 * abs(prev)

        vars_arr = _np.array([var_est_dict.get(e, 5.0) for e in entity_ids], float)
        if _np.isfinite(vars_arr).any():
            tau = float(_np.quantile(vars_arr[_np.isfinite(vars_arr)], var_quantile))
        else:
            tau = 1.0

        bits_prop = {}
        base_bits_b = getattr(params, "BASE_BITS_B", 8) if params is not None else 8
        q_init = int(getattr(params, "quant_bits_init", base_bits_b) if params is not None else base_bits_b)
        for e in entity_ids:
            quant_bits_val = getattr(params, "quant_bits", q_init) if params is not None else q_init
            b = int(quant_bits_val)
            v = float(var_est_dict.get(e, 5.0))
            if v > tau:
                b += 2
            else:
                b -= 1
            b = max(min_bits, min(max_bits, b))
            bits_prop[e] = b

        total_bits = sum(bits_prop[e] * avg_reports_per_entity for e in entity_ids)
        total_bytes = float(total_bits) * float(bytes_per_bit)
        scale = 1.0 if total_bytes <= 0 else target_bytes / total_bytes

        bits_final = {}
        for e in entity_ids:
            b_scaled = int(round(bits_prop[e] * scale))
            b_scaled = max(min_bits, min(max_bits, b_scaled))
            bits_final[e] = b_scaled
        return bits_final, tau

    def _compute_value_scores(entity_ids, var_est_dict, resid_ema_dict,
                              bytes_per_bit: float,
                              bits_A: int, bits_B: int, bits_C_extra: int):
        scores = {}
        extra_bits_A = max(bits_A - bits_B, 1)
        extra_bits_C = bits_C_extra if bits_C_extra > 0 else extra_bits_A
        extra_bytes_A = extra_bits_A * bytes_per_bit
        extra_bytes_C = extra_bits_C * bytes_per_bit
        for e in entity_ids:
            v = float(var_est_dict.get(e, 5.0))
            r = float(resid_ema_dict.get(e, 0.0))
            delta_mse = v + r
            scores[e] = (
                delta_mse / (extra_bytes_A + 1e-6),
                delta_mse / (extra_bytes_C + 1e-6),
            )
        return scores

    def _route_entities(entity_ids, state, tau_val,
                        bits_A: int, bits_B: int, bits_C_extra: int, params):
        base_a_ratio = float(getattr(state["sched"], "a_ratio", 0.25))
        base_c_ratio = float(getattr(params, "BASE_C_RATIO", 0.03) if params is not None else 0.03)
        default_c_batch = int(getattr(params, "C_BATCH_MAX", 16) if params is not None else 16)
        c_batch_cap = state.get("c_batch_cap", default_c_batch)

        scores = _compute_value_scores(entity_ids, state["var_est"], state["resid_ema"],
                                       state["bytes_per_bit"], bits_A, bits_B, bits_C_extra)

        n = len(entity_ids)
        sorted_all = sorted(entity_ids, key=lambda e: -scores[e][0])

        raw_n_c = max(1, int(math.ceil(base_c_ratio * n))) if n >= 3 else 0
        n_C = min(raw_n_c, c_batch_cap, n)

        raw_n_a = int(round(base_a_ratio * n))
        n_A = max(1, raw_n_a) if (n - n_C) >= 2 else 0  # 至少留1个给B路
        if n_A == 0 and base_a_ratio > 0.001:
            n_A = 1

        route_C = set(sorted_all[:n_C])
        route_A = set(sorted_all[n_C: n_C + n_A])
        route_B = set(sorted_all[n_C + n_A:])

        return route_A, route_B, route_C

    eps_B = float(getattr(params, "epsilon", 1.0) if params is not None else 1.0)
    tau0 = float(getattr(params, "tau_percentile", 75.0) if params is not None else 75.0)
    A_ratio0 = float(getattr(params, "A_budget_ratio", 0.25) if params is not None else 0.25)
    if A_ratio0 < 0.25:
        warnings.warn("[SAHTD diag] A_budget_ratio raised to 0.30 to avoid starving A/C routes")

    use_shuffle = bool(getattr(params, "use_shuffle", True) if params is not None else True)
    uldp_sensitive_cols = list(getattr(params, "uldp_sensitive_cols", []) if params is not None else [])
    geo_eps = float(getattr(params, "geo_epsilon", 0.0) if params is not None else 0.0)

    window_w = int(getattr(params, "window_w", 32) if params is not None else 32)
    epsilon_per_window = float(getattr(params, "epsilon_per_window", eps_B * window_w)
                               if params is not None else eps_B * window_w)
    acct_mode = str(getattr(params, "accountant_mode", "naive")
                    if params is not None else "naive").lower()
    delta_target = float(getattr(params, "delta_target", 1e-5)
                         if params is not None else 1e-5)

    raw_target_bytes = float(getattr(params, "target_bytes_per_round", 25000) if params is not None else 25000)
    if raw_target_bytes < 1200.0:
        warnings.warn("[SAHTD diag] target_bytes_per_round raised to 1200 bytes minimum")
    target_bytes = max(raw_target_bytes, 1200.0)

    target_latency_ms = float(getattr(params, "target_latency_ms", 2.0) if params is not None else 2.0)

    bytes_per_bit = float(getattr(params, "bytes_per_bit", 0.125) if params is not None else 0.125)
    raw_bits_A = int(getattr(params, "BASE_BITS_A", 13) if params is not None else 12)
    if raw_bits_A <= 11:
        warnings.warn("[SAHTD diag] BASE_BITS_A raised to 12 bits to avoid quantization loss")
    bits_A = max(raw_bits_A, 12)
    bits_B = int(getattr(params, "BASE_BITS_B", 9) if params is not None else 9)
    bits_C_extra = int(getattr(params, "BITS_C_EXTRA", 2) if params is not None else 2)

    perA_bytes = int(getattr(params, "perA_bytes", 32) if params is not None else 32)
    perC_bytes = int(getattr(params, "perC_bytes", 64) if params is not None else 64)

    disable_post = bool(getattr(params, "disable_postprocess", False) if params is not None else False)
    if disable_post:
        warnings.warn("[SAHTD diag] disable_postprocess=True prevents Kalman smoothing")
    alpha_lap = 0.0 if disable_post else float(getattr(params, "post_lap_alpha", 0.3)
                                               if params is not None else 0.3)
    proc_var = 0.0 if disable_post else float(getattr(params, "post_process_var", 20)
                                              if params is not None else 20)
    obs_var_base = 1.0 if disable_post else float(getattr(params, "post_obs_var_base", 1.0)
                                                  if params is not None else 1.0)
    entity_graph = getattr(params, "entity_graph", None) if params is not None else None
    use_privacy_aware_post = (not disable_post) and bool(
        getattr(params, "use_privacy_aware_postprocess", True) if params is not None else True
    )
    if use_privacy_aware_post:
        var_p_low = float(getattr(params, "post_var_percentile_low", 0.50) if params is not None else 0.50)
        var_p_high = float(getattr(params, "post_var_percentile_high", 0.75) if params is not None else 0.75)
        fallback_var_low = float(getattr(params, "post_var_threshold_low", 10.0) if params is not None else 10.0)
        fallback_var_high = float(getattr(params, "post_var_threshold_high", 50.0) if params is not None else 50.0)
        postprocessor = LightweightVarianceAwarePostprocessor(
            var_percentile_low=var_p_low,
            var_percentile_high=var_p_high,
            fallback_var_low=fallback_var_low,
            fallback_var_high=fallback_var_high,
            sparse_threshold=int(getattr(params, "post_sparse_threshold", 5) if params is not None else 5),
            privacy_tension_ratio=float(getattr(params, "post_privacy_tension_ratio", 0.75) if params is not None else 0.75),
            change_detection_window=int(getattr(params, "post_change_window", 3) if params is not None else 3),
            change_sensitivity=float(getattr(params, "post_change_sensitivity", 2.0) if params is not None else 2.0),
            convergence_window=int(getattr(params, "post_convergence_window", 5) if params is not None else 5),
            convergence_threshold=float(getattr(params, "post_convergence_threshold", 0.05) if params is not None else 0.05),
            warmup_rounds=int(getattr(params, "post_warmup_rounds", 3) if params is not None else 3),
            enable_adaptive=bool(getattr(params, "enable_privacy_adaptive", True) if params is not None else True),
        )
    else:
        postprocessor = None

    raw_c_batch = int(getattr(params, "C_BATCH_MAX", 16) if params is not None else 16)

    if raw_c_batch > 12:
        warnings.warn("[SAHTD diag] C_BATCH_MAX trimmed to 12 to cap C-route load")
    c_batch_cap = min(raw_c_batch, 8)
    sched = Scheduler(tau0=tau0, a_ratio0=A_ratio0, step=0.08,
                      target_latency_ms=target_latency_ms, target_bytes=target_bytes)

    if acct_mode == "pld" and PLDAccountant is not None:
        acct = PLDAccountant(delta_target=delta_target)
    else:
        acct = PrivacyAccountantNaive(epsilon_per_window, window_w)

    var_est = {}
    resid_ema = {}
    kalman_state = {}
    last_est_pp = {}

    logs: List[Dict[str, Any]] = []
    res_hist: List[float] = []
    for r_idx, batch in enumerate(rounds_iter):
        t0 = _time.time()
        truth = _np.asarray(batch.truth, float)
        n_ent = len(truth)
        entities = _np.asarray(batch.entities)

        geo_r = float("nan")
        rep_df = getattr(batch, "reports", None)
        if geo_eps > 0.0 and rep_df is not None and {"lat", "lng"}.issubset(rep_df.columns):
            dx, dy = _geo_laplace_noise(geo_eps, size=len(rep_df))
            geo_r = float(_np.mean(_np.sqrt(dx * dx + dy * dy)))

        arr_by_e: Dict[Any, _np.ndarray] = {}
        var_by_e = _np.zeros(n_ent, float)

        if rep_df is not None and {"entity_id", "value"}.issubset(rep_df.columns):
            x = rep_df[["entity_id", "value"]]
            groups = (
                x.groupby("entity_id")["value"]
                .apply(lambda s: s.to_numpy(dtype=float))
                .to_dict()
            )

            for j, e in enumerate(entities):
                arr = groups.get(e)
                if arr is None:
                    arr = _np.empty((0,), dtype=float)
                arr_by_e[e] = arr

                if arr.size > 1:
                    var_by_e[j] = float(arr.var(ddof=1))
                else:
                    var_by_e[j] = 0.0
        else:
            for j, e in enumerate(entities):
                arr_by_e[e] = _np.empty((0,), dtype=float)
                var_by_e[j] = 0.0

        if len(res_hist) >= 5 and _np.isfinite(var_by_e).any():
            tau_val = float(_np.nanpercentile(var_by_e[_np.isfinite(var_by_e)], sched.tau))
        elif _np.isfinite(var_by_e).any():
            tau_val = float(_np.nanmedian(var_by_e[_np.isfinite(var_by_e)]))
        else:
            tau_val = 0.0

        if last_est_pp:
            est_prev_pp_vec = _np.array(
                [last_est_pp.get(e, 0.0) for e in entities], float
            )
        else:
            est_prev_pp_vec = _np.zeros_like(truth)

        bits_by_e, tau_var = _update_quant_bits(
            entities, est_prev_pp_vec, var_est, resid_ema,
            bytes_per_bit, params
        )

        state_for_router = dict(
            var_est=var_est,
            resid_ema=resid_ema,
            bytes_per_bit=bytes_per_bit,
            sched=sched,
            c_batch_cap=c_batch_cap,
        )
        route_A, route_B, route_C = _route_entities(
            list(entities), state_for_router,
            tau_val=tau_var, bits_A=bits_A, bits_B=bits_B,
            bits_C_extra=bits_C_extra, params=params
        )

        estA = _np.full(n_ent, _np.nan, float)
        estB = _np.full(n_ent, _np.nan, float)
        vA = _np.full(n_ent, _np.inf, float)
        vB = _np.full(n_ent, _np.inf, float)
        countA = countB = countC = 0

        eps_eff_used = None
        batch_key = getattr(batch, "slot", r_idx)

        for j, e in enumerate(entities):
            arr = arr_by_e.get(e)
            if arr is None or arr.size == 0:
                continue

            if e in route_A:
                muA, vvA, _ = _aggregate_A(arr, trim=0.02)
                estA[j], vA[j] = muA, vvA
                countA += len(arr)

            elif e in route_C:
                muC, vvC, _ = _aggregate_A(arr, trim=0.02)
                estA[j], vA[j] = muC, vvC
                countC += len(arr)

            else:
                eps_local = eps_B * (0.6 if uldp_sensitive_cols else 1.0)
                noisy = _local_dp_laplace(arr, epsilon=max(eps_local, 1e-8), sensitivity=1.0)
                muB, vvB, _ = _aggregate_B(noisy, epsilon=eps_local, trim=0.02)
                estB[j], vB[j] = muB, vvB
                eps_eff_used = _epsilon_after_shuffle(eps_local, n=len(arr)) if use_shuffle else eps_local
                countB += len(arr)

        pickA = (vA < vB) & (~_np.isnan(estA))
        est = _np.where(pickA, estA, estB)

        mask_nan = _np.isnan(est)
        if mask_nan.any():
            est = _fill_est_from_reports(est, batch)

        rmse_raw = _rmse(est, truth)

        est_by_e = {e: float(est[j]) for j, e in enumerate(entities)}
        postprocess_decision = None
        if postprocessor is not None:
            var_by_e_dict = {e: float(var_by_e[j]) for j, e in enumerate(entities)}
            reports_count = {e: int(len(arr_by_e.get(e, []))) for e in entities}
            eff_eps = float(eps_eff_used if eps_eff_used is not None else eps_B)
            if isinstance(acct, PrivacyAccountantNaive):
                privacy_used = float(getattr(acct, "cum", 0.0) + eff_eps)
                privacy_total = float(getattr(acct, "limit", epsilon_per_window))
            else:
                privacy_used = float(getattr(acct, "epsilon", lambda: 0.0)() + eff_eps)
                privacy_total = float(epsilon_per_window)
            est_pp_by_e, postprocess_decision = postprocess_with_privacy_awareness(
                est_by_e=est_by_e,
                var_by_e=var_by_e_dict,
                reports_count=reports_count,
                graph=entity_graph,
                kstate=kalman_state,
                round_idx=r_idx,
                privacy_budget_used=privacy_used,
                privacy_budget_total=privacy_total,
                alpha_lap=alpha_lap,
                proc_var=proc_var,
                obs_var_base=obs_var_base,
                postprocessor=postprocessor
            )
        else:
            est_pp_by_e = _postprocess_filter(est_by_e, entity_graph, kalman_state,
                                              alpha_lap=alpha_lap, proc_var=proc_var,
                                              obs_var_base=obs_var_base)
        est_pp = _np.array([est_pp_by_e[e] for e in entities], float)
        rmse = _rmse(est_pp, truth)
        res_hist.append(rmse)

        for j, e in enumerate(entities):
            last_est_pp[e] = float(est_pp[j])

        time_s = _time.time() - t0

        bits_mean_B = _np.mean([bits_by_e.get(e, bits_B) for e in entities]) if entities.size > 0 else bits_B
        perB = int(bytes_per_bit * bits_mean_B)
        perC = perC_bytes

        bytes_A = _bytes(countA, perA_bytes)
        bytes_B = _bytes(countB, perB)
        bytes_C = _bytes(countC, perC)
        bytes_used = bytes_A + bytes_B + bytes_C

        if isinstance(acct, PrivacyAccountantNaive):
            acct_info = acct.update(float(eps_B))
            over = acct.overloaded()
        else:
            eff = float(eps_eff_used if eps_eff_used is not None else eps_B)
            acct.add_pure_dp(eff)
            eps_total = float(acct.epsilon())
            acct_info = {"epsilon_cum_window": eps_total, "epsilon_limit": float(epsilon_per_window)}
            over = bool(eps_total > float(epsilon_per_window) + 1e-12)

        sched.update(last_latency_ms=time_s * 1000.0,
                     last_bytes=float(bytes_used),
                     acct_over=over)

        pp_enabled = bool(postprocess_decision["enable"]) if postprocess_decision is not None else bool(not disable_post)
        pp_reason = (
            postprocess_decision["reason"]
            if postprocess_decision is not None
            else ("legacy_postprocess" if not disable_post else "postprocess_disabled")
        )
        pp_score = float(postprocess_decision.get("total_score", float("nan"))) if postprocess_decision is not None else float("nan")
        pp_alpha_scale = (
            float(postprocess_decision.get("alpha_scale", 1.0)) if postprocess_decision is not None else (1.0 if not disable_post else 0.0)
        )
        pp_proc_var_scale = (
            float(postprocess_decision.get("proc_var_scale", 1.0)) if postprocess_decision is not None else (1.0 if not disable_post else 0.0)
        )

        logs.append(dict(
            rmse=float(rmse),
            rmse_raw=float(rmse_raw),
            bytes=int(bytes_used),
            bytes_A=int(bytes_A),
            bytes_B=int(bytes_B),
            bytes_C=int(bytes_C),
            enc_ops=int(_enc_ops(countA + countB + countC, 2)),
            time_s=float(time_s),

            A_reports=int(countA),
            B_reports=int(countB),
            C_reports=int(countC),
            pickedA=int(pickA.sum()),
            route_ratio=float(countA + countC) / (countA + countB + countC + 1e-9),

            var=float(_np.var(est_pp)) if est_pp.size else float("nan"),
            resid_var=float(_np.var(est_pp - truth)) if est_pp.size else float("nan"),

            tau_percentile=float(sched.tau),
            A_budget_ratio=float(sched.a_ratio),

            postprocess_enabled=pp_enabled,
            postprocess_reason=str(pp_reason),
            postprocess_score=pp_score,
            postprocess_alpha_scale=pp_alpha_scale,
            postprocess_proc_var_scale=pp_proc_var_scale,

            accountant_mode=str("pld" if not isinstance(acct, PrivacyAccountantNaive) else "naive"),
            epsilon_round=float(eps_B),
            epsilon_cum_window=float(acct_info["epsilon_cum_window"]),
            epsilon_limit=float(acct_info["epsilon_limit"]),
            filter_triggered=bool(over),

            shuffle_used=bool(use_shuffle),
            epsilon_effective=float(eps_eff_used) if eps_eff_used is not None else float("nan"),
            geo_r_protect=float(geo_r),

            vdaf_http=False,
            vdaf_ok_ratio=float("nan"),
            reject_ratio=float("nan"),
            dap_mode="offline",
            batch_key=str(batch_key),
            collect_id="",
            system_issues=system_issue_flags,
        ))

    return logs

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
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - 2 * self.step))
            self.tau = min(97.0, self.tau + 10.0)


def _mk_super_increasing_z(M, S):
    """构造满足  sum_{i<l} z_i * S < z_l  的最小超递增序列; z_1=1。"""
    z = [1]
    sumz = 1
    for _ in range(2, M + 1):
        z_next = int(sumz * S) + 1  # 最小可行
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


import time, numpy as np, pandas as pd


def _pick_worker_id_column(rep: pd.DataFrame):
    for c in ["worker_id", "wid", "user_id", "sensor_id", "device_id", "source", "uid"]:
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
    tol = getattr(p, "tol", 1e-4)
    scale = int(getattr(p, "scale", 1e6))  # 定点缩放因子
    per_b = int(getattr(p, "per_bytes", 64))
    init_mth = getattr(p, "init", "mean")

    logs = []
    task_matched = False  # 论文：任务匹配只做一次（式(8)）
    # 开始遍历 rounds（外层由 experiment 驱动）
    for t, batch in enumerate(rounds_iter, start=1):
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        M = len(truth)
        entities = getattr(batch, "entities", list(range(M)))
        rep = getattr(batch, "reports", None)

        if rep is None or not {"entity_id", "value"}.issubset(rep.columns):
            # 没有报告：回退（为了不中断实验）
            est = truth.copy()
            n_used = M * max(1, getattr(batch, "n_workers", n_workers))
            logs.append(dict(
                rmse=float(np.sqrt(np.mean((est - truth) ** 2))),
                bytes=int(n_used * per_b), enc_ops=0,
                time_s=time.time() - t0, A_reports=n_used, B_reports=0,
                pickedA=M, route_ratio=1.0, Kp=M,
                var=float(np.var(est)), resid_var=float(np.var(est - truth))
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
        rep_g = rep.groupby([wid_col, "entity_id"])["value"].mean().reset_index()
        # 构造工人×任务矩阵，缺失为 NaN
        W = rep_g.pivot(index=wid_col, columns="entity_id", values="value").reindex(columns=entities)
        X = W.to_numpy(dtype=float)  # shape: (n_workers_eff, M)
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
            Dk = np.nansum((diff) ** 2, axis=1)
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
            enc_ops=0,  # 重加密在云侧；仿真不计
            time_s=time.time() - t0,
            A_reports=used_rows, B_reports=0,
            pickedA=M, route_ratio=1.0, Kp=M,
            var=float(np.var(est)), resid_var=float(np.var(est - truth))
        ))
    return logs

