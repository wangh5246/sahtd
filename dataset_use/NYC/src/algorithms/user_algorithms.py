from typing import Iterable, List, Dict, Any
import numpy as np, time, math

from SAHTD1.sahtd_x_readable import PrivacyAccountantNaive
from dataset_use.NYC.src.algorithms.pld_accountant import PLDAccountant
# ========= 必要工具函数 =========

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


# ========= SAHTD-Nexus-Lite 主算法 =========

def sa_htd_paper(rounds_iter: Iterable, n_workers: int, params=None):
    """
    SAHTD-Nexus-Lite（用于论文实验的三路调度版本）：
      - A 路：可信、高精度鲁棒聚合；
      - B 路：LDP + 稳健聚合，主力通道；
      - C 路：本地“更强 A 路”模拟，bytes 更贵，用来模拟高价值通道；
      - 自适应量化：约束 bytes_mean 靠近 target_bytes_per_round；
      - 后处理（可关）：1D Kalman + 图拉普拉斯（post-processing，不消耗隐私）；
      - 不对 est 进行 truth 回填，缺失值只用 reports 统计来填补。

    返回 logs: list[dict]，字段兼容原实验脚本，并额外给出：
      - A_reports, B_reports, C_reports
      - bytes_A, bytes_B, bytes_C
      - rmse_raw（后处理前的 RMSE）
    """
    import numpy as _np
    import time as _time

    # ---------- Kalman + 图平滑 ----------
    class _Kalman1DState:
        __slots__ = ("m", "v", "init")

        def __init__(self):
            self.m = 0.0
            self.v = 10.0
            self.init = False

    def _postprocess_filter(est_by_e: dict, graph: dict, kstate: dict,
                            alpha_lap: float, proc_var: float, obs_var_base: float):
        # 允许完全关闭后处理
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

    # ---------- 自适应量化 & 路由 ----------

    def _update_quant_bits(entity_ids, est_vec_prev, var_est_dict,
                           resid_ema_dict, bytes_per_bit: float, params):
        """
        注意：这里不再使用 truth，只利用 **前一轮的估计轨迹** 来更新方差 / 残差规模。
        """
        min_bits = int(getattr(params, "MIN_QUANT_BITS", 6))
        max_bits = int(getattr(params, "MAX_QUANT_BITS", 14))
        target_bytes = float(getattr(params, "target_bytes_per_round", 9e2))
        avg_reports_per_entity = float(getattr(params, "AVG_REPORTS_PER_ENTITY", 10.0))
        var_quantile = float(getattr(params, "VAR_QUANTILE", 0.7))

        est_vec_prev = _np.asarray(est_vec_prev, float)
        # 用“估计增量”近似残差规模（无真值下的 proxy）
        for idx, e in enumerate(entity_ids):
            prev = float(est_vec_prev[idx])
            # 简化：用 |prev| 作为尺度 proxy
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
        q_init = int(getattr(params, "quant_bits_init", getattr(params, "BASE_BITS_B", 8)))
        for e in entity_ids:
            b = int(getattr(params, "quant_bits", q_init))
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
        """
        修改版路由逻辑：全局分层 (Global Tiered Slicing)
        层级 1 (Top): C 路 (处理极高方差/高价值实体)
        层级 2 (Mid): A 路 (处理中高方差实体)
        层级 3 (Low): B 路 (兜底其余实体)
        """
        # 1. 获取参数
        # 注意：这里的 base_a_ratio 理解为“A路独享的比例”，不再是“A+C的总比例”
        base_a_ratio = float(getattr(params, "BASE_A_RATIO",
                                     getattr(state["sched"], "a_ratio", 0.25)))
        base_c_ratio = float(getattr(params, "BASE_C_RATIO", 0.03))
        c_batch_max = int(getattr(params, "C_BATCH_MAX", 16))

        # 2. 计算每个实体的分数 (分数越高越需要高精度通道)
        # scores[e][0] 是 "A路性价比" (DeltaMSE / Bytes_A)，作为通用的难度指标
        scores = _compute_value_scores(entity_ids, state["var_est"], state["resid_ema"],
                                       state["bytes_per_bit"], bits_A, bits_B, bits_C_extra)

        n = len(entity_ids)

        # 3. 全局排序：按分数从高到低
        # 使用 scores[e][0] 作为统一排序标准，因为 A/C 的价值趋势通常是一致的
        sorted_all = sorted(entity_ids, key=lambda e: -scores[e][0])

        # 4. 划定各层级数量 (Counts)
        # C 路数量：受 base_c_ratio 和 c_batch_max 双重限制；若比例>0，则至少保底 1 个
        n_C = min(int(round(base_c_ratio * n)), c_batch_max, n)
        if n_C <= 0 and base_c_ratio > 0.0 and n > 0:
            n_C = min(1, c_batch_max, n)

        # A 路数量：基于 base_a_ratio
        # 修正：确保至少有 1 个 A (如果比例允许)，防止 A 路完全为空
        n_A = int(round(base_a_ratio * n))
        if n_A == 0 and base_a_ratio > 0.001:
            n_A = 1

        # 5. 分层切片 (Slicing)
        # Top-tier -> C
        route_C = set(sorted_all[:n_C])

        # Mid-tier -> A (紧接在 C 后面)
        route_A = set(sorted_all[n_C: n_C + n_A])

        # Rest -> B
        route_B = set(sorted_all[n_C + n_A:])

        return route_A, route_B, route_C

    # ---------- 读超参 ----------
    eps_B = float(getattr(params, "epsilon", 1.0) if params is not None else 1.0)
    tau0 = float(getattr(params, "tau_percentile", 75.0) if params is not None else 75.0)
    A_ratio0 = float(getattr(params, "A_budget_ratio", 0.25) if params is not None else 0.25)

    use_shuffle = bool(getattr(params, "use_shuffle", True) if params is not None else True)
    uldp_sensitive_cols = list(getattr(params, "uldp_sensitive_cols", []) if params is not None else [])
    geo_eps = float(getattr(params, "geo_epsilon", 0.0) if params is not None else 0.0)

    window_w = int(getattr(params, "window_w", 32) if params is not None else 32)
    epsilon_per_window = float(getattr(params, "epsilon_per_window", eps_B * window_w)
                               if params is not None else eps_B * window_w)
    acct_mode = str(getattr(params, "accountant_mode", "pld")
                    if params is not None else "pld").lower()
    delta_target = float(getattr(params, "delta_target", 1e-5)
                         if params is not None else 1e-5)

    target_latency_ms = float(getattr(params, "target_latency_ms", 2.0) if params is not None else 2.0)
    target_bytes = float(getattr(params, "target_bytes_per_round", 9e2) if params is not None else 9e2)

    bytes_per_bit = float(getattr(params, "bytes_per_bit", 0.125))
    bits_A = int(getattr(params, "BASE_BITS_A", 11))
    bits_B = int(getattr(params, "BASE_BITS_B", 9))
    bits_C_extra = int(getattr(params, "BITS_C_EXTRA", 2))

    perA_bytes = int(getattr(params, "perA_bytes", 32))
    perC_bytes = int(getattr(params, "perC_bytes", 64))

    disable_post = bool(getattr(params, "disable_postprocess", False))
    alpha_lap = 0.0 if disable_post else float(getattr(params, "post_lap_alpha", 0.3))
    proc_var = 0.0 if disable_post else float(getattr(params, "post_process_var", 0.5))
    obs_var_base = 1.0 if disable_post else float(getattr(params, "post_obs_var_base", 1.0))
    entity_graph = getattr(params, "entity_graph", None)

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
        truth = _np.asarray(batch.truth, float)   # 只用于离线评估
        n_ent = len(truth)
        entities = _np.asarray(batch.entities)

        geo_r = float("nan")
        rep_df = getattr(batch, "reports", None)
        if geo_eps > 0.0 and rep_df is not None and {"lat", "lng"}.issubset(rep_df.columns):
            dx, dy = _geo_laplace_noise(geo_eps, size=len(rep_df))
            geo_r = float(_np.mean(_np.sqrt(dx * dx + dy * dy)))

        # 按 entity 收集观测数组
        arr_by_e: Dict[Any, _np.ndarray] = {}
        var_by_e = _np.zeros(n_ent, float)

        if rep_df is not None and {"entity_id", "value"}.issubset(rep_df.columns):
            x = rep_df[["entity_id", "value"]]

            # 一次 groupby，避免对每个 entity 全表扫描
            groups = (
                x.groupby("entity_id")["value"]
                .apply(lambda s: s.to_numpy(dtype=float))
                .to_dict()
            )

            for j, e in enumerate(entities):
                arr = groups.get(e)
                if arr is None:
                    # 这个实体在这一轮没人上报
                    arr = _np.empty((0,), dtype=float)
                arr_by_e[e] = arr

                if arr.size > 1:
                    # 如果你原来这里有“按 entity 估 var”的逻辑，就保持：
                    var_by_e[j] = float(arr.var(ddof=1))
                else:
                    var_by_e[j] = 0.0
        else:
            # 没有 reports 的情况：留空
            for j, e in enumerate(entities):
                arr_by_e[e] = _np.empty((0,), dtype=float)
                var_by_e[j] = 0.0

        # tau 估计（用于调度器内部）
        if len(res_hist) >= 5 and _np.isfinite(var_by_e).any():
            tau_val = float(_np.nanpercentile(var_by_e[_np.isfinite(var_by_e)], sched.tau))
        elif _np.isfinite(var_by_e).any():
            tau_val = float(_np.nanmedian(var_by_e[_np.isfinite(var_by_e)]))
        else:
            tau_val = 0.0

        # 上一轮后处理估计（作为无真值下的 proxy）
        if last_est_pp:
            est_prev_pp_vec = _np.array(
                [last_est_pp.get(e, 0.0) for e in entities], float
            )
        else:
            est_prev_pp_vec = _np.zeros_like(truth)

        # 自适应量化（不使用 truth）
        bits_by_e, tau_var = _update_quant_bits(
            entities, est_prev_pp_vec, var_est, resid_ema,
            bytes_per_bit, params
        )

        state_for_router = dict(
            var_est=var_est,
            resid_ema=resid_ema,
            bytes_per_bit=bytes_per_bit,
            sched=sched,
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
                # 轻量版 C 路：不走远程 DAP，只用“更强的 A 路”模拟，并在 bytes 里记更贵
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

        # 关键修改：**不再用 truth 回填 NaN**，改用 _fill_est_from_reports
        mask_nan = _np.isnan(est)
        if mask_nan.any():
            est = _fill_est_from_reports(est, batch)

        rmse_raw = _rmse(est, truth)

        # 后处理（Kalman + 图正则）
        est_by_e = {e: float(est[j]) for j, e in enumerate(entities)}
        est_pp_by_e = _postprocess_filter(est_by_e, entity_graph, kalman_state,
                                          alpha_lap=alpha_lap, proc_var=proc_var,
                                          obs_var_base=obs_var_base)
        est_pp = _np.array([est_pp_by_e[e] for e in entities], float)
        rmse = _rmse(est_pp, truth)
        res_hist.append(rmse)

        for j, e in enumerate(entities):
            last_est_pp[e] = float(est_pp[j])

        time_s = _time.time() - t0

        # 字节估计
        bits_mean_B = _np.mean([bits_by_e.get(e, bits_B) for e in entities]) if entities.size > 0 else bits_B
        perB = int(bytes_per_bit * bits_mean_B)
        perC = perC_bytes

        bytes_A = _bytes(countA, perA_bytes)
        bytes_B = _bytes(countB, perB)
        bytes_C = _bytes(countC, perC)
        bytes_used = bytes_A + bytes_B + bytes_C

        # 会计
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

            accountant_mode=str("pld" if not isinstance(acct, PrivacyAccountantNaive) else "naive"),
            epsilon_round=float(eps_B),
            epsilon_cum_window=float(acct_info["epsilon_cum_window"]),
            epsilon_limit=float(acct_info["epsilon_limit"]),
            filter_triggered=bool(over),

            shuffle_used=bool(use_shuffle),
            epsilon_effective=float(eps_eff_used) if eps_eff_used is not None else float("nan"),
            geo_r_protect=float(geo_r),

            vdaf_http=False,          # 轻量版：不真正用 HTTP
            vdaf_ok_ratio=float("nan"),
            reject_ratio=float("nan"),
            dap_mode="offline",
            batch_key=str(batch_key),
            collect_id=""
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
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - 2*self.step))
            self.tau = min(97.0, self.tau + 10.0)
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

import time, numpy as np, pandas as pd
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
