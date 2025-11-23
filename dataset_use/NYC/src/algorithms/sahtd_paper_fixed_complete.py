# ============================================================================
# SAHTD-Paper 完整修复版本 - 整合所有补丁
# 文件位置：algorithms_bridge.py 或 sahtd_paper_bridge() 所在的文件
# 修复日期：基于研究指导意见
# ============================================================================

from typing import Iterable, List, Dict, Any
import numpy as np
import time
import math
import logging

# 导入必要的模块
from SAHTD1.sahtd_x_readable import PrivacyAccountantNaive
from dataset_use.NYC.src.algorithms.pld_accountant import PLDAccountant

# 导入comprehensive_fix模块
try:
    import sahtd_paper_comprehensive_fix as scf
    HAS_COMPREHENSIVE_FIX = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("sahtd_paper_comprehensive_fix not found, using fallback")
    HAS_COMPREHENSIVE_FIX = False

logger = logging.getLogger(__name__)

# ========= 工具函数（保持不变但补充注释） =========

def _geo_laplace_noise(eps_geo: float, size: int):
    """地理扰动：仅用于记录理论保护半径，不参与聚合计算。"""
    u1 = np.random.random(size=size)
    u2 = np.random.random(size=size)
    r = -(np.log(1.0 - u1) + np.log(1.0 - u2)) / max(eps_geo, 1e-12)
    theta = np.random.uniform(0, 2 * np.pi, size=size)
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return dx, dy


def _safe_array(x):
    """确保输入是有效的numpy数组。"""
    x = np.asarray(x)
    if x.ndim == 0:
        x = x[None]
    if x.size == 0:
        raise ValueError("expected non-empty sample")
    return x


def _rmse(est: np.ndarray, truth: np.ndarray) -> float:
    """计算均方根误差。"""
    est = np.asarray(est, float).ravel()
    truth = np.asarray(truth, float).ravel()
    if est.shape != truth.shape or est.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((est - truth) ** 2)))


def _bytes(n_reports: int, per: int) -> int:
    """计算字节消耗。"""
    return int(n_reports * per)


def _enc_ops(n_reports: int, factor: int = 1) -> int:
    """计算加密操作数。"""
    return int(n_reports * factor)


def _huber(x: np.ndarray, c: float = 1.5) -> float:
    """Huber M-估计器：鲁棒均值。"""
    x = _safe_array(x).astype(float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    r = (x - med) / (1.4826 * mad)
    w = np.clip(c / (np.abs(r) + 1e-12), 0.0, 1.0)
    return float(np.sum(w * x) / (np.sum(w) + 1e-12))


def _local_dp_laplace(x: np.ndarray, epsilon: float, sensitivity: float = 1.0) -> np.ndarray:
    """本地差分隐私：Laplace机制。"""
    x = _safe_array(x).astype(float)
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for LDP")
    scale = float(sensitivity) / float(epsilon)
    noise = np.random.laplace(0.0, scale, size=len(x))
    return x + noise


def _aggregate_B(reports: np.ndarray, epsilon: float,
                 huber_c: float = 1.3, trim: float = 0.02) -> tuple:
    """B通道：LDP后的稳健融合（reports已含噪声）。"""
    reports = _safe_array(reports).astype(float)
    est = _huber(reports, c=huber_c)
    v = float(np.var(reports) + 1e-12)
    return float(est), v, 0


def _aggregate_A(clean_reports: np.ndarray,
                 huber_c: float = 1.6, trim: float = 0.02) -> tuple:
    """A通道：受信聚合（不加噪），稳健中心。"""
    clean_reports = _safe_array(clean_reports).astype(float)
    est = _huber(clean_reports, c=huber_c)
    v = float(np.var(clean_reports) + 1e-12)
    return float(est), v, 0


def _epsilon_after_shuffle(eps_local: float, n: int) -> float:
    """洗牌放大后的等效ε。"""
    if n <= 1:
        return float(eps_local)
    val = abs(math.expm1(float(eps_local)))
    return float(min(eps_local, val / math.sqrt(float(n))))


def _fill_est_from_reports(est: np.ndarray, batch) -> np.ndarray:
    """
    关键修复：使用reports数据填补NaN，不使用真值。
    这是修复#5的实现：避免数据泄露。
    """
    import pandas as pd
    est = np.asarray(est, float).copy()
    rep = getattr(batch, 'reports', None)

    if rep is None or not {'entity_id', 'value'}.issubset(rep.columns):
        if np.isnan(est).all():
            return np.zeros_like(est, dtype=float)
        med = np.nanmedian(est)
        est = np.where(np.isnan(est), med, est)
        return est

    g = rep.groupby('entity_id')['value']
    entities = getattr(batch, 'entities', list(range(len(est))))

    # 按entity单独填
    for j, e in enumerate(entities):
        if np.isnan(est[j]) and e in g.groups:
            arr = g.get_group(e).to_numpy(dtype=float)
            if arr.size:
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)) + 1e-9)
                r = (arr - med) / (1.4826 * mad)
                w = np.clip(1.345 / (np.abs(r) + 1e-12), 0.0, 1.0)
                est[j] = float(np.sum(w * arr) / (np.sum(w) + 1e-12))

    # 全局填补
    if np.isnan(est).any():
        arr_all = rep['value'].to_numpy(dtype=float)
        med = float(np.median(arr_all))
        mad = float(np.median(np.abs(arr_all - med)) + 1e-9)
        r = (arr_all - med) / (1.4826 * mad)
        w = np.clip(1.345 / (np.abs(r) + 1e-12), 0.0, 1.0)
        fill = float(np.sum(w * arr_all) / (np.sum(w) + 1e-12))
        est = np.where(np.isnan(est), fill, est)

    return est


# ========= 调度器（在线参数调整） =========

class Scheduler:
    """
    在线调参器：满足<2ms延迟和字节预算。
    
    这是修复#3的实现：动态调整A/B/C路由比例。
    """
    def __init__(self, tau0=85.0, a_ratio0=0.10, step=0.08,
                 target_latency_ms=2.0, target_bytes=1.8e5):
        self.tau = float(tau0)
        self.a_ratio = float(a_ratio0)
        self.step = float(step)
        self.lat_target = float(target_latency_ms)
        self.bytes_target = float(target_bytes)
        self.update_count = 0

    def update(self, last_latency_ms: float, last_bytes: float, acct_over: bool):
        """根据性能指标动态调整。"""
        self.update_count += 1
        
        # 时延压力
        if last_latency_ms > self.lat_target * 1.05:
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - self.step))
            self.tau = min(95.0, self.tau + 5.0)
            logger.debug(f"[Sched] Latency high ({last_latency_ms:.3f}ms), "
                        f"reducing a_ratio to {self.a_ratio:.4f}")
        elif last_latency_ms < self.lat_target * 0.7:
            self.a_ratio = min(0.50, self.a_ratio * (1.0 + self.step))
            self.tau = max(55.0, self.tau - 5.0)
            logger.debug(f"[Sched] Latency low ({last_latency_ms:.3f}ms), "
                        f"increasing a_ratio to {self.a_ratio:.4f}")
        
        # 字节压力
        if last_bytes > self.bytes_target * 1.05:
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - self.step))
            logger.debug(f"[Sched] Bytes high ({last_bytes:.0f}B), "
                        f"reducing a_ratio to {self.a_ratio:.4f}")
        elif last_bytes < self.bytes_target * 0.7:
            self.a_ratio = min(0.55, self.a_ratio * (1.0 + self.step))
            logger.debug(f"[Sched] Bytes low ({last_bytes:.0f}B), "
                        f"increasing a_ratio to {self.a_ratio:.4f}")
        
        # 隐私超限
        if acct_over:
            self.a_ratio = max(0.05, self.a_ratio * (1.0 - 2 * self.step))
            self.tau = min(92.0, self.tau + 10.0)
            logger.warning(f"[Sched] Privacy over-budget! "
                          f"a_ratio={self.a_ratio:.4f}, tau={self.tau:.1f}")


# ========= Kalman后处理 =========

class KalmanPostProcessor:
    """
    这是修复#5的实现：Kalman滤波 + 图拉普拉斯平滑。
    
    用于恢复被disable_postprocess=True关闭的后处理功能。
    """
    def __init__(self, alpha: float = 0.3, process_var: float = 0.4, 
                 obs_var_base: float = 1.2):
        self.alpha = alpha
        self.process_var = process_var
        self.obs_var_base = obs_var_base
        self.x_filtered = {}
        self.p_filtered = {}

    def kalman_update(self, entity_id: int, measurement: float, 
                     measurement_var: float) -> float:
        """单个entity的Kalman更新步。"""
        if entity_id not in self.x_filtered:
            self.x_filtered[entity_id] = measurement
            self.p_filtered[entity_id] = measurement_var
            return measurement

        x_pred = self.x_filtered[entity_id]
        p_pred = self.p_filtered[entity_id] + self.process_var
        K = p_pred / (p_pred + measurement_var)
        x_new = x_pred + K * (measurement - x_pred)
        p_new = (1 - K) * p_pred

        self.x_filtered[entity_id] = x_new
        self.p_filtered[entity_id] = p_new
        return x_new

    def apply_laplacian_smoothing(self, estimates: Dict[int, float], 
                                 entity_graph: Dict[int, List[int]]) -> Dict[int, float]:
        """图拉普拉斯平滑。"""
        smoothed = {}
        for entity_id, neighbors in entity_graph.items():
            if entity_id not in estimates:
                continue
            self_term = (1 - self.alpha) * estimates[entity_id]
            neighbor_term = 0.0
            if neighbors:
                neighbor_avg = sum(estimates.get(nid, estimates[entity_id]) 
                                  for nid in neighbors) / len(neighbors)
                neighbor_term = self.alpha * neighbor_avg
            smoothed[entity_id] = self_term + neighbor_term
        return smoothed


def _postprocess_filter(est_by_e: dict, graph: dict, kstate: dict,
                       alpha_lap: float, proc_var: float, obs_var_base: float):
    """
    后处理主函数：允许完全关闭。
    这是修复#5的集成点。
    """
    if alpha_lap <= 0.0 and proc_var <= 0.0:
        return est_by_e

    processor = KalmanPostProcessor(alpha=alpha_lap, process_var=proc_var, 
                                   obs_var_base=obs_var_base)
    
    # 先做Kalman
    est_k = {}
    for e, obs in est_by_e.items():
        if e not in kstate:
            kstate[e] = {'m': 0.0, 'v': 10.0, 'init': False}
        
        st = kstate[e]
        if not st['init']:
            st['m'] = float(obs)
            st['v'] = 5.0
            st['init'] = True
            est_k[e] = st['m']
        else:
            est_k[e] = processor.kalman_update(e, float(obs), obs_var_base)
            kstate[e]['m'] = processor.x_filtered[e]
            kstate[e]['v'] = processor.p_filtered[e]

    # 再做拉普拉斯平滑（如果有图）
    if graph is None or alpha_lap <= 0.0:
        return est_k

    est_pp = processor.apply_laplacian_smoothing(est_k, graph)
    return est_pp


# ========= 自适应量化 =========

def _update_quant_bits(entity_ids, est_vec_prev, var_est_dict,
                      resid_ema_dict, bytes_per_bit: float, params):
    """
    这是修复#2的实现：自适应量化，不使用真值。
    
    关键改进：
    - 使用前一轮的估计增量作为方差proxy
    - 自动根据数据方差调整位数
    - 约束总字节数在target范围内
    """
    min_bits = int(getattr(params, "MIN_QUANT_BITS", 6))
    max_bits = int(getattr(params, "MAX_QUANT_BITS", 14))
    target_bytes = float(getattr(params, "target_bytes_per_round", 12000.0))
    avg_reports_per_entity = float(getattr(params, "AVG_REPORTS_PER_ENTITY", 10.0))
    var_quantile = float(getattr(params, "VAR_QUANTILE", 0.75))

    est_vec_prev = np.asarray(est_vec_prev, float)

    # 更新方差估计（基于估计增量）
    for idx, e in enumerate(entity_ids):
        prev = float(est_vec_prev[idx])
        v_old = float(var_est_dict.get(e, 5.0))
        var_est_dict[e] = 0.9 * v_old + 0.1 * (prev * prev)
        r_old = float(resid_ema_dict.get(e, 0.0))
        resid_ema_dict[e] = 0.9 * r_old + 0.1 * abs(prev)

    # 计算tau
    vars_arr = np.array([var_est_dict.get(e, 5.0) for e in entity_ids], float)
    if np.isfinite(vars_arr).any():
        tau = float(np.quantile(vars_arr[np.isfinite(vars_arr)], var_quantile))
    else:
        tau = 1.0

    # 初始位数分配
    bits_prop = {}
    q_init = int(getattr(params, "quant_bits_init", 8))
    for e in entity_ids:
        b = q_init
        v = float(var_est_dict.get(e, 5.0))
        if v > tau:
            b += 2
        else:
            b -= 1
        b = max(min_bits, min(max_bits, b))
        bits_prop[e] = b

    # 缩放以满足字节约束
    total_bits = sum(bits_prop[e] * avg_reports_per_entity for e in entity_ids)
    total_bytes = float(total_bits) * float(bytes_per_bit)
    scale = 1.0 if total_bytes <= 0 else target_bytes / total_bytes

    bits_final = {}
    for e in entity_ids:
        b_scaled = int(round(bits_prop[e] * scale))
        b_scaled = max(min_bits, min(max_bits, b_scaled))
        bits_final[e] = b_scaled

    return bits_final, tau


# ========= 路由选择（修复#4） =========

def _compute_value_scores(entity_ids, var_est_dict, resid_ema_dict,
                         bytes_per_bit: float,
                         bits_A: int, bits_B: int, bits_C_extra: int):
    """计算每个entity的价值得分。"""
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
    这是修复#4的实现：全局分层路由。
    
    三层模式：
    - 层1（Top）：C路 - 高价值实体，高精度
    - 层2（Mid）：A路 - 中等价值实体
    - 层3（Low）：B路 - 低价值实体，主力LDP
    """
    # 获取参数
    base_a_ratio = float(getattr(params, "BASE_A_RATIO",
                                getattr(state.get("sched"), "a_ratio", 0.18)))
    base_c_ratio = float(getattr(params, "BASE_C_RATIO", 0.06))
    c_batch_max = int(getattr(params, "C_BATCH_MAX", 32))

    # 计算得分
    scores = _compute_value_scores(entity_ids, state["var_est"], state["resid_ema"],
                                  state["bytes_per_bit"], bits_A, bits_B, bits_C_extra)

    n = len(entity_ids)
    if n == 0:
        return set(), set(), set()

    # 全局排序
    sorted_all = sorted(entity_ids, key=lambda e: -scores[e][0])

    # 划定各层级数量
    n_C = min(int(round(base_c_ratio * n)), c_batch_max, n)
    if n_C <= 0 and base_c_ratio > 0.0 and n > 0:
        n_C = min(1, c_batch_max, n)

    n_A = int(round(base_a_ratio * n))
    if n_A == 0 and base_a_ratio > 0.001:
        n_A = 1

    # 分层切片
    route_C = set(sorted_all[:n_C])
    route_A = set(sorted_all[n_C: n_C + n_A])
    route_B = set(sorted_all[n_C + n_A:])

    logger.debug(f"[Route] n_A={len(route_A)}, n_B={len(route_B)}, "
                f"n_C={len(route_C)}, total={n}")

    return route_A, route_B, route_C


# ========= 主算法 =========

def sahtd_paper_bridge(rounds_iter: Iterable, n_workers: int, params=None):
    """
    SAHTD-Nexus-Lite：三路调度版本（整合所有修复）。
    
    修复项：
    FIX#1: target_bytes_per_round = 12000 (from 900)
    FIX#2: 量化参数调整 & 自适应量化
    FIX#3: 动态路由比例分配
    FIX#4: 改进的C路触发条件 (0.6倍阈值)
    FIX#5: 启用Kalman后处理 + 图拉普拉斯
    FIX#6: 数据统计计算
    """
    
    # ===== 参数读取 =====
    eps_B = float(getattr(params, "epsilon", 1.0) if params is not None else 1.0)
    tau0 = float(getattr(params, "tau_percentile", 75.0) if params is not None else 75.0)
    
    # FIX#3：使用参数中的BASE_A_RATIO而不是固定值
    A_ratio0 = float(getattr(params, "BASE_A_RATIO", 
                            getattr(params, "A_budget_ratio", 0.18)) 
                    if params is not None else 0.18)

    use_shuffle = bool(getattr(params, "use_shuffle", True) if params is not None else True)
    uldp_sensitive_cols = list(getattr(params, "uldp_sensitive_cols", []) 
                               if params is not None else [])
    geo_eps = float(getattr(params, "geo_epsilon", 0.0) if params is not None else 0.0)

    window_w = int(getattr(params, "window_w", 32) if params is not None else 32)
    epsilon_per_window = float(getattr(params, "epsilon_per_window", eps_B * window_w)
                               if params is not None else eps_B * window_w)
    acct_mode = str(getattr(params, "accountant_mode", "pld")
                    if params is not None else "pld").lower()
    delta_target = float(getattr(params, "delta_target", 1e-5)
                         if params is not None else 1e-5)

    target_latency_ms = float(getattr(params, "target_latency_ms", 2.0) 
                             if params is not None else 2.0)
    
    # FIX#1：关键修复 - 提升字节预算
    target_bytes = float(getattr(params, "target_bytes_per_round", 12000.0) 
                        if params is not None else 12000.0)

    bytes_per_bit = float(getattr(params, "bytes_per_bit", 0.125))
    
    # FIX#2：量化参数调整
    bits_A = int(getattr(params, "BASE_BITS_A", 10))
    bits_B = int(getattr(params, "BASE_BITS_B", 8))
    bits_C_extra = int(getattr(params, "BITS_C_EXTRA", 3))

    perA_bytes = int(getattr(params, "perA_bytes", 32))
    perC_bytes = int(getattr(params, "perC_bytes", 64))

    # FIX#5：启用后处理
    disable_post = bool(getattr(params, "disable_postprocess", False))
    alpha_lap = 0.0 if disable_post else float(getattr(params, "post_lap_alpha", 0.3))
    proc_var = 0.0 if disable_post else float(getattr(params, "post_process_var", 0.4))
    obs_var_base = 1.0 if disable_post else float(getattr(params, "post_obs_var_base", 1.2))
    entity_graph = getattr(params, "entity_graph", None)

    # ===== 初始化 =====
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

    logger.info(f"[SAHTD-Paper] Starting with eps={eps_B}, target_bytes={target_bytes}, "
               f"bits_A={bits_A}, bits_B={bits_B}, bits_C={bits_C_extra}")

    # ===== 主循环 =====
    for r_idx, batch in enumerate(rounds_iter):
        """
        修复说明：
        - r_idx 现已正确定义（来自enumerate）
        - 移除了错误的RouteSelector引入
        - 使用internal _route_entities函数
        """
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        n_ent = len(truth)
        entities = np.asarray(batch.entities)

        # 地理扰动（仅记录）
        geo_r = float("nan")
        rep_df = getattr(batch, "reports", None)
        if geo_eps > 0.0 and rep_df is not None and {"lat", "lng"}.issubset(rep_df.columns):
            dx, dy = _geo_laplace_noise(geo_eps, size=len(rep_df))
            geo_r = float(np.mean(np.sqrt(dx * dx + dy * dy)))

        # 按entity收集观测
        arr_by_e: Dict[Any, np.ndarray] = {}
        var_by_e = np.zeros(n_ent, float)

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
                    arr = np.empty((0,), dtype=float)
                arr_by_e[e] = arr

                if arr.size > 1:
                    var_by_e[j] = float(arr.var(ddof=1))
                else:
                    var_by_e[j] = 0.0
        else:
            for j, e in enumerate(entities):
                arr_by_e[e] = np.empty((0,), dtype=float)
                var_by_e[j] = 0.0

        # tau估计
        if len(res_hist) >= 5 and np.isfinite(var_by_e).any():
            tau_val = float(np.nanpercentile(var_by_e[np.isfinite(var_by_e)], sched.tau))
        elif np.isfinite(var_by_e).any():
            tau_val = float(np.nanmedian(var_by_e[np.isfinite(var_by_e)]))
        else:
            tau_val = 0.0

        # 上一轮后处理估计
        if last_est_pp:
            est_prev_pp_vec = np.array(
                [last_est_pp.get(e, 0.0) for e in entities], float
            )
        else:
            est_prev_pp_vec = np.zeros_like(truth)

        # 自适应量化（FIX#2）
        bits_by_e, tau_var = _update_quant_bits(
            entities, est_prev_pp_vec, var_est, resid_ema,
            bytes_per_bit, params
        )

        # 路由选择（FIX#4）
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

        # 三路聚合
        estA = np.full(n_ent, np.nan, float)
        estB = np.full(n_ent, np.nan, float)
        vA = np.full(n_ent, np.inf, float)
        vB = np.full(n_ent, np.inf, float)
        countA = countB = countC = 0

        eps_eff_used = None
        batch_key = getattr(batch, "slot", r_idx)

        for j, e in enumerate(entities):
            arr = arr_by_e.get(e)
            if arr is None or arr.size == 0:
                continue

            if e in route_A:
                # A路：受信聚合
                muA, vvA, _ = _aggregate_A(arr, trim=0.02)
                estA[j], vA[j] = muA, vvA
                countA += len(arr)

            elif e in route_C:
                # C路：高精度受信聚合（模拟DAP但本地执行）
                muC, vvC, _ = _aggregate_A(arr, trim=0.02)
                estA[j], vA[j] = muC, vvC  # 写入A的位置用于后续比较
                countC += len(arr)

            else:  # route_B
                # B路：LDP + 稳健聚合
                eps_local = eps_B * (0.6 if uldp_sensitive_cols else 1.0)
                noisy = _local_dp_laplace(arr, epsilon=max(eps_local, 1e-8), 
                                         sensitivity=1.0)
                muB, vvB, _ = _aggregate_B(noisy, epsilon=eps_local, trim=0.02)
                estB[j], vB[j] = muB, vvB
                eps_eff_used = (_epsilon_after_shuffle(eps_local, n=len(arr)) 
                               if use_shuffle else eps_local)
                countB += len(arr)

        # 选择较优估计
        pickA = (vA < vB) & (~np.isnan(estA))
        est = np.where(pickA, estA, estB)

        # 关键修复FIX#5：使用reports填补NaN，不使用真值
        mask_nan = np.isnan(est)
        if mask_nan.any():
            est = _fill_est_from_reports(est, batch)

        rmse_raw = _rmse(est, truth)

        # 后处理（FIX#5）：Kalman + 图拉普拉斯平滑
        est_by_e = {e: float(est[j]) for j, e in enumerate(entities)}
        est_pp_by_e = _postprocess_filter(est_by_e, entity_graph, kalman_state,
                                         alpha_lap=alpha_lap, proc_var=proc_var,
                                         obs_var_base=obs_var_base)
        est_pp = np.array([est_pp_by_e[e] for e in entities], float)
        rmse = _rmse(est_pp, truth)
        res_hist.append(rmse)

        # 保存本轮后处理估计供下轮使用
        for j, e in enumerate(entities):
            last_est_pp[e] = float(est_pp[j])

        time_s = time.time() - t0

        # 字节计算
        bits_mean_B = (np.mean([bits_by_e.get(e, bits_B) for e in entities]) 
                      if entities.size > 0 else bits_B)
        perB = int(bytes_per_bit * bits_mean_B)

        bytes_A = _bytes(countA, perA_bytes)
        bytes_B = _bytes(countB, perB)
        bytes_C = _bytes(countC, perC_bytes)
        bytes_used = bytes_A + bytes_B + bytes_C

        # 隐私会计
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

        # 在线调参
        sched.update(last_latency_ms=time_s * 1000.0,
                    last_bytes=float(bytes_used),
                    acct_over=over)

        # 日志记录
        logs.append(dict(
            # 核心指标
            rmse=float(rmse),
            rmse_raw=float(rmse_raw),
            bytes=int(bytes_used),
            bytes_A=int(bytes_A),
            bytes_B=int(bytes_B),
            bytes_C=int(bytes_C),
            enc_ops=int(_enc_ops(countA + countB + countC, 2)),
            time_s=float(time_s),

            # 路由统计
            A_reports=int(countA),
            B_reports=int(countB),
            C_reports=int(countC),
            pickedA=int(pickA.sum()),
            route_ratio=float(countA + countC) / (countA + countB + countC + 1e-9),

            # 方差指标
            var=float(np.var(est_pp)) if est_pp.size else float("nan"),
            resid_var=float(np.var(est_pp - truth)) if est_pp.size else float("nan"),

            # 调度器状态
            tau_percentile=float(sched.tau),
            A_budget_ratio=float(sched.a_ratio),

            # 隐私会计
            accountant_mode=str("pld" if not isinstance(acct, PrivacyAccountantNaive) 
                               else "naive"),
            epsilon_round=float(eps_B),
            epsilon_cum_window=float(acct_info["epsilon_cum_window"]),
            epsilon_limit=float(acct_info["epsilon_limit"]),
            filter_triggered=bool(over),

            # 洗牌和地理
            shuffle_used=bool(use_shuffle),
            epsilon_effective=float(eps_eff_used) if eps_eff_used is not None 
                            else float("nan"),
            geo_r_protect=float(geo_r),

            # 兼容性字段
            vdaf_http=False,
            vdaf_ok_ratio=float("nan"),
            reject_ratio=float("nan"),
            dap_mode="offline",
            batch_key=str(batch_key),
            collect_id=""
        ))

        if (r_idx + 1) % max(1, len(list(rounds_iter)) // 5) == 0:
            logger.info(f"[Round {r_idx}] RMSE={rmse:.4f}, "
                       f"A/B/C={countA}/{countB}/{countC}, "
                       f"Bytes={bytes_used:.0f}, eps={acct_info['epsilon_cum_window']:.3f}")

    logger.info(f"[SAHTD-Paper] Completed {len(logs)} rounds. "
               f"Final RMSE={logs[-1]['rmse']:.4f}")
    return logs
