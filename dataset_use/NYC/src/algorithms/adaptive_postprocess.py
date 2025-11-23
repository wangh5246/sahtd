import numpy as np
from typing import Dict, Any, Optional
import warnings


class AdaptivePostprocessor:
    """
    自适应后处理触发器:
    - 根据RMSE、方差、收敛状态动态决定是否使用后处理
    - 避免对已经良好的估计过度平滑
    """
    
    def __init__(self, 
                 rmse_threshold: float = 40.0,
                 var_threshold: float = 100.0,
                 warmup_rounds: int = 5,
                 enable_adaptive: bool = True):
        """
        参数:
            rmse_threshold: RMSE超过此值才启用后处理(默认40)
            var_threshold: 方差超过此值才启用后处理(默认100)
            warmup_rounds: 前N轮强制不使用后处理(让模型收敛)
            enable_adaptive: 是否启用自适应(False则始终开启后处理)
        """
        self.rmse_threshold = float(rmse_threshold)
        self.var_threshold = float(var_threshold)
        self.warmup_rounds = int(warmup_rounds)
        self.enable_adaptive = bool(enable_adaptive)
        
        # 历史记录
        self.rmse_history = []
        self.var_history = []
        self.trigger_history = []
        
        # 动态阈值(会根据数据自适应调整)
        self.dynamic_rmse_threshold = self.rmse_threshold
        self.dynamic_var_threshold = self.var_threshold
    
    def should_postprocess(self, 
                          round_idx: int,
                          rmse_raw: float,
                          var_est: float,
                          truth_range: Optional[float] = None) -> Dict[str, Any]:
        """
        判断是否需要后处理
        
        返回:
            {
                'enable': bool,  # 是否启用后处理
                'reason': str,   # 触发/不触发的原因
                'alpha_scale': float,  # 平滑强度缩放因子(0-1)
                'proc_var_scale': float,  # Kalman过程噪声缩放因子
            }
        """
        # 如果不启用自适应,始终开启后处理
        if not self.enable_adaptive:
            return {
                'enable': True,
                'reason': 'adaptive_disabled',
                'alpha_scale': 1.0,
                'proc_var_scale': 1.0,
            }
        
        # Warm-up期间不使用后处理
        if round_idx < self.warmup_rounds:
            return {
                'enable': False,
                'reason': f'warmup_period (round {round_idx}/{self.warmup_rounds})',
                'alpha_scale': 0.0,
                'proc_var_scale': 0.0,
            }
        
        # 更新历史
        self.rmse_history.append(rmse_raw)
        self.var_history.append(var_est)
        
        # 动态调整阈值(基于历史数据)
        if len(self.rmse_history) >= 10:
            self._update_dynamic_thresholds()
        
        # ========== 触发条件判断 ========== #
        
        triggers = []
        
        # 条件1: RMSE过高
        if rmse_raw > self.dynamic_rmse_threshold:
            triggers.append(f'rmse_high ({rmse_raw:.2f} > {self.dynamic_rmse_threshold:.2f})')
        
        # 条件2: 方差过大(说明估计不稳定)
        if var_est > self.dynamic_var_threshold:
            triggers.append(f'var_high ({var_est:.2f} > {self.dynamic_var_threshold:.2f})')
        
        # 条件3: 相对误差过大(如果知道truth的范围)
        if truth_range is not None and truth_range > 0:
            relative_rmse = rmse_raw / truth_range
            if relative_rmse > 0.3:  # 相对误差>30%
                triggers.append(f'relative_error_high ({relative_rmse:.2%})')
        
        # 条件4: 最近3轮RMSE未收敛或发散
        if len(self.rmse_history) >= 3:
            recent_trend = np.mean(np.diff(self.rmse_history[-3:]))
            if recent_trend > 0.5:  # 递增趋势
                triggers.append(f'rmse_diverging (trend={recent_trend:.2f})')
        
        # ========== 决策逻辑 ========== #
        
        enable = len(triggers) > 0
        
        if enable:
            # 根据触发条件数量调整平滑强度
            # 触发条件越多,说明问题越严重,需要更强的平滑
            intensity = len(triggers) / 4.0  # 最多4个条件
            alpha_scale = min(1.0, 0.3 + 0.7 * intensity)
            proc_var_scale = min(2.0, 0.5 + 1.5 * intensity)
            
            reason = '; '.join(triggers)
        else:
            alpha_scale = 0.0
            proc_var_scale = 0.0
            reason = f'all_conditions_ok (rmse={rmse_raw:.2f}, var={var_est:.2f})'
        
        decision = {
            'enable': enable,
            'reason': reason,
            'alpha_scale': alpha_scale,
            'proc_var_scale': proc_var_scale,
            'n_triggers': len(triggers),
        }
        
        self.trigger_history.append(decision)
        return decision
    
    def _update_dynamic_thresholds(self):
        """
        根据历史数据动态调整阈值:
        - 如果数据整体偏高,提高阈值(避免过度触发)
        - 如果数据整体偏低,降低阈值(提高敏感度)
        """
        recent_rmse = self.rmse_history[-10:]
        recent_var = self.var_history[-10:]
        
        # RMSE阈值: 使用75分位数作为基准
        rmse_p75 = np.percentile(recent_rmse, 75)
        self.dynamic_rmse_threshold = max(
            self.rmse_threshold,  # 不低于初始设定
            rmse_p75 * 1.2  # 略高于近期p75
        )
        
        # 方差阈值: 类似逻辑
        var_p75 = np.percentile(recent_var, 75)
        self.dynamic_var_threshold = max(
            self.var_threshold,
            var_p75 * 1.5
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """返回触发统计"""
        if not self.trigger_history:
            return {}
        
        total = len(self.trigger_history)
        enabled = sum(1 for d in self.trigger_history if d['enable'])
        
        return {
            'total_rounds': total,
            'postprocess_enabled': enabled,
            'trigger_rate': enabled / total if total > 0 else 0.0,
            'avg_alpha_scale': np.mean([d['alpha_scale'] for d in self.trigger_history]),
            'current_rmse_threshold': self.dynamic_rmse_threshold,
            'current_var_threshold': self.dynamic_var_threshold,
        }


# ========== 集成到主算法的修改 ========== #

def _postprocess_filter_conditional(est_by_e: Dict, graph: Optional[Dict], 
                                   kstate: Dict, round_idx: int,
                                   rmse_raw: float, var_est: float,
                                   alpha_lap: float, proc_var: float, 
                                   obs_var_base: float,
                                   postprocessor: AdaptivePostprocessor,
                                   truth_range: Optional[float] = None):
    """
    带条件触发的后处理函数
    
    参数:
        postprocessor: AdaptivePostprocessor实例
        truth_range: truth数据的范围(max-min),用于相对误差判断
    """
    # 获取触发决策
    decision = postprocessor.should_postprocess(
        round_idx=round_idx,
        rmse_raw=rmse_raw,
        var_est=var_est,
        truth_range=truth_range
    )
    
    # 如果不需要后处理,直接返回原始估计
    if not decision['enable']:
        return est_by_e, decision
    
    # 需要后处理: 根据决策调整参数
    adaptive_alpha = alpha_lap * decision['alpha_scale']
    adaptive_proc_var = proc_var * decision['proc_var_scale']
    
    # ========== Kalman滤波 ========== #
    
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
        
        # Kalman更新(使用自适应参数)
        pred_m = st.m
        pred_v = st.v + adaptive_proc_var
        R = float(obs_var_base)
        K = pred_v / (pred_v + R)
        st.m = float(pred_m + K * (float(obs) - pred_m))
        st.v = float((1.0 - K) * pred_v)
        est_k[e] = st.m
    
    # ========== Graph Laplacian平滑 ========== #
    
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
        est_pp[e] = (1.0 - adaptive_alpha) * val + adaptive_alpha * neigh_mean
    
    return est_pp, decision


# ========== 在sa_htd_paper主函数中的使用 ========== #

def sa_htd_paper_with_adaptive_postprocess(rounds_iter, n_workers: int, params=None):
    """
    集成自适应后处理的SAHTD-Nexus算法
    """
    # ... (前面的初始化代码保持不变) ...
    
    # 🔧 新增: 初始化自适应后处理器
    postprocessor = AdaptivePostprocessor(
        rmse_threshold=float(getattr(params, 'rmse_threshold', 40.0) if params else 40.0),
        var_threshold=float(getattr(params, 'var_threshold', 100.0) if params else 100.0),
        warmup_rounds=int(getattr(params, 'warmup_rounds', 5) if params else 5),
        enable_adaptive=bool(getattr(params, 'enable_adaptive_postprocess', True) if params else True)
    )
    
    # 用于计算truth范围(仅用于相对误差判断)
    truth_min, truth_max = float('inf'), float('-inf')
    
    logs = []
    
    for r_idx, batch in enumerate(rounds_iter):
        # ... (路由、聚合等逻辑保持不变) ...
        
        truth = np.asarray(batch.truth, float)
        
        # 更新truth范围
        truth_min = min(truth_min, float(truth.min()))
        truth_max = max(truth_max, float(truth.max()))
        truth_range = truth_max - truth_min if truth_max > truth_min else None
        
        # ... (est计算逻辑) ...
        
        rmse_raw = _rmse(est, truth)
        var_est = float(np.var(est)) if est.size else float('nan')
        
        # 🔧 条件性后处理
        est_by_e = {e: float(est[j]) for j, e in enumerate(entities)}
        est_pp_by_e, pp_decision = _postprocess_filter_conditional(
            est_by_e, entity_graph, kalman_state, r_idx,
            rmse_raw=rmse_raw,
            var_est=var_est,
            alpha_lap=alpha_lap,
            proc_var=proc_var,
            obs_var_base=obs_var_base,
            postprocessor=postprocessor,
            truth_range=truth_range
        )
        
        est_pp = np.array([est_pp_by_e[e] for e in entities], float)
        rmse = _rmse(est_pp, truth)
        
        # ... (日志记录) ...
        
        logs.append(dict(
            rmse=float(rmse),
            rmse_raw=float(rmse_raw),
            # 🔧 新增: 后处理决策信息
            postprocess_enabled=bool(pp_decision['enable']),
            postprocess_reason=str(pp_decision['reason']),
            postprocess_alpha_scale=float(pp_decision.get('alpha_scale', 0.0)),
            postprocess_n_triggers=int(pp_decision.get('n_triggers', 0)),
            # ... (其他字段) ...
        ))
    
    # 🔧 在最后打印统计信息
    pp_stats = postprocessor.get_statistics()
    if pp_stats:
        warnings.warn(
            f"[Adaptive Postprocess] Triggered in {pp_stats['trigger_rate']:.1%} of rounds "
            f"({pp_stats['postprocess_enabled']}/{pp_stats['total_rounds']})"
        )
    
    return logs


# ========== 参数配置建议 ========== #

"""
在suite_paramgrid_all.py中添加以下参数:

ap.add_argument('--enable_adaptive_postprocess', 
                type=lambda s: str(s).lower() == 'true',
                default=True,
                help="是否启用自适应后处理(True=根据RMSE触发,False=始终开启)")

ap.add_argument('--rmse_threshold', type=float, default=40.0,
                help="RMSE超过此值才启用后处理")

ap.add_argument('--var_threshold', type=float, default=100.0,
                help="方差超过此值才启用后处理")

ap.add_argument('--warmup_rounds', type=int, default=5,
                help="前N轮禁用后处理,让算法先收敛")

# 针对不同数据集的推荐设置:

# NYC数据集(值域较大,需要平滑):
--rmse_threshold 40.0 --var_threshold 100.0 --warmup_rounds 5

# SPBC数据集(值域小,避免过度平滑):
--rmse_threshold 15.0 --var_threshold 30.0 --warmup_rounds 3

# 关闭自适应(总是开启后处理,等价于旧版本):
--enable_adaptive_postprocess false
"""
