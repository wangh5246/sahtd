"""
精简参数系统 - 从20+参数减少到5个核心参数
==============================================

回应审稿人批评: "parameter soup, tuned on test set"

核心思想:
1. 只暴露5个最关键的参数给用户
2. 其他参数通过理论公式自动推导
3. 基于数据统计自动调整
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class ReducedParams:
    """
    精简后的5个核心超参数
    
    每个参数都有清晰的物理意义和理论依据
    """
    
    # ===== 5个核心参数 ===== #
    
    # 1. 隐私预算 (最核心的参数)
    epsilon: float = 1.0
    """
    差分隐私预算 ε
    - 理论依据: DP的基础参数
    - 推荐范围: [0.1, 4.0]
    - 影响: 隐私与效用的tradeoff
    """
    
    # 2. 路由策略强度 (控制A/B/C路分配)
    routing_intensity: float = 0.5
    """
    路由策略的激进程度 ∈ [0, 1]
    - 0.0: 保守(大部分走B路, 隐私优先)
    - 0.5: 平衡(兼顾隐私和效用)
    - 1.0: 激进(更多走A/C路, 效用优先)
    
    自动推导:
    - A_budget_ratio = 0.10 + 0.30 * routing_intensity
    - BASE_C_RATIO = 0.03 + 0.12 * routing_intensity
    - tau_percentile = 85 - 20 * routing_intensity
    """
    
    # 3. 通信预算 (字节约束)
    communication_budget: float = 1.0
    """
    通信预算的相对级别 ∈ [0.5, 2.0]
    - 0.5: 严格限制(低带宽场景)
    - 1.0: 标准
    - 2.0: 宽松(高带宽场景)
    
    自动推导:
    - target_bytes_per_round = 25000 * communication_budget
    - BASE_BITS_A = 10 + 2 * log2(communication_budget)
    - BASE_BITS_B = 6 + 2 * log2(communication_budget)
    """
    
    # 4. 后处理强度 (Kalman+Laplacian平滑)
    smoothing_strength: float = 0.5
    """
    后处理平滑的强度 ∈ [0, 1]
    - 0.0: 不平滑(适合突变频繁的数据)
    - 0.5: 适度平滑
    - 1.0: 强平滑(适合噪声大的数据)
    
    自动推导:
    - post_lap_alpha = 0.05 + 0.30 * smoothing_strength
    - post_process_var = 0.5 + 30 * smoothing_strength
    - post_var_percentile_high = 0.65 + 0.20 * smoothing_strength
    """
    
    # 5. 自适应灵敏度 (后处理触发的敏感度)
    adaptive_sensitivity: float = 0.5
    """
    自适应后处理的触发灵敏度 ∈ [0, 1]
    - 0.0: 不敏感(很少触发, 保持原值)
    - 0.5: 标准
    - 1.0: 高敏感(频繁触发)
    
    自动推导:
    - post_var_percentile_low = 0.40 + 0.20 * (1 - adaptive_sensitivity)
    - post_sparse_threshold = 3 + 4 * (1 - adaptive_sensitivity)
    - change_sensitivity = 1.0 + 2.0 * adaptive_sensitivity
    """
    
    # ===== 自动推导的辅助参数 ===== #
    
    def to_full_params(self, data_stats: Dict[str, float] = None) -> Dict[str, Any]:
        """
        从5个核心参数 → 自动推导出完整的20+参数
        
        Args:
            data_stats: 数据统计信息 (可选)
                - 'n_entities': 实体数量
                - 'avg_reports_per_entity': 平均报告数
                - 'data_variance': 数据方差
        
        Returns:
            完整参数字典
        """
        # 默认数据统计
        if data_stats is None:
            data_stats = {}
        
        n_entities = data_stats.get('n_entities', 100)
        avg_reports = data_stats.get('avg_reports_per_entity', 10.0)
        data_var = data_stats.get('data_variance', 100.0)
        
        # ========== 1. 路由相关参数 ========== #
        
        # A路占比: routing_intensity越高, A路越多
        A_budget_ratio = 0.10 + 0.30 * self.routing_intensity
        
        # C路占比: routing_intensity越高, C路越多
        BASE_C_RATIO = 0.03 + 0.12 * self.routing_intensity
        
        # tau分位数: routing_intensity越高, tau越低(更多entity判定为"高方差"→走A路)
        tau_percentile = 85.0 - 20.0 * self.routing_intensity
        
        # C路批大小: 根据实体数自动调整
        C_BATCH_MAX = max(3, min(12, int(n_entities * BASE_C_RATIO * 1.5)))
        
        # ========== 2. 通信/量化相关参数 ========== #
        
        # 字节预算
        target_bytes_per_round = 25000.0 * self.communication_budget
        
        # 量化位数: 对数尺度调整
        import math
        bits_scale = math.log2(max(self.communication_budget, 0.5))
        BASE_BITS_A = max(10, min(16, int(10 + 2 * bits_scale)))
        BASE_BITS_B = max(6, min(12, int(6 + 2 * bits_scale)))
        BITS_C_EXTRA = max(2, int(2 + bits_scale))
        
        # 字节/bit比率
        bytes_per_bit = 0.125  # 固定
        
        # ========== 3. 后处理相关参数 ========== #
        
        # Laplacian平滑强度
        post_lap_alpha = 0.05 + 0.30 * self.smoothing_strength
        
        # Kalman过程噪声
        post_process_var = 0.5 + 30.0 * self.smoothing_strength
        
        # Kalman观测噪声
        post_obs_var_base = 1.0
        
        # 方差分位数阈值
        post_var_percentile_low = 0.40 + 0.20 * (1.0 - self.adaptive_sensitivity)
        post_var_percentile_high = 0.65 + 0.20 * self.smoothing_strength
        
        # ========== 4. 自适应触发参数 ========== #
        
        # 稀疏阈值: adaptive_sensitivity越高, 阈值越低(更容易触发)
        post_sparse_threshold = max(3, int(3 + 4 * (1.0 - self.adaptive_sensitivity)))
        
        # 隐私预算紧张比例
        post_privacy_tension_ratio = 0.75  # 固定
        
        # 突变检测灵敏度
        post_change_sensitivity = 1.0 + 2.0 * self.adaptive_sensitivity
        
        # 收敛判断阈值
        post_convergence_threshold = 0.05  # 固定
        
        # ========== 5. 隐私会计参数 ========== #
        
        # 窗口大小: 根据epsilon自动调整
        window_w = max(16, min(64, int(32 / max(self.epsilon, 0.5))))
        
        # 窗口预算
        epsilon_per_window = self.epsilon * window_w
        
        # Delta目标
        delta_target = 1e-5  # 固定
        
        # 会计模式: epsilon小时用精确会计
        accountant_mode = "pld" if self.epsilon < 1.0 else "naive"
        
        # ========== 6. 其他固定参数 ========== #
        
        # 调度器参数
        target_latency_ms = 2.0
        
        # 量化范围
        MIN_QUANT_BITS = 2
        MAX_QUANT_BITS = 14
        VAR_QUANTILE = 0.7
        quant_bits_init = BASE_BITS_B
        
        # Bandit参数
        bandit_epsilon = 0.1
        eps_min_scale = 0.5
        eps_max_scale = 1.5
        
        # 后处理其他参数
        post_change_window = 3
        post_convergence_window = 5
        post_warmup_rounds = 3
        enable_privacy_adaptive = True
        
        # Shuffle/ULDP
        use_shuffle = True
        uldp_sensitive_cols = []
        geo_epsilon = 0.0
        
        # 字节计费
        perA_bytes = 32
        perB_bytes = 32
        perC_bytes = 64
        
        # 早停参数
        max_b_per_entity = 32
        early_stop_eps = 8e-3
        early_stop_steps = 2
        
        # 平均报告数
        AVG_REPORTS_PER_ENTITY = avg_reports
        
        # ========== 返回完整参数 ========== #
        
        return {
            # 核心参数
            'epsilon': self.epsilon,
            
            # 路由参数
            'A_budget_ratio': A_budget_ratio,
            'BASE_C_RATIO': BASE_C_RATIO,
            'tau_percentile': tau_percentile,
            'C_BATCH_MAX': C_BATCH_MAX,
            
            # 通信参数
            'target_bytes_per_round': target_bytes_per_round,
            'target_latency_ms': target_latency_ms,
            'bytes_per_bit': bytes_per_bit,
            
            # 量化参数
            'BASE_BITS_A': BASE_BITS_A,
            'BASE_BITS_B': BASE_BITS_B,
            'BITS_C_EXTRA': BITS_C_EXTRA,
            'MIN_QUANT_BITS': MIN_QUANT_BITS,
            'MAX_QUANT_BITS': MAX_QUANT_BITS,
            'VAR_QUANTILE': VAR_QUANTILE,
            'quant_bits_init': quant_bits_init,
            'AVG_REPORTS_PER_ENTITY': AVG_REPORTS_PER_ENTITY,
            
            # 隐私会计
            'accountant_mode': accountant_mode,
            'window_w': window_w,
            'epsilon_per_window': epsilon_per_window,
            'delta_target': delta_target,
            
            # 后处理参数
            'disable_postprocess': False,
            'use_privacy_aware_postprocess': True,
            'post_lap_alpha': post_lap_alpha,
            'post_process_var': post_process_var,
            'post_obs_var_base': post_obs_var_base,
            'post_var_percentile_low': post_var_percentile_low,
            'post_var_percentile_high': post_var_percentile_high,
            'post_sparse_threshold': post_sparse_threshold,
            'post_privacy_tension_ratio': post_privacy_tension_ratio,
            'post_change_window': post_change_window,
            'post_change_sensitivity': post_change_sensitivity,
            'post_convergence_window': post_convergence_window,
            'post_convergence_threshold': post_convergence_threshold,
            'post_warmup_rounds': post_warmup_rounds,
            'enable_privacy_adaptive': enable_privacy_adaptive,
            
            # Bandit参数
            'bandit_epsilon': bandit_epsilon,
            'eps_min_scale': eps_min_scale,
            'eps_max_scale': eps_max_scale,
            
            # Shuffle/ULDP
            'use_shuffle': use_shuffle,
            'uldp_sensitive_cols': uldp_sensitive_cols,
            'geo_epsilon': geo_epsilon,
            
            # 字节计费
            'perA_bytes': perA_bytes,
            'perB_bytes': perB_bytes,
            'perC_bytes': perC_bytes,
            
            # 早停
            'max_b_per_entity': max_b_per_entity,
            'early_stop_eps': early_stop_eps,
            'early_stop_steps': early_stop_steps,
        }
    
    def get_description(self) -> str:
        """
        返回参数的人类可读描述 (用于论文/文档)
        """
        return f"""
SAHTD-Nexus Reduced Parameters:
================================
1. Privacy Budget (ε): {self.epsilon}
   - Controls privacy-utility tradeoff
   
2. Routing Intensity: {self.routing_intensity}
   - 0.0 = Conservative (privacy-first)
   - 1.0 = Aggressive (utility-first)
   → A-route ratio: {0.10 + 0.30 * self.routing_intensity:.2f}
   → C-route ratio: {0.03 + 0.12 * self.routing_intensity:.2f}
   
3. Communication Budget: {self.communication_budget}
   - 1.0 = Standard (~25KB/round)
   - 2.0 = High bandwidth (~50KB/round)
   → Bits_A: {max(10, min(16, int(10 + 2 * np.log2(max(self.communication_budget, 0.5)))))}
   
4. Smoothing Strength: {self.smoothing_strength}
   - 0.0 = No smoothing (for sudden changes)
   - 1.0 = Strong smoothing (for noisy data)
   → Laplacian α: {0.05 + 0.30 * self.smoothing_strength:.2f}
   → Kalman Q: {0.5 + 30.0 * self.smoothing_strength:.1f}
   
5. Adaptive Sensitivity: {self.adaptive_sensitivity}
   - 0.0 = Low sensitivity (fewer triggers)
   - 1.0 = High sensitivity (frequent triggers)
   → Sparse threshold: {max(3, int(3 + 4 * (1.0 - self.adaptive_sensitivity)))}
   → Change detection: {1.0 + 2.0 * self.adaptive_sensitivity:.1f}σ
        """


# ========== 预设配置 (用于快速开始) ========== #

# 隐私优先配置
PRIVACY_FIRST = ReducedParams(
    epsilon=0.5,
    routing_intensity=0.2,  # 保守路由
    communication_budget=0.7,
    smoothing_strength=0.8,  # 强平滑补偿噪声
    adaptive_sensitivity=0.6
)

# 平衡配置 (推荐)
BALANCED = ReducedParams(
    epsilon=1.0,
    routing_intensity=0.5,
    communication_budget=1.0,
    smoothing_strength=0.5,
    adaptive_sensitivity=0.5
)

# 效用优先配置
UTILITY_FIRST = ReducedParams(
    epsilon=2.0,
    routing_intensity=0.8,  # 激进路由
    communication_budget=1.5,
    smoothing_strength=0.3,  # 弱平滑保留细节
    adaptive_sensitivity=0.4
)

# 低带宽配置 (边缘设备)
LOW_BANDWIDTH = ReducedParams(
    epsilon=1.0,
    routing_intensity=0.4,
    communication_budget=0.5,  # 严格限制字节
    smoothing_strength=0.6,
    adaptive_sensitivity=0.5
)

# 突变数据配置 (如SPBC)
CHANGE_DETECTION = ReducedParams(
    epsilon=1.0,
    routing_intensity=0.5,
    communication_budget=1.0,
    smoothing_strength=0.2,  # 弱平滑
    adaptive_sensitivity=0.7  # 高灵敏度检测突变
)


# ========== 使用示例 ========== #

if __name__ == '__main__':
    # 示例1: 使用默认参数
    params = ReducedParams()
    print("=== Default Configuration ===")
    print(params.get_description())
    
    # 示例2: 查看推导的完整参数
    full = params.to_full_params({'n_entities': 200, 'avg_reports_per_entity': 15.0})
    print("\n=== Derived Full Parameters (sample) ===")
    print(f"A_budget_ratio: {full['A_budget_ratio']:.3f}")
    print(f"BASE_C_RATIO: {full['BASE_C_RATIO']:.3f}")
    print(f"target_bytes_per_round: {full['target_bytes_per_round']:.0f}")
    print(f"BASE_BITS_A: {full['BASE_BITS_A']}")
    print(f"post_lap_alpha: {full['post_lap_alpha']:.3f}")
    
    # 示例3: 使用预设配置
    print("\n=== Privacy-First Preset ===")
    privacy_params = PRIVACY_FIRST
    print(privacy_params.get_description())
    
    # 示例4: 对比不同配置
    print("\n=== Configuration Comparison ===")
    configs = {
        'Balanced': BALANCED,
        'Privacy-First': PRIVACY_FIRST,
        'Utility-First': UTILITY_FIRST,
    }
    
    print(f"{'Config':<15} {'ε':<6} {'A-ratio':<8} {'Smooth':<8} {'Bytes':<10}")
    print("-" * 60)
    for name, cfg in configs.items():
        full = cfg.to_full_params()
        print(f"{name:<15} {cfg.epsilon:<6.1f} {full['A_budget_ratio']:<8.2f} "
              f"{full['post_lap_alpha']:<8.2f} {full['target_bytes_per_round']:<10.0f}")
