from reduced_params import *

# 对比不同配置的推导结果
configs = {
    'Balanced': BALANCED,
    'Privacy-First': PRIVACY_FIRST,
    'Utility-First': UTILITY_FIRST,
    'Change Detection': CHANGE_DETECTION,
}

print("=" * 80)
print(f"{'Config':<20} {'ε':<6} {'A%':<6} {'C%':<6} {'Bits_A':<8} {'Smooth_α':<10} {'Bytes':<10}")
print("=" * 80)

for name, cfg in configs.items():
    full = cfg.to_full_params({'n_entities': 150})
    print(f"{name:<20} {cfg.epsilon:<6.1f} "
          f"{full['A_budget_ratio']*100:<6.1f} "
          f"{full['BASE_C_RATIO']*100:<6.1f} "
          f"{full['BASE_BITS_A']:<8} "
          f"{full['post_lap_alpha']:<10.3f} "
          f"{full['target_bytes_per_round']:<10.0f}")

print("=" * 80)
