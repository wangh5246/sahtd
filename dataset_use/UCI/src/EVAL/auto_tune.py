#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动超参数调优脚本
==================

用法:
    python auto_tune.py --dataset nyc --epsilon 1.0 --n_trials 50
    python auto_tune.py --dataset spbc --epsilon 0.5 --n_trials 30

输出:
    - best_params.json: 最优参数
    - optimization_history.csv: 完整搜索历史
    - optimization_trace.png: 优化过程可视化
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path，保证 dataset_use 可被发现
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# 也把当前目录放进去方便相对模块（bayesian_tuner/reduced_params）
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bayesian_tuner import BayesianHyperparameterTuner, create_objective_function_from_suite
from reduced_params import ReducedParams


# ========== 数据集配置 ========== #

DATASET_CONFIGS = {
    'nyc': {
        'reports_csv': '/Users/wanghao/Desktop/SA-HTD/dataset_use/NYC/data/reports.csv',
        'truth_csv': '/Users/wanghao/Desktop/SA-HTD/dataset_use/NYC/data/truth.csv',
        'time_bin': '5min',
        'n_workers': 300,
        'suite': {
            'epsilon': 1.0,  # 会被命令行覆盖
            'rho': 1.0,
            'mal_rate': 0.0,
            'rounds': 12,
        }
    },
    'la': {
        'reports_csv': '/Users/wanghao/Desktop/SA-HTD/dataset_use/metr-LA/data/reports.csv',
        'truth_csv': '/Users/wanghao/Desktop/SA-HTD/dataset_use/metr-LA/data/truth.csv',
        'time_bin': '10min',
        'n_workers': 50,
        'suite': {
            'epsilon': 1.0,
            'rho': 1.0,
            'mal_rate': 0.0,
            'rounds': 12,
        }
    },
    'uci': {
        'reports_csv': '/Users/wanghao/Desktop/SA-HTD/dataset_use/UCI/data/reports.csv',
        'truth_csv': '/Users/wanghao/Desktop/SA-HTD/dataset_use/UCI/data/truth.csv',
        'time_bin': '10min',
        'n_workers': 300,
        'suite': {
            'epsilon': 1.0,
            'rho': 1.0,
            'mal_rate': 0.0,
            'rounds': 12,
        }
    },
'sk': {
        'reports_csv': '/Users/wanghao/Desktop/SA-HTD/dataset_use/skopje-full/data/reports.csv',
        'truth_csv': '/Users/wanghao/Desktop/SA-HTD/dataset_use/skopje-full/data/truth.csv',
        'time_bin': '60min',
        'n_workers': 300,
        'suite': {
            'epsilon': 1.0,
            'rho': 1.0,
            'mal_rate': 0.0,
            'rounds': 12,
        }
    },
}


def main():
    parser = argparse.ArgumentParser(
        description='自动超参数调优 - 贝叶斯优化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # NYC数据集, epsilon=1.0, 50次搜索
  python auto_tune.py --dataset nyc --epsilon 1.0 --n_trials 50
  
  # SPBC数据集, epsilon=0.5, 快速搜索
  python auto_tune.py --dataset spbc --epsilon 0.5 --n_trials 20
  
  # 使用自定义数据路径
  python auto_tune.py --dataset custom \\
    --reports_csv ./data/reports.csv \\
    --truth_csv ./data/truth.csv \\
    --epsilon 1.0 --n_trials 30
        """
    )
    
    # 数据集选择
    parser.add_argument('--dataset', type=str, required=True,
                       choices=list(DATASET_CONFIGS.keys()) + ['custom'],
                       help='数据集名称')
    
    # 自定义数据集路径
    parser.add_argument('--reports_csv', type=str, default='',
                       help='自定义reports.csv路径 (仅当dataset=custom时)')
    parser.add_argument('--truth_csv', type=str, default='',
                       help='自定义truth.csv路径 (仅当dataset=custom时)')
    parser.add_argument('--time_bin', type=str, default='20min',
                       help='时间分箱粒度 (仅当dataset=custom时)')
    parser.add_argument('--n_workers', type=int, default=300,
                       help='工人数量 (仅当dataset=custom时)')
    
    # 优化参数
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='隐私预算 ε')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='贝叶斯优化搜索次数 (推荐30-100)')
    parser.add_argument('--seed', type=int, default=2025,
                       help='随机种子')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='',
                       help='结果输出目录 (默认: ./tuning_results/{dataset}_eps{epsilon})')
    
    # 实验配置
    parser.add_argument('--rho', type=float, default=1.0,
                       help='参与率')
    parser.add_argument('--mal_rate', type=float, default=0.0,
                       help='恶意工人比例')
    parser.add_argument('--rounds', type=int, default=12,
                       help='实验轮数')
    
    args = parser.parse_args()
    
    # ========== 加载配置 ========== #
    
    if args.dataset == 'custom':
        if not args.reports_csv or not args.truth_csv:
            parser.error("使用custom数据集时必须提供--reports_csv和--truth_csv")
        
        config = {
            'reports_csv': args.reports_csv,
            'truth_csv': args.truth_csv,
            'time_bin': args.time_bin,
            'n_workers': args.n_workers,
            'suite': {
                'epsilon': args.epsilon,
                'rho': args.rho,
                'mal_rate': args.mal_rate,
                'rounds': args.rounds,
            }
        }
    else:
        config = DATASET_CONFIGS[args.dataset].copy()
        config['suite']['epsilon'] = args.epsilon
        config['suite']['rho'] = args.rho
        config['suite']['mal_rate'] = args.mal_rate
        config['suite']['rounds'] = args.rounds
    
    # 输出目录
    if not args.output_dir:
        output_dir = f"./tuning_results/{args.dataset}_eps{args.epsilon}"
    else:
        output_dir = args.output_dir
    
    # ========== 打印配置 ========== #
    
    print("\n" + "="*70)
    print("  自动超参数调优 - SAHTD-Nexus")
    print("="*70)
    print(f"数据集: {args.dataset}")
    print(f"  Reports: {config['reports_csv']}")
    print(f"  Truth: {config['truth_csv']}")
    print(f"  时间粒度: {config['time_bin']}")
    print(f"  工人数: {config['n_workers']}")
    print(f"\n实验配置:")
    print(f"  ε = {args.epsilon}")
    print(f"  ρ = {args.rho}")
    print(f"  恶意率 = {args.mal_rate}")
    print(f"  轮数 = {args.rounds}")
    print(f"\n优化设置:")
    print(f"  搜索次数: {args.n_trials}")
    print(f"  随机种子: {args.seed}")
    print(f"  输出目录: {output_dir}")
    print("="*70 + "\n")
    
    # 确认继续
    response = input("开始优化? [y/N]: ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # ========== 创建目标函数 ========== #
    
    print("\n[1/3] 创建目标函数...")
    try:
        objective = create_objective_function_from_suite(
            reports_csv=config['reports_csv'],
            truth_csv=config['truth_csv'],
            time_bin=config['time_bin'],
            n_workers=config['n_workers'],
            suite_config=config['suite'],
            method_name='sa_htd_paper'
        )
        print("✓ 目标函数创建成功")
    except Exception as e:
        print(f"✗ 创建失败: {e}")
        return
    
    # ========== 运行优化 ========== #
    
    print("\n[2/3] 运行贝叶斯优化...")
    tuner = BayesianHyperparameterTuner(
        objective_func=objective,
        n_trials=args.n_trials,
        seed=args.seed,
        output_dir=output_dir
    )
    tuner.set_epsilon(args.epsilon)
    
    best_params, best_score = tuner.optimize(verbose=True)
    
    # ========== 生成可视化 ========== #
    
    print("\n[3/3] 生成可视化...")
    try:
        tuner.plot_optimization_trace(
            save_path=Path(output_dir) / 'optimization_trace.png'
        )
        print("✓ 可视化已保存")
    except Exception as e:
        print(f"⚠ 可视化失败: {e}")
    
    # ========== 总结 ========== #
    
    print("\n" + "="*70)
    print("  优化完成!")
    print("="*70)
    print(f"\n最优RMSE: {best_score:.4f}")
    print(f"\n最优参数 (用于论文):")
    print(best_params.get_description())
    
    print(f"\n推导的完整参数 (前10个):")
    full = best_params.to_full_params()
    for i, (k, v) in enumerate(list(full.items())[:10]):
        print(f"  {k:<30} = {v}")
    print(f"  ... (共{len(full)}个参数)")
    
    print(f"\n结果文件:")
    print(f"  - {output_dir}/best_params.json")
    print(f"  - {output_dir}/optimization_history.csv")
    print(f"  - {output_dir}/optimization_trace.png")
    print(f"  - {output_dir}/best_params_full.json")
    
    print(f"\n下一步:")
    print(f"  1. 查看 optimization_trace.png 确认收敛")
    print(f"  2. 使用 best_params.json 运行完整实验:")
    print(f"     python suite_paramgrid_all.py \\")
    print(f"       --preset balanced \\")
    print(f"       --routing_intensity {best_params.routing_intensity:.3f} \\")
    print(f"       --communication_budget {best_params.communication_budget:.3f} \\")
    print(f"       --smoothing_strength {best_params.smoothing_strength:.3f} \\")
    print(f"       --adaptive_sensitivity {best_params.adaptive_sensitivity:.3f}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
