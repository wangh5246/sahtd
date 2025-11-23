#!/bin/bash
# run_all_experiments.sh
# 一键运行所有IJCAI实验
# 
# 使用方法:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh

set -e  # 遇到错误立即退出

echo "========================================"
echo " IJCAI实验 - 完整运行脚本"
echo "========================================"

# ========== 配置 ========== #

# 路径配置
REPORTS_CSV="/Users/wanghao/Desktop/SA-HTD/dataset_use/UCI/data/reports.csv"
TRUTH_CSV="/Users/wanghao/Desktop/SA-HTD/dataset_use/UCI/data/truth.csv"
OUTPUT_DIR="/Users/wanghao/Desktop/SA-HTD/dataset_use/UCI/results_fair_comparison"

# 实验参数
N_WORKERS=300
TIME_BIN="10min"
NUM_PROCS=2  # 并行进程数

# 方法列表 (你的方法 + 公平对比的baselines)
METHODS="sa_htd_paper,dp_sgd_fair,private_kf_fair,adaptive_dp_fl_fair,eptd,etbp_td,fed_sense"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo ""
echo "[INFO] 配置:"
echo "  数据集: UCI"
echo "  Reports: $REPORTS_CSV"
echo "  Truth: $TRUTH_CSV"
echo "  输出目录: $OUTPUT_DIR"
echo "  方法: $METHODS"
echo "  并行进程: $NUM_PROCS"
echo ""

# ========== P0实验: 主实验 (不同隐私预算) ========== #

echo "========================================"
echo " P0实验: 主实验 (6个epsilon值)"
echo "========================================"

for eps in 0.1 0.3 0.5 1.0 2.0 4.0; do
    echo ""
    echo "[$(date '+%H:%M:%S')] 运行 epsilon=$eps"
    
    python suite_paramgrid_all.py \
        --reports_csv "$REPORTS_CSV" \
        --truth_csv "$TRUTH_CSV" \
        --outdir "$OUTPUT_DIR/main_exp/eps_${eps}" \
        --n_workers $N_WORKERS \
        --time_bin $TIME_BIN \
        --methods "$METHODS" \
        --suites_json "[{\"epsilon\": $eps, \"rho\": 1.0, \"mal_rate\": 0.0, \"rounds\": 12}]" \
        --num_procs $NUM_PROCS \
        --seed 2025
    
    echo "[✓] epsilon=$eps 完成"
done

echo ""
echo "[✓] P0实验完成! 结果: $OUTPUT_DIR/main_exp/"

# ========== P1实验: 不同参与率 ========== #

echo ""
echo "========================================"
echo " P1实验: 不同参与率 (ε=1.0)"
echo "========================================"

for rho in 0.15 0.20 0.25 0.30 0.50 1.0; do
    echo ""
    echo "[$(date '+%H:%M:%S')] 运行 rho=$rho"
    
    python suite_paramgrid_all.py \
        --reports_csv "$REPORTS_CSV" \
        --truth_csv "$TRUTH_CSV" \
        --outdir "$OUTPUT_DIR/participation/rho_${rho}" \
        --n_workers $N_WORKERS \
        --time_bin $TIME_BIN \
        --methods "$METHODS" \
        --suites_json "[{\"epsilon\": 1.0, \"rho\": $rho, \"mal_rate\": 0.0, \"rounds\": 12}]" \
        --num_procs $NUM_PROCS \
        --seed 2025
    
    echo "[✓] rho=$rho 完成"
done

echo ""
echo "[✓] P1实验完成! 结果: $OUTPUT_DIR/participation/"

# ========== P2实验: 对抗鲁棒性 ========== #

echo ""
echo "========================================"
echo " P2实验: 对抗鲁棒性 (ε=1.0, ρ=0.2)"
echo "========================================"

for mal in 0.0 0.1 0.2 0.3; do
    echo ""
    echo "[$(date '+%H:%M:%S')] 运行 mal_rate=$mal"
    
    python suite_paramgrid_all.py \
        --reports_csv "$REPORTS_CSV" \
        --truth_csv "$TRUTH_CSV" \
        --outdir "$OUTPUT_DIR/robustness/mal_${mal}" \
        --n_workers $N_WORKERS \
        --time_bin $TIME_BIN \
        --methods "$METHODS" \
        --suites_json "[{\"epsilon\": 1.0, \"rho\": 0.2, \"mal_rate\": $mal, \"rounds\": 12}]" \
        --num_procs $NUM_PROCS \
        --seed 2025
    
    echo "[✓] mal_rate=$mal 完成"
done

echo ""
echo "[✓] P2实验完成! 结果: $OUTPUT_DIR/robustness/"

# ========== Ablation Study ========== #

echo ""
echo "========================================"
echo " Ablation Study (ε=1.0)"
echo "========================================"

# 只运行你的方法,用不同配置
ABLATION_CONFIGS=(
    "full:--use_reduced_params true --routing_intensity 0.5 --smoothing_strength 0.5"
    "no_routing:--use_reduced_params true --routing_intensity 0.0"
    "no_smoothing:--use_reduced_params true --smoothing_strength 0.0"
    "no_adaptive:--use_reduced_params true --adaptive_sensitivity 0.0"
)

for config in "${ABLATION_CONFIGS[@]}"; do
    name="${config%%:*}"
    params="${config#*:}"
    
    echo ""
    echo "[$(date '+%H:%M:%S')] Ablation: $name"
    
    python suite_paramgrid_all.py \
        --reports_csv "$REPORTS_CSV" \
        --truth_csv "$TRUTH_CSV" \
        --outdir "$OUTPUT_DIR/ablation/$name" \
        --n_workers $N_WORKERS \
        --time_bin $TIME_BIN \
        --methods "sa_htd_paper" \
        --suites_json '[{"epsilon": 1.0, "rho": 1.0, "mal_rate": 0.0, "rounds": 12}]' \
        --num_procs 1 \
        --seed 2025 \
        $params
    
    echo "[✓] Ablation $name 完成"
done

echo ""
echo "[✓] Ablation Study完成! 结果: $OUTPUT_DIR/ablation/"

# ========== 生成汇总报告 ========== #

echo ""
echo "========================================"
echo " 生成汇总报告"
echo "========================================"

python - <<EOF
import pandas as pd
import glob
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")

# 收集所有merged_results.csv
all_results = []
for csv_file in output_dir.rglob("merged_results.csv"):
    df = pd.read_csv(csv_file)
    df['experiment'] = csv_file.parent.parent.name
    all_results.append(df)

if all_results:
    summary = pd.concat(all_results, ignore_index=True)
    summary.to_csv(output_dir / "summary_all.csv", index=False)
    
    # 按方法分组统计
    grouped = summary.groupby(['method', 'epsilon'])['rmse_mean'].agg(['mean', 'std', 'count'])
    print("\n" + "="*70)
    print(" 汇总结果 (RMSE)")
    print("="*70)
    print(grouped)
    print("\n结果已保存: $OUTPUT_DIR/summary_all.csv")
else:
    print("[WARNING] 未找到结果文件")
EOF

# ========== 完成 ========== #

echo ""
echo "========================================"
echo " 所有实验完成!"
echo "========================================"
echo ""
echo "结果目录: $OUTPUT_DIR"
echo ""
echo "下一步:"
echo "1. 查看汇总结果: $OUTPUT_DIR/summary_all.csv"
echo "2. 生成表格和图表 (运行 generate_tables.py)"
echo "3. 更新论文"
echo ""
echo "实验运行完毕! 🎉"
