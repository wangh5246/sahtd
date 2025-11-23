# -*- coding: utf-8 -*-
"""
suite_stats_significance.py
===========================
对某个 results 目录（含 merged_rounds.csv）进行 Friedman 排名检验，并导出方法平均排名和统计量 Q。
* 不依赖 scipy：实现 Friedman 统计量计算；p-value 近似留空或可选。
* 生成：ranks.csv / friedman_summary.txt / table_ranks.tex
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path

def friedman_from_rounds(df: pd.DataFrame, metric='rmse') -> dict:
    # 以 slot 为 block，把同一个 slot 下各方法的 metric 排名（越小越好）
    blocks = []
    for slot, g in df.groupby('slot'):
        g = g[['method', metric]].dropna()
        if g.empty or g['method'].nunique()<2: continue
        # 排名：相同值取平均秩
        ranks = g[metric].rank(method='average', ascending=True)
        blocks.append((g['method'].to_list(), ranks.to_numpy(float)))
    if not blocks:
        return {'k':0,'N':0,'Q':np.nan,'avg_ranks':{}}
    # 聚合到方法级
    methods = sorted({m for b in blocks for m in b[0]})
    m2idx = {m:i for i,m in enumerate(methods)}
    k = len(methods); N = len(blocks)
    R = np.zeros((N,k), float)
    for i,(ml, rk) in enumerate(blocks):
        for m, r in zip(ml, rk):
            R[i, m2idx[m]] = r
    avg_r = R.mean(axis=0)  # 各方法平均秩
    # Friedman 统计量（无 ties 修正）
    Q = (12*N)/(k*(k+1)) * np.sum(avg_r**2) - 3*N*(k+1)
    return {'k':k,'N':N,'Q':float(Q),'avg_ranks':{m:float(avg_r[m2idx[m]]) for m in methods}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results_suite_timebin')
    ap.add_argument('--metric', default='rmse')  # 或 'bytes' 等
    args = ap.parse_args()

    df = pd.read_csv(Path(args.results_dir)/'merged_rounds.csv')
    out = Path(args.results_dir)

    info = friedman_from_rounds(df, metric=args.metric)
    # 保存排名
    ranks_df = pd.DataFrame([{'method':m, 'avg_rank':r} for m,r in info['avg_ranks'].items()]).sort_values('avg_rank')
    ranks_df.to_csv(out/'ranks.csv', index=False)

    # 文本摘要
    with open(out/'friedman_summary.txt','w',encoding='utf-8') as f:
        f.write(f"methods={info['k']} N_blocks={info['N']} Q={info['Q']:.4f}\n")
        for _,row in ranks_df.iterrows():
            f.write(f"{row['method']}: avg_rank={row['avg_rank']:.3f}\n")

    # LaTeX 表（简单版）
    with open(out/'table_ranks.tex','w',encoding='utf-8') as f:
        f.write("\\begin{tabular}{l r}\\hline\n方法 & 平均秩\\\\\\hline\n")
        for _,row in ranks_df.iterrows():
            f.write(f"{row['method']} & {row['avg_rank']:.3f}\\\\\n")
        f.write("\\hline\\end{tabular}\n")
    print(f"已写入：{out/'ranks.csv'}, {out/'friedman_summary.txt'}, {out/'table_ranks.tex'}")

if __name__ == '__main__':
    main()
