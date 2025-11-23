# -*- coding: utf-8 -*-
"""
analysis.py
===========
对各实验套件输出（merged_rounds.csv / merged_results.csv）进行统一分析与可视化。
- 纯本地依赖：numpy / pandas / matplotlib（不使用 seaborn；每图单独 Figure；不指定颜色）。
- 功能：多目录汇总、按套件/方法聚合、Friedman 排名检验、Pareto 前沿、效率指标（RMSE/KB）、
        预算/ε 扫描曲线、鲁棒性对比，以及 LaTeX 表格与图表输出。

用法示例：
python analysis.py \
  --results_dirs results_suite_timebin,results_suite_budget,results_suite_epsilon,results_suite_robust \
  --out_dir analysis_outputs \
  --metric rmse \
  --bootstrap 1000 \
  --alpha 0.05 \
  --save_plots 综合工具
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)

def load_results_dirs(dirs):
    rounds_all, results_all = [], []
    for d in dirs:
        p = Path(d)
        suite = p.name
        r1, r2 = p/"merged_rounds.csv", p/"merged_results.csv"
        if r1.exists():
            df = pd.read_csv(r1)
            df["suite"] = suite
            rounds_all.append(df)
        if r2.exists():
            df2 = pd.read_csv(r2)
            df2["suite"] = suite
            results_all.append(df2)
    rounds_df = pd.concat(rounds_all, ignore_index=True) if rounds_all else pd.DataFrame()
    results_df = pd.concat(results_all, ignore_index=True) if results_all else pd.DataFrame()
    return rounds_df, results_df

def summarize_methods(rounds_df: pd.DataFrame, group_keys=("suite","method")) -> pd.DataFrame:
    if rounds_df.empty: return pd.DataFrame()
    df = rounds_df.copy()
    for col in ["rmse","var","resid_var","bytes","enc_ops","time_s"]:
        if col in df.columns: df[col] = _safe_num(df[col])
    agg = df.groupby(list(group_keys)).agg(
        rmse_mean=("rmse","mean"),
        rmse_std =("rmse","std"),
        var_mean =("var","mean"),
        resid_var_mean=("resid_var","mean"),
        bytes_mean=("bytes","mean"),
        enc_ops_mean=("enc_ops","mean"),
        time_s_mean=("time_s","mean"),
        rounds=("rmse","size")
    ).reset_index()
    if "slot" in df.columns:
        last_rmse = df.sort_values("slot").groupby(list(group_keys))["rmse"].last().reset_index(name="rmse_last")
    else:
        last_rmse = df.groupby(list(group_keys))["rmse"].last().reset_index(name="rmse_last")
    out = pd.merge(agg, last_rmse, on=list(group_keys), how="left")
    out["rmse_per_kb"] = out["rmse_mean"] / (out["bytes_mean"]/1024.0).replace(0, np.nan)
    return out

def friedman_from_rounds(rounds_df: pd.DataFrame, metric="rmse", by_suite=True) -> pd.DataFrame:
    if rounds_df.empty or "method" not in rounds_df.columns: return pd.DataFrame()
    work = rounds_df.copy()
    work[metric] = _safe_num(work[metric])
    suites = [None] if not by_suite else sorted(work["suite"].unique().tolist())
    rows = []
    for suite in suites:
        df = work if suite is None else work[work["suite"]==suite]
        if "slot" not in df.columns:
            df = df.copy(); df["slot"] = df.groupby("method").cumcount()
        blocks = []
        for slot, g in df.groupby("slot"):
            g2 = g[["method",metric]].dropna()
            if g2.empty or g2["method"].nunique()<2: continue
            ranks = g2[metric].rank(method="average", ascending=True)
            blocks.append((g2["method"].tolist(), ranks.to_numpy(float)))
        if not blocks: continue
        methods = sorted({m for b in blocks for m in b[0]})
        m2idx = {m:i for i,m in enumerate(methods)}
        k = len(methods); N = len(blocks)
        R = np.zeros((N,k), float)
        for i,(ml,rk) in enumerate(blocks):
            for m, r in zip(ml, rk):
                R[i, m2idx[m]] = r
        avg_r = R.mean(axis=0)
        Q = (12*N)/(k*(k+1)) * float(np.sum(avg_r**2)) - 3*N*(k+1)
        for m in methods:
            rows.append(dict(suite=("ALL" if suite is None else suite),
                             method=m, avg_rank=float(avg_r[m2idx[m]]),
                             Q=float(Q), k=k, N=N, metric=metric))
    return pd.DataFrame(rows)

def pareto_front(df: pd.DataFrame, rmse_col="rmse_mean", bytes_col="bytes_mean") -> pd.DataFrame:
    if df.empty: return df
    work = df.dropna(subset=[rmse_col, bytes_col]).sort_values([rmse_col, bytes_col], ascending=[True, True]).reset_index(drop=True)
    front_idx, best_bytes = [], np.inf
    for i, row in work.iterrows():
        if row[bytes_col] < best_bytes:
            front_idx.append(i); best_bytes = row[bytes_col]
    return work.loc[front_idx].reset_index(drop=True)

def bootstrap_ci(x: np.ndarray, func=np.mean, iters=1000, alpha=0.05, seed=2025):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size==0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); vals=[]; n=len(x)
    for _ in range(max(10,iters)):
        idx = rng.integers(0, n, size=n); vals.append(func(x[idx]))
    vals = np.sort(np.array(vals, float))
    lo = np.percentile(vals, 100*alpha/2.0)
    hi = np.percentile(vals, 100*(1-alpha/2.0))
    return float(lo), float(hi)

def plot_bar_rmse(summary_df: pd.DataFrame, out_dir: Path, suite: str):
    df = summary_df[summary_df["suite"]==suite].copy()
    if df.empty: return
    df = df.sort_values("rmse_mean")
    plt.figure()
    plt.bar(df["method"], df["rmse_mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMSE(均值)"); plt.title(f"{suite} - 方法 RMSE(均值)")
    plt.tight_layout(); plt.savefig(out_dir / f"{suite}_rmse_bar.png", dpi=180); plt.close()

def plot_scatter_bytes_rmse(summary_df: pd.DataFrame, out_dir: Path, suite:str):
    df = summary_df[summary_df["suite"]==suite].copy()
    if df.empty: return
    plt.figure()
    plt.scatter(df["bytes_mean"], df["rmse_mean"])
    for _, r in df.iterrows():
        plt.annotate(r["method"], (r["bytes_mean"], r["rmse_mean"]), fontsize=8)
    plt.xlabel("平均通信字节"); plt.ylabel("RMSE(均值)"); plt.title(f"{suite} - 通信开销 vs 精度")
    plt.tight_layout(); plt.savefig(out_dir / f"{suite}_bytes_vs_rmse.png", dpi=180); plt.close()

def plot_budget_curve(rounds_df: pd.DataFrame, out_dir: Path):
    df = rounds_df.copy()
    if "sweep_param" not in df.columns or "sweep_value" not in df.columns: return
    df = df[df["sweep_param"]=="budget_bytes"]
    if df.empty: return
    agg = df.groupby("sweep_value")["rmse"].mean().reset_index().sort_values("sweep_value")
    plt.figure()
    plt.plot(agg["sweep_value"], agg["rmse"], marker="o")
    plt.xlabel("budget_bytes"); plt.ylabel("RMSE(均值)"); plt.title("预算扫描：budget_bytes vs RMSE(均值)")
    plt.tight_layout(); plt.savefig(out_dir / "budget_rmse_curve.png", dpi=180); plt.close()

def plot_epsilon_curve(rounds_df: pd.DataFrame, out_dir: Path):
    df = rounds_df.copy()
    if "sweep_param" not in df.columns or "sweep_value" not in df.columns: return
    df = df[df["sweep_param"]=="epsilon"]
    if df.empty: return
    agg = df.groupby(["method","sweep_value"])["rmse"].mean().reset_index()
    for m, g in agg.groupby("method"):
        g = g.sort_values("sweep_value")
        plt.figure()
        plt.plot(g["sweep_value"], g["rmse"], marker="o")
        plt.xlabel("epsilon"); plt.ylabel("RMSE(均值)"); plt.title(f"ε 扫描：{m}")
        plt.tight_layout(); plt.savefig(out_dir / f"epsilon_rmse_curve_{m}.png", dpi=180); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dirs", required=True, help="逗号分隔的结果目录列表")
    ap.add_argument("--out_dir", default="analysis_outputs", help="分析输出目录")
    ap.add_argument("--metric", default="rmse", help="用于排名/检验的指标（默认 rmse）")
    ap.add_argument("--bootstrap", type=int, default=0, help="自举迭代数（0=不计算CI）")
    ap.add_argument("--alpha", type=float, default=0.05, help="自举置信区间的 alpha")
    ap.add_argument("--save_plots", type=int, default=1, help="是否绘图并保存（综合工具=是，0=否）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [d.strip() for d in args.results_dirs.split(",") if d.strip()]
    rounds_df, results_df = load_results_dirs(dirs)

    if not rounds_df.empty:
        rounds_df.to_csv(out_dir / "ALL_rounds_concat.csv", index=False)
    if not results_df.empty:
        results_df.to_csv(out_dir / "ALL_results_concat.csv", index=False)

    summary_df = summarize_methods(rounds_df)
    summary_df.to_csv(out_dir / "summary_by_suite_method.csv", index=False)

    if args.bootstrap and not summary_df.empty:
        rows = []
        for (suite, method), g in rounds_df.groupby(["suite","method"]):
            rmse_vals = _safe_num(g["rmse"]).dropna().to_numpy()
            bytes_vals = _safe_num(g["bytes"]).dropna().to_numpy()
            def _ci(arr):
                if arr.size==0: return (np.nan, np.nan)
                return bootstrap_ci(arr, func=np.mean, iters=args.bootstrap, alpha=args.alpha)
            rmse_ci = _ci(rmse_vals); bytes_ci = _ci(bytes_vals)
            rows.append(dict(suite=suite, method=method, rmse_lo=rmse_ci[0], rmse_hi=rmse_ci[1],
                             bytes_lo=bytes_ci[0], bytes_hi=bytes_ci[1]))
        ci_df = pd.DataFrame(rows)
        ci_df.to_csv(out_dir / "ci_bootstrap.csv", index=False)
        merged = pd.merge(summary_df, ci_df, on=["suite","method"], how="left")
        merged.to_csv(out_dir / "summary_with_ci.csv", index=False)

    fried = friedman_from_rounds(rounds_df, metric=args.metric, by_suite=True)
    if not fried.empty:
        fried.to_csv(out_dir / "friedman_ranks_by_suite.csv", index=False)
        for suite, g in fried.groupby("suite"):
            g2 = g[["method","avg_rank"]].drop_duplicates().sort_values("avg_rank")
            with open(out_dir / f"ranks_{suite}.tex", "w", encoding="utf-8") as f:
                f.write("\\begin{tabular}{l r}\\hline\n方法 & 平均秩\\\\\\hline\n")
                for _, row in g2.iterrows():
                    f.write(f"{row['method']} & {row['avg_rank']:.3f}\\\\\n")
                f.write("\\hline\\end{tabular}\n")

    if not summary_df.empty:
        fronts = []
        for suite, g in summary_df.groupby("suite"):
            pf = pareto_front(g, rmse_col="rmse_mean", bytes_col="bytes_mean")
            pf["suite"] = suite; fronts.append(pf)
        pf_all = pd.concat(fronts, ignore_index=True) if fronts else pd.DataFrame()
        pf_all.to_csv(out_dir / "pareto_fronts.csv", index=False)
        with open(out_dir / "pareto_methods.tex", "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{l l r r}\\hline\n套件 & 方法 & RMSE(均值) & Bytes(均值)\\\\\\hline\n")
            for _, r in pf_all.iterrows():
                f.write(f"{r['suite']} & {r['method']} & {r['rmse_mean']:.6f} & {int(r['bytes_mean'])}\\\\\n")
            f.write("\\hline\\end{tabular}\n")

    if args.save_plots and not summary_df.empty:
        suites = sorted(summary_df["suite"].unique().tolist())
        for s in suites:
            plot_bar_rmse(summary_df, out_dir, s)
            plot_scatter_bytes_rmse(summary_df, out_dir, s)
        if not rounds_df.empty:
            if "sweep_param" in rounds_df.columns and (rounds_df["sweep_param"]=="budget_bytes").any():
                plot_budget_curve(rounds_df, out_dir)
            if "sweep_param" in rounds_df.columns and (rounds_df["sweep_param"]=="epsilon").any():
                plot_epsilon_curve(rounds_df, out_dir)

    md = []
    md.append("# 实验分析摘要\n")
    md.append(f"- 合并目录：{', '.join(dirs)}\n")
    md.append(f"- 输出目录：{out_dir}\n")
    md.append(f"- 统计套件数：{summary_df['suite'].nunique() if not summary_df.empty else 0}\n")
    md.append(f"- 方法数（去重）：{summary_df['method'].nunique() if not summary_df.empty else 0}\n")
    if not summary_df.empty:
        best = summary_df.sort_values(['suite','rmse_mean']).groupby('suite').first().reset_index()
        md.append("\n## 各套件 RMSE(均值) 最优方法\n")
        for _, r in best.iterrows():
            md.append(f"- {r['suite']}: {r['method']} (rmse_mean={r['rmse_mean']:.6f}, bytes_mean={int(r['bytes_mean'])})\n")
    with open(out_dir / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"分析完成，输出位于：{out_dir}")

if __name__ == "__main__":
    main()
