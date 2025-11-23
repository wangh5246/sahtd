#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fair_results.py
--------------------
Read suite outputs and plot fair comparisons:
- RMSE vs Bytes (Pareto-style)
- RMSE distribution (boxplot) by method
- Latency vs Bytes
- Epsilon cumulative vs limit (per method)

Usage:
  python -m src.EVAL.plot_fair_results --root result_fair --out plots
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_all_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for suite_dir in root.iterdir():
        if not suite_dir.is_dir():
            continue
        merged = suite_dir / "merged_results.csv"
        if merged.exists():
            df = pd.read_csv(merged)
            df["suite"] = suite_dir.name
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_rmse_vs_bytes(df: pd.DataFrame, out: Path, title: str):
    methods = sorted(df["method"].unique(), key=lambda m: (m != "sa_htd_paper", m))
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    for m in methods:
        sub = df[df["method"] == m]
        x = sub["bytes_mean"].values
        y = sub["rmse_mean"].values
        plt.plot(x, y, marker="o", linestyle="-", label=m)
    # highlight sa_htd_paper
    if "sa_htd_paper" in methods:
        sub = df[df["method"] == "sa_htd_paper"]
        plt.scatter(sub["bytes_mean"], sub["rmse_mean"], s=64, edgecolors="k", linewidths=1.2)
    plt.xlabel("Bytes (mean per round)")
    plt.ylabel("RMSE (mean)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "rmse_vs_bytes.png")
    plt.close()


def plot_box_rmse(root: Path, out: Path, suite_name: str):
    """Boxplot over rounds for a single suite (choose one)"""
    suite_dir = root / suite_name
    if not suite_dir.exists():
        return
    rounds_files = sorted(suite_dir.glob("rounds_*.csv"))
    data = []
    labels = []
    for f in rounds_files:
        m = f.stem.replace("rounds_", "")
        df = pd.read_csv(f)
        if "rmse" in df.columns and len(df):
            data.append(df["rmse"].values.astype(float))
            labels.append(m)
    if not data:
        return
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("RMSE per round")
    plt.title(f"RMSE distribution (suite: {suite_name})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out / f"rmse_box_{suite_name}.png")
    plt.close()


def plot_latency_vs_bytes(df: pd.DataFrame, out: Path, title: str):
    methods = sorted(df["method"].unique(), key=lambda m: (m != "sa_htd_paper", m))
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    for m in methods:
        sub = df[df["method"] == m]
        x = sub["bytes_mean"].values
        y = sub["time_s_mean"].values
        plt.plot(x, y, marker="o", linestyle="-", label=m)
    if "sa_htd_paper" in methods:
        sub = df[df["method"] == "sa_htd_paper"]
        plt.scatter(sub["bytes_mean"], sub["time_s_mean"], s=64, edgecolors="k", linewidths=1.2)
    plt.xlabel("Bytes (mean per round)")
    plt.ylabel("Latency (s, mean per round)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "latency_vs_bytes.png")
    plt.close()


def plot_epsilon_consumption(root: Path, out: Path, suite_name: str):
    suite_dir = root / suite_name
    if not suite_dir.exists():
        return
    rounds_files = sorted(suite_dir.glob("rounds_*.csv"))
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    for f in rounds_files:
        m = f.stem.replace("rounds_", "")
        df = pd.read_csv(f)
        if "epsilon_cum_window" in df.columns and len(df):
            y = df["epsilon_cum_window"].values.astype(float)
            x = np.arange(len(y))
            plt.plot(x, y, linestyle="-", label=m)
            if "epsilon_limit" in df.columns:
                lim = df["epsilon_limit"].values.astype(float)
                plt.plot(x, lim, linestyle="--", alpha=0.6)
    plt.xlabel("Round index")
    plt.ylabel("ε cumulative (window)")
    plt.title(f"Epsilon cumulative (suite: {suite_name})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"epsilon_cum_{suite_name}.png")
    plt.close()


def main():
    ap = argparse.ArgumentParser("Plotting for fair experiment results")
    ap.add_argument("--root", type=str, required=True, help="result root (the parent of suite folders)")
    ap.add_argument("--out", type=str, required=True, help="output directory for plots")
    ap.add_argument("--pick_suite", type=str, default="", help="one suite name for boxplot/epsilon plots")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    ensure_dir(out)

    df = load_all_summaries(root)
    if df.empty:
        print("[WARN] No merged_results.csv found under", root)
        return

    title = f"Fair Comparison ({df['regime'].iloc[0]})"
    plot_rmse_vs_bytes(df, out, title)
    plot_latency_vs_bytes(df, out, title)

    if args.pick_suite:
        plot_box_rmse(root, out, args.pick_suite)
        plot_epsilon_consumption(root, out, args.pick_suite)

    print("[OK] Plots saved to", out)


if __name__ == "__main__":
    main()