# -*- coding: utf-8 -*-
"""
auditor_dataset_coverage.py
===========================
快速审计你的 CSV 是否被**完整使用**：
- 统计 truth slots 总数、reports 落在 truth slots 的数量及分布
- 输出 coverage_per_slot.csv（每个 slot 的报告条数/实体数）
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports_csv', default='reports.csv')
    ap.add_argument('--truth_csv',   default='truth.csv')
    ap.add_argument('--time_bin',    default='5min')
    ap.add_argument('--outdir',      default='audit_out')
    args = ap.parse_args()

    rep = pd.read_csv(args.reports_csv, parse_dates=["timestamp"])
    tru = pd.read_csv(args.truth_csv, parse_dates=["timestamp"])
    rep["slot"] = rep["timestamp"].dt.floor(args.time_bin)
    tru["slot"] = tru["timestamp"].dt.floor(args.time_bin)

    truth_slots = sorted(tru["slot"].unique().tolist())
    rep_slots = sorted(rep["slot"].unique().tolist())
    inter = sorted(set(truth_slots).intersection(rep_slots))

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    (out/"summary.txt").write_text(
        f"truth slots: {len(truth_slots)}\n"
        f"reports slots: {len(rep_slots)}\n"
        f"intersection slots: {len(inter)}\n",
        encoding="utf-8"
    )

    # 每个 slot 的报告条数 & 实体覆盖数
    cov = rep[rep["slot"].isin(inter)].groupby("slot").agg(
        reports_per_slot=("value","count"),
        entities_covered=("entity_id","nunique"),
        workers_covered=("worker_id","nunique")
    ).reset_index().sort_values("slot")
    cov.to_csv(out/"coverage_per_slot.csv", index=False)
    print(f"审计完成，结果目录：{out}")

if __name__ == '__main__':
    main()
