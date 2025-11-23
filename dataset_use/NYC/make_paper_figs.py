
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_figs.py
------------------
从包含多个实验套件(suite_*)的目录中自动汇总结果，生成适合论文投稿的对比图片（PNG 与 PDF）。
- 读取每个 suite 下的:
  1) merged_results.csv  (各方法的统计量: rmse_mean / rmse_std / bytes_mean / time_s_mean 等)
  2) rounds_*.csv        (各方法的逐轮指标: rmse / var / resid_var / time_s / bytes)
- 输出四类核心图：
  A) RMSE (mean ± std) 柱状图（按方法）
  B) 通信开销（Bytes 平均值）柱状图（按方法）
  C) 时间开销（Time 平均值）柱状图（按方法）
  D) 训练轮次收敛曲线（RMSE vs. round，多方法同图，可选平滑）
- 所有图片同时保存为 PNG（300 DPI）与 PDF（矢量图）。

用法：
  python make_paper_figs.py --data /path/to/root --out out_dir [--smooth 11] [--dpi 300] [--max-suites 0] [--cjk-font /path/to/ttf]

建议：
  1) 如果中文字体缺失，可传入 --cjk-font 指向系统内的中文字体（如 Noto Sans CJK / SimHei）。
  2) 直接将生成的 PDF 用于论文排版，PNG 用于网页或幻灯片预览。
"""
import os
import re
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ============== Matplotlib 全局样式（兼容期刊） ==============
def _maybe_set_cjk_font(cjk_font: Optional[str] = None):
    # 不强制设置颜色；只设置字体与基本样式
    matplotlib.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # 使 PDF 内文字可编辑/检索
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": False,  # 我们会用 tight_layout
    })
    if cjk_font and Path(cjk_font).exists():
        try:
            matplotlib.rcParams["font.sans-serif"] = [cjk_font, "DejaVu Sans", "Arial"]
            matplotlib.rcParams["font.family"] = "sans-serif"
        except Exception:
            pass
        else:
            # mac 上尝试用系统中文字体
            matplotlib.rcParams["font.sans-serif"] = [
                "PingFang SC", "Heiti SC", "Hiragino Sans GB",
                "DejaVu Sans", "Arial"
            ]
            matplotlib.rcParams["font.family"] = "sans-serif"

def _mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _scan_suites(root: Path) -> List[Path]:
    suites = []
    for p in root.rglob("suite_*"):
        if p.is_dir():
            suites.append(p)
    suites = sorted(suites)
    return suites

def _nice_method_name(m: str) -> str:
    mapping = {
        "etbp_td": "ETBP-TD",
        "eptd": "EP-TD",
        "ud_ldp": "UD-LDP",
        "dplp": "DPLP",
        "sa_htd": "SA-HTD",
        "flguard": "FLGuard",
        "draco": "DRACO",
        "mozi": "MOZI",
        "rigl": "RigL",
        "radar": "RADAR",
        "cand": "CAND",
        "napp": "NAPP",
        "random": "Random",
        "e2e": "E2E",
    }
    return mapping.get(str(m).lower(), str(m))

def _suite_tag(suite_dir: Path) -> str:
    # 例如：suite_eps1.0_rho0.2_mal0.1_RALL -> eps1.0_rho0.2_mal0.1
    m = re.search(r"suite_(.*?)_R", suite_dir.name)
    return m.group(1) if m else suite_dir.name

def _best_unit_scale(values: np.ndarray, base_unit: str) -> Tuple[float, str]:
    """
    自动确定数量级显示：Byte/KB/MB 或 秒/ms 等。
    返回 (scale, unit)，其中 y_display = y / scale，ylabel 使用 unit。
    """
    if len(values) == 0:
        return 1.0, base_unit
    v = np.nanmax(values)
    if base_unit.lower() in ["bytes", "byte", "b"]:
        if v >= 1024**2:
            return 1024.0**2, "MB"
        elif v >= 1024:
            return 1024.0, "KB"
        else:
            return 1.0, "B"
    if base_unit.lower() in ["s", "sec", "secs", "seconds"]:
        # 如果都很小，用 ms
        if v < 1.0:
            return 1e-3, "ms"  # y_display = y/1e-3 = y*1000
        return 1.0, "s"
    return 1.0, base_unit

def _bar_with_err(ax, labels: List[str], means: np.ndarray, stds: Optional[np.ndarray],
                  ylabel: str, rotate_xticks: bool = False):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds if stds is not None else None, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if rotate_xticks else 0, ha='right' if rotate_xticks else 'center')
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6)

def _save(fig: matplotlib.figure.Figure, out_base: Path):
    # 注意：不能用 Path.with_suffix 因为文件名中可能包含小数点（如 eps0.5）
    png_path = Path(str(out_base) + ".png")
    pdf_path = Path(str(out_base) + ".pdf")
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

def _load_merged_results(suite_dir: Path) -> Optional[pd.DataFrame]:
    f = suite_dir / "merged_results.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    # 只保留常用列，如果存在
    cols = [c for c in ["method","rmse_mean","rmse_std","rmse_last","bytes_mean","time_s_mean"] if c in df.columns]
    return df[cols].copy()

def _load_rounds_concat(suite_dir: Path) -> Optional[pd.DataFrame]:
    # 合并该 suite 下各方法的 rounds_*.csv，补充 round 序号
    csvs = sorted([p for p in suite_dir.glob("rounds_*.csv") if p.is_file()])
    if not csvs:
        # 兜底尝试 merged_rounds.csv
        m = suite_dir / "merged_rounds.csv"
        if m.exists():
            df = pd.read_csv(m)
            # 若没有显式 round 列，则使用行号当作轮次
            if "round" not in df.columns:
                df = df.copy()
                df["round"] = np.arange(1, len(df)+1)
            return df
        return None
    dfs = []
    for p in csvs:
        d = pd.read_csv(p)
        # 如果没有 round 列，用行号
        if "round" not in d.columns:
            d = d.copy()
            d["round"] = np.arange(1, len(d)+1)
        # method 列可能叫 method 或 method_name
        if "method" not in d.columns and "method_name" in d.columns:
            d["method"] = d["method_name"]
        dfs.append(d)
    cat = pd.concat(dfs, ignore_index=True, axis=0)
    # 仅保留常用列
    keep = [c for c in ["method","round","rmse","var","resid_var","time_s","bytes"] if c in cat.columns]
    return cat[keep].copy()

def plot_suite_figures(suite_dir: Path, out_dir: Path, smooth: int = 0, dpi: int = 300):
    tag = _suite_tag(suite_dir)
    suite_out = out_dir / suite_dir.name
    _mkdir(suite_out)

    # ===== 图 1~3：来自 merged_results.csv =====
    mr = _load_merged_results(suite_dir)
    if mr is not None and not mr.empty:
        # 同步保存汇总表，便于论文表格复现
        mr_out = suite_out / f"{tag}_summary.csv"
        try:
            mr.to_csv(mr_out, index=False)
        except Exception:
            pass
        # 排序：RMSE 越小越好
        if "rmse_mean" in mr.columns:
            mr = mr.sort_values("rmse_mean", ascending=True)
        methods = mr["method"].astype(str).apply(_nice_method_name).tolist()

        # 1) RMSE mean ± std
        if "rmse_mean" in mr.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5), dpi=dpi)
            means = mr["rmse_mean"].to_numpy(dtype=float)
            stds = mr["rmse_std"].to_numpy(dtype=float) if "rmse_std" in mr.columns else None
            _bar_with_err(ax, methods, means, stds, ylabel="RMSE (mean ± std)", rotate_xticks=True)
            ax.set_title(f"Comm. Cost Comparison ({tag})")
            _save(fig, suite_out / f"{tag}_bar_rmse")
            plt.close(fig)

        # 2) Bytes 平均值
        if "bytes_mean" in mr.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5), dpi=dpi)
            vals = mr["bytes_mean"].to_numpy(dtype=float)
            scale, unit = _best_unit_scale(vals, "Bytes")
            _bar_with_err(ax, methods, vals/scale, None, ylabel=f"Comm. Cost / {unit}", rotate_xticks=True)
            ax.set_title(f"Comm. Cost Comparison ({tag})")
            _save(fig, suite_out / f"{tag}_bar_bytes")
            plt.close(fig)

        # 3) Time 平均值
        if "time_s_mean" in mr.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5), dpi=dpi)
            vals = mr["time_s_mean"].to_numpy(dtype=float)
            scale, unit = _best_unit_scale(vals, "s")
            # 注意：_best_unit_scale 返回 (scale, "ms") 时，y_display = y/1e-3 = y*1000
            y_display = vals/scale if unit != "ms" else vals/1e-3
            _bar_with_err(ax, methods, y_display, None, ylabel=f"Time / {unit}", rotate_xticks=True)
            ax.set_title(f"Comm. Cost Comparison ({tag})")
            _save(fig, suite_out / f"{tag}_bar_time")
            plt.close(fig)

    # ===== 图 4：来自 rounds_*.csv =====
    rounds = _load_rounds_concat(suite_dir)
    if rounds is not None and not rounds.empty and "rmse" in rounds.columns:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=dpi)
        # 为避免不同方法曲线长度不一致，分别 groupby 再画
        for m, g in rounds.groupby("method", sort=False):
            x = g["round"].to_numpy()
            y = g["rmse"].to_numpy(dtype=float)
            # 可选平滑
            if smooth and smooth > 1:
                k = int(smooth)
                if k % 2 == 0:
                    k += 1  # kernel 必须奇数
                if k > 1 and len(y) >= k:
                    y = pd.Series(y).rolling(window=k, center=True, min_periods=max(2, k//3)).mean().to_numpy()
            ax.plot(x, y, label=_nice_method_name(str(m)), linewidth=1.6)
        ax.set_xlabel("Round")
        ax.set_ylabel("RMSE")
        ax.set_title(f"Convergence Curves ({tag})")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.legend(frameon=False, ncols=2)
        _save(fig, suite_out / f"{tag}_curve_rmse")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="important", help="根目录（包含多个 suite_* 子目录）")
    parser.add_argument("--out", type=str, default="out_figs", help="输出目录")
    parser.add_argument("--smooth", type=int, default=11, help="收敛曲线的滑动窗口大小（奇数，<=1 表示不平滑）")
    parser.add_argument("--dpi", type=int, default=300, help="PNG 输出 DPI")
    parser.add_argument("--max-suites", type=int, default=0, help="仅处理前 N 个 suite（0 表示全部）")
    parser.add_argument("--cjk-font", type=str, default="", help="可选：中文字体路径（.ttf/.otf）")
    args = parser.parse_args()

    data_root = Path(args.data).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    _mkdir(out_root)

    _maybe_set_cjk_font(args.cjk_font)

    suites = _scan_suites(data_root)
    if args.max_suites and args.max_suites > 0:
        suites = suites[:args.max_suites]

    if not suites:
        print(f"[WARN] 在 {data_root} 下未找到任何 suite_* 目录")
        return

    print(f"[INFO] 发现 {len(suites)} 个 suite：")
    for s in suites:
        print("  -", s.name)

    for s in suites:
        print(f"[INFO] 处理 {s.name} ...")
        plot_suite_figures(s, out_root, smooth=args.smooth, dpi=args.dpi)

    print(f"[OK] 完成。输出目录：{out_root}")

if __name__ == "__main__":
    main()
