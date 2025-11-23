#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiments_fair.py
-----------------------
A reproducible, fairness-aware experiment runner for SA-HTD family on NYC (or similar) datasets.

Highlights:
- Robust CSV loading (arbitrary column names) + global slot indexing from timestamp.
- Two fairness regimes:
    (A) feature-locked: disable C-path & postprocess, compare pure algorithmic performance
    (B) system-level: enable each method's recommended features, but constrain epsilon and average bytes
- Method preflight: dry-run on few slots to validate interface & logs; auto-fix common issues.
- Outputs per-suite rounds_{method}.csv + summary_{method}.csv + merged_results.csv, and FAIR_REPORT.md.

Assumptions:
- src/algorithms/user_algorithms.py has callables: sa_htd_paper(...), sa_htd_plus_x(...) etc.
- src/algorithms/algorithms_bridge.py exposes either <method>_bridge or bridge_call(core=...).
- Each method returns list[dict] or pandas.DataFrame with at least: rmse, bytes, time_s.
"""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- Adjust imports if your repo layout differs ----
from dataset_use.NYC.src.algorithms import algorithms_bridge as bridge
from dataset_use.NYC.src.algorithms import user_algorithms as ualgs


# ================================================================
# 0) Config & Utilities
# ================================================================
@dataclass
class ExperimentConfig:
    # IO
    reports_csv: Path
    truth_csv: Path
    outdir: Path

    # Suites
    methods: List[str]
    eps_list: List[float]
    rho_list: List[float]
    mal_list: List[float]
    time_bin: str
    rounds_per_suite: int
    n_workers: int
    seed: int

    # CSV mapping
    rep_time_col: str = "timestamp"
    rep_entity_col: str = "entity_id"
    rep_worker_col: str = "worker_id"
    rep_value_col: str = "value"
    truth_time_col: str = "timestamp"
    truth_entity_col: str = "entity_id"
    truth_value_col: str = "truth"

    # Fairness regime
    regime: str = "feature_locked"  # "feature_locked" | "system"

    # Global budgets / scheduler targets
    target_latency_ms: float = 2.0
    target_bytes_per_round: float = 1.8e5

    # Privacy
    accountant_mode: str = "pld"  # "pld" | "naive"
    window_w: int = 32
    epsilon_per_window: float = float("nan")
    delta_target: float = 1e-5
    use_shuffle: bool = True
    geo_epsilon: float = 0.0

    # SAHTD++ knobs (will be used or disabled depending on regime)
    A_budget_ratio: float = 0.22
    tau_percentile: float = 75.0

    bytes_per_bit: float = 0.125
    BASE_BITS_A: int = 10
    BASE_BITS_B: int = 8
    BITS_C_EXTRA: int = 2
    MIN_QUANT_BITS: int = 6
    MAX_QUANT_BITS: int = 14
    VAR_QUANTILE: float = 0.7

    post_lap_alpha: float = 0.3
    post_process_var: float = 0.5
    post_obs_var_base: float = 1.0

    # C-path (DAP/VDAF)
    use_vdaf_http: bool = False
    dap_leader_url: str = "http://localhost:8787"
    dap_helper_url: str = ""
    dap_api_token: Optional[str] = None
    dap_mode: str = "dryrun"
    dap_task_id: str = "nyc-speed-2025-ijcai"
    C_BATCH_MIN: int = 8
    C_BATCH_MAX: int = 32
    C_DEADLINE_MS: float = 80.0

    # Optional graph for postprocess
    entity_graph_csv: Optional[Path] = None

    # Subsample within slot (if needed)
    subsample_p: float = 1.0


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def auto_discover_methods(exclude: Optional[List[str]] = None) -> List[str]:
    """Heuristic: list callables in user_algorithms that accept (rounds_iter, n_workers, params)."""
    discovered = []
    for name in dir(ualgs):
        if name.startswith("_"):
            continue
        fn = getattr(ualgs, name)
        if callable(fn):
            discovered.append(name)
    # prefer stable order; keep 'sa_htd_paper' first if present
    discovered = sorted(set(discovered))
    if "sa_htd_paper" in discovered:
        discovered.remove("sa_htd_paper")
        discovered = ["sa_htd_paper"] + discovered
    if exclude:
        discovered = [m for m in discovered if m not in exclude]
    return discovered


# ================================================================
# 1) Load CSVs & make global slots
# ================================================================
def load_reports_and_truth(cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rep = pd.read_csv(cfg.reports_csv, parse_dates=[cfg.rep_time_col])
    tru = pd.read_csv(cfg.truth_csv, parse_dates=[cfg.truth_time_col])

    rep = rep.rename(columns={
        cfg.rep_time_col: "timestamp",
        cfg.rep_entity_col: "entity_id",
        cfg.rep_worker_col: "worker_id",
        cfg.rep_value_col: "value",
    })
    tru = tru.rename(columns={
        cfg.truth_time_col: "timestamp",
        cfg.truth_entity_col: "entity_id",
        cfg.truth_value_col: "truth",
    })

    if "timestamp" not in rep.columns or "timestamp" not in tru.columns:
        raise ValueError("Both reports and truth must contain a timestamp column.")

    bin_delta = pd.to_timedelta(cfg.time_bin)
    bin_sec = bin_delta.total_seconds()
    if bin_sec <= 0:
        raise ValueError(f"Invalid time_bin: {cfg.time_bin}")

    t0 = min(rep["timestamp"].min(), tru["timestamp"].min())
    rep["slot"] = ((rep["timestamp"] - t0).dt.total_seconds() / bin_sec).astype("int64")
    tru["slot"] = ((tru["timestamp"] - t0).dt.total_seconds() / bin_sec).astype("int64")

    rep = rep.sort_values(["slot", "entity_id", "worker_id"]).reset_index(drop=True)
    tru = tru.sort_values(["slot", "entity_id"]).reset_index(drop=True)
    return rep, tru


# ================================================================
# 2) Optional: load neighbor graph for postprocess
# ================================================================
def load_entity_graph_or_none(path: Optional[Path]) -> Optional[Dict[Any, List[Any]]]:
    if path is None:
        return None
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if "src_entity_id" in df.columns and "dst_entity_id" in df.columns:
        s, d = "src_entity_id", "dst_entity_id"
    elif "src" in cols and "dst" in cols:
        s, d = df.columns[cols.index("src")], df.columns[cols.index("dst")]
    else:
        s, d = df.columns[0], df.columns[1]
    g: Dict[Any, List[Any]] = {}
    for _, r in df.iterrows():
        u, v = r[s], r[d]
        g.setdefault(u, []).append(v)
        g.setdefault(v, []).append(u)
    return g


# ================================================================
# 3) Batch & suite transforms
# ================================================================
class Batch:
    __slots__ = ("entities", "truth", "reports", "num_reports", "slot")
    def __init__(self, entities, truth, reports, slot):
        self.entities = list(entities)
        self.truth = np.asarray(truth, float)
        self.reports = reports
        self.num_reports = 0 if reports is None else len(reports)
        self.slot = int(slot)


def apply_rho_and_malicious(rep: pd.DataFrame, rho: float,
                            mal_workers: set, rng: np.random.Generator) -> pd.DataFrame:
    if rep is None or rep.empty:
        return rep
    if "worker_id" not in rep.columns or "value" not in rep.columns:
        return rep
    rep = rep.copy()
    # rho sampling
    workers = rep["worker_id"].unique()
    n_keep = max(1, int(round(rho * len(workers))))
    keep = rng.choice(workers, size=n_keep, replace=False)
    rep = rep[rep["worker_id"].isin(keep)].reset_index(drop=True)
    # malicious injection
    if mal_workers:
        is_mal = rep["worker_id"].isin(mal_workers)
        if is_mal.any():
            grp = rep.groupby("entity_id")["value"]
            mu = grp.transform("mean")
            sd = grp.transform("std").fillna(0.0)
            noise = rng.standard_t(df=5, size=len(rep)) * 3.0
            fake = mu + sd * noise
            rep.loc[is_mal, "value"] = fake[is_mal]
    return rep


def build_rounds_iter(reports: pd.DataFrame, truth: pd.DataFrame,
                      slots: np.ndarray, rho: float, mal: float,
                      seed: int) -> Iterable[Batch]:
    rng = np.random.default_rng(seed)
    mal_workers = set()
    if "worker_id" in reports.columns and mal > 0:
        all_w = reports["worker_id"].unique()
        n_mal = int(round(mal * len(all_w)))
        if n_mal > 0:
            mal_workers = set(rng.choice(all_w, size=n_mal, replace=False))
    for s in slots:
        tru_s = truth[truth["slot"] == s][["entity_id", "truth"]]
        if tru_s.empty:
            continue
        ents = tru_s["entity_id"].astype(str).to_numpy()
        rep_s = reports[(reports["slot"] == s) &
                        (reports["entity_id"].astype(str).isin(ents))].reset_index(drop=True)
        rep_s = apply_rho_and_malicious(rep_s, rho=rho, mal_workers=mal_workers, rng=rng)
        yield Batch(ents, tru_s["truth"].to_numpy(float), rep_s, s)


# ================================================================
# 4) Method resolution & runner
# ================================================================
def resolve_method_callable(method: str):
    """
    适配你当前 algorithms_bridge.py 的接口：
      1) 若存在 bridge.<method>_bridge，优先调用该 bridge（里面已经用 _Spy+bridge_call 封装过）。
      2) 否则，若 user_algorithms 里有同名函数，则直接调用该函数（不再走 bridge_call）。
      3) 否则，若 bridge.generic_bridge 存在，就用 generic_bridge + func_name=method。
    """
    # 1) 先看 bridge 里有没有 <method>_bridge
    fn_bridge = getattr(bridge, f"{method}_bridge", None)
    if callable(fn_bridge):
        # 实验脚本这边只负责传 rounds_iter/n_workers/params，
        # 具体 Spy/bytes 统计都在 bridge 内部完成
        return ("bridge_direct", fn_bridge)

    # 2) 再看 user_algorithms 里有没有这个方法
    core = getattr(ualgs, method, None)
    if callable(core):
        def _direct(rounds_iter, n_workers, params):
            # 这里不再走 bridge_call，直接调用你的核心实现
            return core(rounds_iter, n_workers, params)
        return ("direct", _direct)

    # 3) 最后尝试 bridge.generic_bridge（比如你想比较 eptd 这种）
    gen = getattr(bridge, "generic_bridge", None)
    if callable(gen):
        def _via_generic(rounds_iter, n_workers, params):
            return gen(rounds_iter, n_workers, func_name=method, params=params)
        return ("generic", _via_generic)

    # 4) 实在找不到，就报错
    raise ValueError(
        f"Cannot resolve method '{method}': "
        f"期待 algorithms_bridge.{method}_bridge 或 user_algorithms.{method} 或 generic_bridge(func_name='{method}')"
    )

def build_params(cfg: ExperimentConfig, eps: float,
                 entity_graph: Optional[Dict[Any, List[Any]]],
                 regime: str, seed: int):
    class P: pass
    p = P()
    # common
    p.epsilon = float(eps)
    p.tau_percentile = cfg.tau_percentile
    p.A_budget_ratio = cfg.A_budget_ratio
    p.target_latency_ms = cfg.target_latency_ms
    p.target_bytes_per_round = cfg.target_bytes_per_round

    p.accountant_mode = cfg.accountant_mode
    p.window_w = cfg.window_w
    p.epsilon_per_window = cfg.epsilon_per_window
    p.delta_target = cfg.delta_target
    p.use_shuffle = bool(cfg.use_shuffle)
    p.geo_epsilon = cfg.geo_epsilon

    p.bytes_per_bit = cfg.bytes_per_bit
    p.BASE_BITS_A = cfg.BASE_BITS_A
    p.BASE_BITS_B = cfg.BASE_BITS_B
    p.BITS_C_EXTRA = cfg.BITS_C_EXTRA
    p.MIN_QUANT_BITS = cfg.MIN_QUANT_BITS
    p.MAX_QUANT_BITS = cfg.MAX_QUANT_BITS
    p.VAR_QUANTILE = cfg.VAR_QUANTILE

    p.post_lap_alpha = cfg.post_lap_alpha
    p.post_process_var = cfg.post_process_var
    p.post_obs_var_base = cfg.post_obs_var_base
    p.entity_graph = entity_graph

    p.use_vdaf_http = bool(cfg.use_vdaf_http)
    p.dap_leader_url = cfg.dap_leader_url
    p.dap_helper_url = cfg.dap_helper_url
    p.dap_api_token = cfg.dap_api_token
    p.dap_mode = cfg.dap_mode
    p.dap_task_id = cfg.dap_task_id
    p.C_BATCH_MIN = cfg.C_BATCH_MIN
    p.C_BATCH_MAX = cfg.C_BATCH_MAX
    p.C_DEADLINE_MS = cfg.C_DEADLINE_MS

    p.subsample_p = cfg.subsample_p
    p.rng_seed = seed
    p.n_workers = int(cfg.n_workers)

    # fairness regime toggles
    if regime == "feature_locked":
        # 禁用 SAHTD++ 增强：postprocess & C 路
        p.post_lap_alpha = 0.0
        p.use_vdaf_http = False
    elif regime == "system":
        # 按当前配置启用（已默认），无需特殊处理
        pass
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return p


def run_method_on_suite(cfg: ExperimentConfig, method: str,
                        eps: float, rho: float, mal: float,
                        base_reports: pd.DataFrame, truth: pd.DataFrame,
                        slots: np.ndarray,
                        entity_graph: Optional[Dict[Any, List[Any]]] = None) -> pd.DataFrame:
    seed = cfg.seed + (hash((method, eps, rho, mal, cfg.regime)) % 1000003)
    rounds_iter = build_rounds_iter(base_reports, truth, slots, rho, mal, seed)
    params = build_params(cfg, eps, entity_graph, cfg.regime, seed)
    _, call_fn = resolve_method_callable(method)
    results = call_fn(rounds_iter=rounds_iter, n_workers=cfg.n_workers, params=params)
    df = pd.DataFrame(list(results)) if not isinstance(results, pd.DataFrame) else results.copy()
    df["method"] = method
    df["epsilon"] = eps
    df["rho"] = rho
    df["mal"] = mal
    df["regime"] = cfg.regime
    return df


# ================================================================
# 5) Preflight & fairness checks
# ================================================================
def preflight_method(method: str, cfg: ExperimentConfig,
                     rep: pd.DataFrame, tru: pd.DataFrame,
                     entity_graph: Optional[Dict[Any, List[Any]]]) -> Dict[str, Any]:
    """Run a tiny dry-run to validate interface & logs; return a dict of findings."""
    report: Dict[str, Any] = {"method": method, "ok": True, "notes": []}
    try:
        slots_all = np.sort(rep["slot"].unique())
        slots_small = slots_all[: min(8, len(slots_all))]
        df_small = run_method_on_suite(cfg, method, eps=cfg.eps_list[0],
                                       rho=cfg.rho_list[0], mal=cfg.mal_list[0],
                                       base_reports=rep, truth=tru,
                                       slots=slots_small, entity_graph=entity_graph)
        # Sanity columns
        required = ["rmse", "bytes", "time_s"]
        missing = [c for c in required if c not in df_small.columns]
        if missing:
            report["ok"] = False
            report["notes"].append(f"Missing columns: {missing}")
        # Finite checks
        for col in ["rmse", "bytes", "time_s"]:
            if col in df_small and not np.isfinite(df_small[col].astype(float)).all():
                report["ok"] = False
                report["notes"].append(f"Non-finite in '{col}'")
        # Rounds coverage
        if len(df_small) == 0:
            report["ok"] = False
            report["notes"].append("No rounds produced in preflight")
        report["preview"] = {
            "rmse_mean": float(df_small["rmse"].mean()) if "rmse" in df_small else None,
            "bytes_mean": float(df_small["bytes"].mean()) if "bytes" in df_small else None,
            "time_s_mean": float(df_small["time_s"].mean()) if "time_s" in df_small else None,
            "n_rounds": int(len(df_small)),
        }
    except Exception as e:
        report["ok"] = False
        report["notes"].append(f"Exception: {type(e).__name__}: {e}")
    return report


def write_fair_report(outdir: Path, cfg: ExperimentConfig,
                      preflight_reports: List[Dict[str, Any]],
                      suite_summaries: List[pd.DataFrame]):
    md = []
    md.append("# FAIR REPORT\n")
    md.append(f"- Regime: **{cfg.regime}**\n")
    md.append(f"- Methods: {', '.join(cfg.methods)}\n")
    md.append(f"- Eps list: {cfg.eps_list}; Rho list: {cfg.rho_list}; Mal list: {cfg.mal_list}\n")
    md.append(f"- time_bin: {cfg.time_bin}; rounds_per_suite: {cfg.rounds_per_suite}\n")
    md.append("## Preflight\n")
    for rep in preflight_reports:
        status = "OK" if rep["ok"] else "FAILED"
        md.append(f"### {rep['method']}: {status}\n")
        if "preview" in rep:
            md.append(f"- Preview: {json.dumps(rep['preview'])}\n")
        if rep["notes"]:
            for n in rep["notes"]:
                md.append(f"- Note: {n}\n")
    md.append("\n## Suite Summaries\n")
    for sm in suite_summaries:
        if sm is None or sm.empty:
            continue
        md.append("```\n")
        md.append(sm.to_string(index=False))
        md.append("\n```\n")
    (outdir / "FAIR_REPORT.md").write_text("\n".join(md), encoding="utf-8")


# ================================================================
# 6) Summaries & save
# ================================================================
def summarize_rounds(df_rounds: pd.DataFrame) -> pd.DataFrame:
    if df_rounds.empty:
        return pd.DataFrame()
    def mean(col): return float(df_rounds[col].mean()) if col in df_rounds else float("nan")
    def std(col): return float(df_rounds[col].std()) if col in df_rounds else float("nan")
    return pd.DataFrame([{
        "method": df_rounds["method"].iloc[0],
        "regime": df_rounds["regime"].iloc[0],
        "epsilon": df_rounds["epsilon"].iloc[0],
        "rho": df_rounds["rho"].iloc[0],
        "mal": df_rounds["mal"].iloc[0],
        "n_rounds": len(df_rounds),

        "rmse_mean": mean("rmse"),
        "rmse_std": std("rmse"),
        "rmse_raw_mean": mean("rmse_raw"),
        "bytes_mean": mean("bytes"),
        "bytes_std": std("bytes"),
        "time_s_mean": mean("time_s"),
        "time_s_std": std("time_s"),

        "epsilon_cum_window_mean": mean("epsilon_cum_window"),
        "epsilon_limit_mean": mean("epsilon_limit"),
        "route_ratio_mean": mean("route_ratio"),
        "vdaf_ok_ratio_mean": mean("vdaf_ok_ratio"),
    }])


def save_suite(cfg: ExperimentConfig, suite_dir: Path,
               method_to_rounds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    suite_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for m, df in method_to_rounds.items():
        df.to_csv(suite_dir / f"rounds_{m}.csv", index=False)
        sm = summarize_rounds(df)
        sm.to_csv(suite_dir / f"summary_{m}.csv", index=False)
        summaries.append(sm)
    merged = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    merged.to_csv(suite_dir / "merged_results.csv", index=False)
    return merged


# ================================================================
# 7) Main loop
# ================================================================
def run_all(cfg: ExperimentConfig):
    out = cfg.outdir
    out.mkdir(parents=True, exist_ok=True)

    # data
    rep, tru = load_reports_and_truth(cfg)
    all_slots = np.sort(rep["slot"].unique())
    if cfg.rounds_per_suite > 0:
        slots = all_slots[: cfg.rounds_per_suite]
    else:
        slots = all_slots

    entity_graph = load_entity_graph_or_none(cfg.entity_graph_csv)

    # methods
    methods = cfg.methods or auto_discover_methods()
    # 确保 sa_htd_paper 作为主方法优先
    if "sa_htd_paper" in methods:
        methods = ["sa_htd_paper"] + [m for m in methods if m != "sa_htd_paper"]

    # ---------- 预检：先小范围试跑，看接口/日志是否正常 ----------
    preflight_reports = []
    for m in methods:
        r = preflight_method(m, cfg, rep, tru, entity_graph)
        preflight_reports.append(r)
        print(f"[PREFLIGHT] {m}: ok={r['ok']}, notes={r.get('notes', [])}")

    # 只保留预检通过的方法；如果一个都没有，就退回到“你显式指定的那几个”
    runnable_methods = [r["method"] for r in preflight_reports if r["ok"]]

    if not runnable_methods:
        print("[WARN] 所有方法在预检阶段标记为失败，将仍然尝试运行你显式指定的方法。")
        print("       预检详情会写入 FAIR_REPORT.md，正式运行如有异常会直接抛出来。")
        runnable_methods = methods  # 至少把你命令行里给的 --methods 都跑一遍
    else:
        print("[INFO] 通过预检的方法：", runnable_methods)

    # 主方法 sa_htd_paper 优先，方便后面汇总/画图高亮
    if "sa_htd_paper" in runnable_methods:
        methods = ["sa_htd_paper"] + [m for m in runnable_methods if m != "sa_htd_paper"]
    else:
        methods = runnable_methods

    print(f"[INFO] regime={cfg.regime}; methods={methods}")
    print(f"[INFO] suites: |eps|={len(cfg.eps_list)} |rho|={len(cfg.rho_list)} |mal|={len(cfg.mal_list)}; slots/each={len(slots)}")

    # suites
    all_suite_summaries: List[pd.DataFrame] = []
    for eps in cfg.eps_list:
        for rho in cfg.rho_list:
            for mal in cfg.mal_list:
                suite_name = f"{cfg.regime}_eps{eps}_rho{rho}_mal{mal}_R{len(slots)}"
                suite_dir = out / suite_name
                print(f"[RUN] suite: {suite_name}")

                method_to_rounds: Dict[str, pd.DataFrame] = {}
                for m in methods:
                    print(f"  - method = {m}")
                    df = run_method_on_suite(cfg, m, eps, rho, mal, rep, tru, slots, entity_graph)
                    method_to_rounds[m] = df

                merged = save_suite(cfg, suite_dir, method_to_rounds)
                all_suite_summaries.append(merged)

    write_fair_report(out, cfg, preflight_reports, all_suite_summaries)


# ================================================================
# 8) CLI
# ================================================================
def parse_args() -> ExperimentConfig:
    ap = argparse.ArgumentParser("Fairness-aware experiment runner for SA-HTD family")

    ap.add_argument('--reports_csv', default='/Users/wanghao/Desktop/SA-HTD/dataset_use/NYC/data/reports.csv')
    ap.add_argument('--truth_csv', default='/Users/wanghao/Desktop/SA-HTD/dataset_use/NYC/data/truth.csv')
    ap.add_argument('--outdir', default='/Users/wanghao/Desktop/SA-HTD/dataset_use/NYC/result')

    ap.add_argument("--methods", type=str, default="", help="comma-separated names; empty -> auto discover")
    ap.add_argument("--eps_list", type=str, default="0.5,1.0")
    ap.add_argument("--rho_list", type=str, default="0.2")
    ap.add_argument("--mal_list", type=str, default="0.0")
    ap.add_argument("--time_bin", type=str, default="5min")
    ap.add_argument("--rounds_per_suite", type=int, default=-1)
    ap.add_argument("--n_workers", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2025)

    # CSV mappings
    ap.add_argument("--rep_time_col", type=str, default="timestamp")
    ap.add_argument("--rep_entity_col", type=str, default="entity_id")
    ap.add_argument("--rep_worker_col", type=str, default="worker_id")
    ap.add_argument("--rep_value_col", type=str, default="value")
    ap.add_argument("--truth_time_col", type=str, default="timestamp")
    ap.add_argument("--truth_entity_col", type=str, default="entity_id")
    ap.add_argument("--truth_value_col", type=str, default="truth")

    # regimes
    ap.add_argument("--regime", type=str, choices=["feature_locked", "system"], default="feature_locked")

    # targets
    ap.add_argument("--target_latency_ms", type=float, default=2.0)
    ap.add_argument("--target_bytes_per_round", type=float, default=1.8e5)

    # privacy
    ap.add_argument("--accountant_mode", type=str, default="pld")
    ap.add_argument("--window_w", type=int, default=32)
    ap.add_argument("--epsilon_per_window", type=float, default=float("nan"))
    ap.add_argument("--delta_target", type=float, default=1e-5)
    ap.add_argument("--use_shuffle", action="store_true")
    ap.add_argument("--geo_epsilon", type=float, default=0.0)

    # SAHTD++ knobs
    ap.add_argument("--A_budget_ratio", type=float, default=0.22)
    ap.add_argument("--tau_percentile", type=float, default=75.0)
    ap.add_argument("--bytes_per_bit", type=float, default=0.125)
    ap.add_argument("--BASE_BITS_A", type=int, default=10)
    ap.add_argument("--BASE_BITS_B", type=int, default=8)
    ap.add_argument("--BITS_C_EXTRA", type=int, default=2)
    ap.add_argument("--MIN_QUANT_BITS", type=int, default=6)
    ap.add_argument("--MAX_QUANT_BITS", type=int, default=14)
    ap.add_argument("--VAR_QUANTILE", type=float, default=0.7)

    ap.add_argument("--post_lap_alpha", type=float, default=0.3)
    ap.add_argument("--post_process_var", type=float, default=0.5)
    ap.add_argument("--post_obs_var_base", type=float, default=1.0)

    ap.add_argument("--use_vdaf_http", action="store_true")
    ap.add_argument("--dap_leader_url", type=str, default="http://localhost:8787")
    ap.add_argument("--dap_helper_url", type=str, default="")
    ap.add_argument("--dap_api_token", type=str, default=None)
    ap.add_argument("--dap_mode", type=str, default="dryrun")
    ap.add_argument("--dap_task_id", type=str, default="nyc-speed-2025-ijcai")
    ap.add_argument("--C_BATCH_MIN", type=int, default=8)
    ap.add_argument("--C_BATCH_MAX", type=int, default=32)
    ap.add_argument("--C_DEADLINE_MS", type=float, default=80.0)

    ap.add_argument("--entity_graph_csv", type=str, default=None)
    ap.add_argument("--subsample_p", type=float, default=1.0)

    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else []

    return ExperimentConfig(
        reports_csv=Path(args.reports_csv),
        truth_csv=Path(args.truth_csv),
        outdir=Path(args.outdir),
        methods=methods,
        eps_list=parse_float_list(args.eps_list),
        rho_list=parse_float_list(args.rho_list),
        mal_list=parse_float_list(args.mal_list),
        time_bin=args.time_bin,
        rounds_per_suite=args.rounds_per_suite,
        n_workers=args.n_workers,
        seed=args.seed,
        rep_time_col=args.rep_time_col,
        rep_entity_col=args.rep_entity_col,
        rep_worker_col=args.rep_worker_col,
        rep_value_col=args.rep_value_col,
        truth_time_col=args.truth_time_col,
        truth_entity_col=args.truth_entity_col,
        truth_value_col=args.truth_value_col,
        regime=args.regime,
        target_latency_ms=args.target_latency_ms,
        target_bytes_per_round=args.target_bytes_per_round,
        accountant_mode=args.accountant_mode,
        window_w=args.window_w,
        epsilon_per_window=args.epsilon_per_window,
        delta_target=args.delta_target,
        use_shuffle=bool(args.use_shuffle),
        geo_epsilon=args.geo_epsilon,
        A_budget_ratio=args.A_budget_ratio,
        tau_percentile=args.tau_percentile,
        bytes_per_bit=args.bytes_per_bit,
        BASE_BITS_A=args.BASE_BITS_A,
        BASE_BITS_B=args.BASE_BITS_B,
        BITS_C_EXTRA=args.BITS_C_EXTRA,
        MIN_QUANT_BITS=args.MIN_QUANT_BITS,
        MAX_QUANT_BITS=args.MAX_QUANT_BITS,
        VAR_QUANTILE=args.VAR_QUANTILE,
        post_lap_alpha=args.post_lap_alpha,
        post_process_var=args.post_process_var,
        post_obs_var_base=args.post_obs_var_base,
        use_vdaf_http=bool(args.use_vdaf_http),
        dap_leader_url=args.dap_leader_url,
        dap_helper_url=args.dap_helper_url,
        dap_api_token=args.dap_api_token,
        dap_mode=args.dap_mode,
        dap_task_id=args.dap_task_id,
        C_BATCH_MIN=args.C_BATCH_MIN,
        C_BATCH_MAX=args.C_BATCH_MAX,
        C_DEADLINE_MS=args.C_DEADLINE_MS,
        entity_graph_csv=Path(args.entity_graph_csv) if args.entity_graph_csv else None,
        subsample_p=args.subsample_p,
    )


def main():
    cfg = parse_args()
    np.random.seed(cfg.seed)
    run_all(cfg)


if __name__ == "__main__":
    main()