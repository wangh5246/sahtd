#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A clean, reproducible experiment runner for SA-HTD family (incl. SAHTD++).
- Global slot indexing from timestamp (floor by time_bin).
- Suite = (epsilon, rho, mal), with clear rho sampling + malicious injection.
- Methods are resolved flexibly via algorithms_bridge or direct core.
- Outputs per-suite rounds_{method}.csv and merged_results.csv.

Tested interface assumptions:
- user_algorithms contains: sa_htd_paper(...), sa_htd_plus_x(...)
- algorithms_bridge has either <method>_bridge(...) or bridge_call(core=...)
- Each core returns list[dict] or a pandas.DataFrame with at least:
    rmse, bytes, time_s (and any other fields are preserved)

Author: (your name)
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- your repo structure; adjust if your package path differs ----
from dataset_use.NYC.src.algorithms import algorithms_bridge as bridge
from dataset_use.NYC.src.algorithms import user_algorithms as ualgs


# ================================================================
# 0) Config Dataclass
# ================================================================
@dataclass
class ExperimentConfig:
    reports_csv: Path
    truth_csv: Path
    outdir: Path

    methods: List[str]         # e.g. ["sa_htd_plus_x", "sa_htd_paper"]
    eps_list: List[float]      # ε list
    rho_list: List[float]      # participation rate list
    mal_list: List[float]      # malicious worker ratio list

    time_bin: str              # "5min" / "10min" / "20min" ...
    rounds_per_suite: int      # -1 -> use all slots
    n_workers: int
    seed: int

    # Common SAHTD/SAHTD++ params (can be extended freely)
    # ---- scheduling & budgets ----
    A_budget_ratio: float = 0.22
    tau_percentile: float = 75.0
    target_latency_ms: float = 2.0
    target_bytes_per_round: float = 1.8e5

    # ---- privacy ----
    accountant_mode: str = "pld"       # "pld" | "naive"
    window_w: int = 32
    epsilon_per_window: float = float("nan")
    delta_target: float = 1e-5
    use_shuffle: bool = True
    geo_epsilon: float = 0.0

    # ---- quantization / bits baseline ----
    bytes_per_bit: float = 0.125
    BASE_BITS_A: int = 10
    BASE_BITS_B: int = 8
    BITS_C_EXTRA: int = 2
    MIN_QUANT_BITS: int = 6
    MAX_QUANT_BITS: int = 14
    VAR_QUANTILE: float = 0.7

    # ---- SAHTD++ postprocess (Kalman + Laplacian) ----
    post_lap_alpha: float = 0.3
    post_process_var: float = 0.5
    post_obs_var_base: float = 1.0

    # ---- C path (DAP/VDAF) ----
    use_vdaf_http: bool = False
    dap_leader_url: str = "http://localhost:8787"
    dap_helper_url: str = ""
    dap_api_token: Optional[str] = None
    dap_mode: str = "dryrun"           # "dryrun" | "daphne" | "divviup"
    dap_task_id: str = "nyc-speed-2025-ijcai"
    C_BATCH_MIN: int = 8
    C_BATCH_MAX: int = 32
    C_DEADLINE_MS: float = 80.0

    # ---- data transforms ----
    subsample_p: float = 1.0            # optional subsample within slot
    # optional graph for postprocessing (entity -> neighbors)
    entity_graph_csv: Optional[Path] = None
    # optional rename mapping for CSVs
    rep_time_col: str = "timestamp"
    rep_entity_col: str = "entity_id"
    rep_worker_col: str = "worker_id"
    rep_value_col: str = "value"
    truth_time_col: str = "timestamp"
    truth_entity_col: str = "entity_id"
    truth_value_col: str = "truth"


# ================================================================
# 1) Data Loading: timestamp -> slot (global index)
# ================================================================
def load_reports_and_truth(cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs and produce a global 'slot' integer column for both tables.

    Global slot scheme:
        slot = floor( (timestamp - t0) / time_bin ), t0 = min(all timestamps)
    """
    parse_dt_cols = [cfg.rep_time_col, cfg.truth_time_col]
    reports = pd.read_csv(cfg.reports_csv, parse_dates=[cfg.rep_time_col])
    truth = pd.read_csv(cfg.truth_csv, parse_dates=[cfg.truth_time_col])

    # Rename to canonical names inside this script
    reports = reports.rename(columns={
        cfg.rep_time_col: "timestamp",
        cfg.rep_entity_col: "entity_id",
        cfg.rep_worker_col: "worker_id",
        cfg.rep_value_col: "value",
    })
    truth = truth.rename(columns={
        cfg.truth_time_col: "timestamp",
        cfg.truth_entity_col: "entity_id",
        cfg.truth_value_col: "truth",
    })

    if "timestamp" not in reports.columns or "timestamp" not in truth.columns:
        raise ValueError("Both reports and truth must contain a timestamp column.")

    bin_delta = pd.to_timedelta(cfg.time_bin)
    bin_sec = bin_delta.total_seconds()
    if bin_sec <= 0:
        raise ValueError(f"Invalid time_bin: {cfg.time_bin}")

    t0 = min(reports["timestamp"].min(), truth["timestamp"].min())
    rep_slot = ((reports["timestamp"] - t0).dt.total_seconds() / bin_sec).astype("int64")
    tru_slot = ((truth["timestamp"] - t0).dt.total_seconds() / bin_sec).astype("int64")

    reports["slot"] = rep_slot
    truth["slot"] = tru_slot

    # Sort canonical order
    reports = reports.sort_values(["slot", "entity_id", "worker_id"]).reset_index(drop=True)
    truth = truth.sort_values(["slot", "entity_id"]).reset_index(drop=True)

    return reports, truth


# ================================================================
# 2) Optional: load entity graph (entity -> neighbors) for postprocess
# ================================================================
def load_entity_graph_or_none(path: Optional[Path]) -> Optional[Dict[Any, List[Any]]]:
    if path is None:
        return None
    df = pd.read_csv(path)
    # Expect columns: src_entity_id, dst_entity_id (undirected)
    cols = [c.lower() for c in df.columns]
    if "src_entity_id" in df.columns and "dst_entity_id" in df.columns:
        src_col, dst_col = "src_entity_id", "dst_entity_id"
    elif "src" in cols and "dst" in cols:
        src_col, dst_col = df.columns[cols.index("src")], df.columns[cols.index("dst")]
    else:
        # try first two
        src_col, dst_col = df.columns[0], df.columns[1]

    graph: Dict[Any, List[Any]] = {}
    for _, row in df.iterrows():
        s, d = row[src_col], row[dst_col]
        graph.setdefault(s, []).append(d)
        graph.setdefault(d, []).append(s)
    return graph


# ================================================================
# 3) Batch object & suite transforms (rho/mal)
# ================================================================
class Batch:
    """Minimal batch object expected by user_algorithms.* cores."""
    __slots__ = ("entities", "truth", "reports", "num_reports", "slot")

    def __init__(self, entities: List[Any], truth: np.ndarray,
                 reports: pd.DataFrame, slot: int):
        self.entities = list(entities)
        self.truth = np.asarray(truth, float)
        self.reports = reports
        self.num_reports = 0 if reports is None else len(reports)
        self.slot = int(slot)


def apply_rho_and_malicious_df(rep: pd.DataFrame, rho: float,
                               mal_workers: set, rng: np.random.Generator) -> pd.DataFrame:
    """Apply participation rho (keep a subset of workers) and malicious injections.

    - rho: fraction of workers retained in this slot
    - mal_workers: global set of workers chosen as malicious in this suite
    Injection: mild heavy-tail around per-entity mean (t(df=5)*3.0)
    """
    if rep is None or rep.empty:
        return rep

    if "worker_id" not in rep.columns or "value" not in rep.columns:
        return rep

    rep = rep.copy()
    # 1) rho sampling on workers
    workers = rep["worker_id"].unique()
    n_keep = max(1, int(round(rho * len(workers))))
    keep = rng.choice(workers, size=n_keep, replace=False)
    rep = rep[rep["worker_id"].isin(keep)].reset_index(drop=True)

    # 2) malicious injection
    if mal_workers:
        is_mal = rep["worker_id"].isin(mal_workers)
        if is_mal.any():
            grouped = rep.groupby("entity_id")["value"]
            mean_by_e = grouped.transform("mean")
            std_by_e = grouped.transform("std").fillna(0.0)
            noise = rng.standard_t(df=5, size=len(rep)) * 3.0
            fake_vals = mean_by_e + std_by_e * noise
            rep.loc[is_mal, "value"] = fake_vals[is_mal]

    return rep


def build_rounds_iter_for_suite(reports: pd.DataFrame, truth: pd.DataFrame,
                                slots: np.ndarray, rho: float, mal: float,
                                seed: int) -> Iterable[Batch]:
    """Yield Batch for each slot (entities aligned with truth)."""
    rng = np.random.default_rng(seed)

    # global malicious set (fixed across suite)
    if "worker_id" in reports.columns and mal > 0.0:
        all_workers = reports["worker_id"].unique()
        n_mal = int(round(mal * len(all_workers)))
        mal_workers = set(rng.choice(all_workers, size=n_mal, replace=False)) if n_mal > 0 else set()
    else:
        mal_workers = set()

    for s in slots:
        tru_s = truth[truth["slot"] == s][["entity_id", "truth"]]
        if tru_s.empty:
            continue

        ents = tru_s["entity_id"].astype(str).to_numpy()
        rep_s = reports[(reports["slot"] == s) &
                        (reports["entity_id"].astype(str).isin(ents))].reset_index(drop=True)
        rep_s = apply_rho_and_malicious_df(rep_s, rho=rho, mal_workers=mal_workers, rng=rng)

        batch = Batch(
            entities=ents.tolist(),
            truth=tru_s["truth"].to_numpy(float),
            reports=rep_s,
            slot=int(s),
        )
        yield batch


# ================================================================
# 4) Running one method on a suite
# ================================================================
def _resolve_method_callable(method: str):
    """
    Resolve a callable for the given method name.

    Preferred order:
      1) algorithms_bridge.<method>_bridge
      2) algorithms_bridge.bridge_call(core=<ualgs.fn>)
      3) direct call to <ualgs.fn>
    """
    # 1) try <method>_bridge
    fn = getattr(bridge, f"{method}_bridge", None)
    if callable(fn):
        return ("bridge_direct", fn)

    # 2) try bridge_call with core
    core = getattr(ualgs, method, None)
    if callable(core) and hasattr(bridge, "bridge_call"):
        def _via_bridge(rounds_iter, n_workers, params):
            return bridge.sahtd_paper_bridge(rounds_iter,n_workers,params)
        return "bridge_wrap", _via_bridge

    # 3) direct core
    if callable(core):
        def _direct(rounds_iter, n_workers, params):
            return core(rounds_iter, n_workers, params)
        return ("direct", _direct)

    raise ValueError(f"Cannot resolve method '{method}': "
                     f"expect bridge.{method}_bridge or user_algorithms.{method}")


def run_one_method_on_suite(cfg: ExperimentConfig, method: str,
                            eps: float, rho: float, mal: float,
                            base_reports: pd.DataFrame, truth: pd.DataFrame,
                            slots: np.ndarray,
                            entity_graph: Optional[Dict[Any, List[Any]]] = None) -> pd.DataFrame:
    """Run a method on one (eps, rho, mal) suite and return per-round DataFrame."""

    # Build rounds iterator with rho/mal transforms, stable seed per suite+method
    seed = cfg.seed + (hash((method, eps, rho, mal)) % 1000003)
    rounds_iter = build_rounds_iter_for_suite(
        reports=base_reports, truth=truth, slots=slots,
        rho=rho, mal=mal, seed=seed
    )

    # Build params object (attribute-style)
    class P: pass
    params = P()

    # --- privacy / scheduler / budgets ---
    params.epsilon = float(eps)
    params.tau_percentile = cfg.tau_percentile
    params.A_budget_ratio = cfg.A_budget_ratio
    params.target_latency_ms = cfg.target_latency_ms
    params.target_bytes_per_round = cfg.target_bytes_per_round

    params.accountant_mode = cfg.accountant_mode
    params.window_w = cfg.window_w
    params.epsilon_per_window = cfg.epsilon_per_window
    params.delta_target = cfg.delta_target
    params.use_shuffle = bool(cfg.use_shuffle)
    params.geo_epsilon = cfg.geo_epsilon

    # --- quant / SAHTD++ knobs ---
    params.bytes_per_bit = cfg.bytes_per_bit
    params.BASE_BITS_A = cfg.BASE_BITS_A
    params.BASE_BITS_B = cfg.BASE_BITS_B
    params.BITS_C_EXTRA = cfg.BITS_C_EXTRA
    params.MIN_QUANT_BITS = cfg.MIN_QUANT_BITS
    params.MAX_QUANT_BITS = cfg.MAX_QUANT_BITS
    params.VAR_QUANTILE = cfg.VAR_QUANTILE

    # --- postprocess ---
    params.post_lap_alpha = cfg.post_lap_alpha
    params.post_process_var = cfg.post_process_var
    params.post_obs_var_base = cfg.post_obs_var_base
    params.entity_graph = entity_graph

    # --- C path (DAP/VDAF) ---
    params.use_vdaf_http = bool(cfg.use_vdaf_http)
    params.dap_leader_url = cfg.dap_leader_url
    params.dap_helper_url = cfg.dap_helper_url
    params.dap_api_token = cfg.dap_api_token
    params.dap_mode = cfg.dap_mode
    params.dap_task_id = cfg.dap_task_id
    params.C_BATCH_MIN = cfg.C_BATCH_MIN
    params.C_BATCH_MAX = cfg.C_BATCH_MAX
    params.C_DEADLINE_MS = cfg.C_DEADLINE_MS

    # --- misc ---
    params.subsample_p = cfg.subsample_p
    params.rng_seed = seed
    params.n_workers = int(cfg.n_workers)

    # Resolve callable
    _kind, call_fn = _resolve_method_callable(method)

    # Run
    results = call_fn(rounds_iter=rounds_iter, n_workers=cfg.n_workers, params=params)

    # Normalize to DataFrame
    if isinstance(results, pd.DataFrame):
        df = results.copy()
    else:
        df = pd.DataFrame(list(results))

    # Stamp suite info
    df["method"] = method
    df["epsilon"] = eps
    df["rho"] = rho
    df["mal"] = mal
    return df


# ================================================================
# 5) Summary & Saving
# ================================================================
def summarize_rounds(df_rounds: pd.DataFrame) -> pd.DataFrame:
    if df_rounds.empty:
        return pd.DataFrame()

    def _safe_mean(col: str) -> float:
        return float(df_rounds[col].mean()) if col in df_rounds.columns else float("nan")

    def _safe_std(col: str) -> float:
        return float(df_rounds[col].std()) if col in df_rounds.columns else float("nan")

    summary = {
        "rmse_mean": _safe_mean("rmse"),
        "rmse_std": _safe_std("rmse"),
        "rmse_raw_mean": _safe_mean("rmse_raw"),
        "bytes_mean": _safe_mean("bytes"),
        "bytes_std": _safe_std("bytes"),
        "time_s_mean": _safe_mean("time_s"),
        "time_s_std": _safe_std("time_s"),
        "n_rounds": len(df_rounds),
        "method": df_rounds["method"].iloc[0] if "method" in df_rounds.columns else "",
        "epsilon": df_rounds["epsilon"].iloc[0] if "epsilon" in df_rounds.columns else float("nan"),
        "rho": df_rounds["rho"].iloc[0] if "rho" in df_rounds.columns else float("nan"),
        "mal": df_rounds["mal"].iloc[0] if "mal" in df_rounds.columns else float("nan"),
    }
    return pd.DataFrame([summary])


def save_suite_results(cfg: ExperimentConfig, suite_name: str,
                       method_to_rounds: Dict[str, pd.DataFrame]):
    suite_dir = cfg.outdir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for method, df_rounds in method_to_rounds.items():
        df_rounds.to_csv(suite_dir / f"rounds_{method}.csv", index=False)
        sm = summarize_rounds(df_rounds)
        summaries.append(sm)

    if summaries:
        merged = pd.concat(summaries, ignore_index=True)
        merged.to_csv(suite_dir / "merged_results.csv", index=False)


# ================================================================
# 6) Main driver across suites
# ================================================================
def run_all_suites(cfg: ExperimentConfig):
    reports, truth = load_reports_and_truth(cfg)

    # figure out slots to use
    all_slots = np.sort(reports["slot"].unique())
    if cfg.rounds_per_suite > 0:
        slots = all_slots[: cfg.rounds_per_suite]
    else:
        slots = all_slots

    # entity graph (optional)
    entity_graph = load_entity_graph_or_none(cfg.entity_graph_csv)

    print(f"[INFO] total slots: {len(all_slots)}; per-suite slots: {len(slots)}")
    print(f"[INFO] methods: {cfg.methods}")
    print(f"[INFO] eps_list: {cfg.eps_list} rho_list: {cfg.rho_list} mal_list: {cfg.mal_list}")
    print(f"[INFO] accountant_mode={cfg.accountant_mode}, use_shuffle={cfg.use_shuffle}, use_vdaf_http={cfg.use_vdaf_http}")

    # Iterate suites
    for eps in cfg.eps_list:
        for rho in cfg.rho_list:
            for mal in cfg.mal_list:
                suite_name = f"suite_eps{eps}_rho{rho}_mal{mal}_RALL"
                print(f"[RUN] {suite_name}")

                method_to_rounds: Dict[str, pd.DataFrame] = {}
                for method in cfg.methods:
                    print(f"  - method = {method}")
                    df_rounds = run_one_method_on_suite(
                        cfg=cfg, method=method, eps=eps, rho=rho, mal=mal,
                        base_reports=reports, truth=truth, slots=slots,
                        entity_graph=entity_graph,
                    )
                    method_to_rounds[method] = df_rounds

                save_suite_results(cfg, suite_name, method_to_rounds)


# ================================================================
# 7) CLI
# ================================================================
def parse_args() -> ExperimentConfig:
    ap = argparse.ArgumentParser("Clean experiment runner for SA-HTD / SAHTD++")

    ap.add_argument('--reports_csv', default='/Users/wanghao/Desktop/SA-HTD/dataset_use/NYC/data/reports.csv')
    ap.add_argument('--truth_csv', default='/Users/wanghao/Desktop/SA-HTD/dataset_use/NYC/data/truth.csv')
    ap.add_argument('--outdir', default='/Users/wanghao/Desktop/SA-HTD/dataset_use/NYC/result')

    ap.add_argument("--methods", type=str, default="sa_htd_paper",
                    help="comma-separated method names (e.g. sa_htd_plus_x,sa_htd_paper)")

    ap.add_argument("--eps_list", type=str, default="1.0")
    ap.add_argument("--rho_list", type=str, default="0.2")
    ap.add_argument("--mal_list", type=str, default="0.0")

    ap.add_argument("--time_bin", type=str, default="5min")
    ap.add_argument("--rounds_per_suite", type=int, default=-1)
    ap.add_argument("--n_workers", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2025)

    # budgets / scheduling
    ap.add_argument("--A_budget_ratio", type=float, default=0.22)
    ap.add_argument("--tau_percentile", type=float, default=75.0)
    ap.add_argument("--target_latency_ms", type=float, default=2.0)
    ap.add_argument("--target_bytes_per_round", type=float, default=1.8e5)

    # privacy
    ap.add_argument("--accountant_mode", type=str, default="pld")
    ap.add_argument("--window_w", type=int, default=32)
    ap.add_argument("--epsilon_per_window", type=float, default=float("nan"))
    ap.add_argument("--delta_target", type=float, default=1e-5)
    ap.add_argument("--use_shuffle",default=True ,action="store_true")
    ap.add_argument("--geo_epsilon", type=float, default=0.0)

    # quant / SAHTD++
    ap.add_argument("--bytes_per_bit", type=float, default=0.125)
    ap.add_argument("--BASE_BITS_A", type=int, default=10)
    ap.add_argument("--BASE_BITS_B", type=int, default=8)
    ap.add_argument("--BITS_C_EXTRA", type=int, default=2)
    ap.add_argument("--MIN_QUANT_BITS", type=int, default=6)
    ap.add_argument("--MAX_QUANT_BITS", type=int, default=14)
    ap.add_argument("--VAR_QUANTILE", type=float, default=0.7)

    # postprocess
    ap.add_argument("--post_lap_alpha", type=float, default=0.3)
    ap.add_argument("--post_process_var", type=float, default=0.5)
    ap.add_argument("--post_obs_var_base", type=float, default=1.0)

    # C path
    ap.add_argument("--use_vdaf_http",default=True, action="store_true")
    ap.add_argument("--dap_leader_url", type=str, default="http://localhost:8787")
    ap.add_argument("--dap_helper_url", type=str, default="")
    ap.add_argument("--dap_api_token", type=str, default=None)
    ap.add_argument("--dap_mode", type=str, default="dryrun")
    ap.add_argument("--dap_task_id", type=str, default="nyc-speed-2025-ijcai")
    ap.add_argument("--C_BATCH_MIN", type=int, default=8)
    ap.add_argument("--C_BATCH_MAX", type=int, default=32)
    ap.add_argument("--C_DEADLINE_MS", type=float, default=80.0)

    # transforms
    ap.add_argument("--subsample_p", type=float, default=1.0)
    ap.add_argument("--entity_graph_csv", type=str, default=None)

    # CSV column mapping (only if your input uses different names)
    ap.add_argument("--rep_time_col", type=str, default="timestamp")
    ap.add_argument("--rep_entity_col", type=str, default="entity_id")
    ap.add_argument("--rep_worker_col", type=str, default="worker_id")
    ap.add_argument("--rep_value_col", type=str, default="value")
    ap.add_argument("--truth_time_col", type=str, default="timestamp")
    ap.add_argument("--truth_entity_col", type=str, default="entity_id")
    ap.add_argument("--truth_value_col", type=str, default="truth")

    args = ap.parse_args()

    def parse_float_list(s: str) -> List[float]:
        return [float(x) for x in s.split(",") if x.strip()]

    cfg = ExperimentConfig(
        reports_csv=Path(args.reports_csv),
        truth_csv=Path(args.truth_csv),
        outdir=Path(args.outdir),

        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        eps_list=parse_float_list(args.eps_list),
        rho_list=parse_float_list(args.rho_list),
        mal_list=parse_float_list(args.mal_list),

        time_bin=args.time_bin,
        rounds_per_suite=args.rounds_per_suite,
        n_workers=args.n_workers,
        seed=args.seed,

        A_budget_ratio=args.A_budget_ratio,
        tau_percentile=args.tau_percentile,
        target_latency_ms=args.target_latency_ms,
        target_bytes_per_round=args.target_bytes_per_round,

        accountant_mode=args.accountant_mode,
        window_w=args.window_w,
        epsilon_per_window=args.epsilon_per_window,
        delta_target=args.delta_target,
        use_shuffle=bool(args.use_shuffle),
        geo_epsilon=args.geo_epsilon,

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

        subsample_p=args.subsample_p,
        entity_graph_csv=Path(args.entity_graph_csv) if args.entity_graph_csv else None,

        rep_time_col=args.rep_time_col,
        rep_entity_col=args.rep_entity_col,
        rep_worker_col=args.rep_worker_col,
        rep_value_col=args.rep_value_col,
        truth_time_col=args.truth_time_col,
        truth_entity_col=args.truth_entity_col,
        truth_value_col=args.truth_value_col,
    )
    return cfg


def main():
    cfg = parse_args()
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    run_all_suites(cfg)


if __name__ == "__main__":
    main()