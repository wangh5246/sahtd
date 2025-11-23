# -*- coding: utf-8 -*-
"""
suite_paramgrid_all_nyc.py
======================
在 NYC 数据集上跑一组综合实验：
- 支持多算法：sa_htd_paper（SAHTD-Nexus 论文最终版）、sa_htd、etbp_td、eptd、随机 / LDP 基线等
- --rounds_per_suite <0 表示使用**所有**匹配的时间窗（slot）
"""

import argparse, random, sys, numpy as np, pandas as pd, json
from pathlib import Path

from sympy import false


# Ensure project root is importable so `dataset_use` package resolves on CLI runs.
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from dataset_use.UCI.src.EVAL.reduced_params import BALANCED, PRIVACY_FIRST, UTILITY_FIRST, CHANGE_DETECTION, \
    LOW_BANDWIDTH, ReducedParams
from dataset_use.NYC.src.algorithms import algorithms_bridge as bridge, baselines as RBL
from dataset_use.NYC.src.algorithms.etbp_td import etbp_td, ETBPParams as ETBPParamsStrict
from typing import Tuple as tuple

# ================= 默认 sweep 组合 ================= #

DEFAULT_SUITES = [
    dict(epsilon=0.5, rho=0.15, mal_rate=0.0,  rounds=12),
    dict(epsilon=0.5, rho=0.20, mal_rate=0.1,  rounds=12),
    dict(epsilon=0.5, rho=0.25, mal_rate=0.3,  rounds=12),
    dict(epsilon=1.0, rho=0.15, mal_rate=0.0,  rounds=12),
    dict(epsilon=1.0, rho=0.20, mal_rate=0.1,  rounds=12),
    dict(epsilon=1.0, rho=0.25, mal_rate=0.3,  rounds=12),
    dict(epsilon=2.0, rho=0.15, mal_rate=0.0,  rounds=12),
    dict(epsilon=2.0, rho=0.20, mal_rate=0.1,  rounds=12),
    dict(epsilon=4.0, rho=0.20, mal_rate=0.1,  rounds=12),
]

# ================= 一些小工具 ================= #

def _load_suites_from_json_or_literal(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        import ast
        return ast.literal_eval(s)

def _scalar_num(x, default=None):
    import ast
    if x is None:
        return default
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return _scalar_num(x[0], default=default)
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("(") or s.startswith("[")) and s.endswith((")", "]")):
            try:
                y = ast.literal_eval(s)
                return _scalar_num(y, default=default)
            except Exception:
                pass
        if "," in s and "." not in s and s.count(",") == 1:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return default if default is not None else x
    return x

def _str2bool(val, default=None):
    """
    更健壮地把字符串 / 数值解析为布尔值，支持 yes/no、true/false、1/0 等写法。
    argparse.BoolOptionalAction 在 3.12 前不可用，这里自己实现一个解析器。
    """
    if isinstance(val, bool):
        return val
    if val is None:
        return default if default is not None else False
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "y", "yes", "on"}:
        return True
    if s in {"0", "false", "f", "n", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {val}")

def load_and_bin(reports_csv: str, truth_csv: str, bin_str: str = "20min"):
    rep = pd.read_csv(reports_csv)
    tru = pd.read_csv(truth_csv)

    # —— 统一时间列到 timestamp ——
    def _ensure_time(df: pd.DataFrame, side: str):
        tcol = next(
            (c for c in ["timestamp", "window_start", "time", "date", "datetime",
                         "starttime", "start_time", "count_datetime"]
             if c in df.columns),
            None
        )
        if tcol is None:
            raise ValueError(f"No time-like column found in {side} CSV")
        df["timestamp"] = pd.to_datetime(df[tcol], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        return df

    rep = _ensure_time(rep, "reports")
    tru = _ensure_time(tru, "truth")

    # —— 统一值列名：report → value；y_true/value → truth ——
    if "value" not in rep.columns:
        for c in ["report", "y_report", "y", "count"]:
            if c in rep.columns:
                rep = rep.rename(columns={c: "value"})
                break
    if "value" not in rep.columns:
        raise KeyError("reports CSV缺少值列（尝试了 value/report/y_report/y/count）")

    if "truth" not in tru.columns:
        for c in ["y_true", "value", "gt", "label_true"]:
            if c in tru.columns:
                tru = tru.rename(columns={c: "truth"})
                break
    if "truth" not in tru.columns:
        raise KeyError("truth CSV缺少真值列（尝试了 truth/y_true/value/gt/label_true）")

    # —— 计算 slot ——
    rep["slot"] = rep["timestamp"].dt.floor(bin_str)
    tru["slot"] = tru["timestamp"].dt.floor(bin_str)
    slots = sorted(tru["slot"].unique().tolist())
    return rep, tru, slots

class Batch:
    __slots__ = ('entities','truth','reports','num_reports','slot','by_entity')
    def __init__(self, entities, truth, reports, slot):
        self.entities = entities
        self.truth = truth
        self.reports = reports
        self.num_reports = 0 if reports is None else len(reports)
        self.slot = slot

        # ★ 新增：按 entity 预聚合，之后算法直接用 numpy 数组
        if reports is not None and not reports.empty and \
           {"entity_id","value"}.issubset(reports.columns):
            g = reports.groupby("entity_id")["value"]
            self.by_entity = {
                e: g.get_group(e).to_numpy(dtype=float)
                for e in entities if e in g.groups
            }
        else:
            # 没 report，用 truth 占位，防止后面出空数组
            self.by_entity = {
                e: np.array([truth[i]], dtype=float)
                for i, e in enumerate(entities)
            }

def transform_reports_for_suite(rep_s: pd.DataFrame, mal_workers: set,
                                rho: float, rng: np.random.Generator) -> pd.DataFrame:
    if rep_s is None or rep_s.empty:
        return rep_s
    df = rep_s.copy()

    # 兼容列名：report → value
    if "value" not in df.columns and "report" in df.columns:
        df = df.rename(columns={"report": "value"})

    # 参与率（整窗按工人抽样）
    workers = df["worker_id"].unique().tolist()
    k = int(np.floor(len(workers) * float(rho)))
    if 0 < k < len(workers):
        part_workers = set(rng.choice(workers, size=k, replace=False))
        df = df[df["worker_id"].isin(part_workers)].reset_index(drop=True)

    if df.empty:
        return df

    # 关键：先转 float
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")

    # 稳健中心与尺度（按 entity 聚合）
    g = df.groupby("entity_id")["value"]
    med = g.median()
    mad = g.apply(lambda x: (np.median(np.abs(x - np.median(x))) + 1e-9)) * 1.4826

    is_mal = df["worker_id"].isin(mal_workers)
    if is_mal.any():
        m = df["entity_id"].map(med).to_numpy(dtype=float)
        s = df["entity_id"].map(mad).to_numpy(dtype=float)

        noise = rng.standard_t(df=3, size=len(df)) * 3.0
        new_vals = m + s * noise

        df.loc[is_mal, "value"] = new_vals[is_mal]
        df["value"] = df["value"].clip(lower=0)

    return df

def iter_rounds_suite(rep: pd.DataFrame, tru: pd.DataFrame, slots,
                      suite_idx: int, rounds: int, rho: float,
                      mal_rate: float, seed: int = 2025):
    rng = np.random.default_rng(seed + suite_idx * 97)
    if len(slots) == 0:
        return
    # 选择时窗：rounds < 0 表示使用**所有** slot
    if rounds < 0 or rounds >= len(slots):
        chosen = slots
    else:
        idx = np.linspace(0, len(slots) - 1, num=rounds, dtype=int)
        chosen = [slots[i] for i in idx]

    # 跨本 suite 固定恶意工人集合
    all_workers = rep["worker_id"].unique().tolist() if "worker_id" in rep.columns else []
    mal_k = int(np.floor(len(all_workers) * float(mal_rate)))
    mal_workers = set(rng.choice(all_workers, size=mal_k, replace=False)) if mal_k > 0 else set()

    for s in chosen:
        tru_s = tru[tru["slot"] == s][["entity_id", "truth"]]
        if tru_s.empty:
            continue
        ents = tru_s["entity_id"].to_numpy().tolist()
        rep_s = rep[(rep["slot"] == s) & (rep["entity_id"].isin(ents))].reset_index(drop=True)
        rep_s = transform_reports_for_suite(rep_s, mal_workers=mal_workers, rho=rho, rng=rng)
        yield Batch(entities=ents,
                    truth=tru_s["truth"].to_numpy(dtype=float),
                    reports=rep_s, slot=s)

class _P:
    """简单的命名空间，用来装 params"""
    def __init__(self, **kw):
        self.__dict__.update(kw)

class _Spy:
    """探针式迭代器：让 bridge 能看到所有 batch，用于计算 bytes / enc_ops 等"""
    def __init__(self, it):
        self._it = iter(it)
        self.batches = []

    def __iter__(self):
        return self

    def __next__(self):
        b = next(self._it)
        self.batches.append(b)
        return b

# ==================== 参数构造：统一给 SAHTD-Nexus / 其他算法 ==================== #

def build_params(args, eps):
    """
        使用精简参数系统
        """
    # 🔧 检查是否启用精简参数模式
    use_reduced = getattr(args, 'use_reduced_params', True)

    if use_reduced:
        # 使用5个核心参数
        reduced = ReducedParams(
            epsilon=float(eps),
            routing_intensity=float(getattr(args, 'routing_intensity', 0.5)),
            communication_budget=float(getattr(args, 'communication_budget', 1.0)),
            smoothing_strength=float(getattr(args, 'smoothing_strength', 0.5)),
            adaptive_sensitivity=float(getattr(args, 'adaptive_sensitivity', 0.5))
        )

        # 自动推导完整参数
        full_dict = reduced.to_full_params({
            'n_entities': 100,  # 会在运行时动态更新
            'avg_reports_per_entity': 10.0,
        })

        # 转换为_P对象
        return _P(**full_dict)
    """
    针对 SAHTD-Nexus 的统一参数构造：
    - 把 CLI 中和隐私会计 / 调度 / 量化有关的东西都打包进一个 _P；
    - epsilon_per_window 默认 = eps * window_w。
    """
    epw = getattr(args, "epsilon_per_window", float("nan"))
    if epw != epw or epw is None:  # NaN 检测
        epw = float(eps) * int(getattr(args, "window_w", 32))

    token = getattr(args, "dap_api_token", None)
    api_token = None if (token is None or str(token).strip().lower() in ("", "none")) else str(token)

    uldp_cols_str = str(getattr(args, "uldp_sensitive_cols", ""))
    uldp_cols = [s.strip() for s in uldp_cols_str.split(",") if s.strip()]

    return _P(
        # 基本 DP 预算
        epsilon=float(eps),

        # 调度 / 约束目标
        target_latency_ms=float(getattr(args, "target_latency_ms", 2.0)),
        target_bytes_per_round=float(getattr(args, "target_bytes_per_round", 1.2e4)),

        # 连续流隐私会计
        accountant_mode=str(getattr(args, "accountant_mode", "pld")).lower(),
        window_w=int(getattr(args, "window_w", 32)),
        epsilon_per_window=float(epw),
        delta_target=float(getattr(args, "delta_target", 1e-5)),

        # DAP/VDAF
        use_vdaf_http=bool(getattr(args, "use_vdaf_http", True)),
        dap_mode=str(getattr(args, "dap_mode", "dryrun")),
        dap_leader_url=str(getattr(args, "dap_leader_url", "http://localhost:8787")),
        dap_helper_url=str(getattr(args, "dap_helper_url", "")),
        dap_api_token=api_token,
        dap_task_id="nyc",

        # Shuffle / ULDP / Geo
        use_shuffle=bool(getattr(args, "use_shuffle", True)),
        uldp_sensitive_cols=uldp_cols,
        geo_epsilon=float(getattr(args, "geo_epsilon", 0.0)),

        # SAHTD-Nexus 的路由 / 调度初值
        tau_percentile=float(getattr(args, "tau_percentile", 75.0)),
        A_budget_ratio=float(getattr(args, "A_budget_ratio", 0.18)),

        # 量化 / 自适应相关
        bytes_per_bit=float(getattr(args, "bytes_per_bit", 0.125)),
        BASE_BITS_A=int(getattr(args, "base_bits_a", 10)),
        BASE_BITS_B=int(getattr(args, "base_bits_b", 6)),
        BITS_C_EXTRA=int(getattr(args, "bits_c_extra", 2)),
        MIN_QUANT_BITS=int(getattr(args, "min_quant_bits", 2)),
        MAX_QUANT_BITS=int(getattr(args, "max_quant_bits", 10)),
        VAR_QUANTILE=float(getattr(args, "var_quantile", 0.7)),
        quant_bits_init=int(getattr(args, "quant_bits_init", 6)),

        # Bandit / 自适应 epsilon
        bandit_epsilon=float(getattr(args, "bandit_epsilon", 0.1)),
        eps_min_scale=float(getattr(args, "eps_min_scale", 0.5)),
        eps_max_scale=float(getattr(args, "eps_max_scale", 1.5)),

        # 后处理（Kalman + Graph Laplacian）
        post_lap_alpha=float(getattr(args, "post_lap_alpha", 0.25)),
        post_process_var=float(getattr(args, "post_process_var", 0.3)),
        post_obs_var_base=float(getattr(args, "post_obs_var_base", 1.0)),
        disable_postprocess=_str2bool(getattr(args, "disable_postprocess", False), default=False),
        use_privacy_aware_postprocess=_str2bool(getattr(args, "use_privacy_aware_postprocess", True), default=True),
        post_var_threshold_low=float(getattr(args, "post_var_threshold_low", 20.0)),
        post_var_threshold_high=float(getattr(args, "post_var_threshold_high", 80.0)),
        post_sparse_threshold=int(getattr(args, "post_sparse_threshold", 5)),
        post_privacy_tension_ratio=float(getattr(args, "post_privacy_tension_ratio", 0.75)),
        post_change_window=int(getattr(args, "post_change_window", 3)),
        post_change_sensitivity=float(getattr(args, "post_change_sensitivity", 2.0)),
        post_convergence_window=int(getattr(args, "post_convergence_window", 5)),
        post_convergence_threshold=float(getattr(args, "post_convergence_threshold", 0.05)),
        post_warmup_rounds=int(getattr(args, "post_warmup_rounds", 3)),
        enable_privacy_adaptive=_str2bool(getattr(args, "enable_privacy_adaptive", True), default=True),

        # 字节计费参数（给 Nexus 的 bytes 公式）
        perA_bytes=int(getattr(args, "perA_bytes", 32)),
        perB_bytes=int(getattr(args, "perB_bytes", 32)),
    )

# ==================== 统一算法入口 ==================== #

def run_method(
    rounds_iter,
    name: str,
    n_workers: int,
    args,
    *,
    epsilon=None,
    rho=None,
    mal_rate=None,
    rounds=None,
    slots=None,
    params=None
):
    """
    统一的算法入口。
    - SAHTD-Nexus 用 name = "sa_htd_paper" 或 "sahtd_nexus"；
    - 其他算法保持和你原来工程兼容。
    """
    import pandas as pd
    strict_map = {
        'ud_ldp': RBL.ud_ldp,
        'dplp': RBL.dplp,
        'random_baseline': RBL.random_baseline,
    }

    P = params
    if P is None and epsilon is not None:
        P = build_params(args, epsilon)

    # —— 论文 EPTD —— #
    if name == "eptd":
        return pd.DataFrame(bridge.eptd_bridge(_Spy(rounds_iter), n_workers, P))

    # —— ETBP-TD —— #
    elif name == "etbp_td":
        return pd.DataFrame(etbp_td(_Spy(rounds_iter), n_workers,
                                    params if params is not None else ETBPParamsStrict()))

    # —— 原始 SA-HTD —— #
    elif name == "sa_htd":
        P0 = _P(
            epsilon=epsilon if epsilon is not None else 1.0,
            quant_bits=int(getattr(args, 'quant_bits', 10)),
            deadline_ms=float(getattr(args, 'deadline_ms', 1.0)),
            batch_min=int(getattr(args, 'batch_min', 8)),
            use_shuffle=bool(getattr(args, 'use_shuffle', False)),
            use_vdaf_http=bool(getattr(args, 'use_vdaf_http', False)),
            dap_mode=str(getattr(args, 'dap_mode', 'dryrun')),
            dap_leader_url=getattr(args, 'dap_leader_url', ''),
            dap_helper_url=getattr(args, 'dap_helper_url', ''),
            dap_api_token=getattr(args, 'dap_api_token', ''),
        )
        return pd.DataFrame(bridge.sa_bridge(_Spy(rounds_iter), n_workers, P0))

    # —— SAHTD-X（plus 版本） —— #
    elif name == "sahtd":
        P0 = _P(
            epsilon=epsilon if epsilon is not None else 1.0,
            tau_percentile=getattr(args, 'tau_percentile', 70),
            A_budget_ratio=getattr(args, 'A_budget_ratio', 0.28),
            subsample_p=getattr(args, 'subsample_p', 1.0),
            quant_bits=int(getattr(args, 'quant_bits', 10)),
            deadline_ms=float(getattr(args, 'deadline_ms', 1.0)),
            batch_min=int(getattr(args, 'batch_min', 8)),
            stale_lambda=float(getattr(args, 'stale_lambda', 0.5)),
            early_stop_eps=float(getattr(args, 'early_stop_eps', 1e-4)),
            early_stop_steps=int(getattr(args, 'early_stop_steps', 2)),
            geo_epsilon=float(getattr(args, 'geo_epsilon', 0.0)),
            max_b_per_entity=int(getattr(args, 'max_b_per_entity', 0)),
            target_latency_ms=float(getattr(args, 'target_latency_ms', 2.0)),
            rng_seed=int(getattr(args, 'rng_seed', 2025)),
        )
        return pd.DataFrame(bridge.sahtd_bridge(_Spy(rounds_iter), n_workers, P0))

    # —— SAHTD-Nexus / 论文最终版 —— #
    # —— SAHTD-Nexus（原 sa_htd_paper）：ABC 三路 + 自适应量化 —— #
    elif name == "sa_htd_paper":
        baseP = build_params(args, epsilon)
        if isinstance(baseP, _P):
            base_kwargs = dict(baseP.__dict__)
        else:
            base_kwargs = dict(vars(baseP))

        base_kwargs.update(dict(
            rho=rho,
            mal_rate=mal_rate,
            total_rounds=int(rounds) if rounds is not None else 0,

            # 1. 确保这里不是 0.03，改为从 args 读取，或者硬编码为 0.15
            BASE_C_RATIO=float(getattr(args, 'base_c_ratio', 0.15)),

            # 2. 必须允许大预算！不要写 900.0！
            # 这里直接删掉 target_bytes_per_round 这一行，因为它在 build_params 里已经处理了
            # 如果非要留，改为：
            target_bytes_per_round=float(getattr(args, "target_bytes_per_round", 400000.0)),

            # 3. 修正量化位
            BASE_BITS_A=int(getattr(args, 'base_bits_a', 12)),

            # 4. 修正卡尔曼噪声，给 20.0 以适应突变
            post_process_var=float(getattr(args, 'post_process_var', 20.0)),

            # 其他保持不变...
        ))
        P = _P(**base_kwargs)
        return pd.DataFrame(bridge.sahtd_paper_bridge(_Spy(rounds_iter), n_workers, P))

    # —— 其他基线 —— #
    if name not in ("NewSAHTD", "newsahtd", "new_sa_htd_budgeted"):
        if name == 'pure_ldp' and epsilon is not None:
            return pd.DataFrame(RBL.pure_ldp(_Spy(rounds_iter), n_workers,
                                             RBL.LDPParams(epsilon=float(epsilon))))
        elif name == 'ud_ldp' and epsilon is not None:
            return pd.DataFrame(
                RBL.ud_ldp(_Spy(rounds_iter), n_workers,
                           RBL.UDLDPParams(epsilon_total=float(epsilon) * 3.0)))
        elif name == 'dplp' and epsilon is not None:
            return pd.DataFrame(RBL.dplp(_Spy(rounds_iter), n_workers,
                                         RBL.DPLPParams(epsilon=float(epsilon))))
        elif name == 'fed_sense' and epsilon is not None:
            return pd.DataFrame(RBL.fed_sense(_Spy(rounds_iter), n_workers, None))
        elif name == 'random' and epsilon is not None:
            return pd.DataFrame(RBL.random_baseline(_Spy(rounds_iter), n_workers, None))
        else:
            return pd.DataFrame(strict_map[name](_Spy(rounds_iter), n_workers, P))

    # —— NewSAHTD（老版本接口，保留兼容性） —— #
    if params is None:
        eps_val = epsilon
        Pn = _P(
            epsilon=eps_val,
            rho=rho,
            mal_rate=mal_rate,
            route_ratio_low=0.05,
            route_ratio_high=0.08,
            rho_max=0.35,
            warmup_rounds=2,
            budget_bytes=180_000,
            total_rounds=(int(rounds) if rounds is not None else 0)
        )
    else:
        Pn = params if isinstance(params, _P) else _P(**params)

    logs = bridge.newsahtd_bridge(rounds_iter, n_workers, Pn)
    return pd.DataFrame(logs)

# ==================== 跑一个 suite ==================== #

def run_one_suite(si, suite, rep, tru, slots, methods, args):
    import numpy as np
    import pandas as pd
    from pathlib import Path

    eps = float(_scalar_num(suite.get('epsilon', 1.0), default=1.0))
    rho = float(_scalar_num(suite.get('rho', 1.0), default=1.0))
    mal = float(_scalar_num(suite.get('mal_rate', 0.0), default=0.0))

    # rounds：命令行优先；rounds_per_suite<0 → 全量 slot
    R_cmd = int(args.rounds_per_suite)
    R = R_cmd if R_cmd != 0 else int(suite.get('rounds', 12))
    if R_cmd < 0:
        R = -1

    out_base = Path(args.outdir)
    print(f"== Suite[{si}] eps={eps} rho={rho} mal={mal} rounds={('ALL' if R<0 else R)} ==")
    suite_dir = out_base / f"suite_eps{eps}_rho{rho}_mal{mal}_R{('ALL' if R<0 else R)}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    # 记录 slot 信息
    if R < 0:
        chosen_num = len(slots)
    else:
        chosen_num = min(R, len(slots))
    (suite_dir / '_meta.txt').write_text(
        f"计划使用 slot 数：{chosen_num} / truth slots 总数：{len(slots)}\n",
        encoding='utf-8'
    )

    merged_rounds = []
    merged_results = []

    for name in methods:
        print(f"  - 运行方法: {name}")
        rounds_iter = iter_rounds_suite(
            rep, tru, slots,
            suite_idx=si, rounds=R,
            rho=rho, mal_rate=mal, seed=args.seed
        )
        df = run_method(
            rounds_iter, name, args.n_workers, args,
            epsilon=eps, rho=rho, mal_rate=mal, rounds=R, params=None
        )
        df['method'] = name
        df['suite_idx'] = si
        df['epsilon'] = eps
        df['rho'] = rho
        df['mal_rate'] = mal
        df.to_csv(suite_dir / f"rounds_{name}.csv", index=False)
        merged_rounds.append(df)

        def _num(s):
            return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)

        rm = _num(df['rmse'])
        vm = _num(df['var'])
        rv = _num(df['resid_var'])

        res_row = dict(
            method=name, suite_idx=si, epsilon=eps, rho=rho, mal_rate=mal, rounds=len(df),
            rmse_last=float(rm.iloc[-1]) if not rm.empty else float('nan'),
            rmse_mean=float(rm.mean()) if not rm.empty else float('nan'),
            rmse_std=float(rm.std(ddof=1)) if len(rm) > 1 else float('nan'),
            var_mean=float(vm.mean()) if not vm.empty else float('nan'),
            resid_var_mean=float(rv.mean()) if not rv.empty else float('nan'),
            bytes_mean=float(_num(df['bytes']).mean()),
            enc_ops_mean=float(_num(df['enc_ops']).mean()),
            time_s_mean=float(_num(df['time_s']).mean()),
        )
        merged_results.append(pd.DataFrame([res_row]))

    mr = pd.concat(merged_rounds, ignore_index=True) if merged_rounds else pd.DataFrame()
    if 'slot' in mr.columns:
        mr['slot'] = mr['slot'].astype(str)
    mr.to_csv(suite_dir / 'merged_rounds.csv', index=False)
    res = pd.concat(merged_results, ignore_index=True) if merged_results else pd.DataFrame()
    res.to_csv(suite_dir / 'merged_results.csv', index=False)
    print(f"  输出目录: {suite_dir}")

# ==================== main ==================== #

def main():
    ap = argparse.ArgumentParser()

    # DAP / VDAF
    ap.add_argument('--use_vdaf_http', type=lambda s: str(s).lower() == 'true', default=False)
    ap.add_argument('--dap_mode', default='dryrun')  # dryrun|daphne|divviup|off
    ap.add_argument('--dap_leader_url', default='http://localhost:8787')
    ap.add_argument('--dap_helper_url', default='')
    ap.add_argument('--dap_api_token', default='')

    # 隐私会计
    ap.add_argument('--accountant_mode', default='naive')  # pld|naive
    ap.add_argument('--window_w', type=int, default=32)
    ap.add_argument('--epsilon_per_window', type=float, default=float('nan'))
    ap.add_argument('--delta_target', type=float, default=1e-5)

    # Shuffle / ULDP / Geo
    ap.add_argument('--use_shuffle', type=lambda s: str(s).lower() == 'true', default=True)
    ap.add_argument('--uldp_sensitive_cols', default='')
    ap.add_argument('--geo_epsilon', type=float, default=0.0)

    # 约束目标
    ap.add_argument('--target_latency_ms', type=float, default=2.0)
    ap.add_argument('--target_bytes_per_round', type=float, default=400000.0)

    # NYC 数据集路径
    ap.add_argument('--reports_csv', default='/Users/wanghao/Desktop/SA-HTD/dataset_use/UCI/data/reports.csv')
    ap.add_argument('--truth_csv',   default='/Users/wanghao/Desktop/SA-HTD/dataset_use/UCI/data/truth.csv')
    ap.add_argument('--outdir',      default='/Users/wanghao/Desktop/SA-HTD/dataset_use/UCI/result_new')

    ap.add_argument('--n_workers', type=int, default=300)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--time_bin', default='10min')

    # 方法列表：默认只跑 SAHTD-Nexus
    ap.add_argument('--methods', default='fed_sense,dplp,eptd,ud_ldp,random,etbp_td')
    ap.add_argument('--suites_json', default='')
    ap.add_argument('--suites_first_n', type=int, default=0)
    ap.add_argument('--num_procs', type=int, default=2,
                    help="并行进程数，>1 表示多进程")
    ap.add_argument('--rounds_per_suite', type=int, default=-1,
                    help="<0 表示全部 slot；>0 表示每个 suite 只取该数量的 slot")

    # 与 SA-HTD / SAHTD-X / Nexus 相关的一些参数
    ap.add_argument('--perA_bytes', type=int, default=32)
    ap.add_argument('--perB_bytes', type=int, default=32)
    ap.add_argument('--A_budget_ratio', type=float, default=0.22)
    ap.add_argument('--tau_percentile', type=float, default=75.0)

    # Nexus 量化 / bandit 参数
    ap.add_argument('--bytes_per_bit', type=float, default=0.125)
    ap.add_argument('--base_bits_a', type=int, default=12)
    ap.add_argument('--base_bits_b', type=int, default=6)
    ap.add_argument('--bits_c_extra', type=int, default=2)
    ap.add_argument('--min_quant_bits', type=int, default=2)
    ap.add_argument('--max_quant_bits', type=int, default=10)
    ap.add_argument('--var_quantile', type=float, default=0.7)
    ap.add_argument('--quant_bits_init', type=int, default=6)

    ap.add_argument('--bandit_epsilon', type=float, default=0.1)
    ap.add_argument('--eps_min_scale', type=float, default=0.5)
    ap.add_argument('--eps_max_scale', type=float, default=1.5)

    # 一些老 SAHTD-X 会用到的参数（为了兼容，保留）
    ap.add_argument('--max_b_per_entity', type=int, default=32)
    ap.add_argument('--early_stop_eps', type=float, default=8e-3)
    ap.add_argument('--early_stop_steps', type=int, default=2)
    # —— SAHTD-Nexus 相关：C 路强度、自适应量化、后处理 ——

    # C 路强度：走 C 路的实体比例、C 路 eps 放大倍数、C 路批大小上限
    ap.add_argument('--base_c_ratio', type=float, default=0.05,
                    help="每轮大约多少比例的实体走 C 路（高价值通道），默认 0.05 = 5%")
    ap.add_argument('--c_eps_scale', type=float, default=2.0,
                    help="C 路本地 DP 噪声的 ε 放大倍数，相对 B 路（默认 2.0）")
    ap.add_argument('--c_batch_max', type=int, default=32,
                    help="每轮最多多少实体被分配到 C 路")

    # 量化相关：每 bit 对应的字节数、A/B/C 的基准 bit 数

    ap.add_argument('--avg_reports_per_entity', type=float, default=10.0,
                    help="估计每个实体每轮平均报告数（用于字节预算缩放）")

    # 后处理（Kalman + 图拉普拉斯）相关
    ap.add_argument('--disable_postprocess', type=_str2bool, default=False,
                    help="传 true 关闭 Kalman+Lap 后处理，false 表示开启")
    ap.add_argument('--post_lap_alpha', type=float, default=0.2,
                    help="图拉普拉斯平滑权重 α，越大表示邻居影响越强")
    ap.add_argument('--post_process_var', type=float, default=0.5,
                    help="Kalman 过程噪声方差（状态演化不确定性）")
    ap.add_argument('--post_obs_var_base', type=float, default=1.0,
                    help="Kalman 观测噪声的基准方差")
    ap.add_argument('--use_privacy_aware_postprocess', type=_str2bool, default=True,
                    help="使用隐私/稀疏感知的自适应后处理（结合预算/方差/稀疏度信号）")
    ap.add_argument('--post_var_percentile_low', type=float, default=0.7,
                    help="低方差阈值；高于它开始考虑平滑")
    ap.add_argument('--post_var_percentile_high', type=float, default=0.93,
                    help="高方差阈值；超过视为强平滑需求")
    ap.add_argument('--post_sparse_threshold', type=int, default=0.5,
                    help="每 entity 报告数低于此值视为稀疏")
    ap.add_argument('--post_privacy_tension_ratio', type=float, default=0.75,
                    help="已用预算比例超过该值时加强平滑")
    ap.add_argument('--post_change_window', type=int, default=3,
                    help="突变检测窗口长度")
    ap.add_argument('--post_change_sensitivity', type=float, default=2.0,
                    help="突变敏感度系数，变化量超过 sensitivity*σ 触发")
    ap.add_argument('--post_convergence_window', type=int, default=5,
                    help="收敛检测窗口长度")
    ap.add_argument('--post_convergence_threshold', type=float, default=0.05,
                    help="收敛判定阈值（方差相对变化率）")
    ap.add_argument('--post_warmup_rounds', type=int, default=3,
                    help="后处理触发前的 warmup 轮数")
    ap.add_argument('--enable_privacy_adaptive', type=_str2bool, default=True,
                    help="是否启用自适应后处理；false 则始终开启后处理")
    # ========== 新增: 精简参数模式 ========== #
    ap.add_argument('--use_reduced_params', type=_str2bool, default=True,
                    help="使用精简的5参数系统(推荐)")

    # 精简参数系统的5个核心参数
    ap.add_argument('--routing_intensity', type=float, default=0.5,
                    help="路由策略强度 [0,1]: 0=保守,1=激进")
    ap.add_argument('--communication_budget', type=float, default=1.0,
                    help="通信预算倍数 [0.5,2.0]: 1.0=标准")
    ap.add_argument('--smoothing_strength', type=float, default=0.5,
                    help="后处理平滑强度 [0,1]: 0=不平滑,1=强平滑")
    ap.add_argument('--adaptive_sensitivity', type=float, default=0.5,
                    help="自适应触发灵敏度 [0,1]: 0=低,1=高")

    # 预设配置快捷方式
    ap.add_argument('--preset', type=str, default='',
                    choices=['', 'balanced', 'privacy', 'utility', 'low_bandwidth', 'change_detection'],
                    help="使用预设配置")
    args = ap.parse_args()
    # 🔧 处理预设配置
    if args.preset:
        preset_map = {
            'balanced': BALANCED,
            'privacy': PRIVACY_FIRST,
            'utility': UTILITY_FIRST,
            'low_bandwidth': LOW_BANDWIDTH,
            'change_detection': CHANGE_DETECTION,
        }
        preset = preset_map.get(args.preset)
        if preset:
            # 覆盖命令行参数
            args.routing_intensity = preset.routing_intensity
            args.communication_budget = preset.communication_budget
            args.smoothing_strength = preset.smoothing_strength
            args.adaptive_sensitivity = preset.adaptive_sensitivity
            print(f"[INFO] 使用预设配置: {args.preset}")
            print(preset.get_description())
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_base = Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)

    rep, tru, slots = load_and_bin(args.reports_csv, args.truth_csv, bin_str=args.time_bin)
    print(f"[INFO] truth slots 总数：{len(slots)}")

    suites = DEFAULT_SUITES
    if args.suites_json:
        suites = _load_suites_from_json_or_literal(args.suites_json)
    if args.suites_first_n and args.suites_first_n > 0:
        suites = suites[:args.suites_first_n]
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    # ===== 按需并行跑各个 suite =====
    from multiprocessing import Pool
    tasks = [(si, suite, rep, tru, slots, methods, args)
             for si, suite in enumerate(suites)]

    if args.num_procs and args.num_procs > 1:
        print(f"[INFO] 使用多进程并行运行，每次最多 {args.num_procs} 个进程")
        with Pool(processes=args.num_procs) as pool:
            pool.starmap(run_one_suite, tasks)
    else:
        print("[INFO] 顺序运行各个 suite（num_procs<=1）")
        for t in tasks:
            run_one_suite(*t)

    print(f"[DONE] 输出根目录: {out_base}")

if __name__ == '__main__':
    main()
