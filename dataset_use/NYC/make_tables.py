import re
from pathlib import Path

import pandas as pd

# ============ 1. 根目录路径：改成你自己的 ============
from pathlib import Path

ROOT = Path("/Users/wanghao/Desktop/SA-HTD/dataset_use/NYC/important")
# 例如：ROOT = Path("/home/you/experiments/nyc_suites")

# ============ 2. 解析目录名里的 eps / rho / mal ============

DIR_PATTERN = re.compile(
    r"suite_eps(?P<eps>[0-9.]+)_rho(?P<rho>[0-9.]+)_mal(?P<mal>[0-9.]+)_RALL"
)

# 把文件名映射成你论文里要展示的方法名（按你自己的 baseline 改）
METHOD_NAME_MAP = {
    "rounds_sa_htd_paper": "SAHTD-Nexus (Ours)",
    "rounds_ud_ldp": "UD-LDP",
    "rounds_dplp": "DPLP",
    "rounds_eptd": "EPTD",
    "rounds_etbp_td": "ETBP-TD",
    "rounds_fed_sense": "FedSense",
    "rounds_random": "Random",
}


# ============ 3. 从单个 suite 目录里读取所有 rounds_*.csv ============

def load_metrics_from_rounds(exp_dir: Path):
    """
    从一个 suite 目录（例如 suite_eps0.5_rho0.2_mal0.1_RALL）
    读取里面所有 rounds_*.csv，汇总出每个方法的
    bytes / enc_ops / time 三个指标。

    这里会自动打印每个 csv 的列名，并在多个候选名中找一列，
    避免因为列名细微差异导致 KeyError。
    """
    rows = []

    for csv_path in exp_dir.glob("rounds_*.csv"):
        stem = csv_path.stem  # e.g. "rounds_sa_htd_paper"
        method = METHOD_NAME_MAP.get(stem, stem)

        df = pd.read_csv(csv_path)
        print(f"\n[DEBUG] reading {csv_path.name}")
        print("        columns:", list(df.columns))

        # 我们给每个指标设一组“候选列名”，按顺序寻找
        bytes_candidates   = ["bytes_mean", "bytes", "byte_mean", "total_bytes"]
        enc_ops_candidates = ["enc_ops_mean", "enc_ops", "encrypt_ops"]
        time_candidates    = ["time_s_mean", "time_mean", "time", "time_s","time_ms_mean"]

        def find_col(candidates, kind):
            for c in candidates:
                if c in df.columns:
                    return c
            # 如果全都没找到，就抛错并告诉你在哪个文件出问题
            raise KeyError(
                f"No column for {kind} found in {csv_path.name}. "
                f"Tried: {candidates}"
            )

        bytes_col   = find_col(bytes_candidates,   "bytes")
        enc_ops_col = find_col(enc_ops_candidates, "enc_ops")
        time_col    = find_col(time_candidates,    "time")

        # 按列聚合
        bytes_per_round   = df[bytes_col].mean()
        enc_ops_per_round = df[enc_ops_col].mean()

        # 如果列名里有 "s"（秒），就 *1000 变成 ms，否则直接用
        if "s" in time_col:
            time_ms_per_round = df[time_col].mean() * 1000.0
        else:
            time_ms_per_round = df[time_col].mean()

        rows.append(
            {
                "method": method,
                "bytes_per_round": bytes_per_round,
                "enc_ops_per_round": enc_ops_per_round,
                "time_ms_per_round": time_ms_per_round,
            }
        )

    return rows


# ============ 4. 扫描所有 suite_ 目录，合成大表 ============

def collect_all_results(root: Path = ROOT) -> pd.DataFrame:
    records = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = DIR_PATTERN.fullmatch(d.name)
        if not m:
            continue

        eps = float(m.group("eps"))
        rho = float(m.group("rho"))
        mal = float(m.group("mal"))

        metrics_rows = load_metrics_from_rounds(d)
        for row in metrics_rows:
            row.update({"eps": eps, "rho": rho, "mal": mal})
            records.append(row)

    df = pd.DataFrame(records)
    return df


# ============ 5. 固定一个 eps/rho/mal 生成表格（类似你截图的 Table 3） ============

def human_bytes(x):
    # 把 "每轮字节数" 转成更好看的 M 单位（可按需改）
    return f"{x/1e6:.2f}M"

def human_k(x):
    return f"{x/1e3:.1f}K"


def make_cost_table(df: pd.DataFrame, eps: float, rho: float, mal: float):
    sub = df[
        (df["eps"] == eps)
        & (df["rho"] == rho)
        & (df["mal"] == mal)
    ].copy()

    if sub.empty:
        raise ValueError(f"No experiments found for eps={eps}, rho={rho}, mal={mal}")

    # 方法显示顺序（按你想在表里出现的顺序改）
    method_order = [
        "SAHTD-Nexus (Ours)",
        "DPLP",
        "ETBP-TD",
        "EPTD",
        "FedSense",
        "UD-LDP",
        "Random",
    ]
    sub["method"] = pd.Categorical(sub["method"], categories=method_order, ordered=True)
    sub = sub.sort_values("method")

    table = sub[
        ["method", "bytes_per_round", "enc_ops_per_round", "time_ms_per_round"]
    ].copy()
    table.rename(
        columns={
            "method": "Method",
            "bytes_per_round": "Bytes / round",
            "enc_ops_per_round": "Enc Ops / round",
            "time_ms_per_round": "Time (ms/round)",
        },
        inplace=True,
    )

    print(f"\n=== Raw table for eps={eps}, rho={rho}, mal={mal} ===")
    print(table.to_string(index=False))

    # 生成一个格式化版，更接近论文里的样子
    table_fmt = table.copy()
    table_fmt["Bytes / round"] = table_fmt["Bytes / round"].map(human_bytes)
    table_fmt["Enc Ops / round"] = table_fmt["Enc Ops / round"].map(human_k)
    table_fmt["Time (ms/round)"] = table_fmt["Time (ms/round)"].map(
        lambda x: f"{x:.1f}"
    )

    print("\n=== LaTeX (formatted) ===")
    print(table_fmt.to_latex(index=False, escape=False))

    return table, table_fmt


# ============ 6. 运行示例 ============

if __name__ == "__main__":
    df_all = collect_all_results(ROOT)

    # 举例：生成 eps=1.0, rho=0.2, mal=0.1 的成本表
    make_cost_table(df_all, eps=1.0, rho=0.2, mal=0.1)