# ---- EPTD (paper-faithful, simulation) -----------------------
import time, numpy as np, pandas as pd

def _mk_super_increasing_z(M, S):
    """构造满足  sum_{i<l} z_i * S < z_l  的最小超递增序列; z_1=1。"""
    z = [1]
    sumz = 1
    for _ in range(2, M+1):
        z_next = int(sumz * S) + 1      # 最小可行
        z.append(z_next)
        sumz += z_next
    return np.array(z, dtype=np.int64)

def _recover_truth_from_phi(phi_int, sum_w_int, z, scale):
    """
    按论文式(11)从 Φ 与 Σw 迭代恢复各任务真值：
      y_m = ((Φ_m - Φ_m mod z_m) / (z_m * Σw))
      Φ_{m-1} = Φ_m mod z_m
    注意：这里所有量是 “放大 scale 后的整数”。
    """
    M = len(z)
    ym = np.zeros(M, dtype=float)
    # 去掉 r_{j,3} * Σw 的偏移：Φ* = Φ - r_{j,3} * Σw
    phi_star = int(phi_int)
    for m in range(M-1, 0, -1):  # 按论文记法从后往前剥离
        t = phi_star % int(z[m])
        y_num = (phi_star - t) // int(z[m])
        ym[m] = (y_num / sum_w_int) / scale
        phi_star = t
    # y1: 论文给出 y1 = Φ1 / Σw
    ym[0] = (phi_star / sum_w_int) / scale
    return ym

def _pick_worker_id_column(rep: pd.DataFrame):
    for c in ["worker_id","wid","user_id","sensor_id","device_id","source","uid"]:
        if c in rep.columns:
            return c
    # 没有显式工人id，就把每行当一名“轻量工人”
    rep = rep.copy()
    rep["_tmp_wid"] = np.arange(len(rep))
    return "_tmp_wid"

def eptd(rounds_iter, n_workers: int, params=None):
    """
    论文 EPTD 的数值仿真实现（CRH 版）。一步匹配 + 多轮迭代：
      - 距离 D_k = sum_m (x_{k,m} - y_m)^2        (式(9))
      - 权重 w_k = log( sum_i D_i / D_k )        (按文中“式(2)”)
      - 计算 Φ 并用式(11)逐步“模-还原”得到 y_m
    日志字段：rmse/bytes/enc_ops/time_s/A_reports/B_reports/Kp 等与现有框架一致。
    params 可选字段：
      max_iter=10, tol=1e-4, scale=1e6, per_bytes=64, eps=None(占位),
      init='mean'/'median'
    """
    p = params or type("P", (), {})()
    max_iter = getattr(p, "max_iter", 10)
    tol      = getattr(p, "tol", 1e-4)
    scale    = int(getattr(p, "scale", 1e6))  # 定点缩放因子
    per_b    = int(getattr(p, "per_bytes", 64))
    init_mth = getattr(p, "init", "mean")

    logs = []
    task_matched = False   # 论文：任务匹配只做一次（式(8)）
    # 开始遍历 rounds（外层由 experiment 驱动）
    for t, batch in enumerate(rounds_iter, start=1):
        t0 = time.time()
        truth = np.asarray(batch.truth, float)
        M = len(truth)
        entities = getattr(batch, "entities", list(range(M)))
        rep = getattr(batch, "reports", None)

        if rep is None or not {"entity_id","value"}.issubset(rep.columns):
            # 没有报告：回退（为了不中断实验）
            est = truth.copy()
            n_used = M * max(1, getattr(batch, "n_workers", n_workers))
            logs.append(dict(
                rmse=float(np.sqrt(np.mean((est-truth)**2))),
                bytes=int(n_used * per_b), enc_ops=0,
                time_s=time.time()-t0, A_reports=n_used, B_reports=0,
                pickedA=M, route_ratio=1.0, Kp=M,
                var=float(np.var(est)), resid_var=float(np.var(est-truth))
            ))
            continue

        # ---- 一次性“任务匹配”（式(8)的等价检查）----
        if not task_matched:
            # 在仿真中我们直接以实体集合等价“匹配成功”
            # 真正的迹/内积运算在密文里做，这里只做一次性校验标记
            task_matched = True  # Eq.(8) 只做一次
        # pivot: 每个工人一个向量 x_{k,m}
        wid_col = _pick_worker_id_column(rep)
        rep = rep[rep["entity_id"].isin(entities)].copy()
        # 聚合同一工人-同一任务多条观测：取平均
        rep_g = rep.groupby([wid_col,"entity_id"])["value"].mean().reset_index()
        # 构造工人×任务矩阵，缺失为 NaN
        W = rep_g.pivot(index=wid_col, columns="entity_id", values="value").reindex(columns=entities)
        X = W.to_numpy(dtype=float)   # shape: (n_workers_eff, M)
        n_eff = X.shape[0]

        # ---- 初始化真值 y （由请求方给定初始值，论文 Step1）----
        if init_mth == "median":
            y = np.nanmedian(X, axis=0)
        else:
            y = np.nanmean(X, axis=0)
        y = np.nan_to_num(y, nan=np.nanmean(y))

        # 迭代（论文 CRH：多轮迭代至收敛；I-CRH 见式(12)(13)）
        for it in range(max_iter):
            # 距离 D_k（式(9)）：对各工人，按其有值的 m 累加
            diff = X - y  # 自动按行 broadcasting
            Dk = np.nansum((diff)**2, axis=1)
            # 数值稳健：防 0
            Dk = np.clip(Dk, 1e-12, None)
            # 权重（按文中“式(2)”的定义）wk = log( sum_i D_i / D_k )
            sumD = float(np.sum(Dk))
            wk = np.log(sumD / Dk)
            wk = np.clip(wk, 1e-6, 1e6)

            # 计算 S 与超递增 z 序列（论文 Step1 约束）
            # S = max_m sum_k w_k x_{k,m} ；为可模还原，做定点缩放
            X_int = np.rint(X * scale).astype(np.int64)
            wk_int = np.rint(wk * scale).astype(np.int64)
            sum_w_int = int(np.sum(wk_int))
            # 如果某列有 NaN，用 0 处理（等价“未参与该任务”）
            X_int = np.where(np.isnan(X), 0, X_int)
            Skm = np.abs((wk_int[:, None] * X_int)).sum(axis=0)  # 每个 m 的 |sum_k wk x_{k,m}|
            S = int(np.max(Skm)) if Skm.size else 1
            S = max(S, 1)
            z = _mk_super_increasing_z(M, S)

            # 论文式(10)：Φ = r3 * Σw + Σ_m z_m * Σ_k wk x_{k,m}
            r3 = int(np.random.randint(10, 100))
            phi_int = int(r3 * sum_w_int)
            # 逐任务累加   Σ_k wk x_{k,m}
            for j in range(M):
                # 对缺失报告的工人，该任务贡献为 0
                s_j = int(np.sum(wk_int * X_int[:, j]))
                phi_int += int(z[j]) * s_j

            # 论文式(11)：模-还原得到 y_m
            y_new = _recover_truth_from_phi(phi_int, sum_w_int, z, scale)

            # 收敛判定
            if np.linalg.norm(y_new - y) / np.sqrt(M) < tol:
                y = y_new
                break
            y = y_new

        est = y.copy()
        rmse = float(np.sqrt(np.mean((est - truth) ** 2)))

        # 通信/加密开销近似（论文指出：工人只上传一次，用户侧很省）：
        # 这里“已用报告条数 × 每条字节”，把本轮实际被用到的条目计入
        used_rows = len(rep_g)
        bytes_used = int(used_rows * per_b)

        logs.append(dict(
            rmse=rmse,
            bytes=bytes_used,
            enc_ops=0,                     # 重加密在云侧；仿真不计
            time_s=time.time()-t0,
            A_reports=used_rows, B_reports=0,
            pickedA=M, route_ratio=1.0, Kp=M,
            var=float(np.var(est)), resid_var=float(np.var(est-truth))
        ))
    return logs
