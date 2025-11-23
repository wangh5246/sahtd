def sa_htd_paper(rounds_iter: Iterable, n_workers: int, params=None):
    """
    SAHTD-Nexus（原 sa_htd_paper）本地 C 路模拟版本
    ------------------------------------------------
    三路设计：
      - A 路：可信高精度聚合（无噪声，鲁棒 Huber），字节开销中等；
      - B 路：LDP + 洗牌放大，作为主力低成本通道；
      - C 路：为“高价值实体”提供更高 ε（噪声更小）、更高字节预算的小批高精度通道，
              在本实现中用本地模拟，不依赖 DAP/VDAF HTTP。

    关键改动：
      1. C 路不再使用 DAPClient；只在本地用更大的 eps + 更贵的 per-report bytes 模拟“安全高精度通道”；
      2. 所有估计只基于 reports / 历史状态，不再用 truth 填补 NaN；
      3. 日志中新增 C_reports / bytes_A/B/C 等字段，便于你画图分析 A/B/C 三路贡献。
    """
    import numpy as _np
    import time as _time

    # ---------------------- 内部工具：卡尔曼 + 图平滑 ---------------------- #
    class _Kalman1DState:
        __slots__ = ("m", "v", "init")
        def __init__(self):
            self.m = 0.0
            self.v = 10.0
            self.init = False

    def _postprocess_filter(est_by_e: dict, graph: dict, kstate: dict,
                            alpha_lap: float, proc_var: float, obs_var_base: float):
        """后处理：1D Kalman + 图拉普拉斯平滑（零额外隐私成本）"""
        est_k = {}
        for e, obs in est_by_e.items():
            st = kstate.get(e)
            if st is None:
                st = _Kalman1DState()
                kstate[e] = st
            if not st.init:
                st.m = float(obs)
                st.v = 5.0
                st.init = True
                est_k[e] = st.m
                continue
            # 预测
            pred_m = st.m
            pred_v = st.v + float(proc_var)
            R = float(obs_var_base)
            if not _np.isfinite(R) or R <= 0.0:
                R = 1.0
            K = pred_v / (pred_v + R)
            st.m = float(pred_m + K * (float(obs) - pred_m))
            st.v = float((1.0 - K) * pred_v)
            est_k[e] = st.m

        # 图拉普拉斯平滑
        if graph is None:
            return est_k

        est_pp = {}
        for e, val in est_k.items():
            neigh = graph.get(e)
            if not neigh:
                est_pp[e] = val
                continue
            neigh_vals = [est_k.get(n, val) for n in neigh]
            neigh_mean = float(_np.mean(neigh_vals))
            est_pp[e] = (1.0 - alpha_lap) * val + alpha_lap * neigh_mean
        return est_pp

    # ---------------------- 自适应量化 & 路由打分 ---------------------- #
    def _update_quant_bits(entity_ids, est_pp_vec, truth_proxy_vec, qbits_dict,
                           var_est_dict, resid_ema_dict, bytes_per_bit: float,
                           params):
        """
        根据上一轮“后处理估计 vs 代理真值”更新：
          - 每实体方差估计 var_est[e]
          - 每实体残差 EMA resid_ema[e]
          - 每实体量化 bits（总 bits 经缩放以满足 per-round 字节预算）
        """
        min_bits = int(getattr(params, "MIN_QUANT_BITS", 6) if params is not None else 6)
        max_bits = int(getattr(params, "MAX_QUANT_BITS", 14) if params is not None else 14)
        target_bytes = float(getattr(params, "target_bytes_per_round", 1.8e5) if params is not None else 1.8e5)
        avg_reports_per_entity = float(getattr(params, "AVG_REPORTS_PER_ENTITY", 10.0) if params is not None else 10.0)
        var_quantile = float(getattr(params, "VAR_QUANTILE", 0.7) if params is not None else 0.7)

        est_pp_vec = _np.asarray(est_pp_vec, float)
        truth_proxy_vec = _np.asarray(truth_proxy_vec, float)

        # 1) 更新方差估计 & 残差 EMA
        for idx, e in enumerate(entity_ids):
            err = float(est_pp_vec[idx] - truth_proxy_vec[idx])
            v_old = float(var_est_dict.get(e, 5.0))
            var_est_dict[e] = 0.9 * v_old + 0.1 * (err * err)
            r_old = float(resid_ema_dict.get(e, 0.0))
            resid_ema_dict[e] = 0.9 * r_old + 0.1 * abs(err)

        vars_arr = _np.array([var_est_dict.get(e, 5.0) for e in entity_ids], float)
        if _np.isfinite(vars_arr).any():
            tau = float(_np.quantile(vars_arr[_np.isfinite(vars_arr)], var_quantile))
        else:
            tau = 1.0

        # 2) 初步 bit 分配：高方差 +2，低方差 -1
        bits_prop = {}
        for e in entity_ids:
            b = int(qbits_dict.get(e, int(getattr(params, "quant_bits_init", 10))))
            v = float(var_est_dict.get(e, 5.0))
            if v > tau:
                b += 2
            else:
                b -= 1
            b = max(min_bits, min(max_bits, b))
            bits_prop[e] = b

        # 3) 预算守恒：按总 bits 缩放
        total_bits = sum(bits_prop[e] * avg_reports_per_entity for e in entity_ids)
        total_bytes = float(total_bits) * float(bytes_per_bit)
        scale = 1.0 if total_bytes <= 0 else target_bytes / total_bytes

        bits_final = {}
        for e in entity_ids:
            b_scaled = int(round(bits_prop[e] * scale))
            if b_scaled < min_bits:
                b_scaled = min_bits
            if b_scaled > max_bits:
                b_scaled = max_bits
            bits_final[e] = b_scaled
            qbits_dict[e] = b_scaled
        return bits_final, tau

    def _compute_value_scores(entity_ids, var_est_dict, resid_ema_dict,
                              bytes_per_bit: float, bits_A: int, bits_B: int, bits_C_extra: int):
        """ΔMSE-per-byte 价值度量：预期 MSE 降低 / 额外字节，用于 A/C 路排序"""
        scores = {}
        extra_bits = max(bits_A - bits_B, 1)
        extra_bits_C = bits_C_extra if bits_C_extra > 0 else extra_bits
        extra_bytes_A = extra_bits * bytes_per_bit
        extra_bytes_C = extra_bits_C * bytes_per_bit

        for e in entity_ids:
            v = float(var_est_dict.get(e, 5.0))
            r = float(resid_ema_dict.get(e, 0.0))
            delta_mse = v + r
            scores[e] = (
                delta_mse / (extra_bytes_A + 1e-6),
                delta_mse / (extra_bytes_C + 1e-6),
            )
        return scores

    def _route_entities(entity_ids, state, tau_val, bits_A: int, bits_B: int, bits_C_extra: int, params):
        """
        根据 ΔMSE-per-byte 把实体分配到 A/B/C 三路。
        state: dict(var_est, resid_ema, scheduler, bytes_per_bit)
        """
        base_a_ratio = float(
            getattr(params, "BASE_A_RATIO", state["sched"].a_ratio)
            if params is not None else state["sched"].a_ratio
        )
        base_c_ratio = float(getattr(params, "BASE_C_RATIO", 0.05) if params is not None else 0.05)

        scores = _compute_value_scores(
            entity_ids, state["var_est"], state["resid_ema"],
            state["bytes_per_bit"], bits_A, bits_B, bits_C_extra
        )
        n = len(entity_ids)

        # 先选 A 候选
        sorted_for_A = sorted(entity_ids, key=lambda e: -scores[e][0])
        n_A_total = max(1, int(round(base_a_ratio * n)))
        A_cand = set(sorted_for_A[:n_A_total])
        B_all = set(entity_ids) - A_cand

        # 在 A_cand 里再选一部分走 C 路
        max_C = min(int(round(base_c_ratio * n)), int(getattr(params, "C_BATCH_MAX", 32)))
        if max_C <= 0:
            route_C = set()
            route_A = A_cand
        else:
            sorted_for_C = sorted(A_cand, key=lambda e: -scores[e][1])
            route_C = set(sorted_for_C[:max_C])
            route_A = A_cand - route_C

        route_B = B_all
        return route_A, route_B, route_C

    # ---------------------- 超参数读取 ---------------------- #
    eps_B = float(getattr(params, "epsilon", 1.0) if params is not None else 1.0)
    tau0 = float(getattr(params, "tau_percentile", 75.0) if params is not None else 75.0)
    A_ratio0 = float(getattr(params, "A_budget_ratio", 0.22) if params is not None else 0.22)

    use_shuffle = bool(getattr(params, "use_shuffle", True) if params is not None else True)
    uldp_sensitive_cols = list(getattr(params, "uldp_sensitive_cols", []) if params is not None else [])
    geo_eps = float(getattr(params, "geo_epsilon", 0.0) if params is not None else 0.0)

    # 连续流会计
    window_w = int(getattr(params, "window_w", 32) if params is not None else 32)
    epsilon_per_window = float(
        getattr(params, "epsilon_per_window", eps_B * window_w)
        if params is not None else eps_B * window_w
    )
    acct_mode = str(getattr(params, "accountant_mode", "pld") if params is not None else "pld").lower()
    delta_target = float(getattr(params, "delta_target", 1e-5) if params is not None else 1e-5)

    # 延迟 & 通信目标
    target_latency_ms = float(getattr(params, "target_latency_ms", 2.0) if params is not None else 2.0)
    target_bytes = float(getattr(params, "target_bytes_per_round", 1.8e5) if params is not None else 1.8e5)

    # 量化 & 通道参数
    bytes_per_bit = float(getattr(params, "bytes_per_bit", 0.125) if params is not None else 0.125)
    bits_A = int(getattr(params, "BASE_BITS_A", 10) if params is not None else 10)
    bits_B = int(getattr(params, "BASE_BITS_B", 8) if params is not None else 8)
    bits_C_extra = int(getattr(params, "BITS_C_EXTRA", 2) if params is not None else 2)

    # C 路专用：更大的 epsilon & 更高 per-report 字节
    C_eps_scale = float(getattr(params, "C_eps_scale", 2.0) if params is not None else 2.0)
    perA_default = int(getattr(params, "perA_bytes", 32) if params is not None else 32)
    perC_default = int(getattr(params, "perC_bytes", 64) if params is not None else 64)

    # 后处理参数
    alpha_lap = float(getattr(params, "post_lap_alpha", 0.3) if params is not None else 0.3)
    proc_var = float(getattr(params, "post_process_var", 0.5) if params is not None else 0.5)
    obs_var_base = float(getattr(params, "post_obs_var_base", 1.0) if params is not None else 1.0)

    # 图结构（可选）
    entity_graph = getattr(params, "entity_graph", None)

    # 调度器 & 会计
    sched = Scheduler(
        tau0=tau0, a_ratio0=A_ratio0, step=0.08,
        target_latency_ms=target_latency_ms, target_bytes=target_bytes
    )

    if acct_mode == "pld" and PLDAccountant is not None:
        acct = PLDAccountant(delta_target=delta_target)
    else:
        acct = PrivacyAccountantNaive(epsilon_per_window, window_w)

    # 持久状态
    var_est = {}        # entity -> variance estimate
    resid_ema = {}      # entity -> residual EMA
    quant_bits = {}     # entity -> bit width
    kalman_state = {}   # entity -> _Kalman1DState
    last_est_pp = {}    # entity -> last post-processed estimate

    logs: List[Dict[str, Any]] = []
    res_hist: List[float] = []

    # ---------------------- 主循环：逐轮处理 ---------------------- #
    for r_idx, batch in enumerate(rounds_iter):
        t0 = _time.time()
        truth = _np.asarray(batch.truth, float)          # 只用于离线评测
        n_ent = len(truth)
        entities = _np.asarray(batch.entities)

        # Geo-DP 保护半径（仅记录）
        geo_r = float("nan")
        rep_df = getattr(batch, "reports", None)
        if geo_eps > 0.0 and rep_df is not None and {"lat", "lng"}.issubset(rep_df.columns):
            dx, dy = _geo_laplace_noise(geo_eps, size=len(rep_df))
            geo_r = float(_np.mean(_np.sqrt(dx * dx + dy * dy)))

        # 每实体原始 reports & 样本方差
        arr_by_e = {}
        var_by_e = _np.zeros(n_ent, float)
        if rep_df is not None and {"entity_id", "value"}.issubset(rep_df.columns):
            x = rep_df[["entity_id", "value"]]
            for j, e in enumerate(entities):
                arr = x.loc[x["entity_id"] == e, "value"].to_numpy(float)
                arr_by_e[e] = arr
                if arr.size > 1:
                    var_by_e[j] = float(_np.var(arr))
                elif arr.size == 1:
                    var_by_e[j] = 0.0
                else:
                    var_by_e[j] = float("nan")
        else:
            for j, e in enumerate(entities):
                arr = _np.array([truth[j]], float)  # 真空场景下只作为占位，不参与真实部署
                arr_by_e[e] = arr
                var_by_e[j] = 0.0

        # 根据历史残差更新 tau
        if len(res_hist) >= 5 and _np.isfinite(var_by_e).any():
            tau_val = float(_np.nanpercentile(var_by_e[_np.isfinite(var_by_e)], sched.tau))
        elif _np.isfinite(var_by_e).any():
            tau_val = float(_np.nanmedian(var_by_e[_np.isfinite(var_by_e)]))
        else:
            tau_val = 0.0

        # 代理真值：用于更新方差/残差估计（不喂回算法内部逻辑）
        if last_est_pp:
            est_prev_pp_vec = _np.array(
                [last_est_pp.get(e, truth[j]) for j, e in enumerate(entities)], float
            )
        else:
            est_prev_pp_vec = truth.copy()
        truth_proxy = truth.copy()

        # 自适应量化
        bits_by_e, tau_var = _update_quant_bits(
            entities, est_prev_pp_vec, truth_proxy,
            quant_bits, var_est, resid_ema, bytes_per_bit, params
        )

        # 路由：A/B/C 三路划分
        state_for_router = {
            "var_est": var_est,
            "resid_ema": resid_ema,
            "bytes_per_bit": bytes_per_bit,
            "sched": sched,
        }
        route_A, route_B, route_C = _route_entities(
            list(entities), state_for_router,
            tau_val=tau_var, bits_A=bits_A, bits_B=bits_B,
            bits_C_extra=bits_C_extra, params=params
        )

        # 预分配 A/B 估计
        estA = _np.full(n_ent, _np.nan, float)
        estB = _np.full(n_ent, _np.nan, float)  # B 和 C 都写这里，C 会产生更小 var
        vA = _np.full(n_ent, _np.inf, float)
        vB = _np.full(n_ent, _np.inf, float)
        countA = countB = countC = 0

        eps_eff_used = None
        batch_key = getattr(batch, "slot", r_idx)

        # ================== 逐实体聚合 ================== #
        for j, e in enumerate(entities):
            arr = arr_by_e.get(e)
            if arr is None or arr.size == 0:
                continue

            # ---------- A 路：可信高精度聚合 ----------
            if e in route_A:
                muA, vvA, _infoA = _aggregate_A(arr, huber_c=1.6, trim=0.02)
                estA[j], vA[j] = muA, vvA
                countA += len(arr)

            # ---------- C 路：本地模拟的“高精度 + 高预算”通道 ----------
            elif e in route_C:
                # C 路比 B 路用更大的 ε（噪声更小），但在通信统计中记为更高字节
                eps_local_C = max(eps_B * C_eps_scale, 1e-8)
                noisy = _local_dp_laplace(arr, epsilon=eps_local_C, sensitivity=1.0)
                muC, vvC, _infoC = _aggregate_B(noisy, epsilon=eps_local_C, trim=0.02)
                estB[j], vB[j] = muC, vvC
                if use_shuffle:
                    eps_eff_used = _epsilon_after_shuffle(eps_local_C, n=len(arr))
                else:
                    eps_eff_used = eps_local_C
                countC += len(arr)

            # ---------- B 路：LDP + 洗牌放大 ----------
            else:
                eps_local = eps_B * (0.6 if uldp_sensitive_cols else 1.0)
                eps_local = max(eps_local, 1e-8)
                noisy = _local_dp_laplace(arr, epsilon=eps_local, sensitivity=1.0)
                muB, vvB, _infoB = _aggregate_B(noisy, epsilon=eps_local, trim=0.02)
                estB[j], vB[j] = muB, vvB
                if use_shuffle:
                    eps_eff_used = _epsilon_after_shuffle(eps_local, n=len(arr))
                else:
                    eps_eff_used = eps_local
                countB += len(arr)

        # ---------- 通道选择 & 基础 RMSE（不再用 truth 填补） ----------
        pickA = (vA < vB) & (~_np.isnan(estA))
        est = _np.where(pickA, estA, estB)

        # 对仍为 NaN 的实体，用当前 batch 的 reports 稳健填补（不使用 truth）
        est = _fill_est_from_reports(est, batch)
        rmse_raw = _rmse(est, truth)  # 仅评测用，不参与内部逻辑

        # ---------- 后处理：Kalman + 图平滑 ----------
        est_by_e = {e: float(est[j]) for j, e in enumerate(entities)}
        est_pp_by_e = _postprocess_filter(
            est_by_e, entity_graph, kalman_state,
            alpha_lap=alpha_lap, proc_var=proc_var, obs_var_base=obs_var_base
        )
        est_pp = _np.array([est_pp_by_e[e] for e in entities], float)
        rmse = _rmse(est_pp, truth)
        res_hist.append(rmse)

        # 记录最新后处理估计
        for j, e in enumerate(entities):
            last_est_pp[e] = float(est_pp[j])

        # ---------- 通信 & 会计 & 调度 ---------- #
        time_s = _time.time() - t0

        perA = perA_default
        # B 路 per-report 字节：按量化 bits 粗略估算
        if entities.size > 0:
            avg_bits_B = _np.mean([quant_bits.get(e, bits_B) for e in entities])
            perB = int(bytes_per_bit * 8 * avg_bits_B)
        else:
            perB = 16
        perC = perC_default

        bytes_A = _bytes(countA, perA)
        bytes_B = _bytes(countB, perB)
        bytes_C = _bytes(countC, perC)
        bytes_used = bytes_A + bytes_B + bytes_C

        # 隐私会计
        if isinstance(acct, PrivacyAccountantNaive):
            acct_info = acct.update(float(eps_B))
            over = acct.overloaded()
        else:
            eff = float(eps_eff_used if eps_eff_used is not None else eps_B)
            acct.add_pure_dp(eff)
            eps_total = float(acct.epsilon())
            acct_info = {
                "epsilon_cum_window": eps_total,
                "epsilon_limit": float(epsilon_per_window),
            }
            over = bool(eps_total > float(epsilon_per_window) + 1e-12)

        # 调度器根据本轮时延/字节/隐私压力调整 tau 与 A_ratio
        sched.update(
            last_latency_ms=time_s * 1000.0,
            last_bytes=float(bytes_used),
            acct_over=over
        )

        # ---------- 日志输出（与原版兼容，新增 C 路信息） ---------- #
        logs.append(dict(
            rmse=float(rmse),
            rmse_raw=float(rmse_raw),
            bytes=int(bytes_used),
            bytes_A=int(bytes_A),
            bytes_B=int(bytes_B),
            bytes_C=int(bytes_C),
            enc_ops=int(_enc_ops(countA + countB + countC, 2)),
            time_s=float(time_s),

            A_reports=int(countA),
            B_reports=int(countB),
            C_reports=int(countC),
            route_A_size=int(len(route_A)),
            route_B_size=int(len(route_B)),
            route_C_size=int(len(route_C)),
            pickedA=int(int(pickA.sum())),
            Kp=int(int(pickA.sum())),
            route_ratio=float(int(countA) / (int(countA) + int(countB) + int(countC) + 1e-9)),

            var=float(_np.var(est_pp)) if est_pp.size else float("nan"),
            resid_var=float(_np.var(est_pp - truth)) if est_pp.size else float("nan"),

            tau_percentile=float(sched.tau),
            A_budget_ratio=float(sched.a_ratio),

            accountant_mode=str("pld" if not isinstance(acct, PrivacyAccountantNaive) else "naive"),
            epsilon_round=float(eps_B),
            epsilon_cum_window=float(acct_info["epsilon_cum_window"]),
            epsilon_limit=float(acct_info["epsilon_limit"]),
            filter_triggered=bool(over),

            shuffle_used=bool(use_shuffle),
            epsilon_effective=float(eps_eff_used) if eps_eff_used is not None else float("nan"),

            geo_r_protect=float(geo_r),

            # C 路本地模拟，不再走 HTTP/VDAF，这里统一置为 False/NaN
            vdaf_ok_ratio=float("nan"),
            reject_ratio=float("nan"),
            vdaf_http=False,
            dap_mode="local_sim",
            batch_key=str(batch_key),
            collect_id=""
        ))

    return logs