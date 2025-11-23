# -*- coding: utf-8 -*-
"""
suite_robustness.py
===================
鲁棒性实验：在原始 reports 上按比例注入重尾/异常点，再评测各方法。
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from dataset_use.NYC.src.algorithms import algorithms_bridge as bridge, baselines as RBL
from dataset_use.NYC.src.algorithms.etbp_td import etbp_td, ETBPParams as ETBPParamsStrict

def iter_rounds_timebin(reports_csv: str, truth_csv: str, bin_str: str = "5min", outlier_rate=0.1, outlier_scale=5.0, seed=2025):
    rng = np.random.default_rng(seed)
    rep = pd.read_csv(reports_csv, parse_dates=["timestamp"])
    tru = pd.read_csv(truth_csv, parse_dates=["timestamp"])
    # 注入异常：对 value 以 outlier_rate 的概率加上重尾噪声（Student-t 近似用拉普拉斯/正态放大）
    if 'value' in rep.columns:
        m = rng.random(len(rep)) < outlier_rate
        noise = rng.standard_normal(m.sum()) * outlier_scale
        rep.loc[m, 'value'] = rep.loc[m, 'value'].to_numpy(dtype=float) + noise
    rep["slot"] = rep["timestamp"].dt.floor(bin_str)
    tru["slot"] = tru["timestamp"].dt.floor(bin_str)
    slots = sorted(tru["slot"].unique().tolist())
    for s in slots:
        tru_s = tru[tru["slot"]==s][["entity_id","truth"]]
        if tru_s.empty: continue
        ents = tru_s["entity_id"].to_numpy().tolist()
        rep_s = rep[(rep["slot"]==s) & (rep["entity_id"].isin(ents))].reset_index(drop=True)
        yield type("Batch",(object,),dict(entities=ents, truth=tru_s["truth"].to_numpy(dtype=float), reports=rep_s, slot=s))()

class _Spy:
    def __init__(self, it): self._it=iter(it); self.batches=[]
    def __iter__(self): return self
    def __next__(self): b=next(self._it); self.batches.append(b); return b

def run_method(rounds_iter, name, n_workers, params=None):
    import pandas as pd, numpy as np
    if name == "NewSAHTD":
        df = pd.DataFrame(bridge.newsahtd_bridge(_Spy(rounds_iter), n_workers, params))
    elif name == "EPTD":
        df = pd.DataFrame(bridge.eptd_bridge(_Spy(rounds_iter), n_workers, params))
    elif name == "etbp_td_strict":
        df = pd.DataFrame(etbp_td(_Spy(rounds_iter), n_workers, params if params is not None else ETBPParamsStrict()))
    else:
        strict_map = {
            'eptd_strict': RBL.eptd_strict,
            'pure_ldp_strict': RBL.pure_ldp_strict,
            'ud_ldp_strict': RBL.ud_ldp_strict,
            'dplp_strict': RBL.dplp_strict,
            'fed_sense_strict': RBL.fed_sense_strict,
            'robust_b_strict': RBL.robust_b_strict,
            'random_baseline_strict': RBL.random_baseline_strict,
        }
        if name in strict_map:
            df = pd.DataFrame(strict_map[name](_Spy(rounds_iter), n_workers, params))
        else:
            df = pd.DataFrame(bridge.generic_bridge(_Spy(rounds_iter), n_workers, func_name=name, params=params))
    for col in ['rmse','var','resid_var','bytes','enc_ops','time_s']:
        if col not in df.columns: df[col]=np.nan
    if 'slot' not in df.columns: df['slot']=np.arange(len(df))
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports_csv', default='reports.csv')
    ap.add_argument('--truth_csv',   default='truth.csv')
    ap.add_argument('--outdir',      default='results_suite_robust')
    ap.add_argument('--n_workers',   type=int, default=300)
    ap.add_argument('--seed',        type=int, default=2025)
    ap.add_argument('--time_bin',    default='5min')
    ap.add_argument('--outlier_rate',type=float, default=0.1)
    ap.add_argument('--outlier_scale',type=float, default=5.0)
    ap.add_argument('--methods',     default='NewSAHTD,etbp_td_strict,eptd_strict,robust_b_strict,pure_ldp_strict,ud_ldp_strict,dplp_strict')
    args = ap.parse_args()

    np.random.seed(args.seed)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    mlist = [m.strip() for m in args.methods.split(',') if m.strip()]
    merged_rounds=[]; merged_results=[]
    for name in mlist:
        print(f'== 运行方法: {name} ==')
        rounds_iter = iter_rounds_timebin(args.reports_csv, args.truth_csv, bin_str=args.time_bin, outlier_rate=args.outlier_rate, outlier_scale=args.outlier_scale, seed=args.seed)
        df = run_method(rounds_iter, name, args.n_workers, params=None)
        df['method']=name; merged_rounds.append(df)

        def _num(s): return pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf], np.nan)
        rm=_num(df['rmse']); vm=_num(df['var']); rv=_num(df['resid_var'])
        merged_results.append(pd.DataFrame([{
            'method': name,
            'rmse_mean': float(rm.mean()), 'rmse_std': float(rm.std(ddof=1)) if len(rm)>1 else float('nan'),
            'var_mean': float(vm.mean()), 'resid_var_mean': float(rv.mean()),
            'bytes_mean': float(_num(df['bytes']).mean()), 'enc_ops_mean': float(_num(df['enc_ops']).mean())
        }]))

    mr = pd.concat(merged_rounds, ignore_index=True) if merged_rounds else pd.DataFrame()
    if 'slot' in mr.columns: mr['slot']=mr['slot'].astype(str)
    mr.to_csv(out/'merged_rounds.csv', index=False)
    res = pd.concat(merged_results, ignore_index=True) if merged_results else pd.DataFrame()
    res.to_csv(out/'merged_results.csv', index=False)
    print(f'输出目录: {out}')

if __name__ == '__main__':
    main()
