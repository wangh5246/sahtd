# -*- coding: utf-8 -*-
"""
suite_epsilon_sweep.py
======================
按差分隐私 ε 扫描（适用于 pure_ldp_strict / ud_ldp_strict / dplp_strict 等）。
"""
import argparse, random, numpy as np, pandas as pd
from pathlib import Path
from dataset_use.NYC.src.algorithms import baselines as RBL


def iter_rounds_timebin(reports_csv: str, truth_csv: str, bin_str: str = "5min"):
    rep = pd.read_csv(reports_csv, parse_dates=["timestamp"])
    tru = pd.read_csv(truth_csv, parse_dates=["timestamp"])
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports_csv', default='reports.csv')
    ap.add_argument('--truth_csv',   default='truth.csv')
    ap.add_argument('--outdir',      default='results_suite_epsilon')
    ap.add_argument('--n_workers',   type=int, default=300)
    ap.add_argument('--seed',        type=int, default=2025)
    ap.add_argument('--time_bin',    default='5min')
    ap.add_argument('--methods',     default='pure_ldp_strict,ud_ldp_strict,dplp_strict')
    ap.add_argument('--eps_grid',    default='0.2,0.5,综合工具,2,4')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    grid = [float(x) for x in args.eps_grid.split(',') if x.strip()]
    mlist = [m.strip() for m in args.methods.split(',') if m.strip()]

    merged_rounds=[]; merged_results=[]
    for eps in grid:
        for name in mlist:
            print(f'== {name} @ epsilon={eps} ==')
            rounds_iter = iter_rounds_timebin(args.reports_csv, args.truth_csv, bin_str=args.time_bin)
            if name == 'pure_ldp_strict':
                params = RBL.LDPParams(epsilon=eps)
                df = pd.DataFrame(RBL.pure_ldp_strict(_Spy(rounds_iter), args.n_workers, params))
            elif name == 'ud_ldp_strict':
                params = RBL.UDLDPParams(epsilon_total=eps*3.0)  # 例：总预算=3个窗口 * ε
                df = pd.DataFrame(RBL.ud_ldp_strict(_Spy(rounds_iter), args.n_workers, params))
            elif name == 'dplp_strict':
                params = RBL.DPLPParams(epsilon=eps)
                df = pd.DataFrame(RBL.dplp_strict(_Spy(rounds_iter), args.n_workers, params))
            else:
                continue
            df['method']=name; df['sweep_param']='epsilon'; df['sweep_value']=eps
            merged_rounds.append(df)

            def _num(s): return pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf], np.nan)
            rm=_num(df['rmse']); vm=_num(df['var']); rv=_num(df['resid_var'])
            merged_results.append(pd.DataFrame([{
                'method': name, 'sweep_param':'epsilon', 'sweep_value':eps,
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
