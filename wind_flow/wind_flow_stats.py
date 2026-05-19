"""
Phase 3: lagged correlation and regression statistics linking wind ->
flow -> bottom DO at a mooring. Computed twice -- (a) on the full record,
(b) on the concatenation of all hypoxia-event lead-up windows -- so the
two can be compared.

Outputs:
    LOo/wind_flow/<gtx>/<mooring>_wind_flow_stats_<year>.csv
    LOo/wind_flow/<gtx>/plots/<mooring>_corr_vs_lag_<year>.png

Usage:
    python wind_flow_stats.py -gtx wb1_r0_xn11b -mooring M_inner -year 2017 \
        -wind_mooring_extra M_entrance
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from lo_tools import Lfun


# Pairs we want to evaluate (x_var, y_var, lag-set-name)
HOURLY_PAIRS = [
    ('tau_along', 'along_surface'),
    ('tau_along', 'along_depthavg'),
    ('tau_along', 'along_bottom'),
    ('tau_along', 'bot_DO_mgL'),
    ('tau_across', 'bot_DO_mgL'),
    ('along_bottom', 'bot_DO_mgL'),
]

HOURLY_LAGS_H = np.arange(0, 73)             # 0..72 h
DAILY_LAGS_D  = np.arange(0, 15)             # 0..14 d


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-gtx', '--gtagex', type=str, default='wb1_r0_xn11b')
    p.add_argument('-mooring', type=str, default='M_inner')
    p.add_argument('-year', type=int, default=2017)
    p.add_argument('-ds0', type=str, default='2017.01.01')
    p.add_argument('-ds1', type=str, default='2017.12.31')
    p.add_argument('-wf_fn', type=str, default=None)
    p.add_argument('-wind_mooring_extra', type=str, default=None,
                   help='Second mooring whose wind/stress to also test '
                        'against the primary mooring DO/flow.')
    p.add_argument('-wf_fn_extra', type=str, default=None)
    p.add_argument('-events_csv', type=str, default=None)
    p.add_argument('-out_dir', type=str, default=None)
    p.add_argument('-suffix', type=str, default='_lp',
                   help='Column suffix to use (default "_lp" for Godin).')
    return p.parse_args()


def load_wind_flow(wf_fn):
    ds = xr.open_dataset(wf_fn)
    df = ds.to_dataframe()
    df.index = pd.to_datetime(df.index)
    return df


def lagged_corr(x, y, lag):
    """
    Pearson r of x leading y by `lag` samples (lag>=0 => y(t+lag) vs x(t)).
    """
    if lag == 0:
        a, b = x, y
    else:
        a = x[:-lag]
        b = y[lag:]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 30:
        return np.nan
    a = a[mask]
    b = b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return np.nan
    return float((a * b).sum() / denom)


def corr_curve(df, x_col, y_col, lags):
    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    return np.array([lagged_corr(x, y, int(L)) for L in lags])


def best_lag(lags, rs):
    """Return (lag, r) at the |r| maximum, ignoring NaNs."""
    if np.all(~np.isfinite(rs)):
        return (np.nan, np.nan)
    idx = int(np.nanargmax(np.abs(rs)))
    return (int(lags[idx]), float(rs[idx]))


def event_mask(index, events_df):
    """Boolean array True for timestamps inside any [lead_start, window_end]."""
    mask = np.zeros(len(index), dtype=bool)
    for _, row in events_df.iterrows():
        m = (index >= pd.Timestamp(row.lead_start)) & \
            (index <= pd.Timestamp(row.window_end))
        mask |= m
    return mask


def ols_regression(df, x_cols, y_col):
    """Simple OLS with intercept. Returns dict of coef + R^2."""
    X = df[x_cols].values.astype(float)
    y = df[y_col].values.astype(float)
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if mask.sum() < 30:
        return {c: np.nan for c in x_cols} | {'intercept': np.nan, 'R2': np.nan,
                                              'n': int(mask.sum())}
    X = X[mask]
    y = y[mask]
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    out = {'intercept': float(beta[0]), 'R2': r2, 'n': int(mask.sum())}
    for c, b in zip(x_cols, beta[1:]):
        out[c] = float(b)
    return out


def main():
    args = parse_args()
    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    suf = args.suffix

    if args.wf_fn is not None:
        wf_fn = Path(args.wf_fn)
    else:
        wf_fn = (Ldir['LOo'] / 'wind_flow' / args.gtagex
                 / f'{args.mooring}_wind_flow_{args.ds0}_{args.ds1}.nc')
    if not wf_fn.exists():
        raise FileNotFoundError(f'Wind-flow NC not found: {wf_fn}')

    if args.events_csv is not None:
        events_csv = Path(args.events_csv)
    else:
        events_csv = (Ldir['LOo'] / 'wind_flow' / args.gtagex
                      / f'hypoxia_events_{args.mooring}_{args.year}.csv')
    if not events_csv.exists():
        raise FileNotFoundError(f'Events CSV not found: {events_csv}')

    df = load_wind_flow(wf_fn)
    events = pd.read_csv(events_csv, parse_dates=[
        'event_start', 'event_end', 'lead_start', 'window_end'])

    # Optional second-mooring wind to also test against primary DO/flow
    wind_sources = [(args.mooring, df)]
    if args.wf_fn_extra is not None or args.wind_mooring_extra is not None:
        if args.wf_fn_extra is not None:
            wf_fn_extra = Path(args.wf_fn_extra)
            mooring_extra = (args.wind_mooring_extra
                             or wf_fn_extra.stem.split('_wind_flow')[0])
        else:
            mooring_extra = args.wind_mooring_extra
            wf_fn_extra = (Ldir['LOo'] / 'wind_flow' / args.gtagex
                           / f'{mooring_extra}_wind_flow_'
                             f'{args.ds0}_{args.ds1}.nc')
        if not wf_fn_extra.exists():
            raise FileNotFoundError(
                f'Extra wind-flow NC not found: {wf_fn_extra}')
        df_extra = load_wind_flow(wf_fn_extra)
        # Build a hybrid df: primary flow/DO/strat + extra wind/tau (with
        # rotation onto the *primary*'s depth-avg axis).
        # Easiest path: just swap in the wind/tau columns from df_extra.
        wind_cols = [c for c in df_extra.columns
                     if c.startswith(('Uwind', 'Vwind', 'wind_', 'tau_'))]
        df_hybrid = df.copy()
        for c in wind_cols:
            if c in df_hybrid.columns:
                df_hybrid[c] = df_extra[c].reindex(df_hybrid.index)
        wind_sources.append((mooring_extra, df_hybrid))
        print(f'Will also test wind from {mooring_extra} ({wf_fn_extra})')

    mask_event = event_mask(df.index, events)

    rows = []
    curves = {}   # for the heatmap

    for wind_source, df_use in wind_sources:
        print(f'--- wind source: {wind_source} ---')
        for x_base, y_base in HOURLY_PAIRS:
            xc = x_base + suf
            yc = y_base + suf
            if xc not in df_use.columns or yc not in df_use.columns:
                print(f'  skipping {x_base} -> {y_base}: column missing')
                continue
            # Full record
            rs_full = corr_curve(df_use, xc, yc, HOURLY_LAGS_H)
            lag_f, r_f = best_lag(HOURLY_LAGS_H, rs_full)
            # Event-window subset
            sub = df_use.loc[mask_event]
            rs_evt = corr_curve(sub, xc, yc, HOURLY_LAGS_H)
            lag_e, r_e = best_lag(HOURLY_LAGS_H, rs_evt)
            rows.append({
                'wind_source': wind_source,
                'x': x_base, 'y': y_base, 'window': 'full_year',
                'best_lag_h': lag_f, 'best_r': r_f,
                'n': int(np.isfinite(df_use[xc]).sum()),
            })
            rows.append({
                'wind_source': wind_source,
                'x': x_base, 'y': y_base, 'window': 'leadup_concat',
                'best_lag_h': lag_e, 'best_r': r_e,
                'n': int(np.isfinite(sub[xc]).sum()),
            })
            key_full = f'[{wind_source}] {x_base}->{y_base} full'
            key_evt = f'[{wind_source}] {x_base}->{y_base} leadup'
            curves[key_full] = rs_full
            curves[key_evt] = rs_evt
            print(f'  {x_base:14s} -> {y_base:14s} '
                  f'full: lag={lag_f}h r={r_f:+.3f}   '
                  f'leadup: lag={lag_e}h r={r_e:+.3f}')

    # --- OLS: bot_DO ~ tau_along_lag + tau_across_lag, lag = best from above ---
    # Use best lag from full-record primary-wind tau_along -> bot_DO
    best = next((r for r in rows
                 if r['x'] == 'tau_along' and r['y'] == 'bot_DO_mgL'
                 and r['window'] == 'full_year'
                 and r['wind_source'] == args.mooring), None)
    reg_rows = []
    if best is not None and np.isfinite(best['best_lag_h']):
        L = int(best['best_lag_h'])
        for wind_source, df_use in wind_sources:
            df2 = df_use.copy()
            df2['tau_along_lag']  = df2['tau_along'  + suf].shift(L)
            df2['tau_across_lag'] = df2['tau_across' + suf].shift(L)
            for label, subset in [('full_year', df2),
                                  ('leadup_concat', df2.loc[mask_event])]:
                res = ols_regression(
                    subset, ['tau_along_lag', 'tau_across_lag'],
                    'bot_DO_mgL' + suf)
                res.update({'wind_source': wind_source, 'window': label,
                            'lag_h': L,
                            'predictors': 'tau_along_lag+tau_across_lag',
                            'target': 'bot_DO_mgL' + suf})
                reg_rows.append(res)
                print(f'  OLS [{wind_source} | {label:14s}] lag={L}h '
                      f'R2={res["R2"]:+.3f} '
                      f'beta_along={res["tau_along_lag"]:+.3g} '
                      f'beta_across={res["tau_across_lag"]:+.3g} '
                      f'(n={res["n"]})')

    # --- Save CSV ---
    out_dir = (Path(args.out_dir) if args.out_dir else
               (Ldir['LOo'] / 'wind_flow' / args.gtagex))
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_csv = out_dir / f'{args.mooring}_wind_flow_stats_{args.year}.csv'
    pd.DataFrame(rows).to_csv(corr_csv, index=False)
    print(f'Wrote {corr_csv}')
    if reg_rows:
        reg_csv = out_dir / f'{args.mooring}_wind_flow_regression_{args.year}.csv'
        pd.DataFrame(reg_rows).to_csv(reg_csv, index=False)
        print(f'Wrote {reg_csv}')

    # --- Heatmap of corr vs lag (hourly) ---
    if curves:
        labels = list(curves.keys())
        mat = np.array([curves[k] for k in labels])
        fig, ax = plt.subplots(figsize=(10, 0.4 * len(labels) + 2))
        x_edges = np.concatenate([HOURLY_LAGS_H - 0.5,
                                  [HOURLY_LAGS_H[-1] + 0.5]])
        im = ax.pcolormesh(x_edges, np.arange(len(labels) + 1),
                           mat, cmap='RdBu_r', vmin=-1, vmax=1,
                           shading='flat')
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('lag [hours], x leading y')
        ax.set_title(f'{args.gtagex} {args.mooring} lagged correlation '
                     f'(Godin lowpassed, {args.year})')
        fig.colorbar(im, ax=ax, label='Pearson r')
        fig.tight_layout()
        plot_path = out_dir / 'plots' / f'{args.mooring}_corr_vs_lag_{args.year}.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {plot_path}')


if __name__ == '__main__':
    main()
