"""
wb1_t2_xn11abbur00 (LwSrc = T, WWTPs enter) vs wb1_t0_xn11abbur00 (LwSrc = F),
against the King County + Ecology obs, in the style of the t0/t1 obsmod
comparisons. Every model-obs number uses only the casts BOTH runs have, so a
t2 run still in progress is compared to t0 over the same casts.

Inputs
  1. combined obs+mod pickles for both runs (LO_output/obsmod/combined_<otype>_<year>_<gtx>.p),
     made by LO_user/obsmod/multimodel_compare.py (cast extraction + combine_obs_mod)
  2. the newest reduce_<gtx>_*.nc for each run from 20261023_t2_vs_t0_reduce.py
     (optional; the cast figures are made without it)

Figures -> LO_output/DM_outs/20261023_t2_vs_t0/
  cast-based (per otype):
    stats_<otype>.png (+ stats.csv)   bias / RMSE per variable, shallow vs deep
    scatter_<otype>.png               obs vs t0 | obs vs t2 | (t2 - t0) vs z
    DO_bias_map_<otype>.png           station-mean DO bias t0, t2, and t2 - t0
    bottom_DO_ts_<otype>_p<k>.png     deepest-sample DO per cast: obs, t0, t2
  reduce-based:
    station_ts_<station>.png          daily surface / bottom, all variables, + t2 - t0
    diff_map_<dom>.png                period-mean t2 - t0, surface and bottom
    hyp_days_<dom>.png                hypoxic days t0, t2, t2 - t0
    bottom_DO_diff_monthly_<dom>.png  monthly mean bottom DO t2 - t0
    pc_bottom_DO_ts.png               Penn Cove mean bottom DO, t0 vs t2
  <dom> = full (whole wb1, clipped to wb.p) and pc (Penn Cove window).

Run on apogee (or on the mac after copying back the pickles + reduce files):
    python 20261023_t2_vs_t0_plot.py -gtxs wb1_t0_xn11abbur00,wb1_t2_xn11abbur00
"""

import sys
import argparse
import warnings
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from matplotlib.path import Path as MplPath

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtxs', type=str, default='wb1_t0_xn11abbur00,wb1_t2_xn11abbur00')
parser.add_argument('-years', type=str, default='2024,2025')
parser.add_argument('-otypes', type=str, default='ctd,bottle')
parser.add_argument('-H', '--dividing_depth', type=float, default=10.0)  # shallow/deep split (m)
args = parser.parse_args()

Ldir = Lfun.Lstart()
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', RuntimeWarning)

gtxs = [g.strip() for g in args.gtxs.split(',') if g.strip()]
if len(gtxs) != 2:
    print('*** -gtxs needs exactly two runs: base,new'); sys.exit()
g0, g1 = gtxs
lab = {g: g.split('_')[1] for g in gtxs}       # 't0', 't2'
L0, L1 = lab[g0], lab[g1]
years = [y.strip() for y in args.years.split(',') if y.strip()]
otypes = [o.strip() for o in args.otypes.split(',') if o.strip()]
H = args.dividing_depth

out_dir = Ldir['LOo'] / 'DM_outs' / '20261023_t2_vs_t0'
Lfun.make_dir(out_dir)

DO_UM_TO_MGL = 32.0 / 1000.0
OBS_C = 'k'
COL = {g0: 'tab:red', g1: 'tab:purple'}
SURF_GATE = -5.0      # obs surface sample must be shallower than this (m)

# display variable -> how to build it from a combined-pickle DataFrame
VARS = ['CT', 'SA', 'DO (mg L-1)', 'NO3 (uM)', 'NH4 (uM)', 'DIN (uM)',
        'Chl (mg m-3)', 'TA (uM)', 'DIC (uM)']
LIMS = {
    'SA': (14, 34), 'CT': (4, 20), 'DO (mg L-1)': (0, 15),
    'NO3 (uM)': (0, 45), 'NH4 (uM)': (0, 10), 'DIN (uM)': (0, 50),
    'Chl (mg m-3)': (0, 30), 'TA (uM)': (1500, 2400), 'DIC (uM)': (1500, 2400),
}


def savefig(fig, name):
    fn = out_dir / (name + '.png')
    fig.savefig(fn, dpi=200, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print('Saved %s' % fn)


def df_var(df, vn):
    if vn == 'DO (mg L-1)':
        return df['DO (uM)'] * DO_UM_TO_MGL
    if vn == 'DIN (uM)':
        return df['NO3 (uM)'] + df['NH4 (uM)']
    return df[vn] if vn in df else pd.Series(np.nan, index=df.index)


def stats(o, m):
    good = np.isfinite(o) & np.isfinite(m)
    o, m = o[good], m[good]
    n = len(o)
    if n == 0:
        return dict(n=0, bias=np.nan, rmse=np.nan, r=np.nan)
    d = m - o
    r = np.corrcoef(o, m)[0, 1] if n > 2 and np.std(o) > 0 and np.std(m) > 0 else np.nan
    return dict(n=n, bias=d.mean(), rmse=np.sqrt((d**2).mean()), r=r)


# =============================================================================
# 1. Cast-based comparison on the casts both runs have
# =============================================================================
def load_common(otype):
    """Rows (obs samples) present in BOTH runs' combined pickles."""
    frames = []
    for year in years:
        dd = {}
        for g in gtxs:
            fn = Ldir['LOo'] / 'obsmod' / ('combined_%s_%s_%s.p' % (otype, year, g))
            if fn.is_file():
                dd[g] = pickle.load(open(fn, 'rb'))
        if len(dd) < 2:
            print('  %s %s: missing pickle for %s' % (otype, year,
                  ', '.join(g for g in gtxs if g not in dd)))
            continue
        # combine_obs_mod keeps the original row labels when it drops casts with
        # no model, so rows in both runs share their index
        idx = dd[g0]['obs'].index.intersection(dd[g1]['obs'].index)
        if len(idx) == 0:
            continue
        o0 = dd[g0]['obs'].loc[idx]; o1 = dd[g1]['obs'].loc[idx]
        assert (o0['cid'].values == o1['cid'].values).all() and np.allclose(o0['z'], o1['z'])
        out = o0[['cid', 'time', 'lon', 'lat', 'name', 'z', 'source']].copy()
        out['cast'] = otype + '_' + year + '_' + out['cid'].astype(str)
        for vn in VARS:
            out['obs|' + vn] = df_var(dd[g0]['obs'].loc[idx], vn).astype(float)
            for g in gtxs:
                out[g + '|' + vn] = df_var(dd[g][g].loc[idx], vn).astype(float)
        frames.append(out)
        print('  %s %s: %d common casts (%s had %d, %s had %d)' % (
            otype, year, out['cid'].nunique(), L0, dd[g0]['obs']['cid'].nunique(),
            L1, dd[g1]['obs']['cid'].nunique()))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    df['name'] = df['name'].astype(str)
    return df


def zclass(df):
    return {'all': np.ones(len(df), bool),
            'shallow': (df['z'] > -H).values,
            'deep': (df['z'] <= -H).values}


def stats_table(df, otype):
    rows = []
    for vn in VARS:
        o = df['obs|' + vn].values
        for zc, m in zclass(df).items():
            # same samples for both runs: obs and both models finite
            both = m & np.isfinite(o) & np.isfinite(df[g0 + '|' + vn].values) \
                & np.isfinite(df[g1 + '|' + vn].values)
            for g in gtxs:
                s = stats(o[both], df[g + '|' + vn].values[both])
                rows.append(dict(otype=otype, var=vn, zclass=zc, run=lab[g], **s))
    return pd.DataFrame(rows)


def plot_stats(st, otype):
    vns = [vn for vn in VARS if (st[(st['var'] == vn)]['n'] > 0).any()]
    if not vns:
        return
    zcs = ['shallow', 'deep']
    pfun.start_plot(figsize=(2.3 * len(vns), 6), fs=9)
    fig, axes = plt.subplots(2, len(vns), figsize=(2.3 * len(vns), 6), squeeze=False)
    w = 0.38
    for j, vn in enumerate(vns):
        for i, met in enumerate(['bias', 'rmse']):
            ax = axes[i, j]
            for k, g in enumerate(gtxs):
                vals = [st[(st['var'] == vn) & (st['zclass'] == zc) & (st['run'] == lab[g])][met].iloc[0]
                        for zc in zcs]
                ax.bar(np.arange(2) + (k - 0.5) * w, vals, w, color=COL[g], label=lab[g])
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xticks([0, 1]); ax.set_xticklabels(zcs)
            ax.grid(True, axis='y', alpha=0.3)
            if i == 0:
                n = st[(st['var'] == vn) & (st['zclass'] == 'all') & (st['run'] == L0)]['n'].iloc[0]
                ax.set_title('%s\n(n=%d)' % (vn, n), fontsize=9)
            if j == 0:
                ax.set_ylabel('bias (mod - obs)' if met == 'bias' else 'RMSE')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('%s: %s vs %s on common casts (shallow: z > -%g m)' % (otype, L0, L1, H),
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, 'stats_%s' % otype)


def plot_scatter(df, otype):
    vns = [vn for vn in VARS if np.isfinite(df['obs|' + vn]).any()
           and np.isfinite(df[g0 + '|' + vn]).any()]
    zc = zclass(df)
    pfun.start_plot(figsize=(13, 3.3 * len(vns)), fs=9)
    fig, axes = plt.subplots(len(vns), 3, figsize=(13, 3.3 * len(vns)), squeeze=False)
    for i, vn in enumerate(vns):
        o = df['obs|' + vn].values
        lo, hi = LIMS.get(vn, (np.nanmin(o), np.nanmax(o)))
        for j, g in enumerate(gtxs):
            ax = axes[i, j]
            m = df[g + '|' + vn].values
            for zname, c in [('shallow', 'tab:orange'), ('deep', 'tab:blue')]:
                k = zc[zname]
                ax.plot(o[k], m[k], '.', ms=2, alpha=0.4, color=c,
                        label='z > -%g' % H if zname == 'shallow' else 'z <= -%g' % H)
            ax.plot([lo, hi], [lo, hi], 'k-', lw=0.8)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            s = stats(o, m)
            ax.text(.03, .97, '%s  n=%d\nbias=%.2f  rmse=%.2f  r=%.2f'
                    % (lab[g], s['n'], s['bias'], s['rmse'], s['r']),
                    transform=ax.transAxes, va='top', fontsize=8)
            ax.set_xlabel('obs ' + vn); ax.set_ylabel(lab[g] + ' ' + vn)
        ax = axes[i, 2]
        d = df[g1 + '|' + vn].values - df[g0 + '|' + vn].values
        ax.plot(d, df['z'].values, '.', ms=2, alpha=0.4, color='0.3')
        ax.axvline(0, color='k', lw=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('%s - %s  %s' % (L1, L0, vn)); ax.set_ylabel('z (m)')
        ax.text(.03, .03, 'mean %.3g\nmax |d| %.3g' % (np.nanmean(d), np.nanmax(np.abs(d))),
                transform=ax.transAxes, fontsize=8)
    axes[0, 0].legend(loc='lower right', fontsize=7, markerscale=4)
    fig.suptitle('%s obs vs model on common casts' % otype, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(fig, 'scatter_%s' % otype)


def plot_DO_bias_map(df, otype):
    vn = 'DO (mg L-1)'
    w = df[['name', 'lon', 'lat', 'z']].copy()
    w['b0'] = df[g0 + '|' + vn] - df['obs|' + vn]
    w['b1'] = df[g1 + '|' + vn] - df['obs|' + vn]
    w['d'] = df[g1 + '|' + vn] - df[g0 + '|' + vn]
    w = w.dropna(subset=['b0', 'b1'])
    if len(w) == 0:
        return
    pad_lon = 0.1 * max(w['lon'].max() - w['lon'].min(), 0.05)
    pad_lat = 0.1 * max(w['lat'].max() - w['lat'].min(), 0.05)
    ext = [w['lon'].min() - pad_lon, w['lon'].max() + pad_lon,
           w['lat'].min() - pad_lat, w['lat'].max() + pad_lat]
    pfun.start_plot(figsize=(15, 11), fs=10)
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    for i, (zname, k) in enumerate([('deep (z <= -%g m)' % H, w['z'] <= -H),
                                    ('shallow (z > -%g m)' % H, w['z'] > -H)]):
        s = w[k].groupby('name').agg(lon=('lon', 'mean'), lat=('lat', 'mean'),
                                    b0=('b0', 'mean'), b1=('b1', 'mean'),
                                    d=('d', 'mean'), n=('d', 'count'))
        vb = max(np.nanpercentile(np.abs(np.r_[s['b0'], s['b1']]), 95), 0.1) if len(s) else 1
        vd = max(np.nanmax(np.abs(s['d'])), 0.01) if len(s) else 1
        for j, (col, title, vm) in enumerate([('b0', '%s bias' % L0, vb),
                                              ('b1', '%s bias' % L1, vb),
                                              ('d', '%s - %s' % (L1, L0), vd)]):
            ax = axes[i, j]
            if len(s):
                sc = ax.scatter(s['lon'], s['lat'], c=s[col], cmap='RdBu_r', vmin=-vm, vmax=vm,
                                s=60, edgecolors='k', linewidths=0.4, zorder=5)
                plt.colorbar(sc, ax=ax, shrink=0.7, label='DO (mg/L)')
            pfun.add_coast(ax); ax.axis(ext); pfun.dar(ax)
            ax.set_title('%s, %s (%d stations)' % (title, zname, len(s)), fontsize=10)
    fig.suptitle('%s station-mean DO: bias (model - obs) and run difference, common casts'
                 % otype, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'DO_bias_map_%s' % otype)


def plot_bottom_DO_ts(df, otype):
    vn = 'DO (mg L-1)'
    w = df[np.isfinite(df['obs|' + vn])]
    if len(w) == 0:
        return
    bot = w.loc[w.groupby('cast')['z'].idxmin()].copy()
    bot['b0'] = bot[g0 + '|' + vn] - bot['obs|' + vn]
    st = bot.groupby('name').agg(n=('cast', 'count'), z=('z', 'mean'), b0=('b0', 'mean'))
    st = st[st['n'] >= 2].sort_values('b0', ascending=False)
    stations = st.index.tolist()
    nper = 6
    for p in range(int(np.ceil(len(stations) / nper))):
        page = stations[p * nper:(p + 1) * nper]
        pfun.start_plot(figsize=(14, 2.8 * len(page)), fs=10)
        fig, axes = plt.subplots(len(page), 1, figsize=(14, 2.8 * len(page)),
                                 sharex=True, squeeze=False)
        for ax, stn in zip(axes[:, 0], page):
            s = bot[bot['name'] == stn].sort_values('time')
            ax.plot(s['time'], s['obs|' + vn], 'o-', color=OBS_C, ms=4, label='obs')
            ttl = []
            for g, mk in zip(gtxs, ['s', '^']):
                ax.plot(s['time'], s[g + '|' + vn], mk + '--', color=COL[g], ms=5,
                        mfc='none', label=lab[g])
                ttl.append('bias %s=%.2f' % (lab[g], np.nanmean(s[g + '|' + vn] - s['obs|' + vn])))
            ax.axhline(2.0, color='gray', ls=':', lw=1)
            ax.set_ylabel('DO (mg/L)'); ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8, ncol=3)
            ax.set_title('%s  (n=%d, z=%.0f m | %s)' % (stn, len(s), s['z'].mean(), '  '.join(ttl)),
                         fontsize=10)
        fig.suptitle('%s deepest-sample DO per cast, common casts' % otype, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        savefig(fig, 'bottom_DO_ts_%s_p%d' % (otype, p + 1))


print(' cast-based comparison '.center(60, '='))
common = {}
all_stats = []
for otype in otypes:
    df = load_common(otype)
    if df is None:
        print('  no common casts for %s' % otype)
        continue
    common[otype] = df
    st = stats_table(df, otype)
    all_stats.append(st)
    plot_stats(st, otype)
    plot_scatter(df, otype)
    plot_DO_bias_map(df, otype)
    plot_bottom_DO_ts(df, otype)
if all_stats:
    st = pd.concat(all_stats, ignore_index=True)
    st.to_csv(out_dir / 'stats.csv', index=False, float_format='%.4g')
    print('Saved %s' % (out_dir / 'stats.csv'))
    show = st[st['var'].isin(['DO (mg L-1)', 'SA', 'CT', 'NO3 (uM)', 'NH4 (uM)'])
              & (st['zclass'] != 'all')]
    print(show.pivot_table(index=['otype', 'var', 'zclass'], columns='run',
                           values=['bias', 'rmse']).round(3).to_string())


# =============================================================================
# 2. Reduce-based: daily station series and t2 - t0 maps
# =============================================================================
def newest(g):
    fl = sorted((out_dir).glob('reduce_%s_*.nc' % g), key=lambda p: p.stat().st_mtime)
    return fl[-1] if fl else None


rf = {g: newest(g) for g in gtxs}
if any(v is None for v in rf.values()):
    print('No reduce files for %s; skipping map / station-series figures.'
          % ', '.join(g for g in gtxs if rf[g] is None))
    print('Done.'); sys.exit()
R = {g: xr.open_dataset(rf[g]) for g in gtxs}
for g in gtxs:
    print('reduce: %s' % rf[g].name)
assert (R[g0]['time'].values == R[g1]['time'].values).all(), 'reduce files cover different days'
ok = (R[g0]['day_ok'].values == 1) & (R[g1]['day_ok'].values == 1)
t = pd.to_datetime(R[g0]['time'].values)
period = '%s to %s, %d days' % (t[ok][0].date(), t[ok][-1].date(), ok.sum())
lon = R[g0]['lon_rho'].values; lat = R[g0]['lat_rho'].values
mask = R[g0]['mask_rho'].values == 1


# ---- daily station series ---------------------------------------------------
def station_vars(ds, si):
    """{display var: (time, N)} for station index si."""
    z = ds['stn_z_rho'].values[:, si, :]
    la = float(ds['stn_lat'].values[si]); lo = float(ds['stn_lon'].values[si])
    p = gsw.p_from_z(z, la)
    SA = gsw.SA_from_SP(ds['stn_salt'].values[:, si, :], p, lo, la)
    CT = gsw.CT_from_pt(SA, ds['stn_temp'].values[:, si, :])
    v = {'CT': CT, 'SA': SA,
         'DO (mg L-1)': ds['stn_oxygen'].values[:, si, :] * DO_UM_TO_MGL,
         'NO3 (uM)': ds['stn_NO3'].values[:, si, :],
         'NH4 (uM)': ds['stn_NH4'].values[:, si, :]}
    v['DIN (uM)'] = v['NO3 (uM)'] + v['NH4 (uM)']
    if 'stn_chlorophyll' in ds:
        v['Chl (mg m-3)'] = ds['stn_chlorophyll'].values[:, si, :]
    v['TA (uM)'] = ds['stn_alkalinity'].values[:, si, :]
    v['DIC (uM)'] = ds['stn_TIC'].values[:, si, :]
    return v, z


def obs_level(stn, vn, level):
    tt, vv = [], []
    for otype, df in common.items():
        s = df[(df['name'].str.replace(' ', '_') == stn.replace(' ', '_'))
               & np.isfinite(df['obs|' + vn])]
        for _, gcast in s.groupby('cast'):
            gg = gcast[gcast['z'] >= SURF_GATE] if level == 'surface' else gcast[gcast['z'] < SURF_GATE]
            if len(gg) == 0:
                continue
            row = gg.loc[gg['z'].idxmax()] if level == 'surface' else gg.loc[gg['z'].idxmin()]
            tt.append(row['time']); vv.append(row['obs|' + vn])
    return np.array(tt), np.array(vv)


print(' station series '.center(60, '='))
stations = [str(s) for s in R[g0]['station'].values]
for si, stn in enumerate(stations):
    V = {}
    for g in gtxs:
        V[g], z = station_vars(R[g], si)
    vns = [vn for vn in VARS if vn in V[g0]]
    pfun.start_plot(figsize=(16, 2.0 * len(vns)), fs=9)
    fig, axes = plt.subplots(len(vns), 3, figsize=(16, 2.0 * len(vns)), sharex=True,
                             squeeze=False)
    for i, vn in enumerate(vns):
        for j, (level, k) in enumerate([('surface', -1), ('bottom', 0)]):
            ax = axes[i, j]
            for g in gtxs:
                y = np.where(ok, V[g][vn][:, k], np.nan)
                ax.plot(t, y, '-', color=COL[g], lw=1.0,
                        label='%s (z=%.0f m)' % (lab[g], np.nanmean(z[:, k])))
            ot, ov = obs_level(stn, vn, level)
            if len(ot):
                ax.plot(ot, ov, 'o', color=OBS_C, ms=3.5, label='obs')
            ax.set_ylim(LIMS.get(vn, (None, None))); ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(level, fontweight='bold')
            if j == 0:
                ax.set_ylabel(vn, fontsize=8)
        ax = axes[i, 2]
        for level, k, c in [('surface', -1, 'tab:orange'), ('bottom', 0, 'tab:blue')]:
            d = np.where(ok, V[g1][vn][:, k] - V[g0][vn][:, k], np.nan)
            ax.plot(t, d, '-', color=c, lw=1.0, label=level)
        ax.axhline(0, color='k', lw=0.6); ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title('%s - %s' % (L1, L0), fontweight='bold')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper left', fontsize=8, ncol=len(h))
    h, l = axes[0, 2].get_legend_handles_labels()
    fig.legend(h, l, loc='upper right', fontsize=8, ncol=len(h))
    fig.suptitle('%s daily means (%s)' % (stn, period), fontweight='bold')
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'station_ts_%s' % stn.replace(' ', '_'))


# ---- map windows: rectangular extent, content clipped to wb.p ---------------
def load_poly(name):
    fn = Ldir['LOo'] / 'section_lines' / (name + '.p')
    if not fn.is_file():
        return None
    p = pd.read_pickle(fn)
    return np.column_stack([p['x'].values, p['y'].values])


pts = np.column_stack([lon.ravel(), lat.ravel()])
wb = load_poly('wb')
in_wb = (MplPath(wb).contains_points(pts).reshape(lon.shape) if wb is not None
         else np.ones_like(mask))
if wb is None:
    print('  (no section_lines/wb.p here: maps are not clipped to wb)')
show = mask & in_wb
pc = load_poly('pc')
in_pc = (MplPath(pc).contains_points(pts).reshape(lon.shape) & mask if pc is not None
         else None)

dlon = np.nanmedian(np.diff(lon[0, :])); dlat = np.nanmedian(np.diff(lat[:, 0]))
PAD = 10
if wb is not None:
    full_aa = [wb[:, 0].min() - PAD*dlon, wb[:, 0].max() + PAD*dlon,
               wb[:, 1].min() - PAD*dlat, wb[:, 1].max() + PAD*dlat]
else:
    full_aa = [lon[mask].min(), lon[mask].max(), lat[mask].min(), lat[mask].max()]
if pc is not None:
    pc_aa = [pc[:, 0].min() - PAD*dlon, pc[:, 0].max() + PAD*dlon,
             pc[:, 1].min() - PAD*dlat, pc[:, 1].max() + PAD*dlat]
else:
    pc_aa = [-122.76, -122.63, 48.20, 48.26]
DOMS = {'full': full_aa, 'pc': pc_aa}


def in_window(aa):
    return show & (lon >= aa[0]) & (lon <= aa[1]) & (lat >= aa[2]) & (lat <= aa[3])


# WWTP (direction-2) sources from the trapsN00 forcing, if it is here
wwtp = None
ffl = sorted((Ldir['LOo'] / 'forcing' / 'wb1').glob('f*/trapsN00/rivers.nc'))
if ffl:
    rv = xr.open_dataset(ffl[0])
    k = rv['river_direction'].values == 2
    xi = rv['river_Xposition'].values[k].astype(int)
    et = rv['river_Eposition'].values[k].astype(int)
    wwtp = (lon[et, xi], lat[et, xi])
    rv.close()


def draw(ax, fld, aa, cmap, vmin, vmax, label):
    f = np.where(show, fld, np.nan)
    cs = ax.pcolormesh(lon, lat, f, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
    plt.colorbar(cs, ax=ax, shrink=0.8, label=label)
    if wwtp is not None:
        ax.plot(wwtp[0], wwtp[1], 'x', color='k', ms=4, mew=0.8)
    pfun.add_coast(ax); ax.axis(aa); pfun.dar(ax)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ax.tick_params(labelsize=7)


def sym(fld, win):
    v = np.nanpercentile(np.abs(fld[win]), 99) if np.isfinite(fld[win]).any() else 1
    return v if v > 0 else 1e-6


def period_mean(ds, vname):
    nd = ds['ndays'].values.astype(float)
    a = ds[vname].values
    return np.nansum(a * nd[:, None, None], axis=0) / nd.sum()


MAPV = [('oxygen', 'DO (mg/L)', DO_UM_TO_MGL), ('NO3', 'NO3 (uM)', 1),
        ('NH4', 'NH4 (uM)', 1), ('chlorophyll', 'Chl (mg m-3)', 1),
        ('salt', 'salt (g/kg)', 1), ('temp', 'temp (C)', 1)]
MAPV = [m for m in MAPV if m[0] + '_bot' in R[g0]]

print(' maps '.center(60, '='))
summary = []
for dom, aa in DOMS.items():
    win = in_window(aa)
    # period-mean t2 - t0, surface and bottom
    pfun.start_plot(figsize=(4.0 * len(MAPV), 9), fs=9)
    fig, axes = plt.subplots(2, len(MAPV), figsize=(4.0 * len(MAPV), 9), squeeze=False)
    for j, (v, vl, fac) in enumerate(MAPV):
        for i, lev in enumerate(['sfc', 'bot']):
            d = (period_mean(R[g1], '%s_%s' % (v, lev)) - period_mean(R[g0], '%s_%s' % (v, lev))) * fac
            vm = sym(d, win)
            draw(axes[i, j], d, aa, 'RdBu_r', -vm, vm, vl)
            axes[i, j].set_title('%s %s' % ('surface' if lev == 'sfc' else 'bottom', vl), fontsize=9)
            summary.append(dict(domain=dom, var=vl, level=lev, mean=np.nanmean(d[win]),
                                min=np.nanmin(d[win]), max=np.nanmax(d[win])))
    fig.suptitle('%s - %s period mean (%s); x = WWTP outfalls' % (L1, L0, period),
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, 'diff_map_%s' % dom)

    # hypoxic days
    hd = {g: np.nansum(R[g]['hyp_days'].values, axis=0).astype(float) for g in gtxs}
    pfun.start_plot(figsize=(15, 6), fs=10)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    hmax = max(np.nanmax(np.where(win, hd[g0], np.nan)), np.nanmax(np.where(win, hd[g1], np.nan)), 1)
    dd = hd[g1] - hd[g0]
    vm = max(np.nanmax(np.abs(np.where(win, dd, np.nan))), 1)
    draw(axes[2], dd, aa, 'RdBu_r', -vm, vm, 'days')
    axes[2].set_title('%s - %s hypoxic days' % (L1, L0), fontsize=10)
    for ax, g in zip(axes[:2], gtxs):
        draw(ax, hd[g], aa, 'magma_r', 0, hmax, 'hypoxic days')
        ax.set_title('%s: max %d days, %d cells ever hypoxic' % (
            lab[g], np.nanmax(np.where(win, hd[g], np.nan)), np.sum(win & (hd[g] > 0))),
            fontsize=10)
    fig.suptitle('Hypoxic days, daily-mean DO < 2 mg/L anywhere in column (%s)' % period,
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, 'hyp_days_%s' % dom)

    # monthly bottom DO difference
    months = [str(m) for m in R[g0]['month'].values]
    dm = (R[g1]['oxygen_bot'].values - R[g0]['oxygen_bot'].values) * DO_UM_TO_MGL
    vm = sym(dm[:, win], slice(None))
    nc_ = min(6, len(months)); nr = int(np.ceil(len(months) / nc_))
    pfun.start_plot(figsize=(3.6 * nc_, 3.6 * nr), fs=8)
    fig, axes = plt.subplots(nr, nc_, figsize=(3.6 * nc_, 3.6 * nr), squeeze=False)
    for k, m in enumerate(months):
        ax = axes.ravel()[k]
        draw(ax, dm[k], aa, 'RdBu_r', -vm, vm, 'mg/L')
        ax.set_title('%s (n=%d)' % (m, R[g0]['ndays'].values[k]), fontsize=9)
    for ax in axes.ravel()[len(months):]:
        ax.axis('off')
    fig.suptitle('Monthly mean bottom DO, %s - %s' % (L1, L0), fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, 'bottom_DO_diff_monthly_%s' % dom)

sm = pd.DataFrame(summary)
sm.to_csv(out_dir / 'diff_map_summary.csv', index=False, float_format='%.4g')
print(sm[sm['level'] == 'bot'].round(4).to_string(index=False))

# ---- Penn Cove mean bottom DO ------------------------------------------------
if in_pc is not None:
    pfun.start_plot(figsize=(14, 7), fs=10)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ob = {}
    for g in gtxs:
        a = R[g]['oxygen_bot_daily'].values[:, in_pc] * DO_UM_TO_MGL
        ob[g] = np.where(ok, np.nanmean(a, axis=1), np.nan)
        mn = np.where(ok, np.nanmin(a, axis=1), np.nan)
        axes[0].plot(t, ob[g], '-', color=COL[g], lw=1.3, label='%s cove mean' % lab[g])
        axes[0].plot(t, mn, ':', color=COL[g], lw=1.0, label='%s cove min' % lab[g])
    axes[0].axhline(2, color='gray', ls=':', lw=1)
    axes[0].set_ylabel('bottom DO (mg/L)'); axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=4)
    axes[1].plot(t, ob[g1] - ob[g0], '-', color='k', lw=1.2)
    axes[1].axhline(0, color='k', lw=0.6)
    axes[1].set_ylabel('%s - %s cove mean (mg/L)' % (L1, L0)); axes[1].grid(True, alpha=0.3)
    fig.suptitle('Penn Cove (pc.p, %d cells) daily-mean bottom DO (%s)' % (in_pc.sum(), period),
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, 'pc_bottom_DO_ts')
    print('Penn Cove bottom DO %s - %s: mean %.3f, min %.3f, max %.3f mg/L' % (
        L1, L0, np.nanmean(ob[g1] - ob[g0]), np.nanmin(ob[g1] - ob[g0]),
        np.nanmax(ob[g1] - ob[g0])))

print('Done. Figures in %s' % out_dir)
