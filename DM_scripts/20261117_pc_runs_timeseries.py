"""
Stacked daily time series for every run on hand (cas7_t1_x11ab, cas7_t0_x4b, wb1_t0_xn11abbur00), one color per run:
    Penn Cove bottom DO (pc_west) | Saratoga bottom DO (M5) | Penn Cove drho (pc_west) | cove wind stress east
    (taux) | cove wind stress north (tauy) | cove air T | Penn Cove bottom T (pc_west)
Reads only the pickles already made (no model files):
    DM_outs/20261117_pc_saratoga_strat_DO/*.p   pc_west (+ wb1 M5)   from 20261117_pc_saratoga_strat_DO.py and
                                                                     20261117_wb1_pc_from_boxes.py
    DM_outs/20261117_sp_point_strat_DO/*.p      cas7 M5               from the old 20261117_sp_point_strat_DO.py
    DM_outs/20261116_pc_cas7_bottom_DO_cove/*.p cove air T, taux, tauy from 20261116_pc_cas7_bottom_DO_cove.py
DEFINITIONS as those scripts: bottom = s_rho index 0; drho = sigma0(bottom) - sigma0(top), TEOS-10; taux, tauy = daily
    means of hourly Large & Pond stress components (east, north; + = toward), cove mean (wb1: |tau| x the 'pc'
    polygon wind direction, see 20261117_wb1_pc_from_boxes.py); air T = daily mean of hourly Tair over the cove cells.
    cas7_t1_x11ab daily values are UTC-day averages (ocean_avg); x4b and wb1 are Godin (lowpassed / Godin on hourly).
    The two cas7 runs read the same atm00 forcing for 2013-2019, so their air T and stress lie on top of each other.
OBS     black markers, per cast, read from LO_output/obs (the same files the obsmod comparisons use), 2012-2025:
        kc_whidbeyBasin/ctd (2022-2025 locally; KC's Whidbey program, one cast per station-day):
          PENNCOVEWEST (-122.720, 48.225, to ~16 m; ~0.9 km W of pc_west) -> pc_west bottom DO (deepest sample),
            drho (sigma0 deepest - shallowest sample, gsw.sigma0(SA, CT)) and bottom T (CT of the deepest sample)
          SARATOGARP (-122.55, 48.24, to ~91 m; 2.5 km E of M5) -> M5 panel, DO interpolated to 37.5 m (M5 is
            37-39 m deep), not the cast bottom. Dots.
        nceiSalish/bottle_ctd WOAC station 4 (-122.554, 48.242; 2.2 km from M5; 2014-2024, 7-12 bottles a cast)
            -> M5 panel, DO interpolated to 37.5 m. Open squares. (prism holds duplicates of these cruises.)
        pcRaft/moor/raft_main.nc, the Penn Cove mussel raft sondes (-122.7057, 48.2209; 0.7 km from pc_west),
            hourly 2014-2019 at 1 m and 7 m -> daily means (days with >= 12 h), small grey dots: DO and CT at 7 m
            in the pc_west bottom DO / bottom T panels and sigma0(7 m) - sigma0(1 m) in the drho panel. 7 m is NOT
            the 17 m bottom, so these sit above the model bottom values by construction.
        No Ecology core station is in the cove or near M5 (nearest: SKG003 9 km NE, 25 m; SAR003 17 km S).
        No local wind or air-temperature observations, so the stress and air T panels have none.
        With -anom, obs are shown only for -clim x11ab, as obs minus the cas7_t1_x11ab climatology; never rolled.
OPTIONAL -x0 / -x1 (YYYY.MM.DD) to zoom the time axis, e.g. -x0 2024.01.01 -x1 2025.12.31
         -anom True  plot seasonal anomalies: value minus a day-of-year climatology (mean of each calendar day,
                     Feb 29 its own bin, then a 31-day running mean wrapped around the year end).
         -clim own   (default) each run's own climatology. wb1 has only 2024-25, so its climatology is the mean of
                     two years and its two anomaly years are near mirror images.
         -clim x11ab every run minus the cas7_t1_x11ab climatology (2012-2025), so run-to-run offsets show too.
         -roll N     N-day centered rolling mean of what is plotted (after the anomaly, if -anom), needing at least
                     N/2 days in the window, e.g. -roll 30
OUT     LO_output/DM_outs/20261117_pc_runs_timeseries/pc_runs_timeseries[_anom_<clim>][_roll<N>][_<x0>_<x1>].png
    python 20261117_pc_runs_timeseries.py
"""

import argparse
import glob
import pickle
import gsw
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-x0', type=str, default=None)
parser.add_argument('-x1', type=str, default=None)
parser.add_argument('-anom', type=zfun.boolean_string, default=False)
parser.add_argument('-clim', type=str, default='own')     # own or x11ab
parser.add_argument('-roll', type=int, default=0)         # N-day centered rolling mean, 0 = none
args = parser.parse_args()

Ldir = Lfun.Lstart()
outs = Ldir['LOo'] / 'DM_outs'
out_dir = outs / '20261117_pc_runs_timeseries'
Lfun.make_dir(out_dir)

RUN_COLORS = {'cas7_t1_x11ab': 'tab:blue', 'cas7_t0_x4b': 'tab:red', 'wb1_t0_xn11abbur00': 'tab:green'}
RUN_LABELS = {'cas7_t1_x11ab': 'cas7_t1_x11ab (50% burial)', 'cas7_t0_x4b': 'cas7_t0_x4b (0% burial)',
              'wb1_t0_xn11abbur00': 'wb1_t0_xn11abbur00 (0% burial)'}
HYP = 2.0


def load_point(fn):
    """New multi-station pickle, or a one-station pickle from the old 20261117_sp_point_strat_DO.py."""
    P = pickle.load(open(fn, 'rb'))
    if 'st' not in P:
        keys = ['salt', 'temp', 'oxygen', 'sigma0', 'h']
        P = {'gtagex': P['gtagex'], 'time': P['time'], 'st': {P['sn']: {k: P[k] for k in keys}}}
    return P


# ---- gather one daily Series per (run, variable) ----
S = {g: {} for g in RUN_COLORS}
pt_files = sorted((outs / '20261117_pc_saratoga_strat_DO').glob('*.p')) + \
           sorted((outs / '20261117_sp_point_strat_DO').glob('*.p'))
for fn in pt_files:
    P = load_point(fn)
    g, t = P['gtagex'], pd.to_datetime(P['time'])
    if g not in S:
        continue
    for sn, s in P['st'].items():
        S[g][f'DO_{sn}'] = pd.Series(s['oxygen'][:, 0], index=t)
        S[g][f'drho_{sn}'] = pd.Series(s['sigma0'][:, 0] - s['sigma0'][:, -1], index=t)
        S[g][f'Tbot_{sn}'] = pd.Series(s['temp'][:, 0], index=t)
        S[g][f'h_{sn}'] = s['h']
for fn in sorted((outs / '20261116_pc_cas7_bottom_DO_cove').glob('*_pc_cove_*.p')):
    P = pickle.load(open(fn, 'rb'))
    g = P['gtagex']
    if g in S and 'air' in P:
        S[g]['Tair'] = P['air']['Tair'].set_axis(pd.to_datetime(P['time']))
        for c in ['taux', 'tauy']:
            S[g][c] = P['air'][c].set_axis(pd.to_datetime(P['time']))

def climatology(x):
    """366-value day-of-year climatology (leap-year calendar), 31-day running mean wrapped around the year."""
    doy = pd.to_datetime({'year': 2000, 'month': x.index.month, 'day': x.index.day}).dt.dayofyear.values
    c = pd.Series(x.values).groupby(doy).mean().reindex(range(1, 367))
    c3 = pd.concat([c, c, c]).reset_index(drop=True)
    return c3.rolling(31, center=True, min_periods=10).mean().iloc[366:732].set_axis(range(1, 367))


def load_obs(years=range(2012, 2026)):
    """Per-cast obs, read straight from LO_output/obs (time: tz-aware -> utc=True, then naive).
    Returns {panel key: [(label, marker, Series), ...]}."""
    def read(src, otype):
        fns = [Ldir['LOo'] / 'obs' / src / otype / f'{y}.p' for y in years]
        d = pd.concat([pd.read_pickle(f) for f in fns if f.exists()])
        d['time'] = pd.to_datetime(d['time'], utc=True).dt.tz_localize(None)
        return d.dropna(subset=['z'])

    def do_at(c, z0=-37.5):
        """DO [mg/L] linearly interpolated to z0 in one cast; NaN if the cast does not span z0."""
        c = c.dropna(subset=['DO (uM)']).sort_values('z')
        if len(c) < 2 or c['z'].min() > z0 or c['z'].max() < z0:
            return np.nan
        return float(np.interp(z0, c['z'], c['DO (uM)'])) * 0.032

    kc = read('kc_whidbeyBasin', 'ctd')
    out = {}
    # KC CTD files stamp every sample, so a cast = one station-day (one cast per day at these stations)
    pc = kc[kc['name'] == 'PENNCOVEWEST']
    rows = []
    for d, c in pc.groupby(pc['time'].dt.normalize()):
        c = c.sort_values('z')
        b, top = c.iloc[0], c.iloc[-1]
        rows.append(dict(time=c['time'].min(), DO=b['DO (uM)'] * 0.032, T=b['CT'],
                         drho=gsw.sigma0(b['SA'], b['CT']) - gsw.sigma0(top['SA'], top['CT'])))
    pc = pd.DataFrame(rows).set_index('time').sort_index()
    lab = 'obs: KC PENNCOVEWEST'
    for k, c in [('DO_pc_west', 'DO'), ('drho_pc_west', 'drho'), ('Tbot_pc_west', 'T')]:
        out[k] = [(lab, 'o', pc[c].dropna())]
    sr = kc[kc['name'] == 'SARATOGARP']
    srs = pd.Series({c['time'].min(): do_at(c) for _, c in sr.groupby(sr['time'].dt.normalize())}).dropna()
    wo = read('nceiSalish', 'bottle_ctd')
    wo = wo[np.hypot((wo['lon'] + 122.554) * np.cos(np.deg2rad(48.242)) * 111, (wo['lat'] - 48.242) * 111) < 1]
    wos = pd.Series({c['time'].min(): do_at(c) for _, c in wo.groupby(wo['time'].dt.normalize())}).dropna()
    out['DO_M5'] = [('obs: KC SARATOGARP, 37.5 m', 'o', srs.sort_index()),
                    ('obs: WOAC stn 4, 37.5 m', 's', wos.sort_index())]
    # Penn Cove mussel raft sondes (hourly, 1 m and 7 m), daily means of days with >= 12 h
    rf = xr.open_dataset(Ldir['LOo'] / 'obs' / 'pcRaft' / 'moor' / 'raft_main.nc')
    hr = {v: rf[v].to_pandas() for v in ['CT', 'SA', 'DO (uM)']}          # (time, z) frames
    rf.close()
    sig = pd.DataFrame(gsw.sigma0(hr['SA'].values, hr['CT'].values), index=hr['CT'].index, columns=hr['CT'].columns)
    day = lambda x: x.resample('1D').mean().where(x.resample('1D').count() >= 12).dropna()
    lab = 'obs: mussel raft sonde, daily'
    out['DO_pc_west'].append((lab + ', 7 m', '.', day(hr['DO (uM)'][-7.0] * 0.032)))
    out['Tbot_pc_west'].append((lab + ', 7 m', '.', day(hr['CT'][-7.0])))
    out['drho_pc_west'].append((lab + ', 7 m - 1 m', '.', day(sig[-7.0] - sig[-1.0])))
    return out


OBS = load_obs()


def anomaly(x, clim):
    doy = pd.to_datetime({'year': 2000, 'month': x.index.month, 'day': x.index.day}).dt.dayofyear.values
    return x - clim.reindex(doy).values


if args.anom:
    ref = 'cas7_t1_x11ab'
    S_raw_ref = dict(S[ref])
    # all climatologies first, from the raw series, so the x11ab reference is not already an anomaly
    CL = {g: {k: climatology(S[ref][k] if (args.clim == 'x11ab' and k in S[ref]) else S[g][k])
              for k in S[g] if not k.startswith('h_')} for g in S}
    for g in S:
        for k, c in CL[g].items():
            S[g][k] = anomaly(S[g][k], c)
    OBS = ({k: [(lab, mk, anomaly(v, climatology(S_raw_ref[k]))) for lab, mk, v in L] for k, L in OBS.items()}
           if args.clim == 'x11ab' else {})

if args.roll:
    for g in S:
        for k in [k for k in S[g] if not k.startswith('h_')]:
            S[g][k] = S[g][k].rolling(f'{args.roll}D', center=True, min_periods=args.roll // 2).mean()

PANELS = [('DO_pc_west', 'Penn Cove bottom DO\n(pc_west) [mg/L]', True),
          ('DO_M5', 'Saratoga bottom DO\n(M5) [mg/L]', True),
          ('drho_pc_west', 'Penn Cove $\\Delta\\sigma_0$\nbottom - top [kg m$^{-3}$]', False),
          ('taux', 'Cove wind stress\neast $\\tau_x$ [Pa]', False),
          ('tauy', 'Cove wind stress\nnorth $\\tau_y$ [Pa]', False),
          ('Tair', 'Cove air T [°C]', False),
          ('Tbot_pc_west', 'Penn Cove bottom T\n(pc_west) [°C]', False)]

fig, axs = plt.subplots(len(PANELS), 1, figsize=(15, 2.4 * len(PANELS)), sharex=True)
for ax, (key, ylab, is_do) in zip(axs, PANELS):
    for g, c in RUN_COLORS.items():          # drawing order: long x11ab first, shorter runs on top
        if key in S[g]:
            ax.plot(S[g][key].index, S[g][key].values, '-', color=c, lw=0.6, label=RUN_LABELS[g])
    hobs = []
    for lab, mk, v in OBS.get(key, []):
        sty = (dict(color='0.45', ms=1.5, zorder=4) if mk == '.' else
               dict(color='k', mfc='k' if mk == 'o' else 'none', ms=3.5, zorder=5))
        hobs += ax.plot(v.index, v.values, mk, label=lab, **sty)
    if hobs and key != 'DO_pc_west':
        ax.legend(handles=hobs, loc='upper right', fontsize=8, markerscale=2)
    if args.anom:
        ax.axhline(0, color='gray', lw=0.6)
        ylab = ylab.replace('\n', ' anomaly\n', 1) if '\n' in ylab else ylab + '\nanomaly'
    elif is_do:
        ax.axhline(HYP, color='gray', ls=':', lw=1)
        ax.set_ylim(bottom=0)
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(True, alpha=0.3)
if not args.anom:
    axs[2].set_ylim(bottom=0)
    for ax in axs[3:5]:
        ax.axhline(0, color='gray', lw=0.6)
axs[0].legend(loc='upper right', fontsize=8, ncol=5, markerscale=1.5)

ttl = ('Penn Cove (pc_west, cas7 h 17.0 m / wb1 h 17.2 m) and Saratoga Passage (M5, cas7 h 39.2 m / '
       'wb1 h 37.4 m), daily')
if args.anom:
    ttl += ('\nanomalies from each run\'s own day-of-year climatology' if args.clim == 'own'
            else '\nanomalies from the cas7_t1_x11ab day-of-year climatology')
if args.roll:
    ttl += ('; ' if args.anom else '\n') + f'{args.roll}-day centered rolling mean'
axs[0].set_title(ttl, fontsize=10)
if args.x0 or args.x1:
    axs[-1].set_xlim(pd.Timestamp(args.x0.replace('.', '-')) if args.x0 else None,
                     pd.Timestamp(args.x1.replace('.', '-')) if args.x1 else None)
fig.tight_layout()
suffix = ((f'_anom_{args.clim}' if args.anom else '') + (f'_roll{args.roll}' if args.roll else '')
          + (f'_{args.x0}_{args.x1}' if (args.x0 or args.x1) else ''))
fig_fn = out_dir / f'pc_runs_timeseries{suffix}.png'
fig.savefig(fig_fn, dpi=200, transparent=True)
plt.close(fig)
print(f'saved: {fig_fn}')
for g in S:
    print(g, sorted(k for k in S[g] if not k.startswith('h_')))
