"""
Particle residence times vs exchange-flow flushing times for Penn Cove, release
by release.

Flushing time T = V / Qin, with
  V    volume of the cove landward of pc_lp (tef2 segments pc_cp_m + pc_cp_p +
       pc_lp_m, hourly), averaged over each release's first -win_days
  Qin  inflow through pc_lp, daily, averaged over the same window, two ways:
         TEF   salinity-coordinate Lorenz bulk collapsed to two layers
               (bulk_avg_*, xfun.two_layer) -- a NET (sorted) exchange
         EUL   Eulerian per-cell sign split of the Godin-filtered section
               (extractions_avg_*, xfun.eulerian_bulk mode 'cell') -- a GROSS
               exchange
       as in 20260921_pc_Q_three_methods.py.

CAVEAT, see 20260916_frozen_field_control.py: at pc_lp a frozen salinity field,
whose true exchange is zero, returns TEF Qin = 316.4 m3/s against 358.0 for the
real field (record means, 1000 bins), so ~88 % of TEF Qin is numerical floor.
The record-mean floor-corrected TEF flushing time, V / (Qin - 316.4), is printed
alongside. It is a record-mean number only -- the floor was not computed per
day, so it is not subtracted release by release.

Particle measures per release (whole cove, 20261005_pcmap_reduce.py files):
  still 1/e     time for the still-inside curve to fall below 1/e
  never 1/e     the same for never-left
  mean resid.   area under the still-inside curve to 14 d (mean hours inside
                per particle, re-entry counted; truncated at 14 d)
For a well-mixed box flushed at a constant rate all of these equal V/Qin.

Outputs, to LO_output/DM_outs/20261007_pcmap_retention_vs_tef/:
  pcmap_retention_vs_tef.csv    per release: V, Qin (TEF, EUL), T (TEF, EUL),
                                particle measures, season
  pcmap_retention_vs_tef.png    scatter of each particle measure against
                                T_TEF and T_EUL (1:1 line, coloured by season),
                                and the time series through the year

run 20261007_pcmap_retention_vs_tef.py
"""
import argparse
import importlib.util
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-ds0', default='2024.01.01')
p.add_argument('-ds1', default='2025.12.31')
p.add_argument('-win_days', type=float, default=3.0)
p.add_argument('-year', type=int, default=2025)
p.add_argument('-floor', type=float, default=316.4, help='frozen-field TEF Qin floor at pc_lp [m3/s]')
args = p.parse_args()

_spec = importlib.util.spec_from_file_location('exchange_fun', Path(__file__).parent / '20260916_exchange_fun.py')
xfun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xfun)

Ldir = Lfun.Lstart(gridname='wb1')
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
dates = args.ds0 + '_' + args.ds1
red_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261007_pcmap_retention_vs_tef'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
SEASON = {m: 'Dec-Mar' for m in [12, 1, 2, 3]}
SEASON.update({m: 'Apr-Jul' for m in [4, 5, 6, 7]})
SEASON.update({m: 'Aug-Nov' for m in [8, 9, 10, 11]})
SCOL = {'Dec-Mar': '#4565e8', 'Apr-Jul': '#45a85b', 'Aug-Nov': '#e8455e'}

# ------------------------------------------------------------ V and Qin ---
sg = xr.open_dataset(tef2 / ('segments_%s_wb1_pc1_trapsN00.nc' % dates))
V = pd.Series(sg.volume.sel(seg=['pc_cp_m', 'pc_cp_p', 'pc_lp_m']).sum('seg').values,
              index=pd.to_datetime(sg.time.values))
sg.close()

ds = xr.open_dataset(tef2 / ('bulk_avg_' + dates) / 'pc_lp.nc')
sgn = xfun.INFLOW_SIGN['pc_lp']
two = xfun.two_layer(sgn * ds.q.to_numpy(), ds.salt.to_numpy())
Q_tef = pd.Series(two['Qin'], index=pd.to_datetime(ds.time.to_numpy()))
ds.close()
print('loading the pc_lp extraction for the Eulerian split ...')
S = xfun.load_section('pc_lp', args.gtx, args.ds0, args.ds1, Ldir=Ldir)
E = xfun.eulerian_bulk(S, mode='cell')
Q_eul = pd.Series(E['Qin'], index=pd.to_datetime(xfun.daily_time(S['time'])))
print('record means: V %.3e m3, Qin TEF %.1f, EUL %.1f m3/s; V/Qin TEF %.2f d, EUL %.2f d, '
      'floor-corrected TEF %.1f d (Qin - %.1f = %.1f m3/s)'
      % (V.mean(), Q_tef.mean(), Q_eul.mean(), V.mean() / Q_tef.mean() / 86400,
         V.mean() / Q_eul.mean() / 86400, V.mean() / (Q_tef.mean() - args.floor) / 86400,
         args.floor, Q_tef.mean() - args.floor))


def wmean(s, t0):
    t1 = t0 + pd.Timedelta(days=args.win_days)
    seg = s[(s.index >= t0 - pd.Timedelta(hours=12)) & (s.index < t1 + pd.Timedelta(hours=12))]
    return seg.mean()


# --------------------------------------------------------- particle side ---
keep = set(pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times'
                       / ('pcmap_release_times_%d_every3.csv' % args.year)).sub_tag)
rows = []
for fn in sorted(red_dir.glob('pcmap_3d*.p')):
    D = pickle.load(open(fn, 'rb'))
    m = re.search(r'_([EF])_(\d{4}\.\d{2}\.\d{2})$', D['meta']['dir'])
    if not m or '%s_%s' % (m.group(1), m.group(2)) not in keep:
        continue
    t0 = pd.Timestamp(D['meta']['t0'])
    st, nv = D['curves']['cove']['still'], D['curves']['cove']['never']
    days = np.arange(len(st)) / 24
    ef = lambda c: days[np.where(c < 1 / np.e)[0][0]] if (c < 1 / np.e).any() else np.nan
    v_, qt, qe = wmean(V, t0), wmean(Q_tef, t0), wmean(Q_eul, t0)
    rows.append(dict(t0=t0, set=m.group(1), season=SEASON[t0.month], V=v_, Qin_tef=qt, Qin_eul=qe,
                     T_tef=v_ / qt / 86400, T_eul=v_ / qe / 86400,
                     still_efold=ef(st), never_efold=ef(nv), mean_resid=st[1:].sum() / 24))
R = pd.DataFrame(rows).sort_values('t0').reset_index(drop=True)
R.to_csv(out_dir / 'pcmap_retention_vs_tef.csv', index=False)

PM = [('still_efold', 'still-inside 1/e'), ('never_efold', 'never-left 1/e'),
      ('mean_resid', 'mean residence (to 14 d)')]
print('\nrelease means [d] (n %d): T_TEF %.2f, T_EUL %.2f | %s'
      % (len(R), R.T_tef.mean(), R.T_eul.mean(),
         ', '.join('%s %.2f' % (lab, R[c].mean()) for c, lab in PM)))
print('correlation across releases (r):')
for c, lab in PM:
    print('  %-26s vs T_TEF %+.2f   vs T_EUL %+.2f   | ratio particle/T_TEF %.2f, /T_EUL %.2f'
          % (lab, R[c].corr(R.T_tef), R[c].corr(R.T_eul), (R[c] / R.T_tef).median(), (R[c] / R.T_eul).median()))
print('by season (means, d):')
print(R.groupby('season')[['T_tef', 'T_eul', 'still_efold', 'never_efold', 'mean_resid']].mean()
      .reindex(['Dec-Mar', 'Apr-Jul', 'Aug-Nov']).round(2).to_string())

# ----------------------------------------------------------------- figure ---
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.35)
lim = [0, max(R[['T_tef', 'T_eul'] + [c for c, _ in PM]].max()) * 1.05]
for c_i, (tcol, tlab) in enumerate([('T_tef', 'TEF (net)'), ('T_eul', 'Eulerian (gross)')]):
    ax = fig.add_subplot(gs[0, c_i])
    for pc, mk, plab in [('still_efold', 'o', 'still-inside 1/e'), ('never_efold', '^', 'never-left 1/e'),
                         ('mean_resid', 's', 'mean residence')]:
        for sn, col in SCOL.items():
            q = R[R.season == sn]
            ax.scatter(q[tcol], q[pc], s=14, marker=mk, color=col, alpha=0.6, edgecolor='none',
                       label='%s, %s' % (plab, sn) if c_i == 0 else None)
    ax.plot(lim, lim, color='k', lw=1, ls='--')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('V / Qin, %s [d]' % tlab)
    ax.set_ylabel('particle time scale [d]')
    ax.set_title('particles vs %s flushing time\nr: still %+.2f, never %+.2f, mean %+.2f'
                 % (tlab, R.still_efold.corr(R[tcol]), R.never_efold.corr(R[tcol]),
                    R.mean_resid.corr(R[tcol])), fontsize=10)
    ax.grid(**GRID)
    ax.set_aspect('equal')
    if c_i == 0:
        from matplotlib.lines import Line2D
        hd = [Line2D([], [], ls='', marker='s', color=col, label=sn) for sn, col in SCOL.items()]
        hd += [Line2D([], [], ls='', marker=mk, color='0.4', label=lab)
               for mk, lab in [('o', 'still-inside 1/e'), ('^', 'never-left 1/e'), ('s', 'mean residence')]]
        ax.legend(handles=hd, fontsize=8, loc='upper left')
ax = fig.add_subplot(gs[0, 2])
labs = ['V/Qin TEF', 'V/Qin EUL', 'V/(Qin-floor) TEF', 'still 1/e', 'never 1/e', 'mean resid.']
vals = [R.T_tef.mean(), R.T_eul.mean(), V.mean() / (Q_tef.mean() - args.floor) / 86400,
        R.still_efold.mean(), R.never_efold.mean(), R.mean_resid.mean()]
cols = ['0.3', '0.55', '0.8', '#e8455e', '#4565e8', '#45a85b']
ax.barh(labs[::-1], vals[::-1], color=cols[::-1])
for y, v in enumerate(vals[::-1]):
    ax.text(v, y, ' %.1f' % v, va='center', fontsize=9)
ax.set_xlabel('days (release / record mean)')
ax.set_title('time scales compared\n(floor-corrected TEF: record mean only)', fontsize=10)
ax.grid(axis='x', **GRID)
ax = fig.add_subplot(gs[1, :])
ax.plot(R.t0, R.T_tef, '-', color='0.3', lw=1.4, label='V/Qin, TEF (net)')
ax.plot(R.t0, R.T_eul, '-', color='0.6', lw=1.4, label='V/Qin, Eulerian (gross)')
ax.plot(R.t0, R.still_efold, 'o', ms=3, color='#e8455e', label='particles: still-inside 1/e')
ax.plot(R.t0, R.never_efold, '^', ms=3, color='#4565e8', label='particles: never-left 1/e')
ax.plot(R.t0, R.mean_resid, 's', ms=3, color='#45a85b', label='particles: mean residence (to 14 d)')
ax.set_ylabel('days')
ax.set_title('per release: flushing times over each release\'s first %g d vs particle time scales'
             % args.win_days, fontsize=10)
ax.legend(fontsize=8, ncol=5, loc='upper left')
ax.grid(**GRID)
fig.suptitle('%s Penn Cove: particle residence vs exchange-flow flushing time (pc_lp, cove landward of it)'
             % args.gtx, fontsize=12)
fn_out = out_dir / 'pcmap_retention_vs_tef.png'
fig.savefig(fn_out, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('wrote %s' % fn_out)
