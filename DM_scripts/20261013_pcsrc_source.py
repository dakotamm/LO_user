"""
Penn Cove source water from the pcsrc reduced files (20261013_pcsrc_reduce.py):
what fraction of the water in a target at time T was in upper Saratoga, or
already in the cove, at the release time t0. Mac only.

TARGETS (applied to the hourly J, I, CS, Z of every particle)
  cove        any cell of pcsrc_regions 'cove' (landward of pc_lp)
  inner_bot   inner-cove bottom water, the pcbot definition: a cell of tef2
              pc_cp_m, within 5 m of the bed (hab = Z + h) AND in the lower half
              of the column (cs <= -0.5)

TARGET VOLUME (hourly)
  cove        tef2 segment volumes pc_cp_m + pc_cp_p + pc_lp_m
  inner_bot   sum over pc_cp_m cells of A * min(5, 0.5 * (h + zeta)), with zeta
              the segment-mean surface from the pc_cp_m volume

PARTICLE VOLUME at the release tide: the reduced vol (mean sea level) times
(h + zeta0) / h, zeta0 the segment-mean sea level at t0 from the same tef2
volumes (cove segments for the cove cohort, pc_lp_p for upper Saratoga).
Without it the E set (released at low water) overstates labelled volume ~9%.

FRACTIONS (labelled volume / target volume, tidal-day 25 h running mean)
  from_cove   was in the cove at t0 (pcmap twin cohort)
  from_spu    was in upper Saratoga at t0 (pcsrc cohort)
  unlab       1 - both: was south of sp_mid or in Skagit Bay at t0, or is river
              water added since
  new         1 - from_cove: water that was not in the cove at t0
  spu_share   from_spu / new: the upper-Saratoga share of the cove's SOURCE water
At short lags new is small, so spu_share is noisy there (the ~5% volume noise
seen in pcret is divided by new).

Figure: per release grey, mean black, median dashed; composition stacked.
The share is only drawn where new >= NEW_MIN (0.2) of the target; the CSV
keeps all values.

run 20261013_pcsrc_source.py
run 20261013_pcsrc_source.py -month 8
run 20261013_pcsrc_source.py -test_pcret        (checks against 20261012_pcsrc_spu_feasibility.py)
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lo_tools import Lfun

import pcsrc_regions as pr

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-month', type=int, default=0)
p.add_argument('-set', default='both', choices=['both', 'E', 'F'])
p.add_argument('-days', type=float, default=14.0)
p.add_argument('-test_pcret', action='store_true')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
tag = args.gtx + ('_test_pcret' if args.test_pcret else '')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcsrc_reduce' / tag
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcsrc_source' / tag
Lfun.make_dir(out_dir)

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, h = g.lon_rho.values, g.lat_rho.values, g.h.values
A = 1 / (g.pm.values * g.pn.values)
wet = g.mask_rho.values == 1
g.close()
R = pr.regions(Ldir, lon, lat, wet)
seg = pickle.load(open(sorted((Ldir['LOo'] / 'extract' / 'tef2').glob('seg_info_dict_wb1_pc1_*.p'))[0], 'rb'))
inner = np.zeros(lon.shape, dtype=bool)
a = np.array(seg['pc_cp_m']['ji_list']); inner[a[:, 0], a[:, 1]] = True

segds = xr.open_dataset(Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
                        / 'segments_2024.01.01_2025.12.31_wb1_pc1_trapsN00.nc')
Vseg = segds.volume.to_pandas()
segds.close()
V_cove = Vseg[pr.COVE_SEGS].sum(axis=1)
# inner-cove bottom layer volume from the pc_cp_m mean surface
h_in, A_in = h[inner], A[inner]
zeta_in = (Vseg['pc_cp_m'] - (h_in * A_in).sum()) / A_in.sum()
V_ibot = pd.Series([(A_in * np.minimum(5, 0.5 * (h_in + zt))).sum() for zt in zeta_in.values],
                   index=Vseg.index)
# segment-mean sea level, used to put each particle's volume at its release
# tide: the reduce stores vol at mean sea level (h*A...), but a release at low
# water (the E set, zeta down to -1.7 m) holds less water than that, and the
# excess shows up as labelled volume > target volume
SEGCELLS = {'cove': R['cove'], 'spu': R['spu']}
SEGVOL = {'cove': V_cove, 'spu': Vseg['pc_lp_p']}


def zeta_seg(rn, t0):
    m = SEGCELLS[rn]
    V0 = SEGVOL[rn].reindex([t0], method='nearest').values[0]
    return (V0 - (h[m] * A[m]).sum()) / A[m].sum()


TARGETS = ['cove', 'inner_bot']
NEW_MIN = 0.2
TLAB = {'cove': 'Penn Cove (landward of pc_lp)',
        'inner_bot': 'inner-cove bottom water'}

fns = sorted(red_dir.glob('*.p'))
rows = []
curves = {t: {} for t in TARGETS}
skipped = []
for fn in fns:
    try:
        D = pickle.load(open(fn, 'rb'))
    except (pickle.UnpicklingError, EOFError):
        skipped.append(fn.name)            # e.g. a copy still in progress
        continue
    meta, P, H = D['meta'], D['P'], D['H']
    st = meta['sub_tag']
    if args.month and pd.Timestamp(meta['t0']).month != args.month:
        continue
    if args.set != 'both' and meta.get('set') != args.set:
        continue
    nf = min(meta['nf'], int(round(args.days * 24)) + 1)
    J = H['J'][:nf].astype(int); I = H['I'][:nf].astype(int)
    ok = J >= 0
    Jc, Ic = np.where(ok, J, 0), np.where(ok, I, 0)
    cs = H['CS'][:nf] / 10000
    hab = H['Z'][:nf].astype(float) + h[Jc, Ic]
    in_t = dict(cove=ok & R['cove'][Jc, Ic],
                inner_bot=ok & inner[Jc, Ic] & (hab <= 5) & (cs <= -0.5))
    t = pd.DatetimeIndex(meta['ot'][:nf])
    w = P.vol.values.copy()
    for rn in ['cove', 'spu']:
        m = (P.cohort == rn).values
        hh = h[P.j0.values[m], P.i0.values[m]]
        w[m] *= np.maximum(hh + zeta_seg(rn, t[0]), 0) / hh
    is_cove0 = (P.cohort == 'cove').values
    for tg in TARGETS:
        V = (V_cove if tg == 'cove' else V_ibot).reindex(t, method='nearest').values
        fc = (in_t[tg][:, is_cove0] * w[is_cove0]).sum(axis=1) / V
        fs = (in_t[tg][:, ~is_cove0] * w[~is_cove0]).sum(axis=1) / V
        df = pd.DataFrame(dict(from_cove=fc, from_spu=fs), index=t)
        df = df.rolling(25, center=True, min_periods=25).mean()
        df['unlab'] = 1 - df.from_cove - df.from_spu
        df['new'] = 1 - df.from_cove
        df['spu_share'] = df.from_spu / df.new
        df['days'] = np.arange(nf) / 24
        curves[tg][st] = df
        r = dict(sub_tag=st, set=meta.get('set'), t0=meta['t0'], target=tg)
        for d in [1, 2, 3, 5, 7, 10, 13]:
            k = int(round(d * 24))
            if k < nf:
                for c in ['from_cove', 'from_spu', 'unlab', 'new', 'spu_share']:
                    r['%s_d%d' % (c, d)] = df[c].values[k]
        rows.append(r)

if skipped:
    print('SKIPPED %d unreadable files (incomplete copy?): %s' % (len(skipped), ', '.join(skipped)))
res = pd.DataFrame(rows)
if len(res) == 0:
    raise SystemExit('no reduced releases match in %s' % red_dir)
sfx = ('_m%02d' % args.month if args.month else '') + ('_%s' % args.set if args.set != 'both' else '')
res.to_csv(out_dir / ('pcsrc_source%s.csv' % sfx), index=False)
pd.set_option('display.width', 220)
print('%d releases (%s)' % (res.sub_tag.nunique(), red_dir.name))
for tg in TARGETS:
    s = res[res.target == tg]
    print('\n%s -- mean (median) over releases' % TLAB[tg])
    for d in [1, 3, 7, 13]:
        if 'new_d%d' % d in s:
            print('  day %2d: new %.2f (%.2f)  from_spu %.2f (%.2f)  unlab %.2f (%.2f)  '
                  'spu share of new %.2f (%.2f)'
                  % (d, s['new_d%d' % d].mean(), s['new_d%d' % d].median(),
                     s['from_spu_d%d' % d].mean(), s['from_spu_d%d' % d].median(),
                     s['unlab_d%d' % d].mean(), s['unlab_d%d' % d].median(),
                     s['spu_share_d%d' % d].mean(), s['spu_share_d%d' % d].median()))

# ---------------------------------------------------------------- figure
plt.close('all')
fs_ = 11
plt.rc('font', size=fs_)
C = {'cove': '#52514e', 'spu': '#2a78d6', 'unl': '#d9d8d4'}
fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for r, tg in enumerate(TARGETS):
    cv = curves[tg]
    days = next(iter(cv.values())).days.values
    stack = {c: np.nanmean(np.vstack([cv[k][c].values for k in cv]), axis=0)
             for c in ['from_cove', 'from_spu']}
    ax = axs[r, 0]
    a0, a1 = stack['from_cove'], stack['from_spu']
    ax.fill_between(days, 0, a0, color=C['cove'], lw=0)
    ax.fill_between(days, a0, a0 + a1, color=C['spu'], lw=0)
    ax.fill_between(days, a0 + a1, 1, color=C['unl'], lw=0)
    ax.axhline(1, color='#0b0b0b', lw=0.6)
    ax.set_ylim(0, 1.05); ax.set_xlim(0, args.days)
    ax.set_ylabel('fraction of volume')
    ax.set_title('%s: mean of %d releases' % (TLAB[tg], len(cv)), loc='left', fontsize=fs_)
    ax = axs[r, 1]
    # the share is a ratio over 'new', which is tiny in the first day; don't
    # draw it until new water is at least NEW_MIN of the target
    M = np.vstack([np.where(cv[k]['new'].values >= NEW_MIN, cv[k]['spu_share'].values, np.nan)
                   for k in cv])
    for row in M:
        ax.plot(days, row, color='#b5b3ad', lw=0.6)
    ax.plot(days, np.nanmean(M, axis=0), color='#0b0b0b', lw=2)
    ax.plot(days, np.nanmedian(M, axis=0), color='#0b0b0b', lw=1.5, ls='--')
    ax.set_ylim(0, 1.05); ax.set_xlim(0, args.days)
    ax.set_ylabel('upper-Saratoga share of new water')
    ax.set_title('source of the new water (drawn where new >= %d%%)' % (100 * NEW_MIN),
                 loc='left', fontsize=fs_)
    for ax in axs[r]:
        ax.grid(lw=0.3, alpha=0.5)
for ax in axs[1]:
    ax.set_xlabel('days since release')
axs[0, 0].legend(handles=[Patch(fc=C['cove'], label='in Penn Cove at release'),
                          Patch(fc=C['spu'], label='in upper Saratoga at release'),
                          Patch(fc=C['unl'], label='unlabelled')],
                 loc='upper right', fontsize=fs_ - 1, framealpha=0.95)
axs[0, 1].legend(handles=[Line2D([], [], color='#b5b3ad', lw=0.8, label='release'),
                          Line2D([], [], color='#0b0b0b', lw=2, label='mean'),
                          Line2D([], [], color='#0b0b0b', lw=1.5, ls='--', label='median')],
                 loc='lower left', fontsize=fs_ - 1, framealpha=0.95)
fig.tight_layout()
fn_out = out_dir / ('pcsrc_source%s.png' % sfx)
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('\nsaved %s' % fn_out)
