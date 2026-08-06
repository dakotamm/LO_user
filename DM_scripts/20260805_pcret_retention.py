"""
Penn Cove retention from the pcret particle release: does water stagnate
landward, and does the lateral gyre short-circuit the cove?

Two things are computed for every cohort, and the difference between them is
the whole point:

  still inside(t)   fraction of the cohort inside the cove right now
  never left(t)     fraction that has not once been outside

"Still inside" counts a particle that left and came back as retained. "Never
left" does not. The GAP between the curves is water that crossed the mouth and
returned -- the sloshing that TEF Qin counts as renewal and that the tidal
prism return-flow factor put at about 54%. Salinity cannot separate these two
because water inside and outside the cove has the same salinity; particle
identity can.

Cohorts come from each particle's INITIAL position, which sits exactly on a rho
cell centre. That is robust to the tracker dropping particles on land, which
would break any index-based bookkeeping saved at release time.

Cuts applied:
  by cohort          inner / mid / outer cove, plus Saratoga as control
  by release depth   terciles of cs (fractional depth), so the comparison is
                     fair across boxes of different depth
  by release side    north vs south half, which tests the gyre short-circuit:
                     if the gyre carries water in along the north shore and out
                     along the south without ventilating the interior, north
                     and south releases should behave very differently

run 20260805_pcret_retention.py
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
gtx = 'wb1_t0_xn11abbur00'
trk_dir = Ldir['LOo'] / 'tracks2' / gtx / 'pcret_3d'
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

# ------------------------------------------------------------------ setup ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = g.lon_rho.values
lat = g.lat_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape

tef2 = Ldir['LOo'] / 'extract' / 'tef2'
cands = sorted(tef2.glob('seg_info_dict_wb1_pc1_*.p'))
seg = pickle.load(open(cands[0], 'rb'))
print('segments from %s' % cands[0].name)

COVE = ['pc_cp_m', 'pc_cp_p', 'pc_lp_m']
LABEL = {'pc_cp_m': 'inner', 'pc_cp_p': 'mid', 'pc_lp_m': 'outer',
         'pc_lp_p': 'Saratoga (control)'}
COLOR = {'pc_cp_m': '#009E73', 'pc_cp_p': '#0072B2', 'pc_lp_m': '#CC79A7',
         'pc_lp_p': '#D55E00'}

seg_mask = {}
for s in seg:
    m = np.zeros((NR, NC), dtype=bool)
    a = np.array(seg[s]['ji_list'])
    m[a[:, 0], a[:, 1]] = True
    seg_mask[s] = m
cove_mask = np.zeros((NR, NC), dtype=bool)
for s in COVE:
    cove_mask |= seg_mask[s]


def ji_of(plon, plat):
    """Nearest rho indices on this plaid grid; NaN positions map to -1."""
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.full(plon.shape, -1, dtype=int)
    j = np.full(plon.shape, -1, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    return j, i, ok


def curves(inside):
    """(still inside, never left) as fractions of the cohort, vs time."""
    still = inside.mean(axis=1)
    never = np.minimum.accumulate(inside, axis=0).mean(axis=1)
    return still, never


def efold(frac, hours):
    """Hours until the curve first drops below 1/e; NaN if it never does."""
    k = np.where(frac < 1 / np.e)[0]
    return hours[k[0]] / 24 if len(k) else np.nan


# ------------------------------------------------------------------- load ---
results = {}
for fn in sorted(trk_dir.glob('release_*.nc')):
    rel = fn.stem.replace('release_', '')
    d = xr.open_dataset(fn)
    plon = d.lon.values
    plat = d.lat.values
    cs0 = d.cs.values[0, :]
    d.close()
    hours = np.arange(plon.shape[0])

    j, i, ok = ji_of(plon, plat)
    inside_cove = np.zeros(plon.shape, dtype=bool)
    inside_cove[ok] = cove_mask[j[ok], i[ok]]

    j0, i0 = j[0, :], i[0, :]
    cohort = np.array(['none'] * plon.shape[1], dtype=object)
    for s in seg:
        cohort[seg_mask[s][j0, i0]] = s
    home_sar = np.zeros(plon.shape, dtype=bool)
    home_sar[ok] = seg_mask['pc_lp_p'][j[ok], i[ok]]

    results[rel] = dict(hours=hours, inside=inside_cove, home_sar=home_sar,
                        cohort=cohort, cs0=cs0, lat0=plat[0, :])
    print('%s  cohorts: %s' % (rel, {LABEL[k]: int((cohort == k).sum())
                                     for k in seg if (cohort == k).sum()}))

# ---------------------------------------------------------------- summary ---
print('\n%-12s %-20s %8s %10s %10s %10s'
      % ('release', 'cohort', 'n', 'e-fold d', 'never-left', 'reentry'))
rows = []
for rel, R in results.items():
    for s in COVE + ['pc_lp_p']:
        m = R['cohort'] == s
        if m.sum() == 0:
            continue
        ins = R['home_sar'][:, m] if s == 'pc_lp_p' else R['inside'][:, m]
        still, never = curves(ins)
        rows.append(dict(release=rel, cohort=LABEL[s], n=int(m.sum()),
                         efold=efold(still, R['hours']),
                         efold_never=efold(never, R['hours']),
                         final_still=still[-1], final_never=never[-1]))
        print('  %-10s %-20s %8d %10.2f %10.2f %10.3f'
              % (rel, LABEL[s], m.sum(), rows[-1]['efold'],
                 rows[-1]['efold_never'], still[-1] - never[-1]))
pd.DataFrame(rows).to_csv(out_dir / '20260805_pcret_retention.csv', index=False)

# ----------------------------------------------------------------- figure ---
plt.close('all')
rels = sorted(results)
fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'd']], figsize=(14, 10),
                             layout='constrained')

# a: still-inside by cohort, both releases
A = ax['a']
for rel, ls in zip(rels, ['-', '--']):
    R = results[rel]
    for s in COVE:
        m = R['cohort'] == s
        still, _ = curves(R['inside'][:, m])
        A.plot(R['hours'] / 24, still, ls, lw=2, color=COLOR[s],
               label='%s %s' % (LABEL[s], rel) if ls == '-' else None)
A.axhline(1 / np.e, color='0.5', ls=':', lw=1)
A.text(0.5, 1 / np.e, ' 1/e', color='0.4', va='bottom', fontsize=8)
A.set_xlabel('days since release')
A.set_ylabel('fraction still inside the cove')
A.set_title('a. Retention by cohort\nsolid = %s, dashed = %s' % tuple(rels))
A.grid(color='lightgray', ls='--', alpha=0.5)
A.legend(fontsize=8)

# b: still-inside vs never-left -- the sloshing gap
B = ax['b']
R = results[rels[0]]
for s in COVE:
    m = R['cohort'] == s
    still, never = curves(R['inside'][:, m])
    B.plot(R['hours'] / 24, still, '-', lw=2, color=COLOR[s], label=LABEL[s] + ' still inside')
    B.plot(R['hours'] / 24, never, ':', lw=2, color=COLOR[s], label=LABEL[s] + ' never left')
    B.fill_between(R['hours'] / 24, never, still, color=COLOR[s], alpha=0.12, lw=0)
B.set_xlabel('days since release')
B.set_ylabel('fraction')
B.set_title('b. Shaded = left and came back (%s)' % rels[0])
B.grid(color='lightgray', ls='--', alpha=0.5)
B.legend(fontsize=7, ncol=2)

# c: by release depth tercile, inner cove
C = ax['c']
for rel, ls in zip(rels, ['-', '--']):
    R = results[rel]
    m = R['cohort'] == 'pc_cp_m'
    cs = R['cs0'][m]
    ins = R['inside'][:, m]
    for lo, hi, lab, col in ((-1.01, -2 / 3, 'bottom third', '#3B0F70'),
                             (-2 / 3, -1 / 3, 'middle', '#8C2981'),
                             (-1 / 3, 0.01, 'top third', '#FE9F6D')):
        k = (cs >= lo) & (cs < hi)
        if k.sum() == 0:
            continue
        still, _ = curves(ins[:, k])
        C.plot(R['hours'] / 24, still, ls, lw=2, color=col,
               label='%s (n=%d)' % (lab, k.sum()) if ls == '-' else None)
C.set_xlabel('days since release')
C.set_ylabel('fraction still inside')
C.set_title('c. Inner cove by release depth\n(fractional depth terciles)')
C.grid(color='lightgray', ls='--', alpha=0.5)
C.legend(fontsize=8)

# d: north vs south release -- the gyre short-circuit test
D = ax['d']
for rel, ls in zip(rels, ['-', '--']):
    R = results[rel]
    for s in COVE:
        m = R['cohort'] == s
        la = R['lat0'][m]
        mid = np.median(la)
        for side, k, col in (('north', la >= mid, '#D55E00'),
                             ('south', la < mid, '#0072B2')):
            still, _ = curves(R['inside'][:, m][:, k])
            if s == 'pc_lp_m':
                D.plot(R['hours'] / 24, still, ls, lw=2, color=col,
                       label='outer, %s half' % side if ls == '-' else None)
D.set_xlabel('days since release')
D.set_ylabel('fraction still inside')
D.set_title('d. Outer cove, north vs south release\n(gyre short-circuit test)')
D.grid(color='lightgray', ls='--', alpha=0.5)
D.legend(fontsize=8)

fn_out = out_dir / '20260805_pcret_retention.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn_out)
