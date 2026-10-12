"""
Is seeding upper Saratoga only (tef2 segment pc_lp_p, bounded by pc_lp, sp_mid
and skagit_sp) enough for pcsrc? Answered from the existing pcret runs, which
already seeded the cove segments AND pc_lp_p and tracked them 60 d
(releases 2024.02.15 and 2024.08.15, both on the Mac).

Each particle gets the volume it stands for:
  cove segments (pc_cp_m, pc_cp_p, pc_lp_m)  h*A/n,  n = max(1, round(h/DZ)), DZ = 3
  pc_lp_p (capped at 1500 by stride)          V(pc_lp_p) / N(pc_lp_p)
so the labelled volume inside a box at time T, divided by the box volume
(hourly tef2 segment volume), is the fraction of that box's water that was in
the cove or upper Saratoga at release. The rest is UNLABELLED: water that was
south of sp_mid or in Skagit Bay at t0, plus river water added since.

Also: the tef2 TEF inflow into pc_lp_p through sp_mid vs skagit_sp, the only
bulk handle on where the unlabelled water comes from.

Mac only. python 20261012_pcsrc_spu_feasibility.py
"""
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
gtagex = 'wb1_t0_xn11abbur00'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261012_pcsrc_spu_feasibility'
Lfun.make_dir(out_dir)
trk_dir = Ldir['LOo'] / 'tracks2' / gtagex / 'pcret_3d'
tef_dir = Ldir['LOo'] / 'extract' / gtagex / 'tef2'
DZ = 3
COVE = ['pc_cp_m', 'pc_cp_p', 'pc_lp_m']
RELS = ['2024.02.15', '2024.08.15']

g = xr.open_dataset(trk_dir / 'grid.nc')
lon = g.lon_rho.values; lat = g.lat_rho.values; h = g.h.values
G0 = xr.open_dataset(Ldir['data'] / 'grids' / 'wb1' / 'grid.nc')
A = 1 / (G0.pm.values * G0.pn.values)
lonv = lon[0, :]; latv = lat[:, 0]          # plaid grid
assert np.allclose(lon, lonv[None, :]) and np.allclose(lat, latv[:, None])

seg = pickle.load(open(sorted((Ldir['LOo'] / 'extract' / 'tef2').glob('seg_info_dict_wb1_pc1_*.p'))[0], 'rb'))
SEGID = np.full(lon.shape, -1, dtype=int)
SNAMES = COVE + ['pc_lp_p']
for k, s in enumerate(SNAMES):
    a = np.array(seg[s]['ji_list'])
    SEGID[a[:, 0], a[:, 1]] = k
is_cove = np.isin(SEGID, [0, 1, 2])
is_spu = SEGID == 3

segds = xr.open_dataset(tef_dir / ('segments_2024.01.01_2025.12.31_wb1_pc1_trapsN00.nc'))
Vseg = segds.volume.to_pandas()          # hourly, columns = seg
V_cove_t = Vseg[COVE].sum(axis=1)
V_spu_t = Vseg['pc_lp_p']


def cell_of(x, y):
    """nearest rho cell on the plaid grid; -1 where x is NaN"""
    i = np.clip(np.searchsorted(lonv, x), 1, len(lonv) - 1)
    i -= (np.abs(x - lonv[i - 1]) < np.abs(x - lonv[i]))
    j = np.clip(np.searchsorted(latv, y), 1, len(latv) - 1)
    j -= (np.abs(y - latv[j - 1]) < np.abs(y - latv[j]))
    bad = ~np.isfinite(x) | ~np.isfinite(y)
    return np.where(bad, 0, j), np.where(bad, 0, i), bad


rows = []
curves = {}
for rel in RELS:
    ds = xr.open_dataset(trk_dir / ('release_' + rel + '.nc'))
    x = ds.lon.values; y = ds.lat.values
    t = pd.to_datetime(ds.ot.values, unit='s') if np.issubdtype(ds.ot.dtype, np.number) else pd.to_datetime(ds.ot.values)
    j0, i0, _ = cell_of(x[0], y[0])
    sid = SEGID[j0, i0]
    if (sid < 0).any():
        raise ValueError('%d particles start outside the four segments' % (sid < 0).sum())
    w = np.zeros(len(sid))
    for k in range(3):
        m = sid == k
        n = np.maximum(1, np.round(h[j0[m], i0[m]] / DZ))
        w[m] = h[j0[m], i0[m]] * A[j0[m], i0[m]] / n
    m = sid == 3
    w[m] = V_spu_t.reindex([t[0]], method='nearest').values[0] / m.sum()
    origin = np.where(sid == 3, 'spu', 'cove')
    print('%s: %d particles (cove %d, spu %d); labelled V at t0: cove %.3e vs tef2 %.3e, spu %.3e'
          % (rel, len(w), (origin == 'cove').sum(), m.sum(), w[origin == 'cove'].sum(),
             V_cove_t.reindex([t[0]], method='nearest').values[0], w[m].sum()))

    jj, ii, bad = cell_of(x, y)
    in_cove = is_cove[jj, ii] & ~bad
    in_spu = is_spu[jj, ii] & ~bad
    Vc = V_cove_t.reindex(t, method='nearest').values
    Vs = V_spu_t.reindex(t, method='nearest').values
    out = {}
    for box, inb, V in [('cove', in_cove, Vc), ('spu', in_spu, Vs)]:
        for o in ['cove', 'spu']:
            out[(box, o)] = (inb[:, origin == o] * w[origin == o]).sum(axis=1) / V
    df = pd.DataFrame({'%s_from_%s' % k: v for k, v in out.items()}, index=t)
    # tidal-day (24.84 h) running mean, then sample daily
    dfl = df.rolling(25, center=True, min_periods=25).mean()
    dfl['days'] = (dfl.index - t[0]).total_seconds() / 86400
    curves[rel] = dfl
    for box in ['cove', 'spu']:
        lab = dfl['%s_from_cove' % box] + dfl['%s_from_spu' % box]
        unl = 1 - lab
        r = dict(release=rel, box=box)
        for d in [1, 3, 7, 14, 30, 45, 59]:
            k = np.argmin(np.abs(dfl.days.values - d))
            r['unlab_d%d' % d] = unl.values[k]
        if box == 'cove':
            # NEW water = cove water that was not in the cove at release; its
            # upper-Saratoga share is the labelled part of the cove's source water
            new = 1 - dfl['cove_from_cove']
            share = dfl['cove_from_spu'] / new
            for d in [1, 3, 7, 14, 21, 30]:
                k = np.argmin(np.abs(dfl.days.values - d))
                r['new_d%d' % d] = new.values[k]
                r['spu_share_of_new_d%d' % d] = share.values[k]
        for f in [0.1, 0.25, 0.5]:
            hit = np.where(unl.values >= f)[0]
            r['days_to_unlab_%d%%' % int(100 * f)] = dfl.days.values[hit[0]] if len(hit) else np.nan
        rows.append(r)

res = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print('\nUnlabelled fraction (tidal-day mean) = water that was outside cove + upper Saratoga at release')
print(res.round(2).to_string(index=False))
res.to_csv(out_dir / 'pcsrc_spu_unlabelled.csv', index=False)
print('\nCove SOURCE water: new = not in the cove at release; share = fraction of new water that was in upper Saratoga')
cols = ['release'] + [c for c in res.columns if c.startswith('new_d') or c.startswith('spu_share')]
print(res[res.box == 'cove'][cols].round(2).to_string(index=False))
for rel in RELS:
    curves[rel].to_csv(out_dir / ('pcsrc_spu_curves_%s.csv' % rel))

# ---- TEF inflow to pc_lp_p through sp_mid (+1) and skagit_sp (-1)
def qin(sn, sign):
    b = xr.open_dataset(tef_dir / 'bulk_avg_2024.01.01_2025.12.31' / (sn + '.nc'))
    q = b.q.values * sign
    return pd.Series(np.nansum(np.where(q > 0, q, 0), axis=1), index=pd.to_datetime(b.time.values).normalize())

Q = pd.DataFrame({'sp_mid': qin('sp_mid', 1), 'skagit_sp': qin('skagit_sp', -1)})
Q['frac_sp_mid'] = Q.sp_mid / (Q.sp_mid + Q.skagit_sp)
print('\nTEF Qin into upper Saratoga (m3/s), mean over each 60-d release window:')
for rel in RELS:
    t0 = pd.Timestamp(rel.replace('.', '-'))
    q = Q[(Q.index >= t0) & (Q.index < t0 + pd.Timedelta(days=60))].mean()
    print('  %s  sp_mid %.0f  skagit_sp %.0f  sp_mid share %.2f' % (rel, q.sp_mid, q.skagit_sp, q.frac_sp_mid))
qm = Q.resample('MS').mean()
print('monthly sp_mid share of inflow (2024-2025): min %.2f max %.2f mean %.2f'
      % (qm.frac_sp_mid.min(), qm.frac_sp_mid.max(), qm.frac_sp_mid.mean()))
Q.to_csv(out_dir / 'pcsrc_spu_tef_inflow_daily.csv')

# ---- figure: composition of cove and upper-Saratoga water vs days since release
plt.close('all')
fs = 10
plt.rc('font', size=fs)
C = {'cove': '#52514e', 'spu': '#2a78d6', 'unl': '#d9d8d4'}
fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
for r, rel in enumerate(RELS):
    dfl = curves[rel].dropna()
    for c, box in enumerate(['cove', 'spu']):
        ax = axs[r, c]
        a = dfl['%s_from_cove' % box].values; b = dfl['%s_from_spu' % box].values
        d = dfl.days.values
        ax.fill_between(d, 0, a, color=C['cove'], lw=0)
        ax.fill_between(d, a, a + b, color=C['spu'], lw=0)
        ax.fill_between(d, a + b, 1, color=C['unl'], lw=0)
        ax.axhline(1, color='#0b0b0b', lw=0.6)
        ax.set_ylim(0, 1.05); ax.set_xlim(0, 60)
        ax.grid(lw=0.3, alpha=0.5)
        ax.set_title('%s release - water inside %s' % (rel, 'Penn Cove' if box == 'cove' else 'upper Saratoga (pc_lp_p)'),
                     fontsize=fs, loc='left')
        if r == 1: ax.set_xlabel('days since release')
        if c == 0: ax.set_ylabel('fraction of box volume')
from matplotlib.patches import Patch
axs[0, 1].legend(handles=[Patch(fc=C['cove'], label='was in Penn Cove at release'),
                          Patch(fc=C['spu'], label='was in upper Saratoga at release'),
                          Patch(fc=C['unl'], label='unlabelled (outside both at release)')],
                 loc='lower right', fontsize=fs - 1, framealpha=0.95)
fig.suptitle('pcret (cove + upper Saratoga seeded), tidal-day means', fontsize=fs + 1, x=0.07, ha='left')
fig.tight_layout()
fn = out_dir / 'pcsrc_spu_feasibility.png'
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % fn)
