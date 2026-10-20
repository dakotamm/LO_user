"""
Is DO lower around the loopers at the landmark tau than around the
non-loopers that are still in the cove at the same time?

The tracker does not carry oxygen, so DO is SAMPLED from the model:
  DO_tau   25-h mean of the pc_cove_o2 box oxygen (hourly, 2025) over the tidal
           day centred on tau, at the particle's tidal-day-mean position and
           height at tau (20261013_pcmap_tidalday_reduce.py): nearest wet box
           cell, nearest s_rho layer to its mean cs. Tidally averaged DO at the
           tidally averaged position -- not the DO the particle saw hour by hour.
  DO0      DO at the particle's release position and hour (20261005_pcmap_reduce.py)
  dDO      DO_tau - DO0
Landmark sample as in 20261013_pcmap_trapping_attrib.py (pcmap_tidalday.landmarks):
particles still inside at tau, looped = |loops| >= 1 on the tidal-mean path up to tau.

Loopers sit in the inner cove, where DO differs from the rest, so the
comparison that matters is within the SAME release AND the same place at tau:
  diff_place  looper minus non-looper median DO_tau within release x quadrant
              at tau x current half (cs_tau >= -0.5), averaged over strata
              weighted by the number of loopers (strata with >= 3 of each)
  diff_rel    the same within release only
  ddiff_place the same as diff_place for dDO

Output: LO_output/DM_outs/20261013_pcmap_looping_do_tau/
  pcmap_looping_do_tau.png   median DO_tau of loopers vs non-loopers by
      quadrant x current half at tau; rows = all releases / Aug-Nov,
      columns = tau
  pcmap_looping_do_tau.csv   the medians and the paired differences

  python 20261013_pcmap_looping_do_tau.py
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from lo_tools import Lfun
from pcmap_regions import regions, QNAMES
from pcmap_tidalday import load, dist_to_mouth, landmarks

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-taus', default='3,5,7', help='landmark times [d]')
p.add_argument('-dt', type=float, default=3)
p.add_argument('-dmin', type=float, default=100)
p.add_argument('-box', default='pc_cove_o2_2024.01.01_2025.12.31')
args = p.parse_args()
TAUS = [float(t) for t in args.taus.split(',')]

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcmap_looping_do_tau'
Lfun.make_dir(out_dir)
LAT0 = 48.23

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
DX, DY = 1 / g.pm.values, 1 / g.pn.values
g.close()
lon_ax, lat_ax = lon[0], lat[:, 0]
R_ = regions(Ldir, lon, lat)
cove, QUAD = R_['cove'], R_['QUAD']
dist = dist_to_mouth(cove, DX, DY, np.flatnonzero(cove.any(axis=0))[-1])
jn, in_ = distance_transform_edt(~cove, return_distances=False, return_indices=True)
LON_MOUTH = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()

REL, T = load(Ldir, args.gtx)
A = landmarks(REL, TAUS, args.dt, args.dmin, LON_MOUTH, dist[jn, in_], lon_ax, lat_ax, quad_f=QUAD[jn, in_], lat0=LAT0)
A['surf_tau'] = A.cs_tau >= -0.5
A['season'] = [REL[r]['season'] for r in A.rel.values]
print('%d releases, %d particle-landmarks' % (len(REL), len(A)))

# DO0 from the main reduce, same particle order
rdir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
DO0 = {r: pd.read_pickle(next(rdir.glob(D['name'] + '__*.p')))['P'].DO0.values for r, D in enumerate(REL)}
A['DO0'] = [DO0[r][p_] for r, p_ in zip(A.rel.values, A.p.values)]

# --------------------------------------------------------- DO_tau from the box ---
ds = xr.open_dataset(Ldir['LOo'] / 'extract' / args.gtx / 'box' / (args.box + '.nc'))
blon, blat = ds.lon_rho.values, ds.lat_rho.values
bwet = ds.mask_rho.values == 1
bjn, bin_ = distance_transform_edt(~bwet, return_distances=False, return_indices=True)
NZ = ds.sizes['s_rho']
ot = pd.to_datetime(ds.ocean_time.values)
it0 = np.searchsorted(ot, pd.Timestamp('2025-01-01'))
it1 = np.searchsorted(ot, pd.Timestamp('2026-01-01'))
print('reading oxygen %s to %s' % (ot[it0], ot[it1 - 1]))
OX = ds.oxygen.isel(ocean_time=slice(it0, it1)).values * 32 / 1000 # mg/L, (T, Z, J, I)
ot = ot[it0:it1]
A['DO_tau'] = np.nan
for (r, tau), a in A.groupby(['rel', 'tau']):
    tc = REL[r]['meta']['t0'] + pd.Timedelta(hours=float(a.t_tau_h.iloc[0]))
    k = np.searchsorted(ot, tc)
    assert ot[k] == tc
    F = OX[k - 12:k + 13].mean(axis=0)
    i = np.clip(np.round((a.lon_tau.values - blon[0, 0]) / (blon[0, 1] - blon[0, 0])).astype(int), 0, blon.shape[1] - 1)
    j = np.clip(np.round((a.lat_tau.values - blat[0, 0]) / (blat[1, 0] - blat[0, 0])).astype(int), 0, blat.shape[0] - 1)
    j, i = bjn[j, i], bin_[j, i]
    kz = np.clip(np.floor((a.cs_tau.values + 1) * NZ).astype(int), 0, NZ - 1)
    A.loc[a.index, 'DO_tau'] = F[kz, j, i]
A['dDO'] = A.DO_tau - A.DO0
print('DO_tau missing %.4f' % A.DO_tau.isna().mean())


def paired(a, var, by):
    """Looper-minus-non-looper median of var within strata 'by', weighted by loopers."""
    d, w = [], []
    for _, s in a.groupby(by):
        nl, nn = s.looped.sum(), (~s.looped).sum()
        if nl >= 3 and nn >= 3:
            d.append(s[var][s.looped].median() - s[var][~s.looped].median())
            w.append(nl)
    return (np.average(d, weights=w) if w else np.nan), len(w)


rows = []
for per in ['all', 'Aug-Nov']:
    for tau in TAUS:
        for half, hs in [('surf', True), ('bot', False)]:
            a = A[(A.tau == tau) & (A.surf0 == hs) & ((A.season == per) if per != 'all' else True)]
            dp, ns = paired(a, 'DO_tau', ['rel', 'quad_tau', 'surf_tau'])
            dr, nr = paired(a, 'DO_tau', ['rel'])
            dd, _ = paired(a, 'dDO', ['rel', 'quad_tau', 'surf_tau'])
            rows.append(dict(period=per, tau_d=tau, start_half=half, n=len(a), n_loopers=int(a.looped.sum()),
                             med_DO_tau_loopers=a.DO_tau[a.looped].median(), med_DO_tau_nonloopers=a.DO_tau[~a.looped].median(),
                             diff_rel=dr, n_rel=nr, diff_place=dp, n_strata=ns,
                             med_dDO_loopers=a.dDO[a.looped].median(), med_dDO_nonloopers=a.dDO[~a.looped].median(), ddiff_place=dd))
R = pd.DataFrame(rows)
R.to_csv(out_dir / 'pcmap_looping_do_tau.csv', index=False, float_format='%.4g')
pd.set_option('display.width', 240)
print(R.round(2).to_string(index=False))

# ------------------------------------------------------------------ figure ---
labs = [(q, h) for q in range(4) for h in [True, False]]
fig, axs = plt.subplots(2, len(TAUS), figsize=(6 * len(TAUS), 8), layout='constrained', sharey='row', squeeze=False)
for rr, per in enumerate(['all', 'Aug-Nov']):
    for cc, tau in enumerate(TAUS):
        ax = axs[rr, cc]
        a = A[(A.tau == tau) & ((A.season == per) if per != 'all' else True)]
        for k, (lp, colr, nm) in enumerate([(True, 'tab:blue', 'looped'), (False, '0.6', 'did not loop')]):
            v = [a.DO_tau[(a.quad_tau == q) & (a.surf_tau == h) & (a.looped == lp)].median() for q, h in labs]
            ax.bar(np.arange(len(labs)) + (k - 0.5) * 0.38, v, 0.38, color=colr, label=nm)
        ax.set_xticks(range(len(labs)), ['%s\n%s' % (QNAMES[q], 'surf' if h else 'bot') for q, h in labs], fontsize=8)
        ax.grid(axis='y', color='lightgray', linestyle='--', alpha=0.5)
        r_ = R[(R.period == per) & (R.tau_d == tau)]
        ax.set_title('%s, tau = %g d\nsame release & place, looper - non: surf starters %+.2f, bot starters %+.2f mg/L'
                     % ('All releases' if per == 'all' else 'Aug-Nov', tau, r_[r_.start_half == 'surf'].diff_place.iloc[0],
                        r_[r_.start_half == 'bot'].diff_place.iloc[0]), fontsize=10)
        if cc == 0:
            ax.set_ylabel('Median DO at tau [mg/L]')
axs[0, 0].legend(fontsize=9)
fig.suptitle('DO around particles still in the cove at tau (25-h mean DO at their tidal-day-mean position), by where they are at tau')
fn = out_dir / 'pcmap_looping_do_tau.png'
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))
