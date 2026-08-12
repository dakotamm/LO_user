"""
Retention of inner Penn Cove BOTTOM water, high-DO week vs low-DO week.

THE EXPERIMENT
Two pcbot releases (LO_user/tracker2/experiments.py), identical in space --
the same 1573 particles at the same cells and the same 0.25-4.75 m above the
bed inside pc_cp_m -- launched into two 14-day windows chosen by
20260811_pc_matched_weeks.py to have nearly the same tide:

  hiDO  2024.02.25 13:00 UTC   Penn Cove bottom DO 8.50 mg/L
  loDO  2024.09.03 00:00 UTC   Penn Cove bottom DO 1.23 mg/L

Both start at a higher high water. In the tracker output the two zeta series
correlate at r = 0.98 with an RMS difference of 0.185 m, so the tidal forcing
really is close to common-mode and a difference between the two curves below
is seasonal, not tidal. That is the whole reason for the matching -- see
20260811_pc_matched_weeks.py for how the pair was picked.

The mirror-image experiment -- tide varying, season held fixed -- is a
separate pair of releases with its own script,
20260811_pcbot_retention_springneap.py.

WHY THE RECORDS ARE TRIMMED
tracker2 counts -dtt in calendar days from the START DAY, so a release at
-sh 13 loses those 13 hours off the far end: hiDO returns 324 frames and loDO
337. Comparing curves of different length would put a spurious difference in
every end-of-run number, so both are cut to the common 324 frames (323 h,
13.46 d). Nothing is lost that the tidal match supported anyway.

WHAT IS MEASURED
  still inside   fraction of the cohort in the region right now. Rises and
                 falls with the tide, because a particle can be advected out
                 on an ebb and brought back on the flood.
  never left     fraction that has not left even once (running minimum). The
                 gap between the two IS the tidal reversibility -- water that
                 leaves and comes back is not flushed, and reporting only
                 "still inside" would call it retained while reporting only
                 "never left" would call it lost.
  in bottom      fraction still within HAB_LAYER of the bed, WHEREVER it is --
                 including out in Saratoga Passage. Read it as "still near a
                 bed", not "still in the cove's bottom layer"; the
                 inner+bottom curve is the one that requires both at once and
                 is what an inner-cove DO budget needs.
                 This is the measure that should feel the season most: the
                 same tide against weak winter stratification should mix
                 bottom water upward faster than against a summer pycnocline,
                 and vertical escape is a loss pathway that a map of
                 horizontal position cannot see.

Regions are the tef2 wb1_pc1 segments, the same definition the release used:
  inner cove  pc_cp_m                        (landward of pc_cp)
  whole cove  pc_cp_m + pc_cp_p + pc_lp_m    (landward of pc_lp)

run 20260811_pcbot_retention.py
run 20260811_pcbot_retention.py -hab_layer 3
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-hab_layer', type=float, default=5.0,
               help='height above bed defining "still bottom water" [m]')
p.add_argument('-year', type=int, default=2024, choices=[2024, 2025],
               help='which matched pair to analyse; 2024 and 2025 are two '
                    'independent realisations of the same experiment')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pcbot_retention'
Lfun.make_dir(out_dir)

# Two independent realisations of the same experiment. The 2024 pair was
# selected on RMS(ssh') and the 2025 pair on qprism + diurnal envelope after
# 20260811_pc_matched_weeks.py was revised; the 2024 pair survived the revision
# unchanged because it was already matched to +0.1% in qprism, so the two are
# comparable and the pair of pairs is a replication test, not a methods change.
# 2025 also carries the full matched window: it was run at -dtt 15 so the
# -sh 14 start no longer eats the end (2024 lost 13 h and trims to 13.46 d).
RUNSETS = {
    2024: [('hiDO', 'pcbot_3d_sh13_hiDO', '#4565e8', 'high-DO  2024.02.25'),
           ('loDO', 'pcbot_3d_loDO', '#e8455e', 'low-DO   2024.09.03')],
    2025: [('hiDO', 'pcbot_3d_sh14_hiDO_2025', '#4565e8', 'high-DO  2025.02.16'),
           ('loDO', 'pcbot_3d_sh2_loDO_2025', '#e8455e', 'low-DO   2025.08.27')],
}
RUNS = RUNSETS[args.year]
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)

# ------------------------------------------------------------------ setup ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape

tef2 = Ldir['LOo'] / 'extract' / 'tef2'
seg = pickle.load(open(sorted(tef2.glob('seg_info_dict_wb1_pc1_*.p'))[0], 'rb'))
seg_mask = {}
for s in seg:
    m = np.zeros((NR, NC), dtype=bool)
    a = np.array(seg[s]['ji_list'])
    m[a[:, 0], a[:, 1]] = True
    seg_mask[s] = m
REGION = {'inner cove': seg_mask['pc_cp_m'],
          'whole cove': seg_mask['pc_cp_m'] | seg_mask['pc_cp_p']
          | seg_mask['pc_lp_m']}


def ji_of(plon, plat):
    """Nearest rho indices on this plaid grid; NaN positions map to -1."""
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.full(plon.shape, -1, dtype=int)
    j = np.full(plon.shape, -1, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    return j, i, ok


def efold(frac, hours):
    """Hours until the curve first drops below 1/e, in days; NaN if never."""
    k = np.where(frac < 1 / np.e)[0]
    return hours[k[0]] / 24 if len(k) else np.nan


def half(frac, hours):
    k = np.where(frac < 0.5)[0]
    return hours[k[0]] / 24 if len(k) else np.nan


# ------------------------------------------------------------------- load ---
R = {}
nmin = None
for tag, dname, color, label in RUNS:
    fn = sorted((trk / dname).glob('release_*.nc'))[0]
    d = xr.open_dataset(fn)
    R[tag] = dict(lon=d.lon.values, lat=d.lat.values, cs=d.cs.values,
                  h=d.h.values, zeta=d.zeta.values, z=d.z.values,
                  salt=d.salt.values, ot=pd.to_datetime(d.ot.values, unit='s'),
                  color=color, label=label, fn=fn.name)
    d.close()
    nmin = d.sizes['Time'] if nmin is None else min(nmin, d.sizes['Time'])
print('common record: %d frames = %d h = %.2f d' % (nmin, nmin - 1, (nmin - 1) / 24))

hours = np.arange(nmin)
for tag in R:
    r = R[tag]
    for k in ['lon', 'lat', 'cs', 'h', 'zeta', 'z', 'salt']:
        r[k] = r[k][:nmin, :]
    r['ot'] = r['ot'][:nmin]
    # height above the bed, carried through the run. h in the tracker output is
    # the still-water depth at the particle, so hab = (cs + 1) * (h + zeta)
    # keeps the free surface in it rather than drifting by up to a metre of
    # tidal range -- which at hab = 0.25 m would be a large relative error.
    r['hab'] = (r['cs'] + 1) * (r['h'] + r['zeta'])
    j, i, ok = ji_of(r['lon'], r['lat'])
    r['in'] = {}
    for rn, mask in REGION.items():
        a = np.zeros(r['lon'].shape, dtype=bool)
        a[ok] = mask[j[ok], i[ok]]
        r['in'][rn] = a
    r['bot'] = r['hab'] <= args.hab_layer
    print('%s  %s  NP %d  %s -> %s'
          % (tag, r['fn'], r['lon'].shape[1], r['ot'][0], r['ot'][-1]))

# tidal match, restated from the tracker output itself
za = R['hiDO']['zeta'].mean(axis=1); zb = R['loDO']['zeta'].mean(axis=1)
za = za - za.mean(); zb = zb - zb.mean()
print('tidal match in the runs: r = %.3f, RMS = %.3f m'
      % (np.corrcoef(za, zb)[0, 1], np.sqrt(((za - zb) ** 2).mean())))

# --------------------------------------------------------------- measures ---
for tag in R:
    r = R[tag]
    r['curve'] = {}
    for rn in REGION:
        a = r['in'][rn]
        r['curve'][(rn, 'still')] = a.mean(axis=1)
        r['curve'][(rn, 'never')] = np.minimum.accumulate(a, axis=0).mean(axis=1)
    r['curve'][('bottom', 'still')] = r['bot'].mean(axis=1)
    r['curve'][('bottom', 'never')] = np.minimum.accumulate(
        r['bot'], axis=0).mean(axis=1)
    # bottom water still in the inner cove: both conditions at once, which is
    # the quantity a DO budget for the inner cove actually needs
    both = r['in']['inner cove'] & r['bot']
    r['curve'][('inner+bottom', 'still')] = both.mean(axis=1)
    r['curve'][('inner+bottom', 'never')] = np.minimum.accumulate(
        both, axis=0).mean(axis=1)

rows = []
for what in ['inner cove', 'whole cove', 'bottom', 'inner+bottom']:
    for tag in ['hiDO', 'loDO']:
        r = R[tag]
        s = r['curve'][(what, 'still')]; n = r['curve'][(what, 'never')]
        rows.append(dict(what=what, run=tag, efold_still=efold(s, hours),
                         efold_never=efold(n, hours), half_still=half(s, hours),
                         final_still=s[-1], final_never=n[-1],
                         reentry=s[-1] - n[-1]))
T = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print('\n---- retention (days to 1/e; fractions at %.1f d) ----' % ((nmin - 1) / 24))
print(T.to_string(index=False, float_format=lambda x: '%.3f' % x))
T.to_csv(out_dir / ('pcbot_retention_%d.csv' % args.year), index=False)

# by starting height above the bed: does the deepest water behave differently?
print('\n---- inner-cove retention by starting height above bed ----')
print('%-10s %6s %10s %10s %10s' % ('hab bin', 'n', 'hiDO e-fold', 'loDO e-fold',
                                    'hiDO-loDO'))
bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
hab0 = R['hiDO']['hab'][0, :]
brows = []
for lo, hi in bins:
    m = (hab0 >= lo) & (hab0 < hi)
    if m.sum() == 0:
        continue
    e = {}
    for tag in ['hiDO', 'loDO']:
        e[tag] = efold(R[tag]['in']['inner cove'][:, m].mean(axis=1), hours)
    print('%-10s %6d %10.2f %10.2f %10.2f'
          % ('%g-%g m' % (lo, hi), m.sum(), e['hiDO'], e['loDO'],
             e['hiDO'] - e['loDO']))
    brows.append(dict(hab_lo=lo, hab_hi=hi, n=int(m.sum()),
                      efold_hiDO=e['hiDO'], efold_loDO=e['loDO']))
pd.DataFrame(brows).to_csv(
    out_dir / ('pcbot_retention_by_hab_%d.csv' % args.year), index=False)

# ----------------------------------------------------------------- figure ---
days = hours / 24
fig, axs = plt.subplots(3, 2, figsize=(13, 11), sharex=True)

panels = [('inner cove', 'still inside the inner cove (landward of pc_cp)'),
          ('whole cove', 'still inside the cove (landward of pc_lp)'),
          ('bottom', 'still within %.0f m of the bed (anywhere)'
           % args.hab_layer),
          ('inner+bottom', 'still inner-cove AND bottom water')]
for ax, (what, title) in zip(axs.flatten()[:4], panels):
    for tag, _, color, label in RUNS:
        ax.plot(days, R[tag]['curve'][(what, 'still')], color=color, lw=1.4,
                label=label + '  (still)')
        ax.plot(days, R[tag]['curve'][(what, 'never')], color=color, lw=1.0,
                ls='--', label=label + '  (never left)')
    ax.axhline(1 / np.e, color='0.5', lw=0.8, ls=':')
    ax.set_ylim(0, 1.02); ax.set_ylabel('fraction of cohort')
    ax.set_title(title, fontsize=10); ax.grid(**GRID)
axs[0, 0].legend(fontsize=7, loc='upper right')

ax = axs[2, 0]
# FRACTIONAL height in the column, not metres above the bed. Metres are
# unreadable here: once a particle leaves the cove for 30-100 m of Saratoga
# Passage its height above the bed can grow to tens of metres without it
# having risen at all relative to the water column, so a mean in metres
# mostly plots where the cohort went, not how high in the column it sits.
# cs + 1 is 0 at the bed and 1 at the surface wherever the particle is.
for tag, _, color, label in RUNS:
    ax.plot(days, np.nanmean(R[tag]['cs'] + 1, axis=1), color=color, lw=1.4,
            label=label + '  (all)')
    ins = R[tag]['in']['whole cove']
    fr = np.where(ins, R[tag]['cs'] + 1, np.nan)
    with np.errstate(invalid='ignore'):
        ax.plot(days, np.nanmean(fr, axis=1), color=color, lw=1.0, ls='--',
                label=label + '  (still in cove)')
ax.set_ylim(0, 1)
ax.set_ylabel('height in the column [0 = bed, 1 = surface]')
ax.set_title('how far the cohort climbs off the bed', fontsize=10)
ax.set_xlabel('days from release'); ax.grid(**GRID)
ax.legend(fontsize=7, loc='lower right')

ax = axs[2, 1]
for tag, _, color, label in RUNS:
    z = R[tag]['zeta'].mean(axis=1)
    ax.plot(days, z - z.mean(), color=color, lw=0.9)
ax.set_ylabel("zeta' [m]")
ax.set_title('the matched tide as the two runs actually saw it (r = %.3f)'
             % np.corrcoef(za, zb)[0, 1], fontsize=10)
ax.set_xlabel('days from release'); ax.grid(**GRID)

fig.suptitle('%s: inner Penn Cove bottom-water retention, tide-matched weeks (%d)'
             % (args.gtx, args.year), fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcbot_retention_%d.png' % args.year)
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % fn_out)
