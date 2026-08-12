"""
Retention of inner Penn Cove BOTTOM water, NEAP week vs SPRING week.

THE EXPERIMENT
Two pcbot releases (LO_user/tracker2/experiments.py), identical in space --
the same particles at the same cells and the same 0.25-4.75 m above the bed
inside pc_cp_m -- launched into two adjacent August 2024 windows picked by
20260811_pc_springneap_weeks.py:

  neap    2024.08.08 02:00 UTC   envelope 1.40 m, mean daily range 2.71 m
  spring  2024.08.14 05:00 UTC   envelope 1.95 m, mean daily range 3.65 m

This is the MIRROR IMAGE of 20260811_pcbot_retention.py. That script matched
the tide and let the season vary, so a difference between its curves was
seasonal. Here the season is held as fixed as it can be -- the windows are
seven days apart, bottom DO 2.73 vs 2.62 mg/L, dstrat 1.95 vs 1.99 g/kg,
Skagit 227 vs 210 m3/s -- and the TIDE is the variable, x1.40 in envelope.
A difference between these curves is tidal.

Read the two scripts together: one gives the seasonal sensitivity of
inner-cove bottom-water residence time, the other the tidal sensitivity, and
only the pair says which control dominates.

WHAT TO EXPECT, AND WHAT WOULD BE INTERESTING
Naively a spring tide flushes harder and retention should drop. Two things
could break that. First, a bigger tidal excursion moves water out AND back, so
spring should widen the gap between "still inside" and "never left" without
necessarily moving "never left" much -- that gap is measured here for exactly
this reason. Second, spring tides mix vertically as well as horizontally, and
in Penn Cove the winter-vs-September comparison found VERTICAL escape to be
the loss pathway that mattered; if the spring cohort climbs off the bed faster
than the neap one, the tide is acting through mixing rather than through
advection out of the cove, and the two panels that show height in the column
are where that appears.

WHY THE RECORDS ARE TRIMMED
tracker2 counts -dtt in calendar days from the START DAY, so a release at -sh
h loses those h hours off the far end: at -dtt 7 the neap run (-sh 2) returns
more frames than the spring run (-sh 5). Comparing curves of different length
would put a spurious difference into every end-of-run number, so both are cut
to the common length.

WHAT IS MEASURED
  still inside   fraction of the cohort in the region right now. Rises and
                 falls with the tide, because a particle can be advected out
                 on an ebb and brought back on the flood. At spring this
                 oscillation is larger, which is a real part of the answer,
                 not noise to be smoothed away.
  never left     fraction that has not left even once (running minimum). The
                 gap between the two IS the tidal reversibility -- water that
                 leaves and comes back is not flushed. This experiment should
                 move the gap more than either curve alone.
  in bottom      fraction still within HAB_LAYER of the bed, WHEREVER it is --
                 including out in Saratoga Passage. Read it as "still near a
                 bed", not "still in the cove's bottom layer"; the
                 inner+bottom curve requires both at once and is what an
                 inner-cove DO budget needs.

Regions are the tef2 wb1_pc1 segments, the same definition the release used:
  inner cove  pc_cp_m                        (landward of pc_cp)
  whole cove  pc_cp_m + pc_cp_p + pc_lp_m    (landward of pc_lp)

run 20260811_pcbot_retention_springneap.py
run 20260811_pcbot_retention_springneap.py -hab_layer 3
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
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pcbot_retention_springneap'
Lfun.make_dir(out_dir)

# tag, tracker output directory, colour, legend label. tracker.py builds the
# directory name as <exp>_3d[_sh<h>]_<sub_tag>, and the _sh part is present
# only when the release hour is non-zero -- both of these have one.
RUNS = [('neap', 'pcbot_3d_sh2_neap', '#2b8cbe', 'neap    2024.08.08'),
        ('spring', 'pcbot_3d_sh5_spring', '#d94801', 'spring  2024.08.14')]
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
    # tidal range -- which matters more here than in the matched-week pair,
    # because the two runs differ in tidal range by design.
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

# The forcing contrast, restated from the tracker output itself. This is the
# opposite of the check in 20260811_pcbot_retention.py: there a HIGH r and a
# LOW RMS difference proved the tide was common-mode, here a high r with a
# large std ratio proves the two runs are in semidiurnal phase (both released
# at a higher high water) but differ in amplitude, which is the intended
# design. A low r would mean the phase lock failed and the early curves are
# not comparable.
zn = R['neap']['zeta'].mean(axis=1); zs = R['spring']['zeta'].mean(axis=1)
zn = zn - zn.mean(); zs = zs - zs.mean()
ratio = zs.std() / zn.std()
print('tide in the runs: std %.3f (neap) vs %.3f m (spring), ratio x%.2f; '
      'r = %.3f' % (zn.std(), zs.std(), ratio, np.corrcoef(zn, zs)[0, 1]))

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
    for tag in ['neap', 'spring']:
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
T.to_csv(out_dir / 'pcbot_retention_springneap.csv', index=False)

# The headline contrast, spring minus neap, alongside the tidal ratio that
# produced it. Quoting the difference without the forcing ratio next to it
# invites reading a 0.1 shift as though it came from a doubled tide.
print('\n---- spring minus neap (tidal forcing ratio x%.2f) ----' % ratio)
for what in ['inner cove', 'whole cove', 'bottom', 'inner+bottom']:
    a = T[(T.what == what) & (T.run == 'neap')].iloc[0]
    b = T[(T.what == what) & (T.run == 'spring')].iloc[0]
    print('%-13s e-fold %5.2f -> %5.2f d (%+.2f) | final still %.3f -> %.3f '
          '(%+.3f) | reentry gap %.3f -> %.3f (%+.3f)'
          % (what, a.efold_still, b.efold_still, b.efold_still - a.efold_still,
             a.final_still, b.final_still, b.final_still - a.final_still,
             a.reentry, b.reentry, b.reentry - a.reentry))

# The tidal oscillation of "still inside" itself: at spring the cohort should
# swing in and out of the cove harder each cycle even if the net loss is
# similar. Std of the detrended curve is a compact measure of that swing, and
# separates "flushed" from "sloshed" -- the distinction this pair exists for.
print('\n---- tidal sloshing: std of "still inside" about its own trend ----')
for what in ['inner cove', 'whole cove']:
    out = []
    for tag in ['neap', 'spring']:
        s = R[tag]['curve'][(what, 'still')]
        trend = np.poly1d(np.polyfit(hours, s, 3))(hours)
        out.append((s - trend).std())
    print('  %-12s neap %.4f   spring %.4f   x%.2f'
          % (what, out[0], out[1], out[1] / max(out[0], 1e-9)))

# by starting height above the bed: does the deepest water behave differently?
print('\n---- inner-cove retention by starting height above bed ----')
print('%-10s %6s %12s %12s %12s'
      % ('hab bin', 'n', 'neap e-fold', 'spring e-fold', 'spring-neap'))
bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
hab0 = R['neap']['hab'][0, :]
brows = []
for lo, hi in bins:
    m = (hab0 >= lo) & (hab0 < hi)
    if m.sum() == 0:
        continue
    e = {}
    for tag in ['neap', 'spring']:
        e[tag] = efold(R[tag]['in']['inner cove'][:, m].mean(axis=1), hours)
    print('%-10s %6d %12.2f %12.2f %12.2f'
          % ('%g-%g m' % (lo, hi), m.sum(), e['neap'], e['spring'],
             e['spring'] - e['neap']))
    brows.append(dict(hab_lo=lo, hab_hi=hi, n=int(m.sum()),
                      efold_neap=e['neap'], efold_spring=e['spring']))
pd.DataFrame(brows).to_csv(
    out_dir / 'pcbot_retention_springneap_by_hab.csv', index=False)

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
# This is the panel that says whether a spring tide works by mixing.
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
ax.set_title('the spring-neap contrast as the runs saw it (std x%.2f, r = %.3f)'
             % (ratio, np.corrcoef(zn, zs)[0, 1]), fontsize=10)
ax.set_xlabel('days from release'); ax.grid(**GRID)

fig.suptitle('%s: inner Penn Cove bottom-water retention, neap vs spring week '
             '(both August 2024)' % args.gtx, fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcbot_retention_springneap.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % fn_out)
