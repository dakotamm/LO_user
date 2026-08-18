"""
The two pcbot experiments side by side: season varying at fixed tide, and tide
varying at fixed season.

  column 1  SEASONAL     tide matched, season free
              hiDO  2025.02.16   Penn Cove bottom DO 8.39 mg/L
              loDO  2025.08.27   Penn Cove bottom DO 1.31 mg/L
            windows chosen by 20260811_pc_matched_weeks.py on qprism (-0.3%)
            and the diurnal envelope (+8.1%)

  column 2  SPRING-NEAP  season held, tide free
              neap    2025.07.31
              spring  2025.08.09
            windows chosen by 20260811_pc_springneap_weeks.py, centred on the
            qprism extremum, ~9 days apart so DO, stratification and river are
            as close to common-mode as adjacency allows

  row 1     what fraction of the cohort is still inside the inner cove
  row 2     how far up the water column the cohort sits

Both experiments release the identical 1573 particles in the identical cells
(checked by 20260811_pcbot_release_map.py), so every curve in the figure starts
from the same cohort and the four panels are on one footing.

THE COLUMNS SHARE ONE X LIMIT
The seasonal runs are 14.4 d and the spring-neap runs 7.8 d, and every panel is
drawn to the longer of the two. The spring-neap pair is deliberately short: its
two windows are ADJACENT, so a longer run walks the neap cohort into the
following spring and the contrast it was built to show collapses. Its curves
therefore stop half way across the right-hand panels, and that gap is a fact
about the experiment rather than missing data. The alternative -- giving each
column its own limit -- would put the two on different days-per-inch and make
one set of curves read steeper than the other for no physical reason.

ELEVATION IS FRACTIONAL, NOT METRES. Height above the bed in metres is
unreadable across these runs: once a particle leaves the cove for 30-100 m of
Saratoga Passage its height above the bed grows to tens of metres without it
having risen at all relative to the water column, so a mean in metres plots
where the cohort went rather than how high it sits. cs + 1 is 0 at the bed and
1 at the surface wherever the particle is.

run 20260811_pcbot_retention_grid.py
run 20260811_pcbot_retention_grid.py -year 2024
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
p.add_argument('-year', type=int, default=2025, choices=[2024, 2025],
               help='which seasonal pair fills the left column; 2025 is the '
                    'one that shares a year with the spring-neap pair')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pcbot_retention_grid'
Lfun.make_dir(out_dir)

FS = 15
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)

# Assigned by COLUMN, not by run. The first pass gave the seasonal pair blue/red
# and the spring-neap pair a second blue and a second orange, so the four curves
# read as two blues and two warms and the eye grouped them across columns
# instead of within one. Here the seasonal pair is chromatic (blue = winter,
# red = summer, which the reader already has an intuition for) and the
# spring-neap pair is achromatic, so a glance says which experiment a curve
# belongs to before you read the legend. Spring takes black rather than grey:
# it is the stronger tide and the one whose curve carries the result.
C_HI, C_LO = '#0072B2', '#D62728'        # blue, red
C_NEAP, C_SPRING = '#8C8C8C', '#000000'  # grey, black

# Legend labels carry no dates -- the release dates are in the docstring and in
# the printout, and on a four-curve figure they were the longest thing in the
# panel for the least return.
#
# winter / summer, in the plain-language sense, not the three-season bin names
# used across the Mascarenas et al. figures (Winter Dec-Mar, Spring Apr-Jul,
# Low-DO Aug-Nov). 2025.02.16 sits in that Winter bin either way; 2025.08.27
# sits in Low-DO, and is called summer here because that is what a reader will
# take from a late-August release and because "low DO" is what the panel is
# measuring rather than a name for when it happened.
#
# Neither is called SPRING. That word already means a tide in the right-hand
# column, and one legend cannot carry it as a season and a tide at once.
SEASONAL = {
    2024: [('pcbot_3d_sh13_hiDO', C_HI, 'winter'),
           ('pcbot_3d_loDO', C_LO, 'summer')],
    2025: [('pcbot_3d_sh14_hiDO_2025', C_HI, 'winter'),
           ('pcbot_3d_sh2_loDO_2025', C_LO, 'summer')],
}
COLUMNS = [('seasonal, tide-matched', SEASONAL[args.year]),
           ('spring-neap, phase-matched',
            [('pcbot_3d_sh4_neap', C_NEAP, 'neap'),
             ('pcbot_3d_spring', C_SPRING, 'spring')])]

# ------------------------------------------------------------------ setup ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape

seg = pickle.load(open(sorted((Ldir['LOo'] / 'extract' / 'tef2')
                             .glob('seg_info_dict_wb1_pc1_*.p'))[0], 'rb'))
INNER = np.zeros((NR, NC), dtype=bool)
a = np.array(seg['pc_cp_m']['ji_list'])
INNER[a[:, 0], a[:, 1]] = True


def ji_of(plon, plat):
    """Nearest rho indices on this plaid grid; NaN positions map to -1."""
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.full(plon.shape, -1, dtype=int)
    j = np.full(plon.shape, -1, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    return j, i, ok


def efold(frac, days):
    k = np.where(frac < 1 / np.e)[0]
    return days[k[0]] if len(k) else np.nan


def load(dname):
    fn = sorted((trk / dname).glob('release_*.nc'))[0]
    d = xr.open_dataset(fn)
    out = dict(lon=d.lon.values, lat=d.lat.values, cs=d.cs.values,
               ot=pd.to_datetime(d.ot.values, unit='s'), fn=fn.name)
    d.close()
    return out


# Each column is trimmed to ITS OWN common length. The two experiments are read
# on their own terms, so there is no reason to cut the 14.4 d seasonal pair down
# to the 7.8 d of the spring-neap one; within a column the trim is mandatory,
# because curves of different length would put a spurious difference into every
# end-of-run number.
COL = []
for title, runs in COLUMNS:
    R = {}
    n = min(load(dn)['cs'].shape[0] for dn, _, _ in runs)
    for dname, color, label in runs:
        r = load(dname)
        cs = r['cs'][:n, :]
        j, i, ok = ji_of(r['lon'][:n, :], r['lat'][:n, :])
        ins = np.zeros(cs.shape, dtype=bool)
        ins[ok] = INNER[j[ok], i[ok]]
        R[label] = dict(color=color, still=ins.mean(axis=1),
                        elev=np.nanmean(cs + 1, axis=1))
    COL.append(dict(title=title, n=n, days=np.arange(n) / 24.0, R=R))
    print('%-26s %d frames = %.2f d' % (title, n, (n - 1) / 24))
    for label, r in R.items():
        print('   %-22s e-fold %5.2f d, still %.3f at end, elevation %.2f -> %.2f'
              % (label, efold(r['still'], COL[-1]['days']), r['still'][-1],
                 r['elev'][0], r['elev'][-1]))

# ---------------------------------------------------------------- figure ---
# One x limit for the whole figure, so a day is the same width everywhere and
# the shorter spring-neap runs visibly stop where they stop.
XMAX = max(c['days'][-1] for c in COL)
fig, axs = plt.subplots(2, 2, figsize=(16, 7.5), sharey='row', sharex=True)

for k, C in enumerate(COL):
    # two-line labels: at this panel height the one-line versions are taller
    # than the axes and the two columns' labels run into each other
    for row, (key, ylab) in enumerate(
            [('still', 'fraction still inside\nthe inner cove'),
             ('elev', 'cohort-mean height in the column\n'
                      '[0 = bed, 1 = surface]')]):
        ax = axs[row, k]
        for label, r in C['R'].items():
            ax.plot(C['days'], r[key], color=r['color'], lw=1.8, label=label)
        if key == 'still':
            ax.axhline(1 / np.e, color='0.5', lw=0.9, ls=':')
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(0, 1)
        ax.set_xlim(0, XMAX)
        ax.grid(**GRID)
        ax.tick_params(labelsize=FS - 3)
        if k == 0:
            ax.set_ylabel(ylab, fontsize=FS - 2)
        if row == 1:
            ax.set_xlabel('days from release', fontsize=FS - 2)
    axs[0, k].set_title(C['title'], fontsize=FS)
    axs[0, k].legend(fontsize=FS - 3, loc='upper right', framealpha=0.9)

fig.tight_layout()

# Only the left column carries an axis label and tight_layout takes that space
# out of the left AXES, so equal gridspec cells still leave the two columns
# unequal in width -- which with a shared x limit means unequal days-per-inch.
# Re-lay them equal afterwards, once the label widths are known.
for row in range(2):
    p0 = axs[row, 0].get_position()
    p1 = axs[row, 1].get_position()
    gap = p1.x0 - (p0.x0 + p0.width)
    w = ((p1.x0 + p1.width) - p0.x0 - gap) / 2
    axs[row, 0].set_position([p0.x0, p0.y0, w, p0.height])
    axs[row, 1].set_position([p0.x0 + w + gap, p1.y0, w, p1.height])

dpd = [axs[0, k].get_position().width / XMAX for k in range(2)]
print('days-per-inch check: %.5f vs %.5f (%.2f%% apart)'
      % (dpd[0], dpd[1], 100 * abs(dpd[0] - dpd[1]) / dpd[0]))

fn_out = out_dir / ('pcbot_retention_grid_%d.png' % args.year)
fig.savefig(fn_out, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig)
print('\nwrote %s' % fn_out)
