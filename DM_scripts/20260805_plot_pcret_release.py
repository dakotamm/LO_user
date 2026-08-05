"""
Show where the 'pcret' particles are released, before committing to a run.

Panel a is the map: every release position over Penn Cove bathymetry, coloured
by which tef2 segment it belongs to, with the three cross-cove sections drawn.
Panel b is the side view: particle depth against longitude, over the deepest
bathymetry at each longitude, which is the thing a map cannot show -- whether
the deep water of the inner cove is actually being seeded.

Reads the initial conditions straight from LO_user/tracker2/experiments.py, so
what is plotted is exactly what tracker.py would release, not a reconstruction.

run 20260805_plot_pcret_release.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import xarray as xr

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

Ldir = Lfun.Lstart(gridname='wb1')

# the his file only supplies the grid (h, lon_rho, lat_rho) for the release
fn00 = Ldir['parent'] / 'LO_roms' / 'wb1_t0_xn11ab' / 'f2024.03.01' / 'ocean_his_0002.nc'
if not fn00.is_file():
    raise SystemExit('need a wb1 history file for the grid: %s' % fn00)

exp = Lfun.module_from_file('experiments',
                            Ldir['LOu'] / 'tracker2' / 'experiments.py')
plon, plat, pcs = exp.get_ic(dict(exp_name='pcret', gridname='wb1', fn00=fn00))

dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = dsg.lon_rho.values
lat = dsg.lat_rho.values
mask = dsg.mask_rho.values
h = dsg.h.values
dsg.close()

seg = pickle.load(open(Ldir['LOo'] / 'extract' / 'tef2'
                       / 'seg_info_dict_wb1_pc1_riv00.p', 'rb'))
cells = {k: set(map(tuple, v['ji_list'])) for k, v in seg.items()}

# recover each particle's cohort and depth from its initial position, the same
# way the analysis will have to, so this doubles as a check of that logic
j = np.abs(lat[:, 0][:, None] - plat[None, :]).argmin(axis=0)
i = np.abs(lon[0, :][:, None] - plon[None, :]).argmin(axis=0)
hp = h[j, i]
zp = pcs * hp

SEGS = ['pc_cp_m', 'pc_cp_p', 'pc_lp_m', 'pc_lp_p']
LABEL = {'pc_cp_m': 'inner (pc_cp_m)', 'pc_cp_p': 'mid (pc_cp_p)',
         'pc_lp_m': 'outer (pc_lp_m)', 'pc_lp_p': 'Saratoga control (pc_lp_p)'}
# same CVD-validated order as the flushing figure, landward -> seaward
COLOR = {'pc_cp_m': '#009E73', 'pc_cp_p': '#0072B2', 'pc_lp_m': '#CC79A7',
         'pc_lp_p': '#D55E00'}

cohort = np.array(['OUTSIDE'] * len(plon), dtype=object)
for s in SEGS:
    inseg = np.array([(int(a), int(b)) in cells[s] for a, b in zip(j, i)])
    cohort[inseg] = s

print('\nrelease composition:')
for s in SEGS:
    n = int((cohort == s).sum())
    zz = zp[cohort == s]
    print('  %-16s %5d particles   depth %6.1f to %5.1f m   mean %5.1f'
          % (LABEL[s], n, zz.min(), zz.max(), zz.mean()))
if (cohort == 'OUTSIDE').any():
    print('  WARNING %d particles outside all three segments'
          % int((cohort == 'OUTSIDE').sum()))

# ------------------------------------------------------------------ plot ----
sect_dir = Ldir['LOo'] / 'extract' / 'tef2' / 'sections_wb1_pc1'

plt.close('all')
fig, axes = plt.subplot_mosaic([['map'], ['side']], figsize=(12, 9),
                               layout='constrained')

pad = 0.008
aa = [plon.min() - pad, plon.max() + 3 * pad, plat.min() - pad, plat.max() + pad]

ax = axes['map']
hm = np.ma.masked_where(mask == 0, h)
pc = ax.pcolormesh(lon, lat, hm, cmap='Blues', vmin=0, vmax=60,
                   shading='nearest', zorder=1)
plt.colorbar(pc, ax=ax, shrink=0.8, label='depth [m]')
pfun.add_coast(ax, color='k', linewidth=0.6)
for s in SEGS:
    m = cohort == s
    # jitter within the cell so stacked particles at one cell centre are visible
    jit = 0.0008
    ax.plot(plon[m] + np.random.uniform(-jit, jit, m.sum()),
            plat[m] + np.random.uniform(-jit, jit, m.sum()),
            '.', ms=2.5, color=COLOR[s], label='%s, n=%d' % (LABEL[s], m.sum()),
            zorder=5)
for fn in sorted(sect_dir.glob('*.p')):
    d = pd.read_pickle(fn)
    ax.plot(d.x, d.y, '-', color='magenta', lw=2, zorder=8)
    ax.text(d.x.mean(), d.y.max(), fn.stem, color='magenta', fontsize=9,
            ha='center', va='bottom', fontweight='bold', zorder=9)
pfun.dar(ax)
ax.axis(aa)
# upper left is land here (the north shore of Whidbey), so the legend can sit
# inside without covering any release
ax.legend(loc='upper left', fontsize=8, framealpha=0.92, markerscale=4)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('pcret release positions (jittered within cells for visibility)')

ax = axes['side']
# Deepest bathymetry at each longitude, over the SEGMENT CELLS only.
# A latitude band will not do: once the Saratoga segment is included the band
# spans 48.18-48.30, and at the western longitudes that is no longer Penn Cove
# but Admiralty Inlet on the far side of Whidbey Island, 85+ m deep. That made
# the head of the cove look like a trench.
band = np.zeros(mask.shape, dtype=bool)
for s in SEGS:
    ji = np.array(seg[s]['ji_list'])
    band[ji[:, 0], ji[:, 1]] = True
lon_u = np.unique(lon[band])
hmax = np.array([h[band & (lon == L)].max() for L in lon_u])
ax.fill_between(lon_u, -hmax, -hmax.max() * 1.05, color='0.8', zorder=1)
ax.plot(lon_u, -hmax, '-', color='0.4', lw=1, zorder=2, label='deepest bathymetry')
for s in SEGS:
    m = cohort == s
    ax.plot(plon[m] + np.random.uniform(-0.0008, 0.0008, m.sum()), zp[m],
            '.', ms=3, color=COLOR[s], zorder=5, label=LABEL[s])
for fn in sorted(sect_dir.glob('pc_*.p')):
    d = pd.read_pickle(fn)
    ax.axvline(d.x.iloc[0], color='magenta', lw=1.5, ls='--', zorder=6)
ax.set_xlim(lon_u.min() - 0.005, lon_u.max() + 0.005)
ax.set_ylim(-hmax.max() * 1.05, 2)
ax.set_xlabel('Longitude')
ax.set_ylabel('z [m]')
ax.set_title('Side view: vertical stacking at DZ = 3 m (head at left, mouth at right)')
ax.grid(color='lightgray', linestyle='--', alpha=0.5)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9, markerscale=3)

for k, letter in zip(['map', 'side'], 'ab'):
    axes[k].text(0.008, 1.02, letter, transform=axes[k].transAxes,
                 fontsize=14, fontweight='bold', va='bottom')

out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)
fn_out = out_dir / '20260805_pcret_release.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn_out)
