"""
Residual circulation across the Penn Cove sections: vertical or lateral?

For each cross-cove section, the top row shows the time-mean volume flux per
face-cell as a section view -- distance across the cove on x, sigma level on y,
red into the cove and blue out. The bottom row collapses the same field onto
its two axes:

  the vertical profile   sum over p, plotted against z
  the lateral profile    sum over z, plotted against distance across

If the exchange is a classic two-layer estuarine circulation, the vertical
profile changes sign with depth and the lateral one is flat. If it is a lateral
circulation, the lateral profile changes sign across the cove and the vertical
one is flat. The ratio of their standard deviations, printed and titled, is the
summary statistic.

run 20260805_plot_section_structure.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
      / ('structure_' + args.ds0 + '_' + args.ds1 + '_wb1_pc1.nc'))
if not fn.is_file():
    raise SystemExit('missing %s\nrun reduce_extractions_structure.py on '
                     'apogee and scp it back' % fn)
ds = xr.open_dataset(fn)

SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
COLOR = {'pc_cp': '#009E73', 'pc_lj': '#0072B2', 'pc_lp': '#CC79A7'}

print('%-8s %10s %12s %12s %10s' % ('sect', 'net', 'sd vertical', 'sd lateral',
                                    'lat/vert'))
stats = {}
for sn in SECTS:
    qbar = ds[sn + '_qbar'].values
    pz = np.nansum(qbar, axis=1)
    pp = np.nansum(qbar, axis=0)
    stats[sn] = (np.nanstd(pz), np.nanstd(pp))
    print('  %-8s %10.2f %12.2f %12.2f %10.2f'
          % (sn, np.nansum(qbar), np.nanstd(pz), np.nanstd(pp),
             np.nanstd(pp) / np.nanstd(pz)))

out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

plt.close('all')
fig, axes = plt.subplots(3, len(SECTS), figsize=(5 * len(SECTS), 11),
                         gridspec_kw=dict(height_ratios=[2, 1, 1]))

for k, sn in enumerate(SECTS):
    qbar = ds[sn + '_qbar'].values
    lat = ds[sn + '_lat'].values
    NZ, NP = qbar.shape
    # distance across the section from its southern end
    dist = (lat - lat.min()) * 111.0                     # km
    vmax = np.nanpercentile(np.abs(qbar), 99)

    ax = axes[0, k]
    # positive q is toward the section's plus side; for these N-S cove sections
    # that is eastward, i.e. OUT of the cove, so flip the sign for readability
    cs = ax.pcolormesh(dist, np.arange(NZ), -qbar, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='nearest')
    plt.colorbar(cs, ax=ax, label='mean flux per cell [m$^3$ s$^{-1}$]\n'
                                  'red = into cove')
    ax.set_title('%s\nlateral/vertical = %.2f'
                 % (sn, stats[sn][1] / stats[sn][0]), color=COLOR[sn],
                 fontweight='bold')
    ax.set_xlabel('distance across cove from south [km]')
    ax.set_ylabel('sigma level (0 = bed)')

    ax = axes[1, k]
    pz = -np.nansum(qbar, axis=1)
    ax.plot(pz, np.arange(NZ), '-o', ms=4, color=COLOR[sn])
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('flux into cove [m$^3$ s$^{-1}$]')
    ax.set_ylabel('sigma level')
    ax.set_title('vertical profile (sum over width)', fontsize=10)
    ax.grid(color='lightgray', ls='--', alpha=0.5)

    ax = axes[2, k]
    pp = -np.nansum(qbar, axis=0)
    ax.plot(dist, pp, '-o', ms=4, color=COLOR[sn])
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('distance across cove from south [km]')
    ax.set_ylabel('flux into cove [m$^3$ s$^{-1}$]')
    ax.set_title('lateral profile (sum over depth)', fontsize=10)
    ax.grid(color='lightgray', ls='--', alpha=0.5)

fig.suptitle('Residual circulation across the Penn Cove sections, '
             '2024-2025 mean\nis the exchange organised by depth or across the cove?',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fn_out = out_dir / '20260805_section_structure.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn_out)
