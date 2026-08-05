"""
Top-minus-bottom salinity at each wb1_pc1 section, tidal and subtidal.

Reads the hourly stratification reduced from extractions_avg by
LO_user/extract/tef2/reduce_extractions_strat.py, and shows it two ways:

  hourly    as extracted -- resolves the tidal straining cycle
  subtidal  Godin filtered, the same filter bulk_calc_avg uses

Plotting both matters because they answer different questions. The subtidal
line says how stratified the section is on a seasonal basis; the hourly line
says how much of that is being built and destroyed within each tidal cycle.
A section can be strongly stratified in the mean while the tide overturns it
twice a day, and only the pair shows that.

run 20260805_plot_stratification.py
run 20260805_plot_stratification.py -z0 2024-12-01 -z1 2024-12-16
"""
import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-z0', default='2024-12-01', type=str)
parser.add_argument('-z1', default='2024-12-16', type=str)
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
      / ('strat_' + args.ds0 + '_' + args.ds1 + '_wb1_pc1.nc'))
if not fn.is_file():
    raise SystemExit('missing %s\nrun reduce_extractions_strat.py on apogee '
                     'and scp it back' % fn)

ds = xr.open_dataset(fn)
t = pd.to_datetime(ds.time.values)
sects = list(ds.sect.values)

# same CVD-validated colours as the other figures in this set
COLOR = {'pc_cp': '#009E73', 'pc_lj': '#0072B2', 'pc_lp': '#CC79A7',
         'sp_mid': '#D55E00', 'skagit_sp': '#5D3A9B'}
ORDER = [s for s in ['pc_cp', 'pc_lj', 'pc_lp', 'sp_mid', 'skagit_sp']
         if s in sects]

hourly = {s: pd.Series(ds.dstrat.values[:, sects.index(s)], index=t) for s in ORDER}
# Godin filter, the same one bulk_calc_avg uses, so "subtidal" here means the
# same thing it means everywhere else in this analysis
sub = {s: pd.Series(zfun.lowpass(hourly[s].values, f='godin'), index=t)
       for s in ORDER}

print('%-10s %10s %10s %10s %10s' % ('sect', 'mean sub', 'sd sub', 'sd tidal', 'tidal/sub'))
for s in ORDER:
    tid = hourly[s] - sub[s]           # what the Godin filter removed
    print('  %-10s %10.3f %10.3f %10.3f %10.2f'
          % (s, np.nanmean(sub[s]), np.nanstd(sub[s]), np.nanstd(tid),
             np.nanstd(tid) / np.nanstd(sub[s])))

out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

plt.close('all')
fig, ax = plt.subplot_mosaic([['full'], ['zoom']], figsize=(14, 9),
                             layout='constrained')

A = ax['full']
for s in ORDER:
    A.plot(t, hourly[s].values, lw=0.4, alpha=0.20, color=COLOR[s])
    A.plot(t, sub[s].values, lw=2, color=COLOR[s], label=s)
A.axhline(0, color='k', lw=0.8)
A.set_ylabel(r'$S_{bot} - S_{top}$  [g kg$^{-1}$]')
A.set_title('Top-to-bottom salinity difference\n'
            'faint = hourly (tidal), heavy = Godin filtered (subtidal)')
A.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
A.grid(color='lightgray', ls='--', alpha=0.5)
A.legend(loc='upper left', ncol=5, fontsize=9, framealpha=0.9)
A.margins(x=0.01)

B = ax['zoom']
z = slice(args.z0, args.z1)
for s in ORDER:
    B.plot(hourly[s][z].index, hourly[s][z].values, lw=1.0, alpha=0.65,
           color=COLOR[s])
    B.plot(sub[s][z].index, sub[s][z].values, lw=2.5, color=COLOR[s], label=s)
B.axhline(0, color='k', lw=0.8)
B.set_ylabel(r'$S_{bot} - S_{top}$  [g kg$^{-1}$]')
B.set_title('Zoom %s to %s: tidal straining against the subtidal state'
            % (args.z0, args.z1))
B.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
B.grid(color='lightgray', ls='--', alpha=0.5)
B.legend(loc='upper left', ncol=5, fontsize=9, framealpha=0.9)
B.margins(x=0.01)

for k, letter in zip(['full', 'zoom'], 'ab'):
    ax[k].text(0.006, 1.02, letter, transform=ax[k].transAxes,
               fontsize=14, fontweight='bold', va='bottom')

fn_out = out_dir / '20260805_stratification.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn_out)
