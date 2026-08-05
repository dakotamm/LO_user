"""
Tidal prism transport Qprism at every wb1_pc1 section.

Qprism comes straight out of bulk_calc_avg.py, where it is defined as

    qabs   = |qnet - lowpass(qnet)|          the tidal part of the net transport
    Qprism = 1/2 <qabs>                      Godin filtered, subsampled daily

i.e. conceptually the maximum exchange the tide could produce if every bit of
the tidal excursion were exchanged rather than returned.

NOTE every point plotted here is already tidally averaged. bulk_calc_avg.py
Godin filters qabs and THEN takes every 24th value, so "daily" is the sampling
interval, not raw daily data -- there is no unfiltered series in bulk_avg. The
fortnightly spring-neap cycle survives because Godin removes the diurnal and
semidiurnal bands, not the 14-day modulation.

Panel a is on a log y axis: Qprism spans roughly 300 m3/s at pc_cp to 15000 at
sp_mid, so a linear axis flattens the three Penn Cove sections to nothing. One
axis, log scaled -- not a second y axis.

run 20260805_tef_qprism.py
"""
import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
# window for the zoom panel, to show the fortnightly cycle at readable scale
parser.add_argument('-z0', default='2024.06.01', type=str)
parser.add_argument('-z1', default='2024.08.31', type=str)
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
bulk_dir = (Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
            / ('bulk_avg_' + args.ds0 + '_' + args.ds1))
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

# ordered landward -> seaward, then the two Saratoga bounds
SECTS = ['pc_cp', 'pc_lj', 'pc_lp', 'sp_mid', 'skagit_sp']
# CVD-validated categorical order (validate_palette.js, light surface):
# worst adjacent pair dE 9.6 protan; all checks PASS but a contrast WARN on
# #CC79A7, relieved by the legend and the printed table.
COLORS = ['#009E73', '#0072B2', '#CC79A7', '#D55E00', '#5D3A9B']
color = dict(zip(SECTS, COLORS))

qp = {}
for sn in SECTS:
    ds = xr.open_dataset(bulk_dir / (sn + '.nc'))
    qp[sn] = pd.Series(ds.qprism.values, index=pd.to_datetime(ds.time.values))
    ds.close()
df = pd.DataFrame(qp)

# ---------------------------------------------------------------- summary ---
print('\nQprism [m3 s-1]   %s .. %s' % (args.ds0, args.ds1))
summ = pd.DataFrame({
    'mean': df.mean(), 'min': df.min(), 'max': df.max(),
    'spring/neap': df.max() / df.min(),
    'sd/mean': df.std() / df.mean()})
print(summ.round(2).to_string())
df.to_csv(out_dir / '20260805_tef_qprism.csv')

# ---------------------------------------------------------------- figure ----
plt.close('all')
fig, axes = plt.subplot_mosaic([['full'], ['zoom']], figsize=(13, 8),
                               layout='constrained')

ax = axes['full']
for sn in SECTS:
    ax.plot(df.index, df[sn], lw=0.7, alpha=0.45, color=color[sn])
    ax.plot(df.index, df[sn].rolling(30, center=True, min_periods=10).mean(),
            lw=2, color=color[sn], label=sn)
ax.set_yscale('log')
ax.set_ylabel(r'$Q_{prism}$  [m$^3$ s$^{-1}$]')
ax.set_title('Tidal prism transport at each section\n'
             'thin = Godin-filtered, daily-subsampled (every bulk_avg point); '
             'thick = 30-day rolling mean')
ax.grid(color='lightgray', linestyle='--', alpha=0.5, which='both')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.legend(loc='upper left', bbox_to_anchor=(0, -0.02, 1, 1), ncol=5,
          fontsize=9, framealpha=0.9, mode='expand', borderaxespad=0)
ax.margins(x=0.01)

ax = axes['zoom']
z = df.loc[args.z0.replace('.', '-'):args.z1.replace('.', '-')]
for sn in SECTS:
    ax.plot(z.index, z[sn], '-o', lw=1.5, ms=3, color=color[sn], label=sn)
ax.set_yscale('log')
ax.set_ylabel(r'$Q_{prism}$  [m$^3$ s$^{-1}$]')
ax.set_title('Zoom: %s to %s, showing the fortnightly spring-neap cycle'
             % (args.z0, args.z1))
ax.grid(color='lightgray', linestyle='--', alpha=0.5, which='both')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.margins(x=0.01)

for k, letter in zip(['full', 'zoom'], 'ab'):
    axes[k].text(0.008, 1.02, letter, transform=axes[k].transAxes,
                 fontsize=14, fontweight='bold', va='bottom')

fn_out = out_dir / '20260805_tef_qprism.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn_out)
