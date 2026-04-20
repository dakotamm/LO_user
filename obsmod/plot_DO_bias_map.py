"""
Plot maps showing spatial distribution of DO model bias (model - obs).
Stations where DO is overpredicted are highlighted.

Uses the combined pickle output from combine_obs_mod.py.

Testing on mac:
run plot_DO_bias_map -gtx wb1_t0_xn11ab -year 2024 -otype bottle -test True

"""
import sys
import pandas as pd
import numpy as np
import pickle
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun

# command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. wb1_t0_xn11ab
parser.add_argument('-otype', type=str, default='bottle') # bottle, ctd
parser.add_argument('-year', type=int) # e.g. 2024
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
parser.add_argument('-dividing_depth', type=int, default=10)
args = parser.parse_args()

Ldir = Lfun.Lstart()

if '_mac' in Ldir['lo_env']:
    pass
else:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

in_dir = Ldir['LOo'] / 'obsmod'

year = str(args.year)
gtx = args.gtagex
otype = args.otype
H = args.dividing_depth

# load combined data
in_fn = in_dir / ('combined_' + otype + '_' + year + '_' + gtx + '.p')
df_dict = pickle.load(open(in_fn, 'rb'))

obs = df_dict['obs']
mod = df_dict[gtx]

# check DO is available
if 'DO (uM)' not in obs.columns or 'DO (uM)' not in mod.columns:
    print('DO (uM) not available in data')
    sys.exit()

# compute bias per observation point
obs_do = obs['DO (uM)'].to_numpy()
mod_do = mod['DO (uM)'].to_numpy()
do_bias = mod_do - obs_do  # positive = overpredicted

# mask to valid DO pairs
valid = np.isfinite(obs_do) & np.isfinite(mod_do)

lon = obs['lon'].to_numpy()
lat = obs['lat'].to_numpy()
z = obs['z'].to_numpy()

# where to put output figures
out_dir = Ldir['LOo'] / 'obsmod_val_plots'
Lfun.make_dir(out_dir)

# auto-zoom extents
if valid.any():
    pad_lon = 0.1 * max((lon[valid].max() - lon[valid].min()), 0.05)
    pad_lat = 0.1 * max((lat[valid].max() - lat[valid].min()), 0.05)
    extent = [lon[valid].min() - pad_lon, lon[valid].max() + pad_lon,
              lat[valid].min() - pad_lat, lat[valid].max() + pad_lat]
else:
    extent = [-130, -122, 42, 52]

# ===== Figure 1: DO bias map (all depths) =====
pfun.start_plot(figsize=(16, 8), fs=12)
fig, axes = plt.subplots(1, 3, figsize=(16, 8))

# Panel 1: All stations colored by DO bias
ax = axes[0]
ax.set_title('DO Bias (model - obs) [uM]\nAll depths')
if valid.any():
    vmax = np.nanpercentile(np.abs(do_bias[valid]), 95)
    sc = ax.scatter(lon[valid], lat[valid], c=do_bias[valid],
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    s=10, alpha=0.6)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='DO bias (uM)')
pfun.add_coast(ax)
ax.axis(extent)
pfun.dar(ax)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Panel 2: Deep only (z <= -H)
ax = axes[1]
deep = valid & (z <= -H)
ax.set_title('DO Bias [uM]\nDeep (z < -%d m)' % H)
if deep.any():
    vmax_d = np.nanpercentile(np.abs(do_bias[deep]), 95)
    sc = ax.scatter(lon[deep], lat[deep], c=do_bias[deep],
                    cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d,
                    s=10, alpha=0.6)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='DO bias (uM)')
pfun.add_coast(ax)
ax.axis(extent)
pfun.dar(ax)
ax.set_xlabel('Longitude')
ax.set_ylabel('')

# Panel 3: Shallow only (z > -H)
ax = axes[2]
shallow = valid & (z > -H)
ax.set_title('DO Bias [uM]\nShallow (z >= -%d m)' % H)
if shallow.any():
    vmax_s = np.nanpercentile(np.abs(do_bias[shallow]), 95)
    sc = ax.scatter(lon[shallow], lat[shallow], c=do_bias[shallow],
                    cmap='RdBu_r', vmin=-vmax_s, vmax=vmax_s,
                    s=10, alpha=0.6)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='DO bias (uM)')
pfun.add_coast(ax)
ax.axis(extent)
pfun.dar(ax)
ax.set_xlabel('Longitude')
ax.set_ylabel('')

fig.suptitle('%s %s %s' % (otype, year, gtx), fontweight='bold', fontsize=14)
fig.tight_layout()

ff_str = 'DO_bias_map_' + otype + '_' + year + '_' + gtx

print('Plotting ' + ff_str)
sys.stdout.flush()

if args.testing:
    plt.show()
else:
    plt.savefig(out_dir / (ff_str + '.png'))
    print('Saved to:\n %s' % (str(out_dir / (ff_str + '.png'))))

# ===== Figure 2: Overpredicted stations only =====
pfun.start_plot(figsize=(10, 8), fs=12)
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

overpred = valid & (do_bias > 0)
underpred = valid & (do_bias <= 0)

if underpred.any():
    ax2.plot(lon[underpred], lat[underpred], '.', color='gray',
             markersize=4, alpha=0.3, label='DO underpredicted or matched')
if overpred.any():
    sc2 = ax2.scatter(lon[overpred], lat[overpred], c=do_bias[overpred],
                      cmap='Reds', s=15, alpha=0.7, vmin=0)
    plt.colorbar(sc2, ax=ax2, shrink=0.7, label='DO overprediction (uM)')

ax2.set_title('Stations where DO is overpredicted\n%s %s %s' % (otype, year, gtx),
              fontweight='bold')
pfun.add_coast(ax2)
ax2.axis(extent)
pfun.dar(ax2)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')

n_over = overpred.sum()
n_valid = valid.sum()
ax2.text(0.05, 0.05, '%d / %d stations overpredicted (%.0f%%)' %
         (n_over, n_valid, 100*n_over/max(n_valid,1)),
         transform=ax2.transAxes, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

if overpred.any():
    ax2.legend(loc='upper right', fontsize=9)

fig2.tight_layout()

ff_str2 = 'DO_overpred_map_' + otype + '_' + year + '_' + gtx

print('Plotting ' + ff_str2)
sys.stdout.flush()

if args.testing:
    plt.show()
else:
    plt.savefig(out_dir / (ff_str2 + '.png'))
    print('Saved to:\n %s' % (str(out_dir / (ff_str2 + '.png'))))
