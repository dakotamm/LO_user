"""
Plot maps showing spatial distribution of DO model bias (model - obs).
Observations are grouped by station (rounded lat/lon to ~100m) so each
unique location gets one dot showing its mean bias.

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

# Build a working DataFrame with per-observation bias
# Convert DO from uM to mg/L
DO_UM_TO_MGL = 32.0 / 1000.0
wdf = pd.DataFrame({
    'lon': obs['lon'].values,
    'lat': obs['lat'].values,
    'z': obs['z'].values,
    'obs_DO': obs['DO (uM)'].values * DO_UM_TO_MGL,
    'mod_DO': mod['DO (uM)'].values * DO_UM_TO_MGL,
})
# use station name if available, otherwise group by rounded lat/lon
if 'name' in obs.columns:
    wdf['name'] = obs['name'].values
else:
    wdf['name'] = ''
wdf['bias'] = wdf['mod_DO'] - wdf['obs_DO']
wdf = wdf.dropna(subset=['obs_DO', 'mod_DO'])

# Group by station: round lat/lon to 3 decimal places (~100m)
wdf['lon_r'] = wdf['lon'].round(3)
wdf['lat_r'] = wdf['lat'].round(3)

# Use name if available, otherwise use rounded coords as station key
if wdf['name'].str.len().sum() > 0:
    wdf['station'] = wdf['name']
else:
    wdf['station'] = wdf['lon_r'].astype(str) + '_' + wdf['lat_r'].astype(str)

# Station-level aggregation (all depths)
stn_all = wdf.groupby('station').agg(
    lon=('lon', 'mean'),
    lat=('lat', 'mean'),
    mean_bias=('bias', 'mean'),
    mean_z=('z', 'mean'),
    n_obs=('bias', 'count'),
).reset_index()

# Station-level for deep (z <= -H) and shallow (z > -H)
deep_df = wdf[wdf['z'] <= -H]
shallow_df = wdf[wdf['z'] > -H]

stn_deep = deep_df.groupby('station').agg(
    lon=('lon', 'mean'), lat=('lat', 'mean'),
    mean_bias=('bias', 'mean'), n_obs=('bias', 'count'),
).reset_index() if len(deep_df) > 0 else pd.DataFrame()

stn_shallow = shallow_df.groupby('station').agg(
    lon=('lon', 'mean'), lat=('lat', 'mean'),
    mean_bias=('bias', 'mean'), n_obs=('bias', 'count'),
).reset_index() if len(shallow_df) > 0 else pd.DataFrame()

print('%d unique stations from %d observations' % (len(stn_all), len(wdf)))

# where to put output figures
out_dir = Ldir['LOo'] / 'obsmod_val_plots'
Lfun.make_dir(out_dir)

# auto-zoom extents
pad_lon = 0.1 * max((stn_all['lon'].max() - stn_all['lon'].min()), 0.05)
pad_lat = 0.1 * max((stn_all['lat'].max() - stn_all['lat'].min()), 0.05)
extent = [stn_all['lon'].min() - pad_lon, stn_all['lon'].max() + pad_lon,
          stn_all['lat'].min() - pad_lat, stn_all['lat'].max() + pad_lat]

def _plot_bias_panel(ax, sdf, title):
    """Plot one panel of station-mean DO bias."""
    if len(sdf) == 0:
        ax.set_title(title + '\n(no data)')
        pfun.add_coast(ax)
        ax.axis(extent)
        pfun.dar(ax)
        return
    vmax = np.nanpercentile(np.abs(sdf['mean_bias'].values), 95)
    vmax = max(vmax, 0.1)  # floor in mg/L
    sc = ax.scatter(sdf['lon'], sdf['lat'], c=sdf['mean_bias'],
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    s=30, edgecolors='k', linewidths=0.3, zorder=5)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='mean DO bias (mg/L)')
    ax.set_title(title + '\n(%d stations)' % len(sdf))
    pfun.add_coast(ax)
    ax.axis(extent)
    pfun.dar(ax)

# ===== Figure 1: Station-mean DO bias map (3-panel) =====
pfun.start_plot(figsize=(16, 8), fs=12)
fig, axes = plt.subplots(1, 3, figsize=(16, 8))

_plot_bias_panel(axes[0], stn_all, 'All depths')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

_plot_bias_panel(axes[1], stn_deep, 'Deep (z < -%d m)' % H)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')

_plot_bias_panel(axes[2], stn_shallow, 'Shallow (z >= -%d m)' % H)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')

fig.suptitle('Station-Mean DO Bias (model - obs)\n%s %s %s' % (otype, year, gtx),
             fontweight='bold', fontsize=14)
fig.tight_layout()

ff_str = 'DO_bias_map_' + otype + '_' + year + '_' + gtx

print('Plotting ' + ff_str)
sys.stdout.flush()

if args.testing:
    plt.show()
else:
    plt.savefig(out_dir / (ff_str + '.png'), bbox_inches='tight')
    print('Saved to:\n %s' % (str(out_dir / (ff_str + '.png'))))

# ===== Figure 2: Overpredicted stations, labeled =====
pfun.start_plot(figsize=(10, 8), fs=12)
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

stn_under = stn_all[stn_all['mean_bias'] <= 0]
stn_over = stn_all[stn_all['mean_bias'] > 0]

if len(stn_under) > 0:
    ax2.plot(stn_under['lon'], stn_under['lat'], 'o', color='gray',
             markersize=5, alpha=0.4, label='underpredicted/matched')
if len(stn_over) > 0:
    sc2 = ax2.scatter(stn_over['lon'], stn_over['lat'], c=stn_over['mean_bias'],
                      cmap='Reds', s=40, edgecolors='k', linewidths=0.3,
                      alpha=0.8, vmin=0, zorder=5)
    plt.colorbar(sc2, ax=ax2, shrink=0.7, label='mean DO overprediction (mg/L)')

    # label the top overpredictors
    n_label = min(10, len(stn_over))
    top = stn_over.nlargest(n_label, 'mean_bias')
    for _, row in top.iterrows():
        ax2.annotate(row['station'],
                     (row['lon'], row['lat']),
                     textcoords='offset points', xytext=(5, 5),
                     fontsize=7, color='red', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

ax2.set_title('Stations where DO is overpredicted (station mean)\n%s %s %s' % (otype, year, gtx),
              fontweight='bold')
pfun.add_coast(ax2)
ax2.axis(extent)
pfun.dar(ax2)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')

n_over = len(stn_over)
n_total = len(stn_all)
ax2.text(0.05, 0.05, '%d / %d stations overpredicted (%.0f%%)' %
         (n_over, n_total, 100*n_over/max(n_total,1)),
         transform=ax2.transAxes, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
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
