"""
Plot time series of bottom DO (obs vs model) at each station.

For every station in the combined pickle, finds the deepest observation
per cast and plots obs and model bottom DO as a time series.
Stations are sorted by mean bias (worst overprediction first).

Testing on mac:
run plot_bottom_DO_timeseries -gtx wb1_t0_xn11ab -year 2024 -otype ctd -test True

"""
import sys
import pandas as pd
import numpy as np
import pickle
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str)
parser.add_argument('-otype', type=str, default='ctd')
parser.add_argument('-year', type=int)
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
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

# load combined data
in_fn = in_dir / ('combined_' + otype + '_' + year + '_' + gtx + '.p')
df_dict = pickle.load(open(in_fn, 'rb'))

obs = df_dict['obs']
mod = df_dict[gtx]

if 'DO (uM)' not in obs.columns or 'DO (uM)' not in mod.columns:
    print('DO (uM) not available in data')
    sys.exit()

# Build working dataframe
# Convert DO from uM to mg/L
DO_UM_TO_MGL = 32.0 / 1000.0
wdf = pd.DataFrame({
    'lon': obs['lon'].values,
    'lat': obs['lat'].values,
    'z': obs['z'].values,
    'time': obs['time'].values,
    'cid': obs['cid'].values,
    'obs_DO': obs['DO (uM)'].values * DO_UM_TO_MGL,
    'mod_DO': mod['DO (uM)'].values * DO_UM_TO_MGL,
    'source': obs['source'].values if 'source' in obs.columns else '',
})
if 'name' in obs.columns:
    wdf['station'] = obs['name'].values
else:
    wdf['station'] = obs['lon'].round(3).astype(str) + '_' + obs['lat'].round(3).astype(str)

wdf = wdf.dropna(subset=['obs_DO', 'mod_DO'])
wdf['time'] = pd.to_datetime(wdf['time'])
wdf['bias'] = wdf['mod_DO'] - wdf['obs_DO']

# For each cast (cid), keep only the deepest observation (bottom)
# z is the same in obs and mod (model sampled at nearest z_rho to obs z)
bot = wdf.loc[wdf.groupby('cid')['z'].idxmin()].copy()

# Get station-level stats
stn_stats = bot.groupby('station').agg(
    mean_bias=('bias', 'mean'),
    n_casts=('cid', 'count'),
    lon=('lon', 'mean'),
    lat=('lat', 'mean'),
    mean_z=('z', 'mean'),
).reset_index()

# Only plot stations with >= 2 casts
stn_stats = stn_stats[stn_stats['n_casts'] >= 2]

# Sort by mean bias descending (worst overprediction first)
stn_stats = stn_stats.sort_values('mean_bias', ascending=False).reset_index(drop=True)

stations = stn_stats['station'].tolist()
n_stn = len(stations)

if n_stn == 0:
    print('No stations with >= 2 casts and valid bottom DO')
    sys.exit()

print('%d stations with >= 2 bottom DO casts' % n_stn)

out_dir = Ldir['LOo'] / 'obsmod_val_plots'
Lfun.make_dir(out_dir)

# Plot in pages of up to 6 stations per figure
n_per_page = 6
n_pages = int(np.ceil(n_stn / n_per_page))

for page in range(n_pages):
    i0 = page * n_per_page
    i1 = min(i0 + n_per_page, n_stn)
    page_stations = stations[i0:i1]
    n_sub = len(page_stations)

    pfun.start_plot(figsize=(14, 3 * n_sub), fs=11)
    fig, axes = plt.subplots(n_sub, 1, figsize=(14, 3 * n_sub), sharex=True)
    if n_sub == 1:
        axes = [axes]

    for j, stn in enumerate(page_stations):
        ax = axes[j]
        sdf = bot[bot['station'] == stn].sort_values('time')
        stats = stn_stats[stn_stats['station'] == stn].iloc[0]

        ax.plot(sdf['time'], sdf['obs_DO'], 'o-', color='k',
                markersize=5, label='Obs DO')
        ax.plot(sdf['time'], sdf['mod_DO'], 's--', color='tab:red',
                markersize=5, markerfacecolor='none', label='Model DO')

        ax.axhline(y=2.0, color='gray', linestyle=':', linewidth=1)

        ax.set_ylabel('DO (mg/L)')
        ax.grid(True, alpha=0.3)

        # Secondary axis: sample depth
        ax2 = ax.twinx()
        ax2.plot(sdf['time'], -sdf['z'], '^-', color='tab:blue',
                 markersize=4, alpha=0.6, label='Sample depth')
        ax2.set_ylabel('Depth (m)', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.invert_yaxis()

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper right', fontsize=7, ncol=2)

        ax.set_title('%s  (n=%d, mean z=%.0f m, mean bias=%.2f mg/L)' %
                     (stn, int(stats['n_casts']), stats['mean_z'], stats['mean_bias']),
                     fontweight='bold', fontsize=10, loc='left')

    axes[-1].set_xlabel('Date')
    fig.suptitle('Bottom DO Time Series: %s %s %s (page %d/%d)\nsorted by mean bias (worst overprediction first)' %
                 (otype, year, gtx, page+1, n_pages),
                 fontweight='bold', fontsize=12, y=1.01)
    fig.tight_layout()

    ff_str = 'bottom_DO_ts_%s_%s_%s_p%02d' % (otype, year, gtx, page+1)

    print('Plotting ' + ff_str)
    sys.stdout.flush()

    if args.testing:
        plt.show()
    else:
        plt.savefig(out_dir / (ff_str + '.png'), bbox_inches='tight')
        print('Saved to:\n %s' % (str(out_dir / (ff_str + '.png'))))

print('Done. %d stations across %d pages.' % (n_stn, n_pages))
