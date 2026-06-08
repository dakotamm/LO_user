"""
Plot bottom DO time series comparing obs vs two model versions.

Reads the combined obs+model pickles for two gtagex values and plots
obs, model0, and model1 as time series at each station.
Stations sorted by bias change between models (largest Δbias first).

Testing on mac:
run plot_bottom_DO_ts_compare -gtx0 wb1_t0_xn11abbur00 -gtx1 wb1_t1_xn11abbur00 -year0 2022 -year1 2024 -otype ctd -test True
"""

import sys
import pandas as pd
import numpy as np
import pickle
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gtx0', type=str)
parser.add_argument('-gtx1', type=str)
parser.add_argument('-otype', type=str, default='ctd')
parser.add_argument('-year0', type=int)
parser.add_argument('-year1', type=int, default=0)
parser.add_argument('-stations', type=str, default='')  # comma-separated; empty = all
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

if args.year0 is None:
    print('*** Missing required argument: -year0')
    sys.exit()
if args.year1 == 0:
    args.year1 = args.year0
year_list = list(range(args.year0, args.year1 + 1))

Ldir = Lfun.Lstart()

if '_mac' in Ldir['lo_env']:
    pass
else:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

in_dir = Ldir['LOo'] / 'obsmod'
gtx0 = args.gtx0
gtx1 = args.gtx1
otype = args.otype

DO_UM_TO_MGL = 32.0 / 1000.0

# Load and concatenate data across all years
frames = []
for yr in year_list:
    yr_str = str(yr)
    in_fn0 = in_dir / ('combined_' + otype + '_' + yr_str + '_' + gtx0 + '.p')
    in_fn1 = in_dir / ('combined_' + otype + '_' + yr_str + '_' + gtx1 + '.p')
    missing = False
    for fn, gtx in [(in_fn0, gtx0), (in_fn1, gtx1)]:
        if not fn.is_file():
            print('Missing combined pickle for %s %s: %s' % (gtx, yr_str, fn))
            missing = True
    if missing:
        continue

    df_dict0 = pickle.load(open(in_fn0, 'rb'))
    df_dict1 = pickle.load(open(in_fn1, 'rb'))
    obs  = df_dict0['obs']
    mod0 = df_dict0[gtx0]
    mod1 = df_dict1[gtx1]

    if 'DO (uM)' not in obs.columns:
        print('DO (uM) not available in obs for %s — skipping' % yr_str)
        continue

    wdf = pd.DataFrame({
        'lon':     obs['lon'].values,
        'lat':     obs['lat'].values,
        'z':       obs['z'].values,
        'time':    obs['time'].values,
        'cid':     obs['cid'].values,
        'obs_DO':  obs['DO (uM)'].values  * DO_UM_TO_MGL,
        'mod0_DO': mod0['DO (uM)'].values * DO_UM_TO_MGL,
        'mod1_DO': mod1['DO (uM)'].values * DO_UM_TO_MGL,
    })
    if 'name' in obs.columns:
        wdf['station'] = obs['name'].values
    else:
        wdf['station'] = (obs['lon'].round(3).astype(str) + '_'
                          + obs['lat'].round(3).astype(str))
    frames.append(wdf)

if len(frames) == 0:
    print('No data found for any requested year.')
    sys.exit()

wdf = pd.concat(frames, ignore_index=True)
wdf = wdf.dropna(subset=['obs_DO'])
wdf['time']  = pd.to_datetime(wdf['time'])
wdf['bias0'] = wdf['mod0_DO'] - wdf['obs_DO']
wdf['bias1'] = wdf['mod1_DO'] - wdf['obs_DO']

# Keep only deepest obs per cast (bottom)
bot = wdf.loc[wdf.groupby('cid')['z'].idxmin()].copy()

if args.stations:
    station_filter = [s.strip() for s in args.stations.split(',')]
    bot = bot[bot['station'].isin(station_filter)]
    if len(bot) == 0:
        print('No data found for requested stations: %s' % args.stations)
        sys.exit()

stn_stats = bot.groupby('station').agg(
    mean_bias0=('bias0', 'mean'),
    mean_bias1=('bias1', 'mean'),
    n_casts=('cid', 'count'),
    lon=('lon', 'mean'),
    lat=('lat', 'mean'),
    mean_z=('z', 'mean'),
).reset_index()
stn_stats['bias_diff'] = stn_stats['mean_bias1'] - stn_stats['mean_bias0']
stn_stats = stn_stats[stn_stats['n_casts'] >= 2]
stn_stats = stn_stats.sort_values('bias_diff', ascending=False).reset_index(drop=True)

stations = stn_stats['station'].tolist()
n_stn = len(stations)

if n_stn == 0:
    print('No stations with >= 2 casts and valid bottom DO')
    sys.exit()

print('%d stations with >= 2 bottom DO casts' % n_stn)

out_dir = Ldir['LOo'] / 'obsmod_val_plots'
Lfun.make_dir(out_dir)

def short_gtx(gtx):
    return gtx.split('_')[-1] if '_' in gtx else gtx

lbl0 = short_gtx(gtx0)
lbl1 = short_gtx(gtx1)

year_str = ('%d' % args.year0 if args.year0 == args.year1
            else '%d-%d' % (args.year0, args.year1))

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
                markersize=5, label='Obs')
        ax.plot(sdf['time'], sdf['mod0_DO'], 's--', color='tab:red',
                markersize=5, markerfacecolor='none', label=lbl0)
        ax.plot(sdf['time'], sdf['mod1_DO'], '^--', color='tab:blue',
                markersize=5, markerfacecolor='none', label=lbl1)

        ax.axhline(y=2.0, color='gray', linestyle=':', linewidth=1)
        ax.set_ylabel('DO (mg/L)')
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(sdf['time'], -sdf['z'], 'v-', color='tab:green',
                 markersize=4, alpha=0.5, label='Depth')
        ax2.set_ylabel('Depth (m)', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.invert_yaxis()

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper right', fontsize=7, ncol=2)

        ax.set_title(
            '%s  (n=%d, z=%.0f m | bias %s=%.2f  bias %s=%.2f  Δbias=%.2f mg/L)' % (
                stn, int(stats['n_casts']), stats['mean_z'],
                lbl0, stats['mean_bias0'],
                lbl1, stats['mean_bias1'],
                stats['bias_diff']),
            fontweight='bold', fontsize=9, loc='left')

    axes[-1].set_xlabel('Date')
    fig.suptitle(
        'Bottom DO Comparison: %s %s\nobs (black)  %s (red)  %s (blue)  '
        '[page %d/%d, sorted by Δbias = %s−%s]' % (
            otype, year_str, gtx0, gtx1, page + 1, n_pages, lbl1, lbl0),
        fontweight='bold', fontsize=12, y=1.01)
    fig.tight_layout()

    ff_str = 'bottom_DO_ts_compare_%s_%s_%s_vs_%s_p%02d' % (
        otype, year_str, gtx0, gtx1, page + 1)
    print('Plotting ' + ff_str)
    sys.stdout.flush()

    if args.testing:
        plt.show()
    else:
        plt.savefig(out_dir / (ff_str + '.png'), bbox_inches='tight')
        print('Saved to:\n %s' % (str(out_dir / (ff_str + '.png'))))

print('Done. %d stations across %d pages.' % (n_stn, n_pages))
