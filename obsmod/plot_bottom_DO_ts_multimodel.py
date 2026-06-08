"""
Plot bottom DO time series: obs vs one or more model versions.

Reads the combined obs+model pickles for each gtagex and plots obs
and all models as time series at each station. Works for any number
of models. Stations sorted by mean bias of the first model.

Testing on mac:
run plot_bottom_DO_ts_compare -gtx wb1_t0_xn11abbur00 wb1_t1_xn11abbur00 wb1_t0_xn11ab -year0 2024 -year1 2025 -otype ctd -test True
"""

import sys
import pandas as pd
import numpy as np
import pickle
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gtx', type=str, nargs='+')   # one or more gtagex values
parser.add_argument('-otype', type=str, default='ctd')
parser.add_argument('-year0', type=int)
parser.add_argument('-year1', type=int, default=0)
parser.add_argument('-stations', type=str, default='')  # comma-separated; empty = all
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

if args.gtx is None or len(args.gtx) == 0:
    print('*** Missing required argument: -gtx')
    sys.exit()
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

in_dir   = Ldir['LOo'] / 'obsmod'
gtx_list = args.gtx
otype    = args.otype
n_models = len(gtx_list)

DO_UM_TO_MGL = 32.0 / 1000.0

COLORS  = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
MARKERS = ['s', '^', 'D', 'o', 'v']

def short_gtx(gtx):
    parts = gtx.split('_')
    return '_'.join(parts[-2:]) if len(parts) >= 2 else gtx

def pickle_to_bottom_df(in_fn, gtx):
    """Load combined pickle, return one row per cast (deepest z)."""
    df_dict = pickle.load(open(in_fn, 'rb'))
    obs = df_dict['obs']
    mod = df_dict[gtx]
    if 'DO (uM)' not in obs.columns:
        return None
    wdf = pd.DataFrame({
        'cid':    obs['cid'].values,
        'z':      obs['z'].values,
        'time':   obs['time'].values,
        'lon':    obs['lon'].values,
        'lat':    obs['lat'].values,
        'obs_DO': obs['DO (uM)'].values * DO_UM_TO_MGL,
        'mod_DO': mod['DO (uM)'].values * DO_UM_TO_MGL,
    })
    if 'name' in obs.columns:
        wdf['station'] = obs['name'].values
    else:
        wdf['station'] = (obs['lon'].round(3).astype(str) + '_'
                          + obs['lat'].round(3).astype(str))
    wdf = wdf.dropna(subset=['obs_DO'])
    return wdf.loc[wdf.groupby('cid')['z'].idxmin()].copy()

# Load and concatenate across years; outer-merge all models on cid
frames = []
for yr in year_list:
    yr_str = str(yr)

    model_bottoms = []
    for gtx in gtx_list:
        in_fn = in_dir / ('combined_' + otype + '_' + yr_str + '_' + gtx + '.p')
        bdf = pickle_to_bottom_df(in_fn, gtx) if in_fn.is_file() else None
        model_bottoms.append(bdf)

    if all(b is None for b in model_bottoms):
        print('No data for any model in %s — skipping' % yr_str)
        continue

    # Build merged df: metadata from first available model, one mod_DO_i per model
    merged = None
    for i, bdf in enumerate(model_bottoms):
        mod_col = 'mod_DO_%d' % i
        if bdf is None:
            if merged is not None:
                merged[mod_col] = np.nan
            continue
        bdf = bdf.rename(columns={'mod_DO': mod_col})
        if merged is None:
            merged = bdf
        else:
            merged = merged.merge(bdf[['cid', mod_col]], on='cid', how='outer')

    frames.append(merged)

if len(frames) == 0:
    print('No data found for any requested year.')
    sys.exit()

wdf = pd.concat(frames, ignore_index=True)
wdf['time'] = pd.to_datetime(wdf['time'], utc=True).dt.tz_localize(None)

mod_cols = ['mod_DO_%d' % i for i in range(n_models)]
for i, col in enumerate(mod_cols):
    wdf['bias_%d' % i] = wdf[col] - wdf['obs_DO']

bot = wdf.copy()

if args.stations:
    station_filter = [s.strip() for s in args.stations.split(',')]
    bot = bot[bot['station'].isin(station_filter)]
    if len(bot) == 0:
        print('No data found for requested stations: %s' % args.stations)
        sys.exit()

agg_dict = {
    'n_casts': ('cid', 'count'),
    'mean_z':  ('z', 'mean'),
    'lon':     ('lon', 'mean'),
    'lat':     ('lat', 'mean'),
}
for i in range(n_models):
    agg_dict['mean_bias_%d' % i] = ('bias_%d' % i, 'mean')

stn_stats = bot.groupby('station').agg(**agg_dict).reset_index()
stn_stats = stn_stats[stn_stats['n_casts'] >= 2]
stn_stats = stn_stats.sort_values('mean_bias_0', ascending=False).reset_index(drop=True)

stations = stn_stats['station'].tolist()
n_stn = len(stations)

if n_stn == 0:
    print('No stations with >= 2 casts and valid bottom DO')
    sys.exit()

print('%d stations with >= 2 bottom DO casts' % n_stn)

out_dir = Ldir['LOo'] / 'obsmod_val_plots'
Lfun.make_dir(out_dir)

lbls     = [short_gtx(g) for g in gtx_list]
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
        ax    = axes[j]
        sdf   = bot[bot['station'] == stn].sort_values('time')
        stats = stn_stats[stn_stats['station'] == stn].iloc[0]

        ax.plot(sdf['time'], sdf['obs_DO'], 'o-', color='k',
                markersize=5, label='Obs')
        for i, (col, lbl) in enumerate(zip(mod_cols, lbls)):
            ax.plot(sdf['time'], sdf[col],
                    MARKERS[i % len(MARKERS)] + '--',
                    color=COLORS[i % len(COLORS)],
                    markersize=5, markerfacecolor='none', label=lbl)

        ax.axhline(y=2.0, color='gray', linestyle=':', linewidth=1)
        ax.set_ylabel('DO (mg/L)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        bias_str = '  '.join(['bias %s=%.2f' % (lbls[i], stats['mean_bias_%d' % i])
                               for i in range(n_models)])
        ax.set_title('%s  (n=%d, z=%.0f m | %s)' % (
                         stn, int(stats['n_casts']), stats['mean_z'], bias_str),
                     fontweight='bold', fontsize=9, loc='left')

    axes[-1].set_xlabel('Date')
    model_str = '  '.join(['%s (%s)' % (lbls[i], COLORS[i % len(COLORS)])
                            for i in range(n_models)])
    fig.suptitle('Bottom DO: %s %s\nobs (black)  %s  [page %d/%d]' % (
                     otype, year_str, model_str, page + 1, n_pages),
                 fontweight='bold', fontsize=12, y=1.01)
    fig.tight_layout()

    ff_str = 'bottom_DO_ts_multimodel_%s_%s_%s_p%02d' % (
        otype, year_str, '_vs_'.join(lbls), page + 1)
    print('Plotting ' + ff_str)
    sys.stdout.flush()

    if args.testing:
        plt.show()
    else:
        plt.savefig(out_dir / (ff_str + '.png'), bbox_inches='tight')
        print('Saved to:\n %s' % (str(out_dir / (ff_str + '.png'))))

print('Done. %d stations across %d pages.' % (n_stn, n_pages))
