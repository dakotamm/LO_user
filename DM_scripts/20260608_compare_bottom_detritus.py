"""
Compare bottom detritus between two model versions.

Pure model/model comparison — no obs values used. Station names and cast
locations come from the obs info pickle (metadata only). Cast NetCDF files
produced by extract_casts_fast.py are read for both models.

Available detritus variables: detritus (small), Ldetritus (large)

Testing on mac:
run 20260608_compare_bottom_detritus -gtx0 wb1_t0_xn11abbur00 -gtx1 wb1_t1_xn11abbur00 -source kc_whidbeyBasin -year0 2024 -year1 2025 -otype ctd -test True
run 20260608_compare_bottom_detritus -gtx0 wb1_t0_xn11abbur00 -gtx1 wb1_t1_xn11abbur00 -source kc_whidbeyBasin -year0 2024 -year1 2025 -otype ctd -var Ldetritus -lp 30 -test True
"""

import sys
import pandas as pd
import numpy as np
import xarray as xr
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun, zfun

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gtx0', type=str)
parser.add_argument('-gtx1', type=str)
parser.add_argument('-source', type=str)            # e.g. kc_whidbeyBasin
parser.add_argument('-otype', type=str, default='ctd')
parser.add_argument('-year0', type=int)
parser.add_argument('-year1', type=int, default=0)
parser.add_argument('-var', type=str, default='detritus')  # detritus or Ldetritus
parser.add_argument('-stations', type=str, default='')     # comma-separated; empty = all
parser.add_argument('-lp', type=int, default=0)            # low-pass window in days (0 = off)
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

for a in ['gtx0', 'gtx1', 'source', 'year0']:
    if getattr(args, a) is None:
        print('*** Missing required argument: -%s' % a)
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

gtx0    = args.gtx0
gtx1    = args.gtx1
source  = args.source
otype   = args.otype
varname = args.var
lp_days = args.lp

station_filter = ([s.strip() for s in args.stations.split(',')]
                  if args.stations else None)

# Build cast metadata from info pickles (station names, cids, times — no obs values)
meta_frames = []
for yr in year_list:
    yr_str = str(yr)
    info_fn = Ldir['LOo'] / 'obs' / source / otype / ('info_' + yr_str + '.p')
    if not info_fn.is_file():
        print('No info pickle for %s %s — skipping' % (source, yr_str))
        continue
    info = pd.read_pickle(info_fn)
    mf = pd.DataFrame({
        'cid':     info['cid'].values,
        'z':       info['z'].values,
        'time':    pd.to_datetime(info['time'].values, utc=True).tz_localize(None),
        'station': info['name'].values if 'name' in info.columns else (
                   info['lon'].round(3).astype(str) + '_' + info['lat'].round(3).astype(str)),
        'year':    yr_str,
    })
    meta_frames.append(mf)

if len(meta_frames) == 0:
    print('No info pickles found for source "%s".' % source)
    sys.exit()

meta = pd.concat(meta_frames, ignore_index=True)

# Keep only the deepest level per cast (bottom)
bot_meta = meta.loc[meta.groupby('cid')['z'].idxmin()].copy()

if station_filter:
    bot_meta = bot_meta[bot_meta['station'].isin(station_filter)]
    if len(bot_meta) == 0:
        print('No data found for requested stations: %s' % args.stations)
        sys.exit()

# Read bottom detritus from model cast NetCDF files
records = []
n_missing = {gtx0: 0, gtx1: 0}

for _, row in bot_meta.iterrows():
    cid    = int(row['cid'])
    obs_z  = row['z']
    yr_str = row['year']

    vals = {}
    for gtx in [gtx0, gtx1]:
        cast_fn = (Ldir['LOo'] / 'extract' / gtx / 'cast'
                   / (source + '_' + otype + '_' + yr_str)
                   / (str(cid) + '.nc'))
        val = np.nan
        if cast_fn.is_file():
            ds = xr.open_dataset(cast_fn)
            if varname in ds.data_vars:
                iz  = zfun.find_nearest_ind(ds.z_rho.values, obs_z)
                val = float(ds[varname][iz].values)
            else:
                n_missing[gtx] += 1
            ds.close()
        vals[gtx] = val

    records.append({
        'cid':      cid,
        'station':  row['station'],
        'time':     row['time'],
        'z':        obs_z,
        'mod0_val': vals[gtx0],
        'mod1_val': vals[gtx1],
    })

for gtx, n in n_missing.items():
    if n > 0:
        print('Warning: "%s" not found in %d cast files for %s' % (varname, n, gtx))

if len(records) == 0:
    print('No records found.')
    sys.exit()

df = pd.DataFrame(records)
df = df.dropna(subset=['mod0_val', 'mod1_val'])

if len(df) == 0:
    print('Variable "%s" not found in any cast files. '
          'Check variable name (available: detritus, Ldetritus).' % varname)
    sys.exit()

df['diff'] = df['mod1_val'] - df['mod0_val']

stn_stats = df.groupby('station').agg(
    mean_mod0=('mod0_val', 'mean'),
    mean_mod1=('mod1_val', 'mean'),
    mean_diff=('diff', 'mean'),
    n_casts=('cid', 'count'),
    mean_z=('z', 'mean'),
).reset_index()
stn_stats = stn_stats[stn_stats['n_casts'] >= 2]
stn_stats = stn_stats.sort_values('mean_diff', ascending=False).reset_index(drop=True)

stations = stn_stats['station'].tolist()
n_stn = len(stations)

if n_stn == 0:
    print('No stations with >= 2 casts')
    sys.exit()

print('%d stations with >= 2 bottom %s casts' % (n_stn, varname))

out_dir = Ldir['LOo'] / 'plots'
Lfun.make_dir(out_dir)

def short_gtx(gtx):
    parts = gtx.split('_')
    return '_'.join(parts[-2:]) if len(parts) >= 2 else gtx

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
        sdf   = df[df['station'] == stn].sort_values('time').copy()
        stats = stn_stats[stn_stats['station'] == stn].iloc[0]

        if lp_days > 0:
            sdf = sdf.set_index('time')
            win = '%dD' % lp_days
            sdf['mod0_lp'] = sdf['mod0_val'].rolling(win, center=True, min_periods=1).mean()
            sdf['mod1_lp'] = sdf['mod1_val'].rolling(win, center=True, min_periods=1).mean()
            sdf = sdf.reset_index()

            ax.plot(sdf['time'], sdf['mod0_val'], 's', color='tab:red',
                    markersize=4, alpha=0.3, markerfacecolor='none')
            ax.plot(sdf['time'], sdf['mod1_val'], '^', color='tab:blue',
                    markersize=4, alpha=0.3, markerfacecolor='none')
            ax.plot(sdf['time'], sdf['mod0_lp'], '-', color='tab:red',
                    linewidth=2, label=lbl0)
            ax.plot(sdf['time'], sdf['mod1_lp'], '-', color='tab:blue',
                    linewidth=2, label=lbl1)
        else:
            ax.plot(sdf['time'], sdf['mod0_val'], 's--', color='tab:red',
                    markersize=5, markerfacecolor='none', label=lbl0)
            ax.plot(sdf['time'], sdf['mod1_val'], '^--', color='tab:blue',
                    markersize=5, markerfacecolor='none', label=lbl1)

        ax.set_ylabel('%s (mmol N/m³)' % varname)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        ax.set_title(
            '%s  (n=%d, z=%.0f m | mean %s=%.3f  mean %s=%.3f  Δ=%.3f)' % (
                stn, int(stats['n_casts']), stats['mean_z'],
                lbl0, stats['mean_mod0'],
                lbl1, stats['mean_mod1'],
                stats['mean_diff']),
            fontweight='bold', fontsize=9, loc='left')

    axes[-1].set_xlabel('Date')
    lp_str = '  [%d-day LP filter]' % lp_days if lp_days > 0 else ''
    fig.suptitle(
        'Bottom %s: %s %s %s%s\n%s (red) vs %s (blue)  '
        '[page %d/%d, sorted by Δ = %s−%s]' % (
            varname, source, otype, year_str, lp_str, gtx0, gtx1,
            page + 1, n_pages, lbl1, lbl0),
        fontweight='bold', fontsize=12, y=1.01)
    fig.tight_layout()

    lp_tag = '_lp%dd' % lp_days if lp_days > 0 else ''
    ff_str = '20260608_bottom_%s_%s_%s_%s_%s_vs_%s%s_p%02d' % (
        varname, source, otype, year_str, gtx0, gtx1, lp_tag, page + 1)
    print('Plotting ' + ff_str)
    sys.stdout.flush()

    if args.testing:
        plt.show()
    else:
        plt.savefig(out_dir / (ff_str + '.png'), bbox_inches='tight')
        print('Saved to:\n %s' % (str(out_dir / (ff_str + '.png'))))

print('Done. %d stations across %d pages.' % (n_stn, n_pages))
