"""
Per-station model-comparison time series: ONE figure per station, containing
ALL variables (rows) x {surface, bottom} (columns), with every model run
overlaid plus obs.

Reads the per-station mooring files for each gtagex (one moor job per run).
Restricts to the 15 wb1-domain stations by default. Run on apogee.

    python 20260618_compare_station_timeseries.py \
        -gtxs wb1_t0_xn11abbur00,cas7_t2_x11b
"""

import sys
import re
import argparse
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'val_functions', str(Path(__file__).parent / '20260611_val_functions.py'))
vf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vf)

parser = argparse.ArgumentParser()
parser.add_argument('-gtxs', type=str, default='wb1_t0_xn11abbur00,cas7_t2_x11b')
parser.add_argument('-job', type=str, default=vf.MOOR_JOB)
parser.add_argument('-years', type=str, default='2024,2025')
parser.add_argument('-otypes', type=str, default='ctd,bottle')
parser.add_argument('-wb1_only', default=True, type=Lfun.boolean_string)
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

Ldir = Lfun.Lstart()
if '_mac' in Ldir['lo_env']:
    pass
else:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

gtxs = [g.strip() for g in args.gtxs.split(',') if g.strip()]
years = [y.strip() for y in args.years.split(',') if y.strip()]
otypes = [o.strip() for o in args.otypes.split(',') if o.strip()]
tag = '_vs_'.join(gtxs)
LEVELS = ['surface', 'bottom']

out_dir = vf.out_dir(Ldir)
Lfun.make_dir(out_dir)

# surface/bottom split (m): shallower than this = surface, deeper = bottom
SURF_GATE = -5.0
MODEL_COLORS = ['tab:red', 'tab:purple', 'tab:green', 'tab:orange']
mcolor = {gtx: MODEL_COLORS[i % len(MODEL_COLORS)] for i, gtx in enumerate(gtxs)}
OBS_STYLE = {
    'ctd':    dict(marker='o', color='k',        ms=4, ls=''),
    'bottle': dict(marker='^', color='tab:blue', ms=5, ls='', mfc='none', mew=1.0),
}
DEFAULT_OBS_STYLE = dict(marker='s', color='0.4', ms=4, ls='')


def sanitize(name):
    return str(name).replace(' ', '_')


def load_model(gtx):
    moor_dir = Ldir['LOo'] / 'extract' / gtx / 'moor' / args.job
    model = {}
    if not moor_dir.is_dir():
        print('*** no mooring folder for %s: %s' % (gtx, moor_dir))
        return model
    for fn in sorted(moor_dir.glob('*.nc')):
        station = re.sub(r'_\d{4}\.\d{2}\.\d{2}_\d{4}\.\d{2}\.\d{2}$', '', fn.stem)
        ds = xr.open_dataset(fn)
        lon = float(np.atleast_1d(ds['lon_rho'].values)[0])
        lat = float(np.atleast_1d(ds['lat_rho'].values)[0])
        SA, CT = vf.model_SA_CT(ds, lon, lat)
        zr = ds['z_rho'].values
        levels = {'surface': -1, 'bottom': 0}
        rec = {'time': pd.to_datetime(ds['ocean_time'].values),
               'surface': {}, 'bottom': {},
               'mdepth': {lev: float(np.nanmean(zr[:, k])) for lev, k in levels.items()}}
        for vn in vf.VARS:
            arr = vf.model_var(ds, vn, SA=SA, CT=CT)
            if arr is None:
                continue
            for lev, k in levels.items():
                rec[lev][vn] = np.asarray(arr)[:, k]
        model[station] = rec
        ds.close()
    return model


def load_obs():
    frames = []
    stn_group = {}
    for source in vf.SOURCES:
        for otype in otypes:
            base = Ldir['LOo'] / 'obs' / source / otype
            for year in years:
                fn = base / (year + '.p')
                if not fn.is_file():
                    continue
                df = pd.read_pickle(fn).copy()
                df['station'] = df['name'].map(sanitize)
                df['otype'] = otype
                df['cast'] = source + '_' + otype + '_' + year + '_' + df['cid'].astype(str)
                if 'DIN (uM)' not in df and {'NO3 (uM)', 'NH4 (uM)'} <= set(df.columns):
                    df['DIN (uM)'] = df['NO3 (uM)'] + df['NH4 (uM)']
                for s in df['station'].unique():
                    stn_group[s] = vf.SOURCES[source]
                frames.append(df)
    if not frames:
        return pd.DataFrame(), stn_group
    obs = pd.concat(frames, ignore_index=True)
    obs['time'] = pd.to_datetime(obs['time'], utc=True).dt.tz_localize(None)
    if 'DO (uM)' in obs.columns:
        obs['DO (mg L-1)'] = obs['DO (uM)'] * vf.DO_UM_TO_MGL
    return obs, stn_group


def obs_points(obs, station, vn, level, otype):
    if vn not in obs.columns:
        return np.array([]), np.array([])
    sdf = obs[(obs['station'] == station) & (obs['otype'] == otype)
              & np.isfinite(obs[vn])]
    if len(sdf) == 0:
        return np.array([]), np.array([])
    times, vals = [], []
    for _, g in sdf.groupby('cast'):
        if level == 'surface':
            gg = g[g['z'] >= SURF_GATE]           # within the top |SURF_GATE| m
            if len(gg) == 0:
                continue
            row = gg.loc[gg['z'].idxmax()]
        else:  # bottom: must be BELOW the surface zone (no surface-only dupes)
            gg = g[g['z'] < SURF_GATE]
            if len(gg) == 0:
                continue
            row = gg.loc[gg['z'].idxmin()]
        times.append(row['time']); vals.append(row[vn])
    if not times:
        return np.array([]), np.array([])
    order = np.argsort(times)
    return np.array(times)[order], np.array(vals)[order]


# ---- main --------------------------------------------------------------------
models = {gtx: load_model(gtx) for gtx in gtxs}
models = {gtx: m for gtx, m in models.items() if m}
if not models:
    print('No mooring data found for any gtx; run the extractor first.')
    sys.exit()
obs, stn_group = load_obs()

stations = sorted(set().union(*[set(m) for m in models.values()]))
if args.wb1_only:
    stations = [s for s in stations if s in vf.WB1_STATIONS_SAFE]

nvar = len(vf.VARS)
for station in stations:
    pfun.start_plot(figsize=(13, 2.0 * nvar), fs=9)
    fig, axes = plt.subplots(nvar, len(LEVELS), sharex=True,
                             figsize=(13, 2.0 * nvar))
    axes = np.atleast_2d(axes)
    for i, vn in enumerate(vf.VARS):
        for j, level in enumerate(LEVELS):
            ax = axes[i, j]
            for gtx in models:
                rec = models[gtx].get(station)
                if rec is None or vn not in rec[level]:
                    continue
                ax.plot(rec['time'], rec[level][vn], '-', color=mcolor[gtx],
                        lw=1.0, label='%s (z=%.0f m)' % (gtx, rec['mdepth'][level]))
            for otype in otypes:
                ot, ov = obs_points(obs, station, vn, level, otype)
                if len(ot) == 0:
                    continue
                ax.plot(ot, ov, label='obs %s' % otype,
                        **OBS_STYLE.get(otype, DEFAULT_OBS_STYLE))
            ax.set_ylim(vf.LIMS.get(vn, (None, None)))
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(level, fontweight='bold')
            if j == 0:
                ax.set_ylabel(vn, fontsize=8)
    axes[-1, 0].set_xlabel('Date'); axes[-1, -1].set_xlabel('Date')
    # one shared legend from the top-left panel
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc='upper right', fontsize=8, ncol=len(h))
    fig.suptitle('%s — model comparison time series\n%s'
                 % (station, ' vs '.join(models)), fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    name = 'cmp_station_ts_%s_%s' % (sanitize(station), tag)
    if args.testing:
        plt.show(); plt.close(fig); sys.exit()
    fig.savefig(out_dir / (name + '.png'), bbox_inches='tight')
    print('Saved %s.png' % name)
    plt.close(fig)

print('Done. Figures in %s' % out_dir)
