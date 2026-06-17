"""
Two-model vs obs lowpassed time series at the King County + Ecology stations.

Overlays the lowpassed model line from EACH of two (or more) runs, plus the obs
points, for each variable (CT, SA, DO mg/L) at surface and bottom, split into
King County and Ecology figures.

Reads the per-station mooring files made by the fast extractor
(20260617_extract_moorings_fast.py) / extract_moor, one moor job per gtagex.
Run on apogee.

    python 20260617_compare_models_timeseries.py \
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
parser.add_argument('-gtxs', type=str,
                    default='wb1_t0_xn11abbur00,cas7_t2_x11b')  # comma list
parser.add_argument('-job', type=str, default=vf.MOOR_JOB)
parser.add_argument('-years', type=str, default='2024,2025')
parser.add_argument('-otypes', type=str, default='ctd,bottle')
# restrict to the 15 wb1-domain stations (useful when a gtx is a larger grid)
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

out_dir = vf.out_dir(Ldir)
Lfun.make_dir(out_dir)

SURF_GATE = -10.0
MODEL_COLORS = ['tab:red', 'tab:purple', 'tab:green', 'tab:orange']
OBS_STYLE = {
    'ctd':    dict(marker='o', color='k',        ms=5, ls=''),
    'bottle': dict(marker='^', color='tab:blue', ms=6, ls='', mfc='none', mew=1.2),
}
DEFAULT_OBS_STYLE = dict(marker='s', color='0.4', ms=5, ls='')


def sanitize(name):
    return str(name).replace(' ', '_')


def load_model(gtx):
    """{station: {'time', 'bottom':{vn:arr}, 'surface':{vn:arr}, 'mdepth':{}}}."""
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
        levels = {'bottom': 0, 'surface': -1}
        rec = {'time': pd.to_datetime(ds['ocean_time'].values),
               'bottom': {}, 'surface': {},
               'mdepth': {lev: float(np.nanmean(zr[:, k]))
                          for lev, k in levels.items()}}
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
        label = vf.SOURCES[source]
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
                for s in df['station'].unique():
                    stn_group[s] = label
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
        if level == 'bottom':
            row = g.loc[g['z'].idxmin()]
        else:
            gg = g[g['z'] >= SURF_GATE]
            if len(gg) == 0:
                continue
            row = gg.loc[gg['z'].idxmax()]
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
mcolor = {gtx: MODEL_COLORS[i % len(MODEL_COLORS)] for i, gtx in enumerate(models)}
obs, stn_group = load_obs()

# union of stations across models, grouped by source
stations = sorted(set().union(*[set(m) for m in models.values()]))
if args.wb1_only:
    stations = [s for s in stations if s in vf.WB1_STATIONS_SAFE]
groups = {}
for s in stations:
    groups.setdefault(stn_group.get(s, 'Other'), []).append(s)

for glabel, gstations in groups.items():
    gsafe = glabel.replace(' ', '')
    for level in ['bottom', 'surface']:
        vns = [vn for vn in vf.VARS
               if any(vn in models[g].get(s, {}).get(level, {})
                      for g in models for s in gstations)]
        for vn in vns:
            st_with = [s for s in gstations
                       if any(vn in models[g].get(s, {}).get(level, {}) for g in models)]
            pfun.start_plot(figsize=(14, 2.6*len(st_with)), fs=10)
            fig, axes = plt.subplots(len(st_with), 1, sharex=True,
                                     figsize=(14, 2.6*len(st_with)))
            if len(st_with) == 1:
                axes = [axes]
            for ax, station in zip(axes, st_with):
                for gtx in models:
                    rec = models[gtx].get(station)
                    if rec is None or vn not in rec[level]:
                        continue
                    ax.plot(rec['time'], rec[level][vn], '-', color=mcolor[gtx],
                            lw=1.2,
                            label='%s (z=%.0f m)' % (gtx, rec['mdepth'][level]))
                for otype in otypes:
                    ot, ov = obs_points(obs, station, vn, level, otype)
                    if len(ot) == 0:
                        continue
                    style = OBS_STYLE.get(otype, DEFAULT_OBS_STYLE)
                    ax.plot(ot, ov, label='obs %s' % otype, **style)
                ax.set_ylabel(vn, fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.text(.01, .85, station, transform=ax.transAxes,
                        fontweight='bold', fontsize=9)
                ax.legend(loc='upper right', fontsize=7, ncol=2)
            axes[-1].set_xlabel('Date')
            vsafe = vn.split(' ')[0].replace('/', '')
            fig.suptitle('%s — %s %s — model comparison\n%s'
                         % (glabel, level, vn, ' vs '.join(models)),
                         fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            name = 'cmp_ts_%s_%s_%s_%s' % (gsafe, level, vsafe, tag)
            if args.testing:
                plt.show(); plt.close(fig); sys.exit()
            fig.savefig(out_dir / (name + '.png'), bbox_inches='tight')
            print('Saved %s.png' % name)
            plt.close(fig)

print('Done. Figures in %s' % out_dir)
