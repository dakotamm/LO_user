"""
Lowpassed model time series (surface & bottom) at the King County and Ecology
station locations, with discrete obs overlaid, for wb1_t0_xn11abbur00 2024-2025.

Reads the daily-lowpassed mooring extractions made by
20260611_extract_moorings.py and, for each variable and each level
(bottom = deepest model layer, surface = shallowest), plots a continuous model
line per station. Observations from the raw obs pickles are overlaid as points:
for each cast the bottom value is the deepest finite sample and the surface
value the shallowest finite sample (cf. the reference "Obs Max Depth /
Near Bottom" markers).

Stations are paged (<=6 per figure), produced per level x variable.

    python 20260611_plot_timeseries.py -gtx wb1_t0_xn11abbur00
    python 20260611_plot_timeseries.py -test True   # first figure only
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
parser.add_argument('-gtx', '--gtagex', type=str, default=vf.DEFAULT_GTX)
parser.add_argument('-job', type=str, default=vf.MOOR_JOB)
parser.add_argument('-years', type=str, default='2024,2025')
# ctd and bottle are overlaid as separate obs series (distinct markers). The
# surface depth gate below still keeps deep-only casts out of the surface panel.
parser.add_argument('-otypes', type=str, default='ctd,bottle')
# restrict to the 15 wb1-domain stations (useful when gtx is a larger grid)
parser.add_argument('-wb1_only', default=True, type=Lfun.boolean_string)
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

# surface/bottom split (m, z negative): samples shallower than this count as
# 'surface', deeper as 'bottom'. So a variable sampled only in the top 5 m
# (e.g. KC bottle Chl at ~-1 m) goes to surface only, not duplicated to bottom.
SURF_GATE = -5.0

# marker style per obs type
OBS_STYLE = {
    'ctd':    dict(marker='o', color='k',        ms=5, ls=''),
    'bottle': dict(marker='^', color='tab:blue', ms=6, ls='', mfc='none', mew=1.2),
}
DEFAULT_OBS_STYLE = dict(marker='s', color='tab:green', ms=5, ls='')

Ldir = Lfun.Lstart()
if '_mac' in Ldir['lo_env']:
    pass
else:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

gtx = args.gtagex
years = [y.strip() for y in args.years.split(',') if y.strip()]
otypes = [o.strip() for o in args.otypes.split(',') if o.strip()]

out_dir = vf.out_dir(Ldir)
Lfun.make_dir(out_dir)

moor_dir = Ldir['LOo'] / 'extract' / gtx / 'moor' / args.job


def sanitize(name):
    return str(name).replace(' ', '_')


# ---- load model lowpass series at surface & bottom for each station ----------
def load_model():
    """Return {station: {'time':..., 'bottom':{vn:arr}, 'surface':{vn:arr}}}."""
    model = {}
    if not moor_dir.is_dir():
        print('*** no mooring folder: %s' % moor_dir)
        return model
    for fn in sorted(moor_dir.glob('*.nc')):
        # strip the trailing _<ds0>_<ds1> (each YYYY.MM.DD) from the stem;
        # station names may themselves contain underscores (e.g. Poss_DO-2)
        station = re.sub(r'_\d{4}\.\d{2}\.\d{2}_\d{4}\.\d{2}\.\d{2}$', '', fn.stem)
        ds = xr.open_dataset(fn)
        lon = float(np.atleast_1d(ds['lon_rho'].values)[0])
        lat = float(np.atleast_1d(ds['lat_rho'].values)[0])
        SA, CT = vf.model_SA_CT(ds, lon, lat)        # (time, s_rho)
        zr = ds['z_rho'].values                      # (time, s_rho), ascending
        levels = {'bottom': 0, 'surface': -1}
        rec = {'time': pd.to_datetime(ds['ocean_time'].values),
               'bottom': {}, 'surface': {},
               # mean depth of the sampled sigma layer (m, negative)
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


# ---- load obs (all sources/otypes/years) into one frame ----------------------
def load_obs():
    """Return (obs DataFrame, {station -> source label}). The group map lets us
    split the figures into King County vs Ecology."""
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
                df = pd.read_pickle(fn)
                df = df.copy()
                df['station'] = df['name'].map(sanitize)
                df['otype'] = otype
                df['cast'] = source + '_' + otype + '_' + year + '_' + df['cid'].astype(str)
                if 'DIN (uM)' not in df and {'NO3 (uM)', 'NH4 (uM)'} <= set(df.columns):
                    df['DIN (uM)'] = df['NO3 (uM)'] + df['NH4 (uM)']
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
    """Bottom/surface obs points for one station/variable/otype: per cast take
    the deepest (bottom) or shallowest (surface) finite sample.
    Returns (time, val, z) arrays."""
    if vn not in obs.columns:
        return np.array([]), np.array([]), np.array([])
    sdf = obs[(obs['station'] == station) & (obs['otype'] == otype)
              & np.isfinite(obs[vn])]
    if len(sdf) == 0:
        return np.array([]), np.array([]), np.array([])
    times, vals, zs = [], [], []
    for _, g in sdf.groupby('cast'):
        if level == 'surface':
            gg = g[g['z'] >= SURF_GATE]           # within the top |SURF_GATE| m
            if len(gg) == 0:
                continue
            row = gg.loc[gg['z'].idxmax()]        # shallowest such sample
        else:  # bottom: must be BELOW the surface zone, so a surface-only
               # variable (e.g. KC bottle Chl at ~-1 m) is not duplicated here
            gg = g[g['z'] < SURF_GATE]
            if len(gg) == 0:
                continue
            row = gg.loc[gg['z'].idxmin()]        # deepest such sample
        times.append(row['time']); vals.append(row[vn]); zs.append(row['z'])
    if len(times) == 0:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(times)
    return np.array(times)[order], np.array(vals)[order], np.array(zs)[order]


# ---- main --------------------------------------------------------------------
model = load_model()
if not model:
    print('No mooring data found; run 20260611_extract_moorings.py first.')
    sys.exit()
obs, stn_group = load_obs()

stations = sorted(model.keys())
if args.wb1_only:
    stations = [s for s in stations if s in vf.WB1_STATIONS_SAFE]
n_per_page = 100   # effectively one page per group (all stations together)

# split mooring stations into their source groups (King County / Ecology)
groups = {}
for s in stations:
    groups.setdefault(stn_group.get(s, 'Other'), []).append(s)

for glabel, gstations in groups.items():
    gsafe = glabel.replace(' ', '')
    for level in ['bottom', 'surface']:
        vns = [vn for vn in vf.VARS
               if any(vn in model[s][level] for s in gstations)]
        for vn in vns:
            st_with = [s for s in gstations if vn in model[s][level]]
            n_pages = int(np.ceil(len(st_with) / n_per_page))
            for page in range(n_pages):
                page_st = st_with[page*n_per_page:(page+1)*n_per_page]
                pfun.start_plot(figsize=(14, 2.6*len(page_st)), fs=10)
                fig, axes = plt.subplots(len(page_st), 1, sharex=True,
                                         figsize=(14, 2.6*len(page_st)))
                if len(page_st) == 1:
                    axes = [axes]
                for ax, station in zip(axes, page_st):
                    rec = model[station]
                    ax.plot(rec['time'], rec[level][vn], '-', color='tab:red',
                            lw=1.2, label='model z=%.1f m' % rec['mdepth'][level])
                    for otype in otypes:
                        ot, ov, oz = obs_points(obs, station, vn, level, otype)
                        if len(ot) == 0:
                            continue
                        style = OBS_STYLE.get(otype, DEFAULT_OBS_STYLE)
                        ax.plot(ot, ov, label='%s z=%.0f m' % (otype, np.nanmean(oz)),
                                **style)
                    ax.set_ylabel(vn, fontsize=8)
                    ax.set_ylim(vf.LIMS.get(vn, (None, None)))
                    ax.grid(True, alpha=0.3)
                    ax.text(.01, .85, station, transform=ax.transAxes,
                            fontweight='bold', fontsize=9)
                    ax.legend(loc='upper right', fontsize=7, ncol=2)
                axes[-1].set_xlabel('Date')
                vsafe = vn.split(' ')[0].replace('/', '')
                fig.suptitle('%s — %s %s — %s %s'
                             % (glabel, level, vn, gtx, args.job),
                             fontweight='bold')
                fig.tight_layout(rect=[0, 0, 1, 0.98])
                name = 'ts_%s_%s_%s_%s' % (gsafe, level, vsafe, gtx)
                if args.testing:
                    plt.show(); plt.close(fig); sys.exit()
                fig.savefig(out_dir / (name + '.png'), bbox_inches='tight')
                print('Saved %s.png' % name)
                plt.close(fig)

print('Done. Figures in %s' % out_dir)
