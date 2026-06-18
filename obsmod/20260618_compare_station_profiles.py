"""
Per-station model-comparison profiles: ONE figure per station, containing ALL
variables as panels, each showing the period-mean profile (+/- 95% CI) for obs
and for every model run.

Obs are pooled across otypes (ctd + bottle) so every variable appears in the
single figure. Model profiles come from the per-cast extractions (native sigma
layers). Restricts to the 15 wb1-domain stations by default. Run on apogee.

    python 20260618_compare_station_profiles.py \
        -gtxs wb1_t0_xn11abbur00,cas7_t2_x11b
"""

import sys
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

out_dir = vf.out_dir(Ldir)
Lfun.make_dir(out_dir)

OBS_C = 'k'
MODEL_COLORS = ['tab:red', 'tab:purple', 'tab:green', 'tab:orange']
mcolor = {gtx: MODEL_COLORS[i % len(MODEL_COLORS)] for i, gtx in enumerate(gtxs)}
ZBIN = 2.0


def sanitize(name):
    return str(name).replace(' ', '_').replace('/', '-')


def obs_cast_dict(obs_cast):
    d = {'z': obs_cast['z'].to_numpy()}
    for vn in vf.VARS:
        d[vn] = vf.obs_var(obs_cast, vn)
    return d


def model_cast_dict(ds, lon, lat):
    SA, CT = vf.model_SA_CT(ds, lon, lat)
    z = ds['z_rho'].values
    d = {'z': z}
    for vn in vf.VARS:
        mk = vf.model_var(ds, vn, SA=SA, CT=CT)
        d[vn] = mk if mk is not None else np.full(np.shape(z), np.nan)
    return d


def load_all():
    """Pool across otypes. Returns (obs_site, {gtx: model_site}),
    each {station: [cast dicts]}."""
    obs_site = {}
    model_site = {gtx: {} for gtx in gtxs}
    for source in vf.SOURCES:
        for otype in otypes:
            for year in years:
                base = Ldir['LOo'] / 'obs' / source / otype
                info_fn = base / ('info_' + year + '.p')
                obs_fn = base / (year + '.p')
                if not info_fn.is_file() or not obs_fn.is_file():
                    continue
                info = pd.read_pickle(info_fn)
                obs = pd.read_pickle(obs_fn)
                for cid in info.index:
                    station = info.loc[cid, 'name']
                    if args.wb1_only and station not in vf.WB1_STATIONS:
                        continue
                    obs_cast = obs.loc[obs.cid == cid, :]
                    if len(obs_cast) == 0:
                        continue
                    obs_site.setdefault(station, []).append(obs_cast_dict(obs_cast))
                    lon = float(obs_cast['lon'].iloc[0])
                    lat = float(obs_cast['lat'].iloc[0])
                    for gtx in gtxs:
                        fn = (Ldir['LOo'] / 'extract' / gtx / 'cast'
                              / (source + '_' + otype + '_' + year) / (str(int(cid)) + '.nc'))
                        if not fn.is_file():
                            continue
                        ds = xr.open_dataset(fn)
                        model_site[gtx].setdefault(station, []).append(
                            model_cast_dict(ds, lon, lat))
                        ds.close()
    return obs_site, model_site


def avg_profile(casts, vn):
    z = np.concatenate([c['z'] for c in casts])
    v = np.concatenate([np.asarray(c[vn], dtype=float) for c in casts])
    good = np.isfinite(z) & np.isfinite(v)
    if good.sum() == 0:
        return None
    z, v = z[good], v[good]
    edges = np.arange(np.floor(z.min() / ZBIN) * ZBIN, ZBIN, ZBIN)
    if len(edges) < 2:
        return None
    idx = np.digitize(z, edges)
    zc, mean, ci = [], [], []
    for b in range(1, len(edges)):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        vb = v[m]
        zc.append((edges[b-1] + edges[b]) / 2)
        mean.append(np.nanmean(vb))
        ci.append(1.96 * np.nanstd(vb, ddof=1) / np.sqrt(n) if n >= 2 else 0.0)
    if not zc:
        return None
    return np.array(zc), np.array(mean), np.array(ci)


# ---- main --------------------------------------------------------------------
obs_site, model_site = load_all()
if not obs_site:
    print('No obs found.'); sys.exit()

nvar = len(vf.VARS)
ncol = 3
nrow = int(np.ceil(nvar / ncol))

for station in sorted(obs_site, key=str):
    obs_casts = obs_site[station]
    pfun.start_plot(figsize=(5 * ncol, 4 * nrow), fs=11)
    fig, axes = plt.subplots(nrow, ncol, sharey=True, figsize=(5 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, vn in enumerate(vf.VARS):
        ax = axes[i]
        series = [(obs_casts, OBS_C, 'obs')]
        series += [(model_site[gtx].get(station, []), mcolor[gtx], gtx) for gtx in gtxs]
        for casts, col, lab in series:
            if not casts:
                continue
            r = avg_profile(casts, vn)
            if r is None:
                continue
            zc, mean, ci = r
            ax.plot(mean, zc, '-', color=col, lw=2, label=lab)
            ax.fill_betweenx(zc, mean - ci, mean + ci, color=col, alpha=0.15)
        ax.set_xlim(vf.LIMS.get(vn, (None, None)))
        ax.set_ylim(vf.DEPTH_LIM)
        ax.set_xlabel(vn); ax.grid(True, alpha=0.3)
        ax.text(.03, .03, vn, transform=ax.transAxes, fontweight='bold')
        if i % ncol == 0:
            ax.set_ylabel('z (m)')
    for k in range(nvar, len(axes)):
        axes[k].axis('off')
    axes[0].legend(loc='lower right', fontsize=9)
    fig.suptitle('%s — period mean ± 95%% CI\n%s'
                 % (station, ' vs '.join(gtxs)), fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    name = 'cmp_station_prof_%s_%s' % (sanitize(station), tag)
    if args.testing:
        plt.show(); plt.close(fig); sys.exit()
    fig.savefig(out_dir / (name + '.png'), bbox_inches='tight')
    print('Saved %s.png' % name)
    plt.close(fig)

print('Done. Figures in %s' % out_dir)
