"""
Two-model vs obs profile comparison at the King County + Ecology stations.

Per site and per otype (ctd / bottle as separate figures), one panel per
variable (CT, SA, DO mg/L):

  Set 1  "avg"     : period-mean profile with a 95% CI band for obs and for
                     EACH model run (model native sigma layers; obs on obs
                     depths), all pooled over the site's casts.
  Set 2  "closest" : the cast closest in time to a target date (default
                     2025-12-03), obs profile and each model's profile.

Reads raw obs pickles + per-cast model extractions for each run, matching cast
ids across runs. Restricts to the 15 wb1-domain stations by default (-wb1_only).
Run on apogee.

    python 20260617_compare_models_profiles.py \
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
parser.add_argument('-gtxs', type=str,
                    default='wb1_t0_xn11abbur00,cas7_t2_x11b')
parser.add_argument('-years', type=str, default='2024,2025')
parser.add_argument('-otypes', type=str, default='ctd,bottle')
parser.add_argument('-date', type=str, default='2025.12.03')  # set-2 target
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
target_date = pd.Timestamp(args.date.replace('.', '-'))
tag = '_vs_'.join(gtxs)

out_dir = vf.out_dir(Ldir)
Lfun.make_dir(out_dir)

OBS_C = 'k'
MODEL_COLORS = ['tab:red', 'tab:purple', 'tab:green', 'tab:orange']
mcolor = {gtx: MODEL_COLORS[i % len(MODEL_COLORS)] for i, gtx in enumerate(gtxs)}
ZBIN = 2.0


def cast_dict(z, ds_or_df, t, lon=None, lat=None):
    """Build {'z', 'time', <vn>...} from either an obs DataFrame or a model
    cast Dataset (when lon/lat are given, treat as model and convert)."""
    d = {'z': z, 'time': t}
    if lon is None:
        for vn in vf.VARS:
            d[vn] = vf.obs_var(ds_or_df, vn)
    else:
        SA, CT = vf.model_SA_CT(ds_or_df, lon, lat)
        for vn in vf.VARS:
            mk = vf.model_var(ds_or_df, vn, SA=SA, CT=CT)
            d[vn] = mk if mk is not None else np.full(np.shape(z), np.nan)
    return d


def load_all(otype):
    """Return (obs_site, {gtx: model_site}); each is {station: [cast dicts]}."""
    obs_site = {}
    model_site = {gtx: {} for gtx in gtxs}
    for source in vf.SOURCES:
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
                t = pd.to_datetime(obs_cast['time'].iloc[0], utc=True).tz_localize(None)
                obs_site.setdefault(station, []).append(
                    cast_dict(obs_cast['z'].to_numpy(), obs_cast, t))
                lon = float(obs_cast['lon'].iloc[0])
                lat = float(obs_cast['lat'].iloc[0])
                for gtx in gtxs:
                    fn = (Ldir['LOo'] / 'extract' / gtx / 'cast'
                          / (source + '_' + otype + '_' + year) / (str(int(cid)) + '.nc'))
                    if not fn.is_file():
                        continue
                    ds = xr.open_dataset(fn)
                    model_site[gtx].setdefault(station, []).append(
                        cast_dict(ds['z_rho'].values, ds, t, lon=lon, lat=lat))
                    ds.close()
    return obs_site, model_site


def avg_profile(casts, vn):
    """Pool casts' (z, vn) samples, bin by depth -> (zc, mean, ci95)."""
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


def closest_cast(casts):
    return min(casts, key=lambda c: abs(c['time'] - target_date))


def new_fig():
    n = len(vf.VARS)
    pfun.start_plot(figsize=(5 * n, 6), fs=12)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]
    return fig, axes


def finish(fig, axes, title, name):
    for ax, vn in zip(axes, vf.VARS):
        ax.set_xlim(vf.LIMS.get(vn, (None, None)))
        ax.set_xlabel(vn); ax.grid(True, alpha=0.3)
        ax.text(.03, .03, vn, transform=ax.transAxes, fontweight='bold')
    axes[0].set_ylabel('z (m)')
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if args.testing:
        plt.show()
    else:
        fig.savefig(out_dir / (name + '.png'), bbox_inches='tight')
        print('Saved %s.png' % name)
    plt.close(fig)


def st_safe(station):
    return str(station).replace(' ', '_').replace('/', '-')


def plot_avg(station, obs_casts, model_casts, otype):
    fig, axes = new_fig()
    for ax, vn in zip(axes, vf.VARS):
        series = [(obs_casts, OBS_C, 'obs')]
        series += [(model_casts.get(gtx, []), mcolor[gtx], gtx) for gtx in gtxs]
        for casts, col, lab in series:
            if not casts:
                continue
            r = avg_profile(casts, vn)
            if r is None:
                continue
            zc, mean, ci = r
            ax.plot(mean, zc, '-', color=col, lw=2, label=lab)
            ax.fill_betweenx(zc, mean - ci, mean + ci, color=col, alpha=0.15)
    finish(fig, axes, '%s  %s — period mean ± 95%% CI\n%s'
           % (station, otype, ' vs '.join(gtxs)),
           'cmp_prof_avg_%s_%s_%s' % (st_safe(station), otype, tag))


def plot_closest(station, obs_casts, model_casts, otype):
    fig, axes = new_fig()
    oc = closest_cast(obs_casts)
    mc = {gtx: closest_cast(model_casts[gtx])
          for gtx in gtxs if model_casts.get(gtx)}
    for ax, vn in zip(axes, vf.VARS):
        ax.plot(oc[vn], oc['z'], '-o', color=OBS_C, lw=1.5, ms=3, label='obs')
        for gtx in gtxs:
            if gtx in mc:
                ax.plot(mc[gtx][vn], mc[gtx]['z'], '--', color=mcolor[gtx],
                        lw=1.5, label=gtx)
    finish(fig, axes, '%s  %s — cast closest to %s\n%s'
           % (station, otype, target_date.date(), ' vs '.join(gtxs)),
           'cmp_prof_closest_%s_%s_%s' % (st_safe(station), otype, tag))


# ---- main --------------------------------------------------------------------
for otype in otypes:
    obs_site, model_site = load_all(otype)
    if not obs_site:
        print('No obs for %s' % otype); continue
    for station in sorted(obs_site, key=str):
        obs_casts = obs_site[station]
        model_casts = {gtx: model_site[gtx].get(station, []) for gtx in gtxs}
        plot_avg(station, obs_casts, model_casts, otype)
        plot_closest(station, obs_casts, model_casts, otype)
        if args.testing:
            sys.exit()

print('Done. Figures in %s' % out_dir)
