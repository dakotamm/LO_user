"""
Obs vs model profile comparison for the King County (kc_whidbeyBasin) and
Ecology (ecology_nc) stations, run wb1_t0_xn11abbur00, 2024-2025.

Two sets of figures, each per site and per otype (ctd and bottle as separate
plots), with one panel per variable (SA, CT, DO):

  Set 1  "avg"     : period-averaged profiles. All of the site's casts are
                     pooled and binned by depth; the obs mean profile and the
                     model mean profile are each drawn with a 95% confidence
                     band.
  Set 2  "closest" : the single cast closest in time to a target date
                     (default 2025-12-03) at the site, obs profile and the
                     matching model profile.

Reads raw obs pickles + per-cast model extractions directly (so cast ids map
cleanly to the cast .nc files). Run on apogee.

    python 20260611_plot_profiles.py -gtx wb1_t0_xn11abbur00
    python 20260611_plot_profiles.py -test True   # show first figure only
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

# load the shared helper module living next to this script (name starts with a
# digit, so it cannot be imported normally)
_spec = importlib.util.spec_from_file_location(
    'val_functions', str(Path(__file__).parent / '20260611_val_functions.py'))
vf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vf)

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str, default=vf.DEFAULT_GTX)
parser.add_argument('-years', type=str, default='2024,2025')
parser.add_argument('-otypes', type=str, default='ctd,bottle')
parser.add_argument('-date', type=str, default='2025.12.03')  # set-2 target date
# restrict to the 15 wb1-domain stations (useful when gtx is a larger grid)
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

gtx = args.gtagex
years = [y.strip() for y in args.years.split(',') if y.strip()]
otypes = [o.strip() for o in args.otypes.split(',') if o.strip()]
target_date = pd.Timestamp(args.date.replace('.', '-'))

out_dir = vf.out_dir(Ldir)
Lfun.make_dir(out_dir)

OBS_C, MOD_C = 'k', 'tab:red'   # obs / model colors
ZBIN = 2.0                      # depth bin size [m] for averaging


def load_cast(obs_cast, ds):
    """Build obs (on obs z) and model-native (on z_rho) profile DataFrames for
    one cast, columns = vf.VARS, plus the cast time."""
    lon = float(obs_cast['lon'].iloc[0])
    lat = float(obs_cast['lat'].iloc[0])
    SA, CT = vf.model_SA_CT(ds, lon, lat)
    zk = ds['z_rho'].values
    oz = obs_cast['z'].to_numpy()
    obs_d, mod_d = {}, {}
    for vn in vf.VARS:
        obs_d[vn] = vf.obs_var(obs_cast, vn)
        mk = vf.model_var(ds, vn, SA=SA, CT=CT)
        mod_d[vn] = mk if mk is not None else np.full(zk.shape, np.nan)
    obs_df = pd.DataFrame(obs_d); obs_df['z'] = oz
    mod_df = pd.DataFrame(mod_d); mod_df['z'] = zk
    t = pd.to_datetime(obs_cast['time'].iloc[0], utc=True).tz_localize(None)
    return {'obs': obs_df, 'mod': mod_df, 'time': t}


def load_site_casts(otype):
    """Return {station: [cast dicts]} pooled across all years for one otype."""
    site = {}
    for source in vf.SOURCES:
        for year in years:
            base = Ldir['LOo'] / 'obs' / source / otype
            info_fn = base / ('info_' + year + '.p')
            obs_fn = base / (year + '.p')
            if not info_fn.is_file() or not obs_fn.is_file():
                continue
            info = pd.read_pickle(info_fn)
            obs = pd.read_pickle(obs_fn)
            cast_dir = (Ldir['LOo'] / 'extract' / gtx / 'cast'
                        / (source + '_' + otype + '_' + year))
            for cid in info.index:
                fn = cast_dir / (str(int(cid)) + '.nc')
                if not fn.is_file():
                    continue
                obs_cast = obs.loc[obs.cid == cid, :]
                if len(obs_cast) == 0:
                    continue
                ds = xr.open_dataset(fn)
                station = info.loc[cid, 'name']
                site.setdefault(station, []).append(load_cast(obs_cast, ds))
                ds.close()
    return site


def avg_profile(casts, vn, key):
    """Pool all casts' (z, vn) samples (key 'obs' or 'mod'), bin by depth, and
    return (zc, mean, ci95). ci95 is the 95% CI half-width (0 where n<2)."""
    zs, vs = [], []
    for c in casts:
        df = c[key]
        zs.append(df['z'].to_numpy()); vs.append(df[vn].to_numpy())
    z = np.concatenate(zs); v = np.concatenate(vs)
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


def new_fig():
    n = len(vf.VARS)
    pfun.start_plot(figsize=(5 * n, 6), fs=12)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]
    return fig, axes


def finish(fig, axes, title, name):
    for ax, vn in zip(axes, vf.VARS):
        ax.set_xlim(vf.LIMS.get(vn, (None, None)))
        ax.set_ylim(vf.DEPTH_LIM)
        ax.set_xlabel(vn); ax.grid(True, alpha=0.3)
        ax.text(.03, .03, vn, transform=ax.transAxes, fontweight='bold')
    axes[0].set_ylabel('z (m)')
    axes[0].legend(loc='lower right', fontsize=9)
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if args.testing:
        plt.show()
    else:
        fig.savefig(out_dir / (name + '.png'), bbox_inches='tight')
        print('Saved %s.png' % name)
    plt.close(fig)


def st_safe(station):
    return str(station).replace(' ', '_').replace('/', '-')


def plot_avg(station, casts, otype):
    fig, axes = new_fig()
    for ax, vn in zip(axes, vf.VARS):
        for key, col, lab in [('obs', OBS_C, 'obs'), ('mod', MOD_C, 'model')]:
            r = avg_profile(casts, vn, key)
            if r is None:
                continue
            zc, mean, ci = r
            ax.plot(mean, zc, '-', color=col, lw=2, label=lab)
            ax.fill_betweenx(zc, mean - ci, mean + ci, color=col, alpha=0.2)
    finish(fig, axes,
           '%s  %s  %s — period mean ± 95%% CI (n=%d casts)'
           % (station, otype, gtx, len(casts)),
           'profiles_avg_%s_%s_%s' % (st_safe(station), otype, gtx))


def plot_closest(station, casts, otype):
    c = min(casts, key=lambda c: abs(c['time'] - target_date))
    fig, axes = new_fig()
    for ax, vn in zip(axes, vf.VARS):
        odf, mdf = c['obs'], c['mod']
        ax.plot(odf[vn], odf['z'], '-o', color=OBS_C, lw=1.5, ms=3, label='obs')
        ax.plot(mdf[vn], mdf['z'], '--', color=MOD_C, lw=1.5, label='model')
    finish(fig, axes,
           '%s  %s  %s — cast closest to %s (%s)'
           % (station, otype, gtx, target_date.date(), c['time'].date()),
           'profiles_closest_%s_%s_%s' % (st_safe(station), otype, gtx))


# ---- main --------------------------------------------------------------------
for otype in otypes:
    site = load_site_casts(otype)
    if args.wb1_only:
        site = {st: c for st, c in site.items() if st in vf.WB1_STATIONS}
    if not site:
        print('No data for %s' % otype); continue
    for station in sorted(site, key=str):
        casts = site[station]
        plot_avg(station, casts, otype)
        plot_closest(station, casts, otype)
        if args.testing:
            sys.exit()

print('Done. Figures in %s' % out_dir)
