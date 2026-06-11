"""
Obs vs model profile comparison for the King County (kc_whidbeyBasin) and
Ecology (ecology_nc) CTD/bottle stations, run wb1_t0_xn11abbur00, 2024-2025.

For every in-domain cast it builds three profiles per variable:
  - obs            : observed values on observed depths
  - model (k)      : model values on the model's native sigma layers (z_rho)
  - model (Z)      : model values interpolated onto the observed depths

It then makes, for each otype x year:
  (1) "together"   : one multi-panel (variable) figure per group, all stations
                     overlaid as binned mean profiles with a shaded obs spread
                     envelope (obs solid, model dashed, colored by station).
  (2) "separately" : one multi-panel figure per station, every cast drawn
                     individually (obs solid, model dashed).
Each is produced in two depth modes: 'k' (native model layers) and 'Z' (on obs
depths), and for three groups: King County, Ecology, and combined.

Reads raw obs pickles + per-cast model extractions directly (not the combined
pickle) so cast ids map cleanly to the cast .nc files. Run on apogee.

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

out_dir = vf.out_dir(Ldir)
Lfun.make_dir(out_dir)

# station color lookup (stable across figures)
_palette = plt.get_cmap('tab20')
def station_color(stations):
    return {s: _palette(i % 20) for i, s in enumerate(sorted(stations))}


def load_cast(obs_cast, ds):
    """Build obs / model-native / model-on-obs-z profile DataFrames for one cast.
    Returns dict of three DataFrames indexed by depth z, columns = vf.VARS."""
    lon = float(obs_cast['lon'].iloc[0])
    lat = float(obs_cast['lat'].iloc[0])
    SA, CT = vf.model_SA_CT(ds, lon, lat)
    zk = ds['z_rho'].values                      # native model depths (ascending)
    oz = obs_cast['z'].to_numpy()                # obs depths

    obs_d, modk_d, modz_d = {}, {}, {}
    for vn in vf.VARS:
        obs_d[vn] = vf.obs_var(obs_cast, vn)
        mk = vf.model_var(ds, vn, SA=SA, CT=CT)
        if mk is None:
            modk_d[vn] = np.full(zk.shape, np.nan)
            modz_d[vn] = np.full(oz.shape, np.nan)
        else:
            modk_d[vn] = mk
            # interpolate native column onto obs depths (zk ascending)
            modz_d[vn] = np.interp(oz, zk, mk, left=np.nan, right=np.nan)

    obs_df = pd.DataFrame(obs_d); obs_df['z'] = oz
    modk_df = pd.DataFrame(modk_d); modk_df['z'] = zk
    modz_df = pd.DataFrame(modz_d); modz_df['z'] = oz
    return {'obs': obs_df, 'mod_k': modk_df, 'mod_Z': modz_df}


def load_group(otype, year):
    """Return {source: {station: [cast dicts]}} for one otype/year."""
    data = {}
    for source in vf.SOURCES:
        base = Ldir['LOo'] / 'obs' / source / otype
        info_fn = base / ('info_' + year + '.p')
        obs_fn = base / (year + '.p')
        if not info_fn.is_file() or not obs_fn.is_file():
            continue
        info = pd.read_pickle(info_fn)
        obs = pd.read_pickle(obs_fn)
        cast_dir = Ldir['LOo'] / 'extract' / gtx / 'cast' / (source + '_' + otype + '_' + year)
        src_d = {}
        for cid in info.index:
            fn = cast_dir / (str(int(cid)) + '.nc')
            if not fn.is_file():
                continue
            obs_cast = obs.loc[obs.cid == cid, :]
            if len(obs_cast) == 0:
                continue
            ds = xr.open_dataset(fn)
            station = info.loc[cid, 'name']
            src_d.setdefault(station, []).append(load_cast(obs_cast, ds))
            ds.close()
        if src_d:
            data[source] = src_d
    return data


def bin_profile(casts, vn, zkey):
    """Bin a list of cast DataFrames (key zkey in {'obs','mod_k','mod_Z'}) into
    2 m depth bins. Returns (zc, mean, lo, hi) of the variable vn, or None."""
    zs, vs = [], []
    for c in casts:
        df = c[zkey]
        zs.append(df['z'].to_numpy())
        vs.append(df[vn].to_numpy())
    z = np.concatenate(zs); v = np.concatenate(vs)
    good = np.isfinite(z) & np.isfinite(v)
    if good.sum() == 0:
        return None
    z, v = z[good], v[good]
    edges = np.arange(np.floor(z.min() / 2) * 2, 2, 2.0)
    if len(edges) < 2:
        return None
    idx = np.digitize(z, edges)
    zc, mean, lo, hi = [], [], [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() == 0:
            continue
        zc.append((edges[b-1] + edges[b]) / 2)
        mean.append(np.nanmean(v[m])); lo.append(np.nanmin(v[m])); hi.append(np.nanmax(v[m]))
    if not zc:
        return None
    return np.array(zc), np.array(mean), np.array(lo), np.array(hi)


def active_vars(group):
    """Variables that have any finite obs OR model data anywhere in the group."""
    out = []
    for vn in vf.VARS:
        ok = False
        for stations in group.values():
            for casts in stations.values():
                for c in casts:
                    if vf.has_data(c['obs'][vn].to_numpy()) or vf.has_data(c['mod_k'][vn].to_numpy()):
                        ok = True; break
                if ok: break
            if ok: break
        if ok:
            out.append(vn)
    return out


def panel_grid(n):
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    return nrow, ncol


def save(fig, name):
    if args.testing:
        plt.show()
    else:
        fn = out_dir / (name + '.png')
        fig.savefig(fn, bbox_inches='tight')
        print('Saved %s' % fn.name)
    plt.close(fig)


def plot_together(group, vns, mode, tag, title):
    """All stations overlaid as binned mean profiles, one panel per variable."""
    modkey = 'mod_k' if mode == 'k' else 'mod_Z'
    stations = [s for stns in group.values() for s in stns]
    cmap = station_color(stations)
    nrow, ncol = panel_grid(len(vns))
    pfun.start_plot(figsize=(5*ncol, 4*nrow), fs=11)
    fig = plt.figure()
    for i, vn in enumerate(vns):
        ax = fig.add_subplot(nrow, ncol, i+1)
        for stns in group.values():
            for station, casts in stns.items():
                col = cmap[station]
                ob = bin_profile(casts, vn, 'obs')
                mo = bin_profile(casts, vn, modkey)
                if ob is not None:
                    zc, mean, lo, hi = ob
                    ax.fill_betweenx(zc, lo, hi, color=col, alpha=0.12)
                    ax.plot(mean, zc, '-', color=col, lw=2,
                            label=station if i == 0 else None)
                if mo is not None:
                    zc, mean, lo, hi = mo
                    ax.plot(mean, zc, '--', color=col, lw=1.5)
        ax.set_xlim(vf.LIMS.get(vn, (None, None)))
        ax.set_xlabel(vn); ax.grid(True, alpha=0.3)
        if i % ncol == 0:
            ax.set_ylabel('z (m)')
        ax.text(.03, .03, vn, transform=ax.transAxes, fontweight='bold')
    fig.legend(loc='lower right', fontsize=7, ncol=2,
               title='solid=obs  dashed=model')
    fig.suptitle(title + '  [mode=%s]' % mode, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, 'profiles_together_%s_%s' % (tag, mode))


def plot_separately(group, vns, mode, base_tag):
    """One figure per station, every cast drawn individually."""
    modkey = 'mod_k' if mode == 'k' else 'mod_Z'
    nrow, ncol = panel_grid(len(vns))
    for stns in group.values():
        for station, casts in stns.items():
            pfun.start_plot(figsize=(5*ncol, 4*nrow), fs=11)
            fig = plt.figure()
            for i, vn in enumerate(vns):
                ax = fig.add_subplot(nrow, ncol, i+1)
                for c in casts:
                    odf, mdf = c['obs'], c[modkey]
                    ax.plot(odf[vn], odf['z'], '-', color='k', lw=1, alpha=0.5)
                    ax.plot(mdf[vn], mdf['z'], '--', color='tab:red', lw=1, alpha=0.5)
                ax.set_xlim(vf.LIMS.get(vn, (None, None)))
                ax.set_xlabel(vn); ax.grid(True, alpha=0.3)
                if i % ncol == 0:
                    ax.set_ylabel('z (m)')
                ax.text(.03, .03, vn, transform=ax.transAxes, fontweight='bold')
            fig.suptitle('%s  (n=%d casts)  black=obs  red dashed=model  [mode=%s]'
                         % (station, len(casts), mode), fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            st_safe = str(station).replace(' ', '_').replace('/', '-')
            save(fig, 'profiles_sep_%s_%s_%s' % (base_tag, st_safe, mode))
            if args.testing:
                return  # only the first station when testing


# ---- main loop ---------------------------------------------------------------
for otype in otypes:
    for year in years:
        data = load_group(otype, year)
        if not data:
            print('No data for %s %s' % (otype, year)); continue

        # groups: each source on its own, plus combined
        groups = {}
        for source, label in vf.SOURCES.items():
            if source in data:
                groups[label] = {source: data[source]}
        if len(data) > 1:
            groups['Combined'] = data

        for label, group in groups.items():
            vns = active_vars(group)
            if not vns:
                continue
            tag = '%s_%s_%s_%s' % (label.replace(' ', ''), otype, year, gtx)
            title = '%s  %s %s  %s' % (label, otype, year, gtx)
            for mode in ['k', 'Z']:
                plot_together(group, vns, mode, tag, title)
                plot_separately(group, vns, mode, tag)
                if args.testing:
                    break
            if args.testing:
                break
        if args.testing:
            break
    if args.testing:
        break

print('Done. Figures in %s' % out_dir)
