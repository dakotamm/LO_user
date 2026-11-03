"""
Plot the time-varying ellipses from 20261027_tidal_ellipses_windowed.py.

  python 20261027_tidal_ellipses_windowed_plot.py -job pc_cove -dom box -fld bar
  python 20261027_tidal_ellipses_windowed_plot.py -job pc_cove -dom box -fld top
  python 20261027_tidal_ellipses_windowed_plot.py -job sp_head_tide -dom sp_head -fld bar

-dom / -fld as in 20261008_tidal_ellipses_plot.py (box or a section_lines
polygon; bar, surf/bot for sp_head_tide, top/bot from the 3D fit for pc_cove).
Area means are over the cells drawn (wet, inside wb.p, inside -dom's polygon).
-cons  constituents for the monthly map grids (default M2,K1)

Figures (to LO_output/DM_outs/20261027_tidal_ellipses_windowed/):
  monthly_maps_<job>_<dom>_<fld>_<CON>.png  semi-major axis + ellipses, one
      panel per month, one color scale and one glyph scale for all panels
  monthly_series_<job>_<dom>_<fld>.png     area-mean semi-major axis by
      month (absolute and relative to each constituent's 2-year mean), the same
      for zeta (the forcing), eccentricity, Eulerian residual speed
  monthly_vertical_<job>_<CON>.png          (3D) month x depth at the mouth
      column: section-mean semi-major axis and eccentricity
  bands_index_<job>.png                     the spring/neap (D2) and
      tropic/equatorial (D1) indices and the tercile bins
  springneap_<job>_<dom>_<fld>.png          D2 ellipses spring vs neap
  diurnal_<job>_<dom>_<fld>.png             D1 ellipses tropic vs equatorial,
      and the current form ratio D1/D2
  bands_vertical_<job>.png                  (3D) mouth-section profiles,
      spring vs neap (D2) and tropic vs equatorial (D1)
  bands_residual_<job>_<dom>_<fld>.png      Eulerian residual in each bin
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
import cmocean

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-job', default='pc_cove', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-dom', default='box', type=str)
parser.add_argument('-fld', default='bar', type=str)
parser.add_argument('-cons', default='M2,K1', type=str)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
ell_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'tidal_ellipses'
stem = args.job + '_' + args.ds0 + '_' + args.ds1
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_tidal_ellipses_windowed'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
PAD_CELLS = 10
NGLYPH = 16
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
# Okabe-Ito, fixed order (validated: CVD dE >= 8.5); semidiurnal blue/green, diurnal orange, M4 purple
SCOL = {'M2': '#0072B2', 'S2': '#56B4E9', 'N2': '#009E73', 'K1': '#D55E00', 'O1': '#E69F00', 'M4': '#CC79A7'}
BINCOL = {'spring': '#0072B2', 'neap': '#56B4E9', 'tropic': '#D55E00', 'equatorial': '#E69F00', 'all': 'k'}
FLDLAB = {'bar': 'depth-averaged', 'surf': 'top layer', 'top': 'top layer', 'bot': 'bottom layer'}

def save(fig, name):
    fn = out_dir / name
    fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white') # white, as 20261008_tidal_ellipses_plot.py
    plt.close(fig)
    print('Saved ' + str(fn))

M = xr.open_dataset(ell_dir / (stem + '_monthly.nc')) if (ell_dir / (stem + '_monthly.nc')).is_file() else None
Bd = xr.open_dataset(ell_dir / (stem + '_bands.nc')) if (ell_dir / (stem + '_bands.nc')).is_file() else None
ref = M if M is not None else Bd
lon = ref.lon_rho.values
lat = ref.lat_rho.values
dx = np.nanmedian(np.diff(lon[0]))
dy = np.nanmedian(np.diff(lat[:, 0]))
plon, plat = pfun.get_plon_plat(lon, lat)

def poly_mask(df):
    pth = MplPath(np.column_stack([df.x.values, df.y.values]))
    return pth.contains_points(np.column_stack([lon.ravel(), lat.ravel()])).reshape(lon.shape)

wb = pd.read_pickle(sect_dir / 'wb.p')
show = (ref.mask_rho.values == 1) & poly_mask(wb)
if args.dom == 'box':
    aa = [plon.min(), plon.max(), plat.min(), plat.max()]
    region = show.copy()
else:
    df = wb if args.dom in ['full', 'wb'] else pd.read_pickle(sect_dir / (args.dom + '.p'))
    aa = [df.x.min() - PAD_CELLS * dx, df.x.max() + PAD_CELLS * dx, df.y.min() - PAD_CELLS * dy, df.y.max() + PAD_CELLS * dy]
    aa = [max(aa[0], plon.min()), min(aa[1], plon.max()), max(aa[2], plat.min()), min(aa[3], plat.max())]
    region = show & poly_mask(df)
inwin = (lon >= aa[0]) & (lon <= aa[1]) & (lat >= aa[2]) & (lat <= aa[3])
region &= inwin
lat0 = 0.5 * (aa[2] + aa[3])
m_per_deg = 111.32e3
cell_m = dy * m_per_deg
step = max(1, int(round((aa[1] - aa[0]) / dx / NGLYPH)))
asp = (aa[3] - aa[2]) / ((aa[1] - aa[0]) * np.cos(np.pi * lat0 / 180))

def get(ds, vn, **sel):
    """Field vn for -fld; top/bot come from the 3D fit when there is no 2D one."""
    if args.fld in ['top', 'bot'] and (args.fld + '_' + vn) not in ds:
        return ds['3d_' + vn].sel(**sel).isel(s_rho=-1 if args.fld == 'top' else 0).values
    return ds[args.fld + '_' + vn].sel(**sel).values

def masked(f):
    return np.where(show, f, np.nan)

def amean(f):
    return np.nanmean(f[..., region], axis=-1)

def setup(ax, ticks=True):
    pfun.add_coast(ax, color='gray', linewidth=0.5)
    ax.axis(aa)
    pfun.dar(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    if not ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(**GRID)

def letter(ax, s):
    ax.text(0.025, 0.95, s, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

def nice(x):
    e = 10 ** np.floor(np.log10(x))
    return max(m for m in [1, 2, 5, 10] if m * e <= x) * e

def glyphs(ax, Lsmaj, Lsmin, theta, big, refbar=True):
    """Ellipse outlines on every step-th cell; `big` [m/s] fills half the glyph
    spacing, so panels sharing `big` share the glyph scale."""
    jj, ii = np.meshgrid(np.arange(0, lon.shape[0], step), np.arange(0, lon.shape[1], step), indexing='ij')
    sel = show[jj, ii] & inwin[jj, ii] & np.isfinite(Lsmaj[jj, ii])
    jj, ii = jj[sel], ii[sel]
    sc = 0.5 * step * cell_m / big
    ph = np.linspace(0, 2 * np.pi, 41)
    segs = []
    for j, i in zip(jj, ii):
        a, b, th = Lsmaj[j, i], Lsmin[j, i], np.deg2rad(theta[j, i])
        x, y = a * np.cos(ph), b * np.sin(ph)
        X = (x * np.cos(th) - y * np.sin(th)) * sc / (m_per_deg * np.cos(np.deg2rad(lat[j, i])))
        Y = (x * np.sin(th) + y * np.cos(th)) * sc / m_per_deg
        segs.append(np.column_stack([lon[j, i] + X, lat[j, i] + Y]))
    ax.add_collection(LineCollection(segs, colors='k', linewidths=0.5, zorder=3))
    if refbar:
        r = nice(big)
        x0 = aa[0] + 0.06 * (aa[1] - aa[0])
        y0 = aa[2] + 0.05 * (aa[3] - aa[2])
        L = r * sc / (m_per_deg * np.cos(np.deg2rad(lat0)))
        ax.plot([x0, x0 + L], [y0, y0], '-', color='k', linewidth=2, zorder=4)
        ax.text(x0 + L / 2, y0 + 0.015 * (aa[3] - aa[2]), '%g cm/s' % (100 * r), ha='center', va='bottom', fontsize=8)

def pct(f, p):
    v = f[region & np.isfinite(f)] if f.ndim == 2 else f[:, region].ravel()
    v = v[np.isfinite(v)]
    return np.percentile(v, p)

def log_cbar(fig, cs, ax, rr, label):
    """colorbar for a LogNorm ratio centred on 1, with plain decimal ticks."""
    cb = fig.colorbar(cs, ax=ax, shrink=0.85, label=label)
    t = np.unique(np.round(np.geomspace(1 / rr, rr, 5), 2))
    cb.set_ticks(t)
    cb.set_ticklabels(['%.2f' % v for v in t])
    cb.minorticks_off()

def end_label(ax, x, y, s, color):
    k = np.flatnonzero(np.isfinite(y))[-1]
    ax.annotate(s, (x[k], y[k]), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color='0.2')
    ax.plot(x[k], y[k], 'o', ms=3, color=color)

# mouth column for 3D profiles (same as 20261008: last rho column west of pc_lp)
HAS3D = '3d_Lsmaj' in ref
if HAS3D:
    pc_lp = pd.read_pickle(sect_dir / 'pc_lp.p')
    imouth = np.flatnonzero(lon[0] < pc_lp.x.mean())[-1]
    jm = np.flatnonzero(show[:, imouth])
    z0m = np.nanmean(ref.z0_rho.values[:, jm, imouth], axis=1) # section-mean layer depth
    def mouth(f):
        """(..., s_rho, eta, xi) -> section mean (..., s_rho) at the mouth column"""
        return np.nanmean(f[..., jm, imouth], axis=-1)

# ============================================================== monthly ===
if M is not None:
    MON = pd.DatetimeIndex(M.month.values)
    NM = len(MON)
    mlab = [t.strftime('%b %Y') for t in MON]

    # ---- map grids
    NCOL = 6
    nrow = int(np.ceil(NM / NCOL))
    pw = 2.6
    for con in args.cons.split(','):
        L = get(M, 'Lsmaj', con=con)
        Ln, th = get(M, 'Lsmin', con=con), get(M, 'theta', con=con)
        vmax = pct(100 * L, 98)
        fig, axs = plt.subplots(nrow, NCOL, figsize=(pw * NCOL, pw * asp * nrow + 1.2), layout='constrained', squeeze=False)
        for n in range(nrow * NCOL):
            ax = axs.flat[n]
            if n >= NM:
                ax.set_visible(False)
                continue
            cs = ax.pcolormesh(plon, plat, masked(100 * L[n]), cmap=cmocean.cm.speed, vmin=0, vmax=vmax, shading='flat')
            setup(ax, ticks=(n % NCOL == 0) and (n // NCOL == nrow - 1))
            glyphs(ax, L[n], Ln[n], th[n], vmax / 100, refbar=(n == 0))
            ax.set_title(mlab[n], fontsize=9)
        fig.colorbar(cs, ax=axs, shrink=0.6, label='%s semi-major axis [cm/s]' % con)
        fig.suptitle('%s tidal ellipses by month (%s): %s' % (con, FLDLAB[args.fld], args.gtagex))
        save(fig, 'monthly_maps_%s_%s_%s_%s.png' % (args.job, args.dom, args.fld, con))

    # ---- area-mean series
    SER = ['M2', 'S2', 'N2', 'K1', 'O1', 'M4']
    x = MON + pd.Timedelta(days=14)
    fig, axd = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'f']], layout='constrained', figsize=(13, 11))
    for con in SER:
        L = amean(get(M, 'Lsmaj', con=con))
        ci = amean(get(M, 'Lsmaj_ci', con=con))
        zA = amean(M.zeta_A.sel(con=con).values)
        c = SCOL[con]
        axd['a'].fill_between(x, 100 * (L - ci), 100 * (L + ci), color=c, alpha=0.15, lw=0)
        axd['a'].plot(x, 100 * L, '-', color=c, lw=2, label=con)
        end_label(axd['a'], x, 100 * L, con, c)
        r = 100 * (L / np.nanmean(L) - 1)
        axd['b'].plot(x, r, '-', color=c, lw=2, label=con)
        end_label(axd['b'], x, r, con, c)
        rz = 100 * (zA / np.nanmean(zA) - 1)
        axd['c'].plot(x, rz, '-', color=c, lw=2, label=con)
        end_label(axd['c'], x, rz, con, c)
    for con in ['M2', 'K1']:
        e = amean(get(M, 'Lsmin', con=con) / get(M, 'Lsmaj', con=con))
        axd['d'].plot(x, e, '-', color=SCOL[con], lw=2, label=con)
        end_label(axd['d'], x, e, con, SCOL[con])
        th = get(M, 'theta', con=con)
        thm = (np.rad2deg(np.angle(np.nanmean(np.exp(2j * np.deg2rad(th[:, region])), axis=1))) / 2) % 180
        axd['e'].plot(x, thm, '-', color=SCOL[con], lw=2, label=con)
        end_label(axd['e'], x, thm, con, SCOL[con])
    spd = amean(np.hypot(get(M, 'umean'), get(M, 'vmean')))
    axd['f'].plot(x, 100 * spd, '-', color='k', lw=2)
    axd['a'].set_ylabel('semi-major axis [cm/s]')
    axd['a'].set_title('current: area-mean semi-major axis (band = 95% CI)', fontsize=10)
    axd['b'].set_ylabel('% of 2-year mean')
    axd['b'].set_title('current: relative to each constituent\'s mean', fontsize=10)
    axd['c'].set_ylabel('% of 2-year mean')
    axd['c'].set_title('zeta (forcing): relative to each constituent\'s mean', fontsize=10)
    axd['d'].set_ylabel('Lsmin / Lsmaj  (+ counterclockwise)')
    axd['d'].set_title('current: area-mean eccentricity', fontsize=10)
    axd['e'].set_ylabel('inclination [deg CCW from east]')
    axd['e'].set_title('current: area axial-mean inclination', fontsize=10)
    axd['f'].set_ylabel('speed [cm/s]')
    axd['f'].set_title('Eulerian residual: area-mean speed of the monthly-mean velocity', fontsize=10)
    for k in 'bc':
        axd[k].axhline(0, color='0.5', lw=0.8)
    for n, k in enumerate('abcdef'):
        axd[k].grid(**GRID)
        letter(axd[k], k)
        if k in 'abcde':
            axd[k].legend(fontsize=8, ncol=3 if k in 'abc' else 2, loc='upper left', bbox_to_anchor=(0.06, 1.0))
    fig.suptitle('Monthly tidal fits (%s, %s, %d cells): %s' % (FLDLAB[args.fld], args.dom, region.sum(), args.gtagex))
    save(fig, 'monthly_series_%s_%s_%s.png' % (args.job, args.dom, args.fld))

    # ---- month x depth at the mouth (3D)
    if HAS3D:
        for con in args.cons.split(','):
            L = mouth(M['3d_Lsmaj'].sel(con=con).values) # (month, s_rho)
            E = mouth((M['3d_Lsmin'] / M['3d_Lsmaj']).sel(con=con).values)
            fig, axd = plt.subplot_mosaic([['a'], ['b']], layout='constrained', figsize=(12, 7))
            for k, f, cmap, (v0, v1), lab in [('a', 100 * L, cmocean.cm.speed, (0, np.nanmax(100 * L)), 'semi-major axis [cm/s]'),
                                              ('b', E, cmocean.cm.balance, (-np.nanmax(np.abs(E)), np.nanmax(np.abs(E))), 'Lsmin / Lsmaj  (+ counterclockwise)')]:
                ax = axd[k]
                cs = ax.pcolormesh(x, z0m, f.T, cmap=cmap, vmin=v0, vmax=v1, shading='nearest')
                fig.colorbar(cs, ax=ax, label=lab)
                ax.set_ylabel('section-mean z at zeta = 0 [m]')
                ax.grid(**GRID)
                letter(ax, k)
            fig.suptitle('%s at the Penn Cove mouth (rho column %d, section mean of %d cells), by month: %s' % (con, imouth, len(jm), args.gtagex))
            save(fig, 'monthly_vertical_%s_%s.png' % (args.job, con))

# ================================================================ bands ===
if Bd is not None:
    TW = pd.DatetimeIndex(Bd.time.values)
    # ---- indices
    fig, axd = plt.subplot_mosaic([['a'], ['b']], layout='constrained', figsize=(13, 7), sharex=True)
    for k, b, hi, lo in [('a', 'D2', 'spring', 'neap'), ('b', 'D1', 'tropic', 'equatorial')]:
        ax = axd[k]
        y = Bd['zidx_' + b].values
        ax.plot(TW, y, '-', color='0.6', lw=1)
        for nm in [hi, lo]:
            f = Bd['is_' + nm].values == 1
            ax.plot(TW[f], y[f], 'o', ms=4, color=BINCOL[nm], label='%s (n = %d)' % (nm, f.sum()))
        ax.plot(TW, y - Bd['zidx_' + b + '_anom'].values, '-', color='k', lw=1.5, label='31-day running mean')
        ax.set_ylabel('box-mean %s zeta amplitude [m]' % b)
        ax.grid(**GRID)
        ax.legend(fontsize=8, ncol=3, loc='upper right')
        letter(ax, k)
    axd['a'].set_title('spring / neap = top / bottom third of the D2 anomaly from its running mean', fontsize=10)
    axd['b'].set_title('tropic / equatorial = top / bottom third of the D1 anomaly', fontsize=10)
    fig.suptitle('Band-fit indices (%s): %s' % (args.job, args.gtagex))
    save(fig, 'bands_index_%s.png' % args.job)

    # ---- spring/neap (D2) and tropic/equatorial (D1) maps
    def pair_maps(band, hi, lo, fname, title, extra=None):
        rows = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']] + ([['j', 'k', 'l']] if extra else [])
        fig, axd = plt.subplot_mosaic(rows, layout='constrained', figsize=(15, (5 * asp + 0.6) * len(rows) + 0.8))
        L = {s: get(Bd, 'Lsmaj', bin=s, band=band) for s in [hi, lo]}
        Ln = {s: get(Bd, 'Lsmin', bin=s, band=band) for s in [hi, lo]}
        th = {s: get(Bd, 'theta', bin=s, band=band) for s in [hi, lo]}
        lag = {s: get(Bd, 'lag', bin=s, band=band) for s in [hi, lo]}
        zA = {s: Bd.zeta_A.sel(bin=s, band=band).values for s in [hi, lo]}
        vmax = max(pct(100 * L[hi], 98), pct(100 * L[lo], 98))
        for k, s in [('a', hi), ('b', lo)]:
            cs = axd[k].pcolormesh(plon, plat, masked(100 * L[s]), cmap=cmocean.cm.speed, vmin=0, vmax=vmax, shading='flat')
            glyphs(axd[k], L[s], Ln[s], th[s], vmax / 100)
            axd[k].set_title('%s: %s semi-major axis (zeta %.2f m)' % (s, band, amean(zA[s])), fontsize=10)
        fig.colorbar(cs, ax=[axd['a'], axd['b']], shrink=0.85, label='%s semi-major axis [cm/s]' % band)
        # current ratio over the local zeta ratio: 1 = current scales with the forcing
        R = (L[hi] / L[lo]) / (zA[hi] / zA[lo])
        rr = max(1.1, pct(np.maximum(R, 1 / R), 98))
        cs = axd['c'].pcolormesh(plon, plat, masked(R), cmap=cmocean.cm.balance, norm=LogNorm(1 / rr, rr), shading='flat')
        log_cbar(fig, cs, axd['c'], rr, '(current %s/%s) / (zeta %s/%s)' % (hi, lo, hi, lo))
        axd['c'].set_title('current ratio %.2f, zeta ratio %.2f (area means)' % (amean(L[hi]) / amean(L[lo]), amean(zA[hi]) / amean(zA[lo])), fontsize=10)
        E = {s: Ln[s] / L[s] for s in [hi, lo]}
        for k, s in [('d', hi), ('e', lo)]:
            cs = axd[k].pcolormesh(plon, plat, masked(E[s]), cmap=cmocean.cm.balance, vmin=-1, vmax=1, shading='flat')
            axd[k].set_title('%s: Lsmin / Lsmaj (+ CCW)' % s, fontsize=10)
        fig.colorbar(cs, ax=[axd['d'], axd['e']], shrink=0.85, label='eccentricity')
        dE = E[hi] - E[lo]
        de = max(0.02, pct(np.abs(dE), 98))
        cs = axd['f'].pcolormesh(plon, plat, masked(dE), cmap=cmocean.cm.balance, vmin=-de, vmax=de, shading='flat')
        fig.colorbar(cs, ax=axd['f'], shrink=0.85, label='%s - %s' % (hi, lo))
        axd['f'].set_title('eccentricity difference', fontsize=10)
        for k, s in [('g', hi), ('h', lo)]:
            cs = axd[k].pcolormesh(plon, plat, masked(lag[s]), cmap=cmocean.cm.phase, vmin=0, vmax=360, shading='flat')
            axd[k].set_title('%s: lag of max eastward current after HW' % s, fontsize=10)
        fig.colorbar(cs, ax=[axd['g'], axd['h']], shrink=0.85, label='phase lag [deg]')
        dl = (lag[hi] - lag[lo] + 180) % 360 - 180
        dd = max(2.0, pct(np.abs(dl), 98))
        cs = axd['i'].pcolormesh(plon, plat, masked(dl), cmap=cmocean.cm.balance, vmin=-dd, vmax=dd, shading='flat')
        fig.colorbar(cs, ax=axd['i'], shrink=0.85, label='%s - %s [deg]' % (hi, lo))
        axd['i'].set_title('lag difference (360 deg = %.1f h)' % {'D1': 24.84, 'D2': 12.42}[band], fontsize=10)
        if extra:
            extra(fig, axd)
        for n, k in enumerate(sum(rows, [])):
            setup(axd[k])
            letter(axd[k], k)
        fig.suptitle(title)
        save(fig, fname)

    pair_maps('D2', 'spring', 'neap', 'springneap_%s_%s_%s.png' % (args.job, args.dom, args.fld),
        'Semidiurnal (D2) ellipses, spring vs neap (%s): %s' % (FLDLAB[args.fld], args.gtagex))

    def form_ratio(fig, axd):
        """Current form ratio D1/D2 semi-major axis: the diurnal inequality of the current."""
        F = {s: get(Bd, 'Lsmaj', bin=s, band='D1') / get(Bd, 'Lsmaj', bin=s, band='D2') for s in ['tropic', 'equatorial']}
        Fz = {s: Bd.zeta_A.sel(bin=s, band='D1').values / Bd.zeta_A.sel(bin=s, band='D2').values for s in F}
        vmax = max(pct(F['tropic'], 98), pct(F['equatorial'], 98))
        for k, s in [('j', 'tropic'), ('k', 'equatorial')]:
            cs = axd[k].pcolormesh(plon, plat, masked(F[s]), cmap=cmocean.cm.amp, vmin=0, vmax=vmax, shading='flat')
            axd[k].set_title('%s: current D1/D2 (zeta D1/D2 %.2f)' % (s, amean(Fz[s])), fontsize=10)
        fig.colorbar(cs, ax=[axd['j'], axd['k']], shrink=0.85, label='semi-major axis D1 / D2')
        r = F['tropic'] / Fz['tropic'] / (F['equatorial'] / Fz['equatorial'])
        rr = max(1.1, pct(np.maximum(r, 1 / r), 98))
        cs = axd['l'].pcolormesh(plon, plat, masked(r), cmap=cmocean.cm.balance, norm=LogNorm(1 / rr, rr), shading='flat')
        log_cbar(fig, cs, axd['l'], rr, 'current/zeta form ratio, tropic / equatorial')
        axd['l'].set_title('1 = current inequality follows the zeta inequality', fontsize=10)

    pair_maps('D1', 'tropic', 'equatorial', 'diurnal_%s_%s_%s.png' % (args.job, args.dom, args.fld),
        'Diurnal (D1) ellipses, tropic vs equatorial (%s): %s' % (FLDLAB[args.fld], args.gtagex), extra=form_ratio)

    # ---- residual by bin
    BB = ['spring', 'neap', 'tropic', 'equatorial']
    UV = {s: (get(Bd, 'umean', bin=s), get(Bd, 'vmean', bin=s)) for s in BB}
    spd = {s: np.hypot(*UV[s]) for s in BB}
    big = max(pct(spd[s], 98) for s in BB)
    fig, axd = plt.subplot_mosaic([['spring', 'neap'], ['tropic', 'equatorial']], layout='constrained', figsize=(12, 2 * (6 * asp + 0.6) + 0.8))
    jj, ii = np.meshgrid(np.arange(0, lon.shape[0], step), np.arange(0, lon.shape[1], step), indexing='ij')
    for n, s in enumerate(BB):
        ax = axd[s]
        u, v = UV[s]
        cs = ax.pcolormesh(plon, plat, masked(100 * spd[s]), cmap=cmocean.cm.speed, vmin=0, vmax=100 * big, shading='flat')
        setup(ax)
        sel = show[jj, ii] & inwin[jj, ii] & np.isfinite(u[jj, ii])
        q = ax.quiver(lon[jj, ii][sel], lat[jj, ii][sel], u[jj, ii][sel], v[jj, ii][sel], scale=big / 0.3, scale_units='inches', width=0.004, color='k', zorder=3)
        ax.quiverkey(q, 0.12, 0.06, nice(big), '%g cm/s' % (100 * nice(big)), labelpos='N', coordinates='axes', fontproperties={'size': 8})
        ax.set_title('%s (area-mean speed %.2f cm/s)' % (s, 100 * amean(spd[s])), fontsize=10)
        letter(ax, 'abcd'[n])
    fig.colorbar(cs, ax=list(axd.values()), shrink=0.85, label='Eulerian residual speed [cm/s]')
    fig.suptitle('Eulerian residual (mean of 3-day window means) by bin (%s): %s' % (FLDLAB[args.fld], args.gtagex))
    save(fig, 'bands_residual_%s_%s_%s.png' % (args.job, args.dom, args.fld))

    # ---- mouth profiles (3D)
    if HAS3D:
        fig, axd = plt.subplot_mosaic([['a', 'b', 'c'], ['d', 'e', 'f']], layout='constrained', figsize=(13, 9))
        for row, band, hi, lo in [('abc', 'D2', 'spring', 'neap'), ('def', 'D1', 'tropic', 'equatorial')]:
            for s in [hi, lo]:
                L = mouth(Bd['3d_Lsmaj'].sel(bin=s, band=band).values)
                E = mouth(Bd['3d_Lsmin'].sel(bin=s, band=band).values) / L
                for k, f in zip(row, [100 * L, L / np.nanmean(L), E]):
                    axd[k].plot(f, z0m, '-', color=BINCOL[s], lw=2, label=s)
            axd[row[0]].set_xlabel('%s semi-major axis [cm/s]' % band)
            axd[row[1]].set_xlabel('%s semi-major axis / its depth mean' % band)
            axd[row[2]].set_xlabel('%s Lsmin / Lsmaj (+ CCW)' % band)
            axd[row[2]].axvline(0, color='0.5', lw=0.8)
        for k in 'abcdef':
            axd[k].set_ylabel('section-mean z at zeta = 0 [m]')
            axd[k].grid(**GRID)
            axd[k].legend(fontsize=8, loc='lower right')
            letter(axd[k], k)
        fig.suptitle('Penn Cove mouth (rho column %d, %d cells), section-mean profiles: %s' % (imouth, len(jm), args.gtagex))
        save(fig, 'bands_vertical_%s.png' % args.job)
