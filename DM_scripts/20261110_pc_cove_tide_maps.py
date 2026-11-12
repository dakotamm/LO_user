"""
Tidal (and S1) harmonic MAPS of Penn Cove in the flow-defined layers of
20261110_pc_cove_layer_maps.py (depth mean, upper, lower), of wb1_t0_xn11abbur00,
2024-2025 hourly snapshots. Step 4 of the whole-cove plan.

WHY REFIT. The 20261008_tidal_ellipses.py fits carry no S1, so their K1 / P1 absorb the
daily wind; and they are on s-levels, not the flow-defined layers used here.

FITS (utide, as 20261103_pc_section_tides.py: OLS, nodal corrections, no trend, the 11
constituents + S1; a series is fitted only if it has data in >= 99% of hours, so the 38
shallow edge cells whose lower layer empties at low water are left blank, not biased)
of the layer velocity u = Ux / H, v = Vy / H at each cell:
    along-cove u_al = u ax + v ay, (ax, ay) = (-0.9789, -0.2043), + toward the head
        scalar fit: amplitude A and Greenwich phase g
    (u, v) 2-D fit: ellipse major / minor axis (minor + = counterclockwise), inclination
and the mouth transport Q (pc_lp, + into the cove), the phase reference.

PHASE as the LAG behind the same constituent of mouth Q (20261103_pc_section_tides.py):
    lag = g - g_Q wrapped to (-180, 180]; + = peaks later; +-180 = counterflow;
    blank where A < its 95% confidence half-width.
S1 instead: hour of maximum INFLOW (u_al > 0), PST, calibrated on a synthetic series.

FIGURES (LO_output/DM_outs/20261110_pc_cove_tide_maps/), one per constituent in SHOW:
    tide_map_[con].png   rows depth mean | upper | lower; columns along-cove amplitude |
                         lag behind mouth Q (S1: hour of max inflow) | ellipses (red CCW,
                         blue CW) on the major-axis amplitude
CACHE  tide_map_fits.p (REFIT = True to redo)

run 20261110_pc_cove_tide_maps.py [-Nproc 8]
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')        # one BLAS thread: utide is ~3x faster this way
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from time import time as clock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import xarray as xr
import utide
import cmocean

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
REFIT = False
LAT = 48.23
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head
CONS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'MS4', 'M6', 'S1']
KW = dict(method='ols', conf_int='linear', constit=CONS, nodal=True, trend=False, verbose=False)
SHOW = ['M2', 'S2', 'K1', 'O1', 'S1', 'M4']
LAYS = ['depth mean', 'upper', 'lower']
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2'}

Ldir = Lfun.Lstart(gridname='wb1')
lay_fn = Ldir['LOo'] / 'DM_outs' / '20261110_pc_cove_layer_maps' / 'layers_hourly.nc'
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_' + DS0 + '_' + DS1 + '.nc')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_cove_tide_maps'
out_dir.mkdir(parents=True, exist_ok=True)
T = xr.open_dataset(lay_fn).time.values


# ------------------------------------------------------------------ fit helpers (fit, s1 calibration, lag as 20261103_pc_section_tides.py)
def fit(x):
    """scalar fit; (4, ncon) = A, A_ci, g, g_ci in CONS order; NaN unless >= 99% of hours valid"""
    out = np.full((4, len(CONS)), np.nan)
    ok = np.isfinite(x)
    if ok.mean() < 0.99:
        return out
    c = utide.solve(T[ok], x[ok], lat=LAT, **KW)
    ii = [list(c.name).index(n) for n in CONS]
    out[:] = [c.A[ii], c.A_ci[ii], c.g[ii], c.g_ci[ii]]
    return out


def fit2(u, v):
    """2-D fit; (5, ncon) = Lsmaj, Lsmaj_ci, Lsmin, theta, g; NaN unless >= 99% of hours valid"""
    out = np.full((5, len(CONS)), np.nan)
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.mean() < 0.99:
        return out
    c = utide.solve(T[ok], u[ok], v[ok], lat=LAT, **KW)
    ii = [list(c.name).index(n) for n in CONS]
    out[:] = [c.Lsmaj[ii], c.Lsmaj_ci[ii], c.Lsmin[ii], c.theta[ii], c.g[ii]]
    return out


def work(args):
    u, v = args
    return fit(u * AXIS[0] + v * AXIS[1]), fit2(u, v)


def s1_hour_calibration():
    """utide's S1 Greenwich phase of a synthetic series that peaks at 06 UTC every day."""
    t = T.astype('datetime64[s]').astype(float) / 3600
    c = utide.solve(T, np.cos(2 * np.pi * (t - 6) / 24), lat=LAT, **KW)
    return c.g[list(c.name).index('S1')]


def lag(F, FQ, n):
    """lag of constituent n behind the same constituent of mouth Q, (-180, 180]; NaN if not significant"""
    i = CONS.index(n)
    d = (F[..., 2, i] - FQ[2, i] + 180) % 360 - 180
    return np.where(F[..., 0, i] >= F[..., 1, i], d, np.nan)


def s1_hour(F, g6):
    """hour of maximum inflow of S1, PST (UTC-8); NaN if not significant"""
    i = CONS.index('S1')
    h_utc = 6 + (F[..., 2, i] - g6) / 15
    return np.where(F[..., 0, i] >= F[..., 1, i], (h_utc - 8) % 24, np.nan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-Nproc', default=8, type=int)
    NPROC = parser.parse_args().Nproc

    d = xr.open_dataset(lay_fn)
    JJ, II = d.eta.values, d.xi.values
    NC = len(JJ)

    # ------------------------------------------------------------------ fits (cached)
    cache_fn = out_dir / 'tide_map_fits.p'
    if cache_fn.exists() and not REFIT:
        FIT = pickle.load(open(cache_fn, 'rb'))
        print('loaded fits from ' + str(cache_fn))
    else:
        Ux, Vy, H = d.Ux.values.astype(float), d.Vy.values.astype(float), d.H.values.astype(float)
        Ux, Vy, H = [np.concatenate([a.sum(axis=1, keepdims=True), a], axis=1) for a in (Ux, Vy, H)]   # depth mean, upper, lower
        with np.errstate(invalid='ignore', divide='ignore'):
            uh = np.where(H > 0, Ux / H, np.nan)
            vh = np.where(H > 0, Vy / H, np.nan)
        tt0 = clock()
        jobs = [(uh[:, L, c], vh[:, L, c]) for L in range(3) for c in range(NC)]
        with ProcessPoolExecutor(NPROC) as ex:
            res = list(ex.map(work, jobs, chunksize=8))
        FA = np.array([r[0] for r in res]).reshape(3, NC, 4, len(CONS))
        FE = np.array([r[1] for r in res]).reshape(3, NC, 5, len(CONS))
        FQ = fit(d.Qmouth.values)
        FIT = dict(FA=FA, FE=FE, FQ=FQ, S1_G6=s1_hour_calibration())
        pickle.dump(FIT, open(cache_fn, 'wb'))
        print('%d fits in %.0f s; saved %s' % (2 * len(jobs), clock() - tt0, cache_fn))
    FA, FE, FQ = FIT['FA'], FIT['FE'], FIT['FQ']

    # check: depth-mean M2 ellipses vs the 20261008 ubar / vbar fits (M2 is not touched by adding S1)
    old = xr.open_dataset(Ldir['LOo'] / 'extract' / GTAGEX / 'tidal_ellipses' / ('pc_cove_%s_%s.nc' % (DS0, DS1)))
    i2, o2 = CONS.index('M2'), list(old.con.values).index('M2')
    a_old = old.bar_Lsmaj.values[o2][JJ, II]
    a_new = FE[0, :, 0, i2]
    ok = np.isfinite(a_old) & np.isfinite(a_new)
    print('check M2 depth-mean major axis vs 20261008 ubar fit: r %.4f, median ratio new/old %.3f (%d cells)'
          % (np.corrcoef(a_old[ok], a_new[ok])[0, 1], np.median(a_new[ok] / a_old[ok]), ok.sum()))
    for n in ['M2', 'K1', 'S1', 'M4']:
        i = CONS.index(n)
        print('%s: mouth Q %.0f m3/s; along-cove amplitude median depth mean %.1f, upper %.1f, lower %.1f cm/s; blank cells %s'
              % (n, FQ[0, i], *(100 * np.nanmedian(FA[L, :, 0, i]) for L in range(3)), [int(np.isnan(FA[L, :, 0, i]).sum()) for L in range(3)]))

    # ------------------------------------------------------------------ plot helpers
    g = xr.open_dataset(box_fn)
    LON, LAT_, MASK = g.lon_rho.values, g.lat_rho.values, g.mask_rho.values == 1
    g.close()
    KMX = 111.32 * np.cos(np.deg2rad(LAT))    # km per degree of longitude
    CELL = 0.2                                # ~cell size [km]

    def to2d(a):
        out = np.full(LON.shape, np.nan)
        out[JJ, II] = a
        return out

    def basemap(ax):
        ax.pcolormesh(LON, LAT_, np.where(MASK, np.nan, 1.0), cmap='Greys', vmin=0, vmax=2.5, shading='nearest')
        for SN, XI_U in SECT_XI_U.items():
            jw = np.flatnonzero(MASK[:, XI_U] & MASK[:, XI_U + 1])
            lo = 0.5 * (LON[0, XI_U] + LON[0, XI_U + 1])
            ax.plot([lo, lo], [LAT_[jw.min(), 0] - 0.0009, LAT_[jw.max(), 0] + 0.0009], color=SCOL[SN], lw=1.5)
        ax.set_xlim(LON[0, 0], LON[0, -1])
        ax.set_ylim(LAT_[0, 0], LAT_[-1, 0])
        ax.set_aspect(1 / np.cos(np.deg2rad(LAT)))
        ax.tick_params(labelsize=6)

    def ellipses(ax, E, n, scale):
        """tidal ellipses of constituent n as polygons drawn in km (isotropic on the map); red CCW, blue CW"""
        i = CONS.index(n)
        s = np.linspace(0, 2 * np.pi, 41)
        for c in range(NC):
            a, b, th = E[c, 0, i], E[c, 2, i], np.deg2rad(E[c, 3, i])
            if not np.isfinite(a) or a < E[c, 1, i]:
                continue
            ex, ey = a * np.cos(s) * scale, b * np.sin(s) * scale
            x = ex * np.cos(th) - ey * np.sin(th)
            y = ex * np.sin(th) + ey * np.cos(th)
            ax.plot(LON[JJ[c], II[c]] + x / KMX, LAT_[JJ[c], II[c]] + y / 111.32, lw=0.6,
                    color='tab:red' if b > 0 else 'tab:blue')

    # ------------------------------------------------------------------ one figure per constituent
    for n in SHOW:
        i = CONS.index(n)
        vmax = 100 * np.nanpercentile(FA[:, :, 0, i], 99)
        scale = 0.48 * CELL / np.nanpercentile(FE[:, :, 0, i], 99)
        fig, axs = plt.subplots(3, 3, figsize=(15, 10), layout='constrained')
        for L, lay in enumerate(LAYS):
            ax = axs[L, 0]
            basemap(ax)
            ca = ax.pcolormesh(LON, LAT_, to2d(100 * FA[L, :, 0, i]), cmap=cmocean.cm.speed, vmin=0, vmax=vmax, shading='nearest')
            ax.set_title('%s, %s: along-cove amplitude' % (n, lay), fontsize=9)
            ax = axs[L, 1]
            basemap(ax)
            if n == 'S1':
                cp = ax.pcolormesh(LON, LAT_, to2d(s1_hour(FA[L], FIT['S1_G6'])), cmap='twilight', vmin=0, vmax=24, shading='nearest')
                ax.set_title('S1, %s: hour of maximum inflow (PST)' % lay, fontsize=9)
            else:
                cp = ax.pcolormesh(LON, LAT_, to2d(lag(FA[L], FQ, n)), cmap='twilight_shifted', vmin=-180, vmax=180, shading='nearest')
                ax.set_title('%s, %s: lag behind mouth Q' % (n, lay), fontsize=9)
            ax = axs[L, 2]
            basemap(ax)
            ax.pcolormesh(LON, LAT_, to2d(np.where(np.isfinite(FE[L, :, 0, i]), 0.0, np.nan)), cmap='Greys', vmin=0, vmax=1, shading='nearest', alpha=0.15)
            ellipses(ax, FE[L], n, scale)
            ax.set_title('%s, %s: ellipses (red CCW, blue CW)' % (n, lay), fontsize=9)
        fig.colorbar(ca, ax=axs[:, 0], location='bottom', shrink=0.8, label='%s amplitude of along-cove u [cm s$^{-1}$]' % n)
        fig.colorbar(cp, ax=axs[:, 1], location='bottom', shrink=0.8,
                     label='S1 hour of max inflow [PST]' if n == 'S1' else 'lag behind mouth Q [deg], + later, $\\pm$180 counterflow; blank = not significant')
        fig.suptitle('%s of Penn Cove velocity in flow-defined layers (split at each column\'s median divide), utide 2024-2025 hourly, %s\n'
                     'mouth Q %s amplitude %.0f m$^3$ s$^{-1}$; lower-layer cells that dry at low water left blank'
                     % (n, GTAGEX, n, FQ[0, i]), fontsize=10)
        fig.savefig(out_dir / ('tide_map_%s.png' % n), dpi=200, transparent=True)
        plt.close(fig)
        print('saved tide_map_%s.png' % n)

    # ------------------------------------------------------------------ numbers: lag statistics, upper vs lower
    for n in ['M2', 'K1', 'O1', 'S1']:
        i = CONS.index(n)
        if n == 'S1':
            hu, hl = s1_hour(FA[1], FIT['S1_G6']), s1_hour(FA[2], FIT['S1_G6'])
            dh = (hu - hl + 12) % 24 - 12
            print('S1 hour of max inflow (PST) median: upper %.1f, lower %.1f; layers > 8 h apart (near opposite) in %.0f%% of cells; upper/lower amp median %.2f'
                  % (np.nanmedian(hu), np.nanmedian(hl), 100 * np.nanmean(np.abs(dh) > 8), np.nanmedian(FA[1, :, 0, i] / FA[2, :, 0, i])))
        else:
            lu, ll = lag(FA[1], FQ, n), lag(FA[2], FQ, n)
            dl = (lu - ll + 180) % 360 - 180
            print('%s lag behind mouth Q median: depth mean %+.0f, upper %+.0f, lower %+.0f deg; |upper - lower| > 90 deg in %.0f%% of cells; upper/lower amp median %.2f'
                  % (n, np.nanmedian(lag(FA[0], FQ, n)), np.nanmedian(lu), np.nanmedian(ll), 100 * np.nanmean(np.abs(dl) > 90),
                     np.nanmedian(FA[1, :, 0, i] / FA[2, :, 0, i])))
