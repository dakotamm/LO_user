"""
Figures for 20261110_pc_velocity_eof.py (EOFs of Penn Cove velocity, wb1_t0_xn11abbur00
2024-2025), the plots of the UW data-analysis Lab 8-1 for each field:
    variance explained   S**2 / sum(S**2) vs mode (line + markers), and its cumulative sum
    EOFs                 the spatial patterns, here in m/s per 1 std of PC (cm/s on the plots)
    PCs                  the time series (unit variance), full record (+ a 15-day zoom, hourly)
Mode 0 = the time mean that was removed before the SVD (the residual circulation, the
Step 3 maps), drawn above the EOFs for reference.

SIGN: PC > 0 = upper-layer water toward the head, so the EOF shows the flow when its PC is
positive and the reverse when it is negative.
SECTIONS: each looking INTO the cove (west), north at RIGHT, as 20261103; + into the cove.
COVE: color = along-cove component (+ toward the head), arrows = the vector; columns upper |
lower layer.
COLORS: mean RdBu_r, EOFs PuOr_r (anomalies). One color scale per row (shared across
sections / layers).

INPUT  LO_output/DM_outs/20261110_pc_velocity_eof/eof_<band>.p
       LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (grid only)
OUTPUT LO_output/DM_outs/20261110_pc_velocity_eof/
    var_explained_<band>.png, eof_sections_<band>.png, eof_cove_<band>.png, pcs_<band>.png

run 20261110_pc_velocity_eof_plot.py -band hourly
"""
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-band', default='hourly', choices=['hourly', 'subtidal'])
BAND = parser.parse_args().band

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2', 'cove': 'k'}
NP = 4                                    # EOFs plotted
ZOOM = ('2024-08-25', '2024-09-08')       # hourly PC zoom (around the standard cycle, 2024-08-31)

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_velocity_eof'
with open(out_dir / ('eof_%s.p' % BAND), 'rb') as f:
    E = pickle.load(f)
g = xr.open_dataset(Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
LON, LAT, MASK = g.lon_rho.values, g.lat_rho.values, g.mask_rho.values == 1
g.close()
FIELDS = list(SECT_XI_U) + ['cove']
TIME = pd.to_datetime(E['time'])
al = lambda u, v: u * AXIS[0] + v * AXIS[1]


def lab(R, i):
    """panel label: mode 0 = mean, else EOF i and its variance explained"""
    return 'mean (mode 0)' if i == 0 else 'EOF%d  %.1f%%' % (i, 100 * R['var_exp'][i - 1])


def field(R, i):
    """mode 0 = the time mean, else EOF i (physical units, m/s)"""
    return R['mean'] if i == 0 else R['eof'][:, i - 1]


def savefig(fig, name):
    fn = out_dir / ('%s_%s.png' % (name, BAND))
    fig.savefig(fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved ' + str(fn))


# ------------------------------------------------------------------ variance explained (the lab's S**2 / sum(S**2) plot) and its cumulative sum
NV = 20
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
for F in FIELDS:
    ve = 100 * E[F]['var_exp'][:NV]
    axs[0].plot(np.arange(1, NV + 1), ve, '-o', color=SCOL[F], ms=4, lw=1.2, label=F)
    axs[1].plot(np.arange(1, NV + 1), np.cumsum(ve), '-o', color=SCOL[F], ms=4, lw=1.2, label=F)
axs[0].set_ylabel('variance explained [%]', fontsize=9)
axs[1].set_ylabel('cumulative variance explained [%]', fontsize=9)
axs[1].set_ylim(top=100)
for ax in axs:
    ax.set_xlabel('mode', fontsize=9)
    ax.set_xticks(np.arange(1, NV + 1, 2))
    ax.grid(color='0.85', lw=0.5)
    ax.tick_params(labelsize=8)
axs[0].legend(fontsize=8, frameon=False)
fig.suptitle('Penn Cove velocity EOFs, %s band, %s 2024-2025 (sections: u normal, area-weighted; cove: u, v of both layers, volume-weighted)'
             % (BAND, GTAGEX), fontsize=9)
savefig(fig, 'var_explained')


# ------------------------------------------------------------------ section EOFs: rows mean, EOF1..NP; columns pc_cp | pc_lj | pc_lp
WMAX = max(E[S]['dy'].sum() for S in SECT_XI_U) / 1000
ZMIN = min(E[S]['z0'].min() for S in SECT_XI_U)
fig, axs = plt.subplots(NP + 1, 3, figsize=(13, 2.3 * (NP + 1)), sharey=True, layout='constrained')
for r in range(NP + 1):
    vm = 100 * max(np.abs(field(E[S], r)).max() for S in SECT_XI_U)
    for c, S in enumerate(SECT_XI_U):
        R, ax = E[S], axs[r, c]
        fld = 100 * field(R, r).reshape(R['shape'])
        xc = np.cumsum(R['dy']) / 1000 - R['dy'] / 2000                # face centres, km from the north end
        X = np.broadcast_to(xc[None, :], R['z0'].shape)
        cs = ax.pcolormesh(X, R['z0'], fld, cmap='RdBu_r' if r == 0 else 'PuOr_r', vmin=-vm, vmax=vm, shading='nearest')
        if fld.min() < 0 < fld.max():
            ax.contour(X, R['z0'], fld, levels=[0], colors='k', linewidths=0.7)
        ax.axhline(-R['D'], color='k', ls='--', lw=0.8)
        ax.set_xlim(WMAX, 0)                                            # reversed: north at right = looking into the cove
        ax.set_ylim(ZMIN - 1, 0)
        ax.set_facecolor('0.75')
        ax.tick_params(labelsize=7)
        ax.set_title('%s  %s' % (S, lab(R, r)), fontsize=8, loc='left', color='k')
        if r == NP:
            ax.set_xlabel('km from the north shore (north at right)', fontsize=7)
        if c == 0:
            ax.set_ylabel('z [m]', fontsize=7)
    fig.colorbar(cs, ax=axs[r, :], shrink=0.9, pad=0.01,
                 label=('mean u [cm s$^{-1}$]' if r == 0 else 'u per 1 std of PC\n[cm s$^{-1}$]'))
fig.suptitle('Penn Cove section-normal velocity EOFs, %s band, %s 2024-2025; looking into the cove, + = into the cove; '
             'sign: PC > 0 = upper-layer inflow; dashed = median divide' % (BAND, GTAGEX), fontsize=9)
savefig(fig, 'eof_sections')


# ------------------------------------------------------------------ cove EOFs: rows mean, EOF1..NP; columns upper | lower layer
def to2d(a):
    out = np.full(LON.shape, np.nan)
    out[JJ, II] = a
    return out


def basemap(ax):
    land = np.where(MASK, np.nan, 1.0)
    ax.pcolormesh(LON, LAT, land, cmap='Greys', vmin=0, vmax=2.5, shading='nearest')
    for SN, XI_U in SECT_XI_U.items():
        jw = np.flatnonzero(MASK[:, XI_U] & MASK[:, XI_U + 1])
        lo = 0.5 * (LON[0, XI_U] + LON[0, XI_U + 1])
        ax.plot([lo, lo], [LAT[jw.min(), 0] - 0.0009, LAT[jw.max(), 0] + 0.0009], color=SCOL[SN], lw=1.5)
    ax.set_xlim(LON[0, 0], LON[0, -1])
    ax.set_ylim(LAT[0, 0], LAT[-1, 0])
    ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
    ax.tick_params(labelsize=6)


R = E['cove']
JJ, II, N = R['eta'], R['xi'], R['ncell']
uv = lambda f, L: (f[2 * L * N:(2 * L + 1) * N], f[(2 * L + 1) * N:(2 * L + 2) * N])   # layer L = 0 upper, 1 lower
fig, axs = plt.subplots(NP + 1, 2, figsize=(8.5, 2.6 * (NP + 1)), layout='constrained')
for r in range(NP + 1):
    f = field(R, r)
    vm = 100 * max(np.abs(al(*uv(f, L))).max() for L in (0, 1))
    sp = max(np.hypot(*uv(f, L)).max() for L in (0, 1))
    for L, name in enumerate(['upper', 'lower']):
        ax = axs[r, L]
        u, v = uv(f, L)
        basemap(ax)
        cs = ax.pcolormesh(LON, LAT, to2d(100 * al(u, v)), cmap='RdBu_r' if r == 0 else 'PuOr_r', vmin=-vm, vmax=vm, shading='nearest')
        ax.quiver(LON[JJ, II], LAT[JJ, II], u, v, scale=sp * 12, scale_units='width', width=0.003, color='k')
        ax.set_title('%s layer  %s' % (name, lab(R, r)), fontsize=8, loc='left')
    fig.colorbar(cs, ax=axs[r, :], shrink=0.9, pad=0.01,
                 label=('mean along-cove u\n[cm s$^{-1}$]' if r == 0 else 'along-cove u per\n1 std of PC [cm s$^{-1}$]'))
fig.suptitle('Penn Cove layer-velocity EOFs (u, v of both layers in one SVD), %s band, %s 2024-2025\n'
             'color = along-cove component (+ toward the head), arrows = vector (one arrow scale per row)\nsign: PC > 0 = upper layer toward the head'
             % (BAND, GTAGEX), fontsize=9)
savefig(fig, 'eof_cove')


# ------------------------------------------------------------------ PCs: rows EOF1..NP; full record (+ 15-day zoom with mouth Q, hourly)
ncol = 2 if BAND == 'hourly' else 1
fig, axs = plt.subplots(NP, ncol, figsize=(7 * ncol + 2, 1.9 * NP), sharex='col', squeeze=False, layout='constrained')
z = (TIME >= ZOOM[0]) & (TIME < ZOOM[1])
Qn = E['Qmouth'] / np.nanstd(E['Qmouth'])
for r in range(NP):
    for c in range(ncol):
        ax = axs[r, c]
        if c == 1:
            ax.plot(TIME[z], Qn[z], color='0.6', lw=1.6, label='mouth Q / std')
        for F in FIELDS:
            R = E[F]
            t, pc = TIME[R['ok']], R['pc'][r]
            m = slice(None) if c == 0 else z[R['ok']]
            ax.plot(t[m], pc[m], color=SCOL[F], lw=0.4 if c == 0 else 1.0,
                    label='%s (%.0f%%)' % (F, 100 * R['var_exp'][r]))
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel('PC%d' % (r + 1), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color='0.9', lw=0.5)
        ax.legend(fontsize=6, frameon=False, ncol=5, loc='upper left')
axs[0, 0].set_title('full record', fontsize=9, loc='left')
if ncol == 2:
    axs[-1, 1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axs[-1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axs[0, 1].set_title('%s to %s' % ZOOM, fontsize=9, loc='left')
fig.suptitle('Penn Cove velocity PCs (unit variance), %s band, %s; PC > 0 = upper layer toward the head' % (BAND, GTAGEX), fontsize=9)
savefig(fig, 'pcs')
