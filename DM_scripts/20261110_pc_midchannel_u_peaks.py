"""
The cross-channel averaged velocity profile at the three Penn Cove sections at the PEAK hour of every flood
and every ebb in the run (2024-2025): one figure for floods, one for ebbs; rows pc_cp | pc_lj |
pc_lp (west to east, top to bottom), x = time of each tide's peak, y = z, colour = u normal to the
section, + = into the cove (west) drawn BLUE, out RED (cmocean balance_r). The same panels as
20261110_pc_midchannel_u_timeseries.py, but one column per tide instead of one per hour.

TIDES  floods / ebbs = runs of the TIDAL mouth transport (utide fit of pc_lp Q without S1,
    pc_tidal_cycles.mouth_fit; same zero crossings as 20261105_pc_standard_cycle) > 0 / < 0; runs
    cut by the record ends are dropped. PEAK hour = max (flood) / min (ebb) of the MODEL hourly
    mouth Q (column_metrics.nc, col 30 = pc_lp) inside the run. One mouth clock for all three
    sections.
PROFILE  cross-channel average over the whole section, U_b = Q_b / A_b in 1 m bins of depth below
    the moving surface (as 20261110_pc_midchannel_u_timeseries.py); bins wetted less than half a
    face are blank; bin b at z = zs - depth, zs = width-mean surface.
GRID  contiguous columns, no gaps: each tide's column runs between the midpoints to its
    neighbours, and z_w at a column edge = mean of the two tides it separates (as the hourly
    version does in time). One colour scale for both figures (99th percentile of |u|), one depth
    axis for all rows. Top strip: the peak mouth Ubar = Q / A of each tide.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (peak hours only)
       LO_output/DM_outs/20261110_pc_column_metrics/column_metrics.nc (hourly mouth Q)
OUTPUT LO_output/DM_outs/20261110_pc_midchannel_u_peaks/
    peaks.npz                 cached profiles (delete to re-read the box)
    profile_u_flood.png, profile_u_ebb.png
    across_ubar_flood.png, across_ubar_ebb.png
ACROSS-CHANNEL version: y = latitude of each u face (one latitude axis for all rows, north at the
    top), colour = that face's depth-averaged u = sum_k u dz / sum_k dz at the peak hour; own colour
    scale (99th percentile of |depth-averaged u|, shared by flood and ebb).

run 20261110_pc_midchannel_u_peaks.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
import cmocean
import utide

from lo_tools import Lfun
from pc_tidal_cycles import mouth_fit, CONS

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                 # section-positive (east) is out of the cove
FACES = {                                 # u faces in the pc_cove box, north to south; west to east
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
COL_IN, COL_OUT = '#2166AC', '#B2182B'   # blue = entering the cove, red = leaving
EDGES = np.arange(0, 31, 1.0)             # profile bins, depth below the surface [m]
NBLK = 200                                # peak hours read from the box per block

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
cm_fn = Ldir['LOo'] / 'DM_outs' / '20261110_pc_column_metrics' / 'column_metrics.nc'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_midchannel_u_peaks'
out_dir.mkdir(parents=True, exist_ok=True)
cache = out_dir / 'peaks.npz'

# ------------------------------------------------------------------ peak hours of every flood and ebb
cm = xr.open_dataset(cm_fn)
Q = cm.Q.sel(col=30).to_numpy()                                      # + into the cove
T = pd.DatetimeIndex(cm.time.values)
cm.close()
Qt = utide.reconstruct(T.values, mouth_fit(), constit=[c for c in CONS if c != 'S1'], verbose=False).h
cuts = np.flatnonzero(np.diff(np.sign(Qt)) != 0) + 1
PK = {'flood': [], 'ebb': []}
for a, b in zip(cuts[:-1], cuts[1:]):                                # complete runs only
    if Qt[a:b].mean() > 0:
        PK['flood'].append(a + np.argmax(Q[a:b]))
    else:
        PK['ebb'].append(a + np.argmin(Q[a:b]))
PK = {k: np.array(v) for k, v in PK.items()}
for k, v in PK.items():
    print('%d %ss; peak mouth Q median %+.0f m3/s' % (len(v), k, np.median(Q[v])))

# ------------------------------------------------------------------ cross-channel averaged profiles at those hours
def binned(u, zw, dy):
    """U_b = Q_b / A_b in the EDGES bins below the moving surface, returned bottom to top as (u (t, b),
    z edges (t, b + 1)) on z = zs - depth; NaN where a bin is wetted less than half a face."""
    dz = np.diff(zw, axis=1)
    q = u * dz * dy
    dw = zw[:, -1:, :] - zw                                          # depth below the surface of each interface
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    assert d_bot.max() < EDGES[-1]
    nb = len(EDGES) - 1
    Qb, Ab = np.zeros((len(u), nb)), np.zeros((len(u), nb))
    for b in range(nb):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None)
        Qb[:, b] = (q * ov / dz).sum(axis=(1, 2))
        Ab[:, b] = (ov * dy).sum(axis=(1, 2))
    assert np.allclose(Qb.sum(axis=1), q.sum(axis=(1, 2)))
    Ub = np.where(Ab >= 0.5 * dy.mean(), Qb / np.maximum(Ab, 1e-12), np.nan)
    zs = (zw[:, -1, :] * dy).sum(axis=-1) / dy.sum()
    return Ub[:, ::-1], zs[:, None] - EDGES[None, ::-1]


C = dict(np.load(cache)) if cache.exists() else {}
if 'binned' not in C:                                                # no cache, or one from before the cross-channel average
    ds = xr.open_dataset(in_fn)
    assert (pd.DatetimeIndex(ds.ocean_time.values) == T).all()
    C = {}
    for kind, idx in PK.items():
        for SN, F in FACES.items():
            for v in ['u', 'zw', 'ubar', 'uface']:
                C['%s_%s_%s' % (v, kind, SN)] = []
        for i0 in range(0, len(idx), NBLK):                          # each hour chunk holds the whole box
            it = idx[i0:i0 + NBLK]
            u_all = SIGN * ds.u.isel(ocean_time=it).to_numpy().astype(float)        # (t, k, eta_u, xi_u)
            zw_all = ds.z_w.isel(ocean_time=it).to_numpy()                           # (t, w, eta, xi)
            for SN, F in FACES.items():
                XI_U, ETA = F['XI_U'], F['ETA']
                u = u_all[:, :, ETA, XI_U]
                zw = zw_all[:, :, ETA][..., [XI_U, XI_U + 1]].mean(axis=-1)
                dy = (1 / ds.pn.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy()).mean(axis=-1)
                dz = np.diff(zw, axis=1)
                C['ubar_%s_%s' % (kind, SN)].append((u * dz * dy).sum(axis=(1, 2)) / (dz * dy).sum(axis=(1, 2)))
                C['uface_%s_%s' % (kind, SN)].append((u * dz).sum(axis=1) / dz.sum(axis=1))   # (t, p) depth average of each face
                ub, ze = binned(u, zw, dy)
                C['u_%s_%s' % (kind, SN)].append(ub)
                C['zw_%s_%s' % (kind, SN)].append(ze)
            print('  %s %d / %d' % (kind, min(i0 + NBLK, len(idx)), len(idx)))
    C = {k: np.concatenate(v) for k, v in C.items()}
    for SN, F in FACES.items():
        C['lat_' + SN] = ds.lat_u.isel(eta_u=F['ETA'], xi_u=F['XI_U']).to_numpy()   # north to south
    ds.close()
    C.update(idx_flood=PK['flood'], idx_ebb=PK['ebb'], binned=np.array(True))
    np.savez(cache, **C)
for kind in PK:
    assert (C['idx_' + kind] == PK[kind]).all()

# ------------------------------------------------------------------ figures
umax = np.ceil(100 * np.nanpercentile(np.abs(np.concatenate([C[k].ravel() for k in C if k.startswith('u_')])), 99))
norm = matplotlib.colors.Normalize(-umax, umax)
cmap = cmocean.cm.balance_r
zbot = min(np.nanmin(np.where(np.isfinite(C['u_' + k[3:]]), C[k][:, :-1], np.nan)) for k in C if k.startswith('zw_')) - 0.5   # deepest wet bin
ztop = max(C[k][:, -1].max() for k in C if k.startswith('zw_')) + 0.5
print('colour limit +-%.0f cm/s' % umax)

for kind, idx in PK.items():
    tp = mdates.date2num(T[idx])
    dt = np.diff(tp)
    xe = np.concatenate([[tp[0] - dt[0] / 2], 0.5 * (tp[1:] + tp[:-1]), [tp[-1] + dt[-1] / 2]])   # (n + 1,) shared edges
    print('%s: tide spacing %.2f to %.2f d' % (kind, dt.min(), dt.max()))

    fig = plt.figure(figsize=(18, 10), layout='constrained')
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 2, 2, 2])
    ax_t = fig.add_subplot(gs[0])
    axs = [fig.add_subplot(gs[c + 1], sharex=ax_t) for c in range(3)]
    for SN, colr in [('pc_cp', '#009E73'), ('pc_lj', '#CC79A7'), ('pc_lp', '#E69F00')]:
        ax_t.plot(tp, 100 * C['ubar_%s_%s' % (kind, SN)], color=colr, lw=0.6, marker='.', ms=2, label='Ubar ' + SN)
    ax_t.axhline(0, color='k', lw=0.5)
    ax_t.set_ylabel('Ubar = Q / A at the\npeak hour [cm s$^{-1}$]\n+ in')
    ax_t.legend(fontsize=8, loc='upper left', ncol=3)
    ax_t.grid(True, alpha=0.3)
    ax_t.tick_params(labelbottom=False)

    for ax, SN in zip(axs, FACES):
        u = 100 * C['u_%s_%s' % (kind, SN)]                          # (n, k)
        zw = C['zw_%s_%s' % (kind, SN)]                              # (n, w)
        Y = np.concatenate([zw[:1], 0.5 * (zw[1:] + zw[:-1]), zw[-1:]], axis=0)   # (n + 1, w) at the column edges
        X = np.broadcast_to(xe[:, None], Y.shape)
        ax.pcolormesh(X.T, Y.T, np.ma.masked_invalid(u).T, cmap=cmap, norm=norm, shading='flat', edgecolors='face', linewidth=0, rasterized=True)
        ax.set_ylim(zbot, ztop)
        ax.set_ylabel('z [m]')
        ax.set_title('%s: cross-channel average U$_b$ = Q$_b$ / A$_b$, 1 m bins below the surface' % NAME[SN], fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)
        if SN != 'pc_lp':
            ax.tick_params(labelbottom=False)
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax_t.set_xlim(xe[0], xe[-1])

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axs, location='right', shrink=0.8, pad=0.01, extend='both')
    cb.set_label('cross-channel averaged u normal to the section [cm s$^{-1}$]\n+ into the cove (west)')
    cb.ax.text(0.5, 1.02, 'IN', transform=cb.ax.transAxes, ha='center', va='bottom', fontweight='bold', color=COL_IN)
    cb.ax.text(0.5, -0.02, 'OUT', transform=cb.ax.transAxes, ha='center', va='top', fontweight='bold', color=COL_OUT)
    fig.suptitle('Cross-channel averaged velocity profile at the peak hour of every %s (%d), %s, hourly ocean_his\n'
                 'peak = max |mouth Q| within each %s of the tidal (utide, no S1) mouth transport; blue = into the cove, red = out'
                 % (kind.upper(), len(tp), GTAGEX, kind), fontsize=11)
    fn = out_dir / ('profile_u_%s.png' % kind)
    fig.savefig(fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % fn)

# ------------------------------------------------------------------ across-channel figures: depth-averaged u per face
umax = np.ceil(100 * np.percentile(np.abs(np.concatenate([C[k].ravel() for k in C if k.startswith('uface_')])), 99))
norm = matplotlib.colors.Normalize(-umax, umax)
LATE = {}
for SN in FACES:
    lat = C['lat_' + SN]
    LATE[SN] = np.concatenate([[lat[0] - (lat[1] - lat[0]) / 2], 0.5 * (lat[1:] + lat[:-1]), [lat[-1] + (lat[-1] - lat[-2]) / 2]])
ylo = min(v.min() for v in LATE.values())
yhi = max(v.max() for v in LATE.values())
print('across-channel colour limit +-%.0f cm/s' % umax)

for kind, idx in PK.items():
    tp = mdates.date2num(T[idx])
    dt = np.diff(tp)
    xe = np.concatenate([[tp[0] - dt[0] / 2], 0.5 * (tp[1:] + tp[:-1]), [tp[-1] + dt[-1] / 2]])

    fig = plt.figure(figsize=(18, 10), layout='constrained')
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 2, 2, 2])
    ax_t = fig.add_subplot(gs[0])
    axs = [fig.add_subplot(gs[c + 1], sharex=ax_t) for c in range(3)]
    for SN, colr in [('pc_cp', '#009E73'), ('pc_lj', '#CC79A7'), ('pc_lp', '#E69F00')]:
        ax_t.plot(tp, 100 * C['ubar_%s_%s' % (kind, SN)], color=colr, lw=0.6, marker='.', ms=2, label='Ubar ' + SN)
    ax_t.axhline(0, color='k', lw=0.5)
    ax_t.set_ylabel('Ubar = Q / A at the\npeak hour [cm s$^{-1}$]\n+ in')
    ax_t.legend(fontsize=8, loc='upper left', ncol=3)
    ax_t.grid(True, alpha=0.3)
    ax_t.tick_params(labelbottom=False)

    for ax, SN in zip(axs, FACES):
        ax.pcolormesh(xe, LATE[SN], 100 * C['uface_%s_%s' % (kind, SN)].T, cmap=cmap, norm=norm, shading='flat',
                      edgecolors='face', linewidth=0, rasterized=True)
        ax.set_ylim(ylo, yhi)                                        # north at the top
        ax.set_ylabel('latitude [deg N]\nS \u2193   N \u2191')
        ax.set_title('%s: depth-averaged u of each face (%d faces)' % (NAME[SN], len(C['lat_' + SN])), fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)
        if SN != 'pc_lp':
            ax.tick_params(labelbottom=False)
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax_t.set_xlim(xe[0], xe[-1])

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axs, location='right', shrink=0.8, pad=0.01, extend='both')
    cb.set_label('depth-averaged u normal to the section [cm s$^{-1}$]\n+ into the cove (west)')
    cb.ax.text(0.5, 1.02, 'IN', transform=cb.ax.transAxes, ha='center', va='bottom', fontweight='bold', color=COL_IN)
    cb.ax.text(0.5, -0.02, 'OUT', transform=cb.ax.transAxes, ha='center', va='top', fontweight='bold', color=COL_OUT)
    fig.suptitle('Depth-averaged velocity across each section at the peak hour of every %s (%d), %s, hourly ocean_his\n'
                 'peak = max |mouth Q| within each %s of the tidal (utide, no S1) mouth transport; blue = into the cove, red = out'
                 % (kind.upper(), len(tp), GTAGEX, kind), fontsize=11)
    fn = out_dir / ('across_ubar_%s.png' % kind)
    fig.savefig(fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % fn)
