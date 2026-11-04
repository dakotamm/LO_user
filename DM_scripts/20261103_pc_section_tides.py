"""
Tidal constituents of the section-normal velocity at the three Penn Cove
sections, pc_cp (off Coupeville), pc_lj (middle) and pc_lp (mouth), of
wb1_t0_xn11abbur00: cross-sections and profiles of amplitude and phase, laid
out like 20261103_pc_flood_ebb_sections.py and 20261103_pc_section_profiles.py.

FITS (utide, 2024-2025 hourly, OLS, nodal corrections, no trend; the 11
constituents of 20261008_tidal_ellipses.py + S1, as in 20261027_pc_diurnal_S1.py,
so the daily wind signal is not absorbed by K1 / P1). All fits are scalar, of the
velocity NORMAL to the section at its own u-faces (not the rho-point ellipses of
20261008, which average two faces):
    Q(t)        section-total transport int u dA        -> the phase reference
    u_c(t)      each element (face p, s-level k)         -> cross-sections
    U_b(t)      = Q_b / A_b per 1 m bin (as for velocity) -> profiles; NaN in an
                hour when the bin holds no water (utide skips those hours)

PHASE is shown as the LAG behind the same constituent of Q,
    lag = g - g_Q, wrapped to (-180, 180] deg        + = peaks later than the transport
so 0 = in step with the section's tidal transport and +-180 = counterflow (this
water flows out while the section as a whole floods). Greenwich phase itself is
an arbitrary reference. Phase is blanked where the amplitude is not significant
(A < its 95% confidence half-width).

S1 IS THE EXCEPTION: it is the daily (wind) signal, mostly two-layer, so its net
transport is ~10 m3/s with a +-20 deg phase and is no reference. S1 phase is shown
instead as the HOUR OF MAXIMUM INFLOW, local standard time (PST = UTC-8), 0-24 h.
Greenwich phase -> clock hour is calibrated by fitting a synthetic series that
peaks at a known hour (no reliance on the astronomical argument convention).
Phase lines on the profiles are broken where they wrap.

COORDINATES, BINS, COLUMNS, HATCHING: as in 20261103_pc_section_profiles.py
(depth below the moving surface; height above the bed for the two-face columns;
1 m bins split by overlap; bins not full every hour hatched).

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc, hourly ocean_his
       SNAPSHOTS; box indices as in 20261103_pc_eulerian_u0u1u2.py.

SIGN  u is positive into Penn Cove (west). Amplitude and phase have no direction,
so their axes are not reversed.

LAYOUT sections are COLUMNS west to east (pc_cp | pc_lj | pc_lp); cross-section
panels look into the cove (west) with north at the right, each as wide as the
section really is.

OUTPUT LO_output/DM_outs/20261103_pc_section_tides/
    xsec_[con].png         cross-sections, rows amplitude | phase lag (S1: hour of max inflow), for con in SHOW
    width_tides.png        whole-width profiles, rows amplitude | phase lag (tides) | S1 hour of max inflow
    [col]_tides.png        two-face column, depth below the surface; col = north, center, south
    [col]_tides_bed.png    the same column, height above the bed
    tide_fits.p            every fit (A, A_ci, g, g_ci), reloaded on later runs (REFIT = True to redo)

run 20261103_pc_section_tides.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')        # one BLAS thread: utide is ~3x faster this way
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import utide
import cmocean
import pickle
from time import time

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove
# box indices of each section's u-faces (global i - 37, global j - 209), north to south;
# listed west to east (inner cove to mouth), which is the column order of every figure
FACES = {
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
REFIT = False                             # True: redo the fits even if tide_fits.p exists
LAT = 48.23                               # latitude for the nodal corrections (the cove spans < 0.04 deg)
CONS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'MS4', 'M6', 'S1']
KW = dict(method='ols', conf_int='linear', constit=CONS, nodal=True, trend=False, verbose=False)
SHOW = ['M2', 'S2', 'K1', 'O1', 'S1', 'M4']
COLOR = {'M2': '#0072B2', 'S2': '#56B4E9', 'K1': '#D55E00', 'O1': '#E69F00', 'S1': '#009E73', 'M4': '#CC79A7'}
EDGES = np.arange(0, 31, 1.0)             # bin edges [m]
DMID = 0.5 * (EDGES[1:] + EDGES[:-1])     # bin centres
NB = len(DMID)
COLS = {'north': 1 / 3, 'center': 1 / 2, 'south': 2 / 3}   # point on the section, fraction of the width from the north end
COLNAME = {'north': 'north 1/3 point', 'center': 'centre', 'south': 'south 1/3 point'}

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261103_pc_section_tides'
out_dir.mkdir(parents=True, exist_ok=True)


def isum(x):
    """Integral over the section, int ... dA, as a sum over elements (k, p)."""
    return x.sum(axis=(-2, -1))


def bin_transport(q, dz, dy, lo, hi, P):
    """Q_b(t), A_b(t): each cell's transport and area split among the bins by
    overlap, summed over the faces P only (as in 20261103_pc_section_profiles.py)."""
    q, dz, lo, hi = q[:, :, P], dz[:, :, P], lo[:, :, P], hi[:, :, P]
    Qb = np.zeros((len(q), NB))
    Ab = np.zeros((len(q), NB))
    for b in range(NB):
        ov = np.clip(np.minimum(hi, EDGES[b + 1]) - np.maximum(lo, EDGES[b]), 0, None)
        Qb[:, b] = isum(q * ov / dz)
        Ab[:, b] = isum(ov * dy[None, None, P])
    return Qb, Ab


def fit(T, x):
    """utide fit of one scalar series; returns (4, ncon) = A, A_ci, g, g_ci in CONS order.
    Only fitted if the series has data in >= 99% of hours: a gap that follows the tide
    (a bin that is dry at low water) would bias a harmonic fit."""
    out = np.full((4, len(CONS)), np.nan)
    ok = np.isfinite(x)
    if ok.mean() < 0.99:
        return out
    c = utide.solve(T[ok], x[ok], lat=LAT, **KW)
    ii = [list(c.name).index(n) for n in CONS]
    out[:] = [c.A[ii], c.A_ci[ii], c.g[ii], c.g_ci[ii]]
    return out


def s1_hour_calibration():
    """utide's S1 Greenwich phase of a synthetic series that peaks at 06 UTC every day."""
    t = T.astype('datetime64[s]').astype(float) / 3600   # hours since 1970-01-01 00 UTC
    c = utide.solve(T, np.cos(2 * np.pi * (t - 6) / 24), lat=LAT, **KW)
    return c.g[list(c.name).index('S1')]


def s1_hour(F):
    """Hour of maximum inflow of S1, PST (UTC-8), 0-24; NaN where not significant.
    S1 runs 15 deg per hour and a larger phase is later, so hour_UTC = 6 + (g - g6) / 15."""
    i = CONS.index('S1')
    h_utc = 6 + (F[..., 2, i] - S1_G6) / 15
    return np.where(F[..., 0, i] >= F[..., 1, i], (h_utc - 8) % 24, np.nan)


def broken(x, jump):
    """Insert NaN where a wrapped phase jumps by more than `jump`, so lines do not cross the plot."""
    x = np.array(x, float)
    j = np.flatnonzero(np.abs(np.diff(x)) > jump)
    x[j + 1] = np.nan
    return x


def lag(F, FQ, n):
    """Phase lag of constituent n behind the same constituent of the section transport,
    wrapped to (-180, 180] deg; NaN where the amplitude is not significant (A < A_ci)."""
    i = CONS.index(n)
    d = (F[..., 2, i] - FQ[2, i] + 180) % 360 - 180
    return np.where(F[..., 0, i] >= F[..., 1, i], d, np.nan)


# ------------------------------------------------------------------ fits (cached: plotting can be rerun without refitting)
cache_fn = out_dir / 'tide_fits.p'
if cache_fn.exists() and not REFIT:
    R = pickle.load(open(cache_fn, 'rb'))
    print('loaded fits from ' + str(cache_fn))
else:
    ds = xr.open_dataset(in_fn)
    T = ds.ocean_time.values
    R = {}
    time0 = time()
    for SN in FACES:
        XI_U, ETA = FACES[SN]['XI_U'], FACES[SN]['ETA']
        IRM, IRP = XI_U, XI_U + 1             # rho cells west (minus) and east (plus) of the face

        # ------------------------------------------------------------------ load snapshots
        u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)                 # (t, k, p) + into the cove
        zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().mean(axis=-1)          # (t, w, p) at the face
        pn = ds.pn.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()                         # (p, 2)
        dy = (1 / pn).mean(axis=-1)               # (p,) face width
        dz = np.diff(zw, axis=1)                  # (t, k, p) cell thickness
        q = u * dz * dy[None, None, :]            # (t, k, p) transport per element [m3/s]
        dw = zw[:, -1:, :] - zw                   # (t, w, p) depth below the surface
        hab = zw - zw[:, :1, :]                   # (t, w, p) height above the bed
        NK, NP = u.shape[1:]

        # ------------------------------------------------------------------ fit: section transport (the phase reference)
        FQ = fit(T, isum(q))                      # (4, ncon)

        # ------------------------------------------------------------------ fit: every element of the cross-section
        FX = np.full((NK, NP, 4, len(CONS)), np.nan)
        for k in range(NK):
            for p in range(NP):
                FX[k, p] = fit(T, u[:, k, p])
        print('%s: %d element fits, %.0f s so far' % (SN, NK * NP, time() - time0))

        # ------------------------------------------------------------------ fit: every bin of every profile
        xc = np.cumsum(dy) - dy / 2               # (p,) face centres, m from the north end
        PROF = {'width': (np.arange(NP), dw[:, 1:, :], dw[:, :-1, :])}                   # depth: top, bottom of each cell
        for col, frac in COLS.items():
            P = np.sort(np.argsort(np.abs(xc - frac * dy.sum()))[:2])
            PROF[col] = (P, dw[:, 1:, :], dw[:, :-1, :])
            PROF[col + '_bed'] = (P, hab[:, :-1, :], hab[:, 1:, :])                       # height: bottom, top
        out = {}
        for key, (P, lo, hi) in PROF.items():
            Qb, Ab = bin_transport(q, dz, dy, lo, hi, P)
            Ub = np.where(Ab > 0, Qb / np.where(Ab > 0, Ab, 1), np.nan)                    # (t, b) bin velocity, NaN when dry
            keep = Ab.mean(axis=0) >= dy.mean() * 1.0                                      # as for the velocity profiles
            FB = np.full((NB, 4, len(CONS)), np.nan)
            for b in np.flatnonzero(keep):
                FB[b] = fit(T, Ub[:, b])
            out[key] = dict(F=FB, P=P, full=np.where(keep, Ab.mean(axis=0) / np.maximum(Ab.max(axis=0), 1e-12), np.nan))
        print('%s: profile fits done, %.0f s so far' % (SN, time() - time0))

        R[SN] = dict(FQ=FQ, FX=FX, prof=out, dy=dy, zw0=zw.mean(axis=0), zbed=zw[:, 0, :].mean(axis=0))
        i2 = CONS.index('M2')
        print('  Q: M2 %.0f m3/s, K1 %.0f, S1 %.0f, O1 %.0f' % tuple(FQ[0, CONS.index(n)] for n in ['M2', 'K1', 'S1', 'O1']))
    ds.close()
    pickle.dump(R, open(cache_fn, 'wb'))
    print('saved fits to ' + str(cache_fn))


# ------------------------------------------------------------------ S1 clock calibration
_ds = xr.open_dataset(in_fn)
T = _ds.ocean_time.values
_ds.close()
S1_G6 = s1_hour_calibration()
print('S1 calibration: a series peaking at 06 UTC has Greenwich phase %.1f deg' % S1_G6)


# ------------------------------------------------------------------ plot: cross-sections, one figure per constituent
WR = [R[SN]['dy'].sum() for SN in FACES]  # column widths = real section widths, west to east


def draw_xsec(ax, fld, dy, zw0, cmap, vmin, vmax):
    """Each face a column of cells between its time-mean interfaces; north at right."""
    xe = np.concatenate([[0], np.cumsum(dy)]) / 1000
    for p in range(len(dy)):
        cs = ax.pcolormesh(xe[p:p + 2], zw0[:, p], fld[:, p:p + 1], cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    ax.set_xlim(xe[-1], xe[0])           # reversed: north at right = looking west, into the cove
    ax.set_facecolor('0.85')             # blanked (insignificant) phase shows grey
    ax.grid(True, alpha=0.3)
    return cs


for n in SHOW:
    i = CONS.index(n)
    vmax = np.nanpercentile(np.concatenate([100 * R[SN]['FX'][:, :, 0, i].ravel() for SN in FACES]), 99)
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharey=True, layout='constrained', gridspec_kw={'width_ratios': WR})
    for j, SN in enumerate(FACES):
        FX, dy, zw0 = R[SN]['FX'], R[SN]['dy'], R[SN]['zw0']
        ca = draw_xsec(axs[0, j], 100 * FX[:, :, 0, i], dy, zw0, cmocean.cm.speed, 0, vmax)
        if n == 'S1':
            cp = draw_xsec(axs[1, j], s1_hour(FX), dy, zw0, 'twilight', 0, 24)
            axs[1, j].set_title('%s: S1 hour of maximum inflow (PST)' % NAME[SN], fontsize=9)
        else:
            cp = draw_xsec(axs[1, j], lag(FX, R[SN]['FQ'], n), dy, zw0, 'twilight_shifted', -180, 180)
            axs[1, j].set_title('%s: %s phase lag behind Q' % (NAME[SN], n), fontsize=9)
        axs[0, j].set_title('%s: %s amplitude (Q: %.0f m$^3$ s$^{-1}$)' % (NAME[SN], n, R[SN]['FQ'][0, i]), fontsize=9)
        axs[1, j].set_xlabel('distance from north shore [km]')
    for ax in axs[:, 0]:
        ax.set_ylabel('z [m]')
    fig.colorbar(ca, ax=axs[0, :], location='right', label='%s amplitude of u [cm s$^{-1}$]' % n)
    if n == 'S1':
        cb = fig.colorbar(cp, ax=axs[1, :], location='right', label='hour of max inflow [PST = UTC-8]\ngrey = not significant')
        cb.set_ticks([0, 6, 12, 18, 24])
    else:
        cb = fig.colorbar(cp, ax=axs[1, :], location='right', label='lag behind Q [deg]\n+ = later; $\\pm$180 = counterflow; grey = not significant')
        cb.set_ticks([-180, -90, 0, 90, 180])
    fig.suptitle('%s of the section-normal velocity, utide 2024-2025 hourly snapshots; sections west to east, each looking into the cove (west), north at right; %s'
                 % (n, GTAGEX), fontsize=10)
    fig.savefig(out_dir / ('xsec_%s.png' % n), dpi=200, transparent=True)
    plt.close(fig)


# ------------------------------------------------------------------ plot: profiles, one line per constituent
def hatch_partial(ax, full):
    """Hatch the bins that are not full every hour (full < 0.99)."""
    for b in np.flatnonzero(full < 0.99):
        ax.axhspan(EDGES[b], EDGES[b + 1], facecolor='none', edgecolor='0.6', hatch='///', lw=0, zorder=0)


def plot_prof(key, fname, where, bed=False):
    fig, axs = plt.subplots(3, 3, figsize=(13, 14), sharey=True, layout='constrained')
    for j, SN in enumerate(FACES):
        pr = R[SN]['prof'][key + '_bed' if bed else key]
        sub = '' if key == 'width' else ', faces p = %d-%d' % tuple(pr['P'])
        for n in SHOW:
            i = CONS.index(n)
            axs[0, j].plot(100 * pr['F'][:, 0, i], DMID, color=COLOR[n], lw=2, label=n)
            if n == 'S1':
                axs[2, j].plot(broken(s1_hour(pr['F']), 12), DMID, color=COLOR[n], lw=2, marker='o', ms=3, label=n)
            else:
                axs[1, j].plot(broken(lag(pr['F'], R[SN]['FQ'], n), 180), DMID, color=COLOR[n], lw=2, label=n)
        for ax in axs[:, j]:
            hatch_partial(ax, pr['full'])
            ax.grid(True, alpha=0.3)
        axs[0, j].set_xlim(0, None)
        axs[1, j].set_xlim(-180, 180)
        axs[1, j].set_xticks([-180, -90, 0, 90, 180])
        axs[1, j].axvline(0, color='k', lw=0.6)
        axs[0, j].set_title(NAME[SN] + sub + ': amplitude', fontsize=9)
        axs[1, j].set_title(NAME[SN] + sub + ': phase lag behind Q', fontsize=9)
        axs[2, j].set_xlim(0, 24)
        axs[2, j].set_xticks([0, 6, 12, 18, 24])
        axs[2, j].set_title(NAME[SN] + sub + ': S1 hour of maximum inflow', fontsize=9)
        axs[0, j].set_xlabel(r'amplitude of u [cm s$^{-1}$]', fontsize=8)
        axs[1, j].set_xlabel('tidal lag behind Q [deg], + = later, $\\pm$180 = counterflow', fontsize=8)
        axs[2, j].set_xlabel('S1 hour of max inflow [PST = UTC-8]', fontsize=8)
    axs[0, 0].plot([], [], color='none', label='hatched: bin not\nfull every hour')
    axs[0, 0].legend(fontsize=7, loc='lower right' if not bed else 'upper right')
    if not bed:
        axs[0, 0].invert_yaxis()
    for ax in axs[:, 0]:
        ax.set_ylabel('height above the bed [m]' if bed else 'depth below the surface [m]')
    fig.suptitle('Tidal constituents of the section-normal velocity, %s, 1 m bins (%s), utide 2024-2025 hourly snapshots, %s'
                 % (where, 'height above the bed' if bed else 'depth below the moving surface', GTAGEX), fontsize=10)
    fig.savefig(out_dir / fname, dpi=200, transparent=True)
    plt.close(fig)


plot_prof('width', 'width_tides.png', 'whole width')
for col in COLS:
    plot_prof(col, col + '_tides.png', COLNAME[col] + ' (two faces)')
    plot_prof(col, col + '_tides_bed.png', COLNAME[col] + ' (two faces)', bed=True)
print('saved to ' + str(out_dir))
