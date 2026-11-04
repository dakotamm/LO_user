"""
Gradient Richardson number Ri = N^2 / S^2 at the three Penn Cove sections, pc_cp
(off Coupeville), pc_lj (middle) and pc_lp (mouth), of wb1_t0_xn11abbur00, for
flood, ebb and all hours, as cross-sections and profiles laid out like
20261103_pc_section_shear.py and 20261103_pc_section_strat.py.

LAYERS. Everything is computed hourly, within each face, between the centres of
cells k and k+1 (the same layers as the shear and N^2 scripts):
    N^2 = -(g / rho0) d(sigma0)/dz          sigma0 from TEOS-10, salt/temp averaged onto the face
    S^2 = (du/dz)^2 + (dv/dz)^2              FULL shear: u normal to the section at the face, and
                                             v along it, the mean of the 4 v points around the face
    Ri  = N^2 / S^2
The along-section v is included because mixing feels the total shear; leaving it
out underestimates S^2 and overestimates Ri (its share is printed). Masked v
points (land) are 0, the no-normal-flow value.

AVERAGING. Ri itself is not averaged: it runs to infinity as S^2 -> 0 (a few
slack hours would dominate) and is negative where N^2 < 0. Instead, area-weighted
as for the shear and N^2,
    f_crit = share of the area-hours with Ri < 1/4      (N^2 < 0 counts: unstable)
    Ri_bar = <N^2> / <S^2>                              Ri of the typical state, log scale
per element (cross-sections) and per 1 m bin (profiles, split by overlap; bins
not full every hour hatched). No Ri is resolved below the bottom cell centre or
above the top one (grey).

FLOOD / EBB from the section-total transport Q(t) = int u dA (hourly):
flood Q > 0 (into the cove), ebb Q < 0; "all" = every hour.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc, hourly ocean_his
       SNAPSHOTS (u, v, salt, temp, z_rho, z_w); box indices as in
       20261103_pc_eulerian_u0u1u2.py. A u-face between rho (j, i) and (j, i+1) is
       surrounded by v at (eta_v, xi_v) = (j-1, i), (j-1, i+1), (j, i), (j, i+1).

LAYOUT sections are COLUMNS west to east (pc_cp | pc_lj | pc_lp); cross-section
panels look into the cove (west) with north at the right, each as wide as the
section really is.

OUTPUT LO_output/DM_outs/20261103_pc_section_Ri/
    section_fcrit.png    cross-sections of f_crit; rows flood, ebb, all
    section_Ribar.png    cross-sections of <N^2>/<S^2> (log); rows flood, ebb, all
    width_Ri.png         whole-width profiles, rows f_crit | <N^2>/<S^2>, lines flood / ebb / all
    [col]_Ri.png         two-face column, depth below the surface; col = north, center, south
    [col]_Ri_bed.png     the same column, height above the bed

run 20261103_pc_section_Ri.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import xarray as xr
import cmocean
import gsw

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove; v is not flipped (+ = north)
# box indices of each section's u-faces (global i - 37, global j - 209), north to south;
# listed west to east (inner cove to mouth), which is the column order of every figure
FACES = {
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
G, RHO0 = 9.81, 1025.0                    # gravity [m/s2], reference density [kg/m3]
LON, LAT = -122.69, 48.23                 # one position for the cove (for SA_from_SP)
RI_CRIT = 0.25
EDGES = np.arange(0, 31, 1.0)             # bin edges [m]
DMID = 0.5 * (EDGES[1:] + EDGES[:-1])     # bin centres
NB = len(DMID)
COLS = {'north': 1 / 3, 'center': 1 / 2, 'south': 2 / 3}   # point on the section, fraction of the width from the north end
COLNAME = {'north': 'north 1/3 point', 'center': 'centre', 'south': 'south 1/3 point'}
PHASES = ['flood', 'ebb', 'all']
COLOR = {'flood': '#D55E00', 'ebb': '#0072B2', 'all': 'k'}

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261103_pc_section_Ri'
out_dir.mkdir(parents=True, exist_ok=True)


def isum(x):
    """Sum over elements (k, p) of a section."""
    return x.sum(axis=(-2, -1))


def bin_sums(fields, dy, lo, hi, P):
    """Split each layer among the bins by overlap, over the faces P only. lo < hi is
    the layer's extent in the bin coordinate. Returns, per hour and bin, the area
    A = sum ov dy and, for each field X, sum X ov dy."""
    fields = [X[:, :, P] for X in fields]
    lo, hi = lo[:, :, P], hi[:, :, P]
    A = np.zeros((len(lo), NB))
    XA = np.zeros((len(fields), len(lo), NB))
    for b in range(NB):
        ova = np.clip(np.minimum(hi, EDGES[b + 1]) - np.maximum(lo, EDGES[b]), 0, None) * dy[None, None, P]   # (t, k, p) m2
        A[:, b] = isum(ova)
        for n, X in enumerate(fields):
            XA[n, :, b] = isum(X * ova)
    return A, XA


def phase_stats(A, XA, MASK, amin):
    """Per bin and phase: f_crit = sum crit / sum A, Ri_bar = sum N2 / sum S2.
    XA holds the sums of (crit, N2, S2). Bins with mean area < amin are dropped."""
    out = {}
    for ph in PHASES:
        m = MASK[ph]
        Am = A[m].mean(axis=0)
        keep = Am >= amin
        C, N, S = (XA[n][m].sum(axis=0) for n in range(3))
        out[ph] = dict(fcrit=np.where(keep, C / np.maximum(A[m].sum(axis=0), 1e-12), np.nan),
                       Ribar=np.where(keep, N / np.maximum(S, 1e-30), np.nan),
                       full=np.where(keep, Am / np.maximum(A.max(axis=0), 1e-12), np.nan), frac=m.mean())
    return out


ds = xr.open_dataset(in_fn)
R = {}
for SN in FACES:
    XI_U, ETA = FACES[SN]['XI_U'], FACES[SN]['ETA']
    IRM, IRP = XI_U, XI_U + 1             # rho cells west (minus) and east (plus) of the face
    at_face = lambda vn: ds[vn].isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().astype(float).mean(axis=-1)

    # ------------------------------------------------------------------ load snapshots
    u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)              # (t, k, p) + into the cove
    v4 = [ds.v.isel(eta_v=ETA + dj, xi_v=[IRM, IRP]).to_numpy().astype(float) for dj in (-1, 0)]   # 2 x (t, k, p, 2)
    v = np.nan_to_num(np.concatenate(v4, axis=-1)).mean(axis=-1)                     # (t, k, p) along-section, + north; land = 0
    salt, temp = at_face('salt'), at_face('temp')
    zw, zr = at_face('z_w'), at_face('z_rho')
    pn = ds.pn.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()
    dy = (1 / pn).mean(axis=-1)               # (p,) face width
    dz = np.diff(zw, axis=1)
    Q = isum(u * dz * dy[None, None, :])      # (t,) section-total transport: sets flood / ebb
    MASK = {'flood': Q > 0, 'ebb': Q < 0, 'all': np.ones(len(Q), bool)}

    # ------------------------------------------------------------------ N^2, S^2, Ri on the layers between cell centres
    SA = gsw.SA_from_SP(salt, -zr, LON, LAT)
    sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, temp))
    dzr = np.diff(zr, axis=1)                 # (t, k-1, p)
    N2 = -(G / RHO0) * np.diff(sig, axis=1) / dzr
    Su2 = (np.diff(u, axis=1) / dzr)**2
    Sv2 = (np.diff(v, axis=1) / dzr)**2
    S2 = Su2 + Sv2
    crit = (N2 < RI_CRIT * S2).astype(float)  # Ri < 1/4, written so S2 = 0 never divides; N2 < 0 counts
    W = dzr
    print('\n%s: dv/dz carries %.0f%% of S^2; Ri < 1/4 in %.0f%% of layer-hours; <N2>/<S2> = %.2f (u-only shear would give %.2f)'
          % (SN, 100 * (Sv2 * W).sum() / (S2 * W).sum(), 100 * (crit * W).sum() / W.sum(),
             (N2 * W).sum() / (S2 * W).sum(), (N2 * W).sum() / (Su2 * W).sum()))

    # ------------------------------------------------------------------ cross-sections: per element, per phase
    xs = {}
    for ph in PHASES:
        m = MASK[ph]
        w = W[m]
        xs[ph] = dict(fcrit=(crit[m] * w).sum(axis=0) / w.sum(axis=0),
                      Ribar=(N2[m] * w).sum(axis=0) / (S2[m] * w).sum(axis=0),
                      zr=zr[m].mean(axis=0), zbed=zw[m, 0, :].mean(axis=0), zsurf=zw[m, -1, :].mean(axis=0))

    # ------------------------------------------------------------------ profiles: whole width, then each two-face column
    zeta, zbed = zw[:, -1:, :], zw[:, :1, :]
    EXT = {'d': (zeta - zr[:, 1:, :], zeta - zr[:, :-1, :]), 'h': (zr[:, :-1, :] - zbed, zr[:, 1:, :] - zbed)}
    xc = np.cumsum(dy) - dy / 2               # (p,) face centres, m from the north end
    PROF = {'width': (np.arange(len(dy)), 'd')}
    for col, frac in COLS.items():
        P = np.sort(np.argsort(np.abs(xc - frac * dy.sum()))[:2])
        PROF[col] = (P, 'd')
        PROF[col + '_bed'] = (P, 'h')
    pr = {}
    for key, (P, c) in PROF.items():
        pr[key] = dict(P=P, st=phase_stats(*bin_sums([crit, N2, S2], dy, *EXT[c], P), MASK, 0.25 * dy.mean()))
        if c == 'h':
            pr[key]['zb'] = (zr[:, 0, P] - zw[:, 0, P]).mean()                 # bottom cell centre above the bed
    R[SN] = dict(xs=xs, prof=pr, dy=dy)
ds.close()


# ------------------------------------------------------------------ plot: cross-sections
WR = [R[SN]['dy'].sum() for SN in FACES]  # column widths = real section widths, west to east
LRC = np.log10(RI_CRIT)


def draw_xsec(ax, d, dy, fld, cmap, vmin, vmax, line):
    """Each face a column of layers between its mean cell centres, grey where no Ri is
    resolved; one contour at `line`; north at right."""
    F = np.log10(np.clip(d[fld], 1e-3, None)) if fld == 'Ribar' else d[fld]
    xe = np.concatenate([[0], np.cumsum(dy)]) / 1000
    for p in range(len(dy)):
        cs = ax.pcolormesh(xe[p:p + 2], d['zr'][:, p], F[:, p:p + 1], cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax.fill_between(xe[p:p + 2], d['zbed'][p], d['zr'][0, p], color='0.8', lw=0)
        ax.fill_between(xe[p:p + 2], d['zr'][-1, p], d['zsurf'][p], color='0.8', lw=0)
    X = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), F.shape)
    ax.contour(X, 0.5 * (d['zr'][1:] + d['zr'][:-1]), F, levels=[line], colors='k', linewidths=1.2)
    ax.set_xlim(xe[-1], xe[0])           # reversed: north at right = looking west, into the cove
    ax.grid(True, alpha=0.3)
    return cs


for fld, cmap, vmin, vmax, line, lab, fname in [
        ('fcrit', cmocean.cm.amp, 0, 1, 0.5, 'share of hours with Ri < 1/4 (black: 0.5)', 'section_fcrit.png'),
        ('Ribar', cmocean.cm.balance, LRC - 1.5, LRC + 1.5, LRC, r'log$_{10}$ <N$^2$>/<S$^2$> (black: Ri = 1/4; red = more stable)', 'section_Ribar.png')]:
    fig, axs = plt.subplots(3, 3, figsize=(14, 9), sharey=True, layout='constrained', gridspec_kw={'width_ratios': WR})
    for i, ph in enumerate(PHASES):
        for j, SN in enumerate(FACES):
            cs = draw_xsec(axs[i, j], R[SN]['xs'][ph], R[SN]['dy'], fld, cmap, vmin, vmax, line)
            axs[i, j].set_title('%s: %s (%.0f%% of hours)' % (NAME[SN], ph, 100 * R[SN]['prof']['width']['st'][ph]['frac']), fontsize=9)
            if i == 2:
                axs[i, j].set_xlabel('distance from north shore [km]')
        axs[i, 0].set_ylabel('z [m]')
    cb = fig.colorbar(cs, ax=axs, location='bottom', shrink=0.5, aspect=40, label=lab + '; grey = no resolved Ri')
    if fld == 'Ribar':
        tk = np.log10([0.025, 0.1, 0.25, 1, 2.5])
        cb.set_ticks(tk)
        cb.set_ticklabels(['0.025', '0.1', '0.25', '1', '2.5'])
    fig.suptitle('Richardson number Ri = N$^2$/S$^2$ (full shear, u and v) on the sections, west to east, each looking into the cove (west; north at right); hourly snapshots, %s, %s to %s'
                 % (GTAGEX, DS0, DS1), fontsize=10)
    fig.savefig(out_dir / fname, dpi=200, transparent=True)
    plt.close(fig)


# ------------------------------------------------------------------ plot: profiles, rows f_crit | <N^2>/<S^2>
def hatch_partial(ax, full):
    """Hatch the bins that are not full every hour (full < 0.99)."""
    for b in np.flatnonzero(full < 0.99):
        ax.axhspan(EDGES[b], EDGES[b + 1], facecolor='none', edgecolor='0.6', hatch='///', lw=0, zorder=0)


def plot_prof(key, fname, where, bed=False):
    fig, axs = plt.subplots(2, 3, figsize=(13, 10), sharey=True, layout='constrained')
    for j, SN in enumerate(FACES):
        pr = R[SN]['prof'][key + '_bed' if bed else key]
        st = pr['st']
        sub = '' if key == 'width' else ', faces p = %d-%d' % tuple(pr['P'])
        for i, (fld, lab) in enumerate([('fcrit', 'share of hours with Ri < 1/4'), ('Ribar', r'<N$^2$>/<S$^2$>')]):
            ax = axs[i, j]
            hatch_partial(ax, st['all']['full'])
            if bed:
                ax.axhspan(0, pr['zb'], color='0.88', zorder=0)
                ax.text(0.02, pr['zb'] / 2, 'no resolved Ri below the bottom cell centre', transform=ax.get_yaxis_transform(),
                        fontsize=6, va='center', color='0.3')
            for ph in PHASES:
                ax.plot(st[ph][fld], DMID, color=COLOR[ph], lw=2, label='%s (%.0f%% of hours)' % (ph, 100 * st[ph]['frac']))
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(lab, fontsize=8)
            ax.set_title(NAME[SN] + sub + ': ' + lab, fontsize=9)
        axs[0, j].set_xlim(0, 1)
        axs[1, j].set_xscale('log')
        axs[1, j].set_xlim(0.05, 5)
        axs[1, j].set_xticks([0.1, 0.25, 1, 2.5])
        axs[1, j].set_xticklabels(['0.1', '0.25', '1', '2.5'])
        axs[1, j].xaxis.set_minor_formatter(plt.NullFormatter())
        axs[1, j].axvline(RI_CRIT, color='k', lw=1, ls='--')
    axs[0, 0].legend(fontsize=7, loc='lower right' if not bed else 'upper right')
    if not bed:
        axs[0, 0].invert_yaxis()
    for ax in axs[:, 0]:
        ax.set_ylabel('height above the bed [m]' if bed else 'depth below the surface [m]')
    fig.suptitle('Ri = N$^2$/S$^2$ (full shear), %s, 1 m bins (%s); dashed: Ri = 1/4; %s, %s to %s'
                 % (where, 'height above the bed' if bed else 'depth below the moving surface', GTAGEX, DS0, DS1), fontsize=10)
    fig.savefig(out_dir / fname, dpi=200, transparent=True)
    plt.close(fig)


plot_prof('width', 'width_Ri.png', 'whole width')
for col in COLS:
    plot_prof(col, col + '_Ri.png', COLNAME[col] + ' (two faces)')
    plot_prof(col, col + '_Ri_bed.png', COLNAME[col] + ' (two faces)', bed=True)
print('\nsaved to ' + str(out_dir))
