"""
Cross-section averaged velocity profiles U(d) at the three Penn Cove sections,
pc_lp (mouth), pc_lj (middle) and pc_cp (off Coupeville), of wb1_t0_xn11abbur00:
the average flood, the average ebb, and the residual.

VERTICAL COORDINATE d = depth below the moving free surface, in 1 m bins.
Averaging along s-levels instead would mix depths: one s-level sits at 8 m on
a shallow edge face and at 15 m on the deep centre face.

BINS. Each ROMS cell c = (k, p) on the section spans depths d_top..d_bot below
the surface. Its transport q_c = u_c dA_c is split among the bins it overlaps,
in proportion to the overlap (exact if u is uniform within the cell):
    Q_b(t) = sum_c q_c * ov_cb / dz_c         transport through bin b    [m3/s]
    A_b(t) = sum_c dy_p * ov_cb               wetted area of bin b       [m2]
    ov_cb  = overlap of cell c with bin b, in m of depth
so sum_b Q_b = int u dA and sum_b A_b = int dA, exactly, every hour.

FLOOD / EBB from the section-total transport Q(t) = int u dA (hourly):
flood Q > 0 (into the cove), ebb Q < 0.

PROFILE over the hours in a phase (flood, ebb, or all hours = residual):
    U_b = sum_t Q_b / sum_t A_b              transport-weighted, so that
    sum_b U_b * mean(A_b) = mean Q           exactly (printed as a check).
Bins whose mean wetted area is less than one face (dy x 1 m) are dropped.

SAMPLING BIAS AT THE END OF A PROFILE. The free surface moves with the tide and
the bed does not, so a bin at the far end of the coordinate (the deepest bins
below the surface, the highest bins above the bed) holds water only when the
tide is high. Its average then covers only the high-water hours of a phase
(e.g. late flood), while the bins in between cover every hour. Measured per bin as
    full_b = mean_t A_b / max_t A_b           (1 = full every hour)
and bins with full_b < 0.99 are HATCHED on the plots, labeled with full_b.

ELEMENTS (as in 20261013_maccready2011_sec2.py)
    dA_c(t) = dy_p * dz_c(t)
    dy_p    = 0.5 * (1/pn[irm] + 1/pn[irp])
    dz_c(t) = 0.5 * (Hz[irm] + Hz[irp]),  Hz = diff(z_w)  (ROMS u-point thickness)

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       hourly ocean_his SNAPSHOTS (on the hour). Box indices as in
       20261103_pc_eulerian_u0u1u2.py; p = 0 is the north end.

SIGN  u is positive UP-ESTUARY = into Penn Cove (the sections' own positive
direction is east = out of the cove, so it is flipped). On the profile plots the
velocity axis is REVERSED so that west (+, into the cove) is at the left and east
at the right, matching the west-to-east panel order.

COLUMN PROFILES (centre, north 1/3, south 1/3). The same bins and averaging,
but summing only over the two faces whose centres are closest to a point on the
section: the midpoint, the point 1/3 of the width from the north end, and the
point 1/3 from the south end. Each is a ~400 m wide column. On the 8-face
sections (pc_lj, pc_cp) the north and south columns each share one face with
the centre column. Flood / ebb are still set by the section-total Q.

CENTRE PROFILE vs HEIGHT ABOVE THE BED. The two centre faces again, binned by
height above the bed (their beds differ by < 0.5 m, so the coordinate is the
same across the column). ROMS does not set u = 0 at the bed: this run uses
quadratic drag (UV_QDRAG, RDRG2 = Cd = 3.0e-3, BLANK.in), a stress on the bottom
cell, so the lowest resolved velocity (bin centre z_b = 0.5 m) is nonzero.
Below z_b the profile is EXTENDED, not resolved, with a log layer
    u(z) = U_b ln(z / z0) / ln(z_b / z0),   z0 = z_b exp(-kappa / sqrt(Cd))
i.e. the roughness that gives the same bed stress as the model's drag law
(z0 ~ 3e-4 m; Zob = 0.02 m in BLANK.in is used only by the GLS closure, and as
a log-layer z0 would imply Cd ~ 0.015, 5x the drag the model applies).
Drawn dashed over a grey band.

OUTPUT LO_output/DM_outs/20261103_pc_section_profiles/
    section_profiles.png    whole width. top: flood, ebb, residual; bottom: residual alone
    [col]_profiles.png      the column's two faces, same layout; col = center, north, south
    [col]_profiles_bed.png  the column's two faces vs height above the bed, with the
                            inferred log layer below the lowest bin; bottom row = lowest 3 m

run 20261103_pc_section_profiles.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove
# box indices of each section's u-faces (global i - 37, global j - 209), north to south;
# listed west to east (inner cove to mouth), which is the panel order in every figure
FACES = {
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
EDGES = np.arange(0, 31, 1.0)             # bin edges, depth below the surface [m]
DMID = 0.5 * (EDGES[1:] + EDGES[:-1])     # bin centres
NB = len(DMID)
KAPPA = 0.41                              # von Karman
CD = 3.0e-3                               # RDRG2, quadratic bottom drag coefficient (BLANK.in)
COLS = {'north': 1 / 3, 'center': 1 / 2, 'south': 2 / 3}   # point on the section, fraction of the width from the north end
COLNAME = {'north': 'north 1/3 point', 'center': 'centre', 'south': 'south 1/3 point'}
PHASES = ['flood', 'ebb', 'residual']
COLOR = {'flood': '#D55E00', 'ebb': '#0072B2', 'residual': 'k'}

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261103_pc_section_profiles'
out_dir.mkdir(parents=True, exist_ok=True)


def isum(x):
    """Integral over the section, int ... dA, as a sum over elements (k, p)."""
    return x.sum(axis=(-2, -1))


def bin_transport(q, dz, dy, lo, hi, P):
    """Q_b(t), A_b(t): each cell's transport and area split among the bins by
    overlap, summed over the faces P only. lo < hi is each cell's extent in the
    bin coordinate (depth below the surface, or height above the bed)."""
    q, dz, lo, hi = q[:, :, P], dz[:, :, P], lo[:, :, P], hi[:, :, P]
    Qb = np.zeros((len(q), NB))               # (t, b) transport through bin b
    Ab = np.zeros((len(q), NB))               # (t, b) wetted area of bin b
    for b in range(NB):
        ov = np.clip(np.minimum(hi, EDGES[b + 1]) - np.maximum(lo, EDGES[b]), 0, None)   # (t, k, p) m
        Qb[:, b] = isum(q * ov / dz)
        Ab[:, b] = isum(ov * dy[None, None, P])
    return Qb, Ab


def profiles(Qb, Ab, Q, amin):
    """flood / ebb / residual profiles U_b = sum_t Q_b / sum_t A_b; bins with mean
    wetted area < amin are dropped."""
    MASK = {'flood': Q > 0, 'ebb': Q < 0, 'residual': np.ones(len(Q), bool)}
    prof = {}
    for ph in PHASES:
        m = MASK[ph]
        Abm = Ab[m].mean(axis=0)              # (b,) mean wetted area over the phase
        U = Qb[m].sum(axis=0) / np.where(Abm > 0, Ab[m].sum(axis=0), np.nan)
        prof[ph] = dict(U=np.where(Abm >= amin, U, np.nan), Abm=Abm, frac=m.mean(),
                        Q=Q[m].mean(), Qcol=Qb[m].sum(axis=1).mean(),
                        full=np.where(Abm >= amin, Abm / np.maximum(Ab.max(axis=0), 1e-12), np.nan))   # (b,) how full the bin is, on average
        print('    %-8s %4.1f%% of hours, mean Q %+8.2f m3/s, sum_b U_b * mean A_b = %+8.2f m3/s'
              % (ph, 100 * m.mean(), Qb[m].sum(axis=1).mean(), np.nansum(U * Abm)))
    return prof


ds = xr.open_dataset(in_fn)
R = {}
for SN in FACES:
    XI_U, ETA = FACES[SN]['XI_U'], FACES[SN]['ETA']
    IRM, IRP = XI_U, XI_U + 1             # rho cells west (minus) and east (plus) of the face

    # ------------------------------------------------------------------ load snapshots
    u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)   # (t, k, p)
    zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().mean(axis=-1)   # (t, w, p) at the face
    pn = ds.pn.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()             # (p, 2)

    dz = np.diff(zw, axis=1)                  # (t, k, p)
    dy = (1 / pn).mean(axis=-1)               # (p,)
    dA = dz * dy[None, None, :]               # (t, k, p)
    q = u * dA                                # (t, k, p) transport per element [m3/s]

    # ------------------------------------------------------------------ depth below the surface of each cell
    dw = zw[:, -1:, :] - zw                   # (t, w, p), 0 at the surface, water depth at the bed
    d_bot = dw[:, :-1, :]                     # (t, k, p) lower face of cell k (k = 0 is the bed)
    d_top = dw[:, 1:, :]                      # (t, k, p) upper face
    assert d_bot.max() < EDGES[-1]            # every cell falls inside the bins
    h = -zw[0, 0, :]                          # (p,) depth at the face

    # ------------------------------------------------------------------ whole width
    ALL = np.arange(u.shape[2])
    Qb, Ab = bin_transport(q, dz, dy, d_top, d_bot, ALL)
    Q = isum(q)                               # (t,) section-total transport: sets flood / ebb
    print('\n' + SN)
    print('  max |sum_b Q_b - int u dA| = %.1e m3/s, max |sum_b A_b - int dA| = %.1e m2'
          % (np.abs(Qb.sum(axis=1) - Q).max(), np.abs(Ab.sum(axis=1) - isum(dA)).max()))
    print('  whole width')
    R[SN] = {'width': profiles(Qb, Ab, Q, dy.mean() * 1.0)}

    # ------------------------------------------------------------------ columns: the two faces nearest each point
    xc = np.cumsum(dy) - dy / 2               # (p,) face centres, m from the north end
    hab = zw - zw[:, :1, :]                   # (t, w, p) height above the bed, 0 at the bed
    assert hab.max() < EDGES[-1]
    for col, frac in COLS.items():
        P = np.sort(np.argsort(np.abs(xc - frac * dy.sum()))[:2])
        R[SN][col + '_P'] = P
        print('  %s: faces p = %d, %d (h %.1f, %.1f m)' % (COLNAME[col], P[0], P[1], h[P[0]], h[P[1]]))
        Qb, Ab = bin_transport(q, dz, dy, d_top, d_bot, P)                       # depth below the surface
        R[SN][col] = profiles(Qb, Ab, Q, dy.mean() * 1.0)
        Qb, Ab = bin_transport(q, dz, dy, hab[:, :-1, :], hab[:, 1:, :], P)     # height above the bed
        print('  %s vs height above the bed (bottom cell %.2f m thick)' % (COLNAME[col], dz[:, 0, P].mean()))
        R[SN][col + '_bed'] = profiles(Qb, Ab, Q, dy.mean() * 1.0)
ds.close()

# ------------------------------------------------------------------ plot
def hatch_partial(ax, full, label=True):
    """Hatch the bins that are not full every hour (full_b < 0.99), with full_b written in."""
    for b in np.flatnonzero(full < 0.99):
        ax.axhspan(EDGES[b], EDGES[b + 1], facecolor='none', edgecolor='0.6', hatch='///', lw=0, zorder=0)
        if label:
            ax.text(0.98, DMID[b], 'full %.0f%%' % (100 * full[b]), transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=6, color='0.35')


# top row: flood, ebb and residual together; bottom row: residual alone, zoomed
def plot(key, fname, what):
    fig, axs = plt.subplots(2, 3, figsize=(13, 9), sharey=True, layout='constrained')
    for j, SN in enumerate(FACES):
        sub = '' if key == 'width' else ', faces p = %d-%d' % tuple(R[SN][key + '_P'])
        ax = axs[0, j]
        for ph in PHASES:
            p = R[SN][key][ph]
            ax.plot(100 * p['U'], DMID, color=COLOR[ph], lw=2,
                    label='%s (%.0f%% of hours, mean Q %+.0f m$^3$ s$^{-1}$)' % (ph, 100 * p['frac'], p['Qcol']))
        hatch_partial(ax, R[SN][key]['residual']['full'])
        ax.set_title(NAME[SN] + sub, fontsize=10)
        ax.fill_between([], [], facecolor='none', edgecolor='0.6', hatch='///', label='bin not full every hour:\naverages only higher-water hours')
        ax.legend(fontsize=7, loc='lower left')
        ax = axs[1, j]
        hatch_partial(ax, R[SN][key]['residual']['full'])
        ax.plot(100 * R[SN][key]['residual']['U'], DMID, color=COLOR['residual'], lw=2)
        ax.set_title(NAME[SN] + sub + ': residual', fontsize=10)
        ax.set_xlabel(r'$\leftarrow$ west, into the cove (+)    velocity [cm s$^{-1}$]    out (-), east $\rightarrow$', fontsize=8)
    for ax in axs.flat:
        ax.invert_xaxis()                     # west (+, into the cove) at left, east at right
        ax.axvline(0, color='k', lw=0.6)
        ax.grid(True, alpha=0.3)
    axs[0, 0].invert_yaxis()                  # depth increases downward (shared y)
    axs[0, 0].set_ylabel('depth below the surface [m]')
    axs[1, 0].set_ylabel('depth below the surface [m]')
    fig.suptitle('%s velocity, U = sum Q_b / sum A_b in 1 m depth bins below the moving surface, hourly snapshots, %s, %s to %s'
                 % (what.capitalize(), GTAGEX, DS0, DS1), fontsize=10)
    fig.savefig(out_dir / fname, dpi=200, transparent=True)
    plt.close(fig)


# ------------------------------------------------------------------ plot vs height above the bed, with the inferred log layer
ZB = DMID[0]                              # lowest resolved point: centre of the 0-1 m bin
Z0 = ZB * np.exp(-KAPPA / np.sqrt(CD))    # roughness matching the model's quadratic drag
ZL = np.geomspace(Z0, ZB, 60)             # heights for the log layer
FL = np.log(ZL / Z0) / np.log(ZB / Z0)    # log-layer shape, 0 at z0, 1 at z_b
print('\nlog layer: z_b = %.2f m, z0 = %.1e m' % (ZB, Z0))


def plot_bed(col, fname):
    fig, axs = plt.subplots(2, 3, figsize=(13, 10), layout='constrained',
                            gridspec_kw={'height_ratios': [2, 1]})
    for j, SN in enumerate(FACES):
        for i, ztop in enumerate([None, 3.0]):
            ax = axs[i, j]
            ax.axhspan(0, ZB, color='0.88', zorder=0)
            hatch_partial(ax, R[SN][col + '_bed']['residual']['full'], label=(i == 0))
            for ph in PHASES:
                p = R[SN][col + '_bed'][ph]
                ax.plot(100 * p['U'], DMID, color=COLOR[ph], lw=2, marker='o' if ztop else None, ms=4,
                        label='%s (mean Q %+.0f m$^3$ s$^{-1}$)' % (ph, p['Qcol']))
                ax.plot(100 * p['U'][0] * np.concatenate([[0], FL]), np.concatenate([[0], ZL]),
                        color=COLOR[ph], lw=2, ls='--')
            ax.invert_xaxis()                 # west (+, into the cove) at left, east at right
            ax.axvline(0, color='k', lw=0.6)
            ax.grid(True, alpha=0.3)
            if ztop:
                ax.set_ylim(0, ztop)
                ax.text(0.02, ZB / 2 / ztop, 'not resolved: log layer from the\nmodel drag law (Cd = %.0e, z$_0$ = %.0e m)' % (CD, Z0),
                        transform=ax.transAxes, fontsize=7, va='center', color='0.3')
                ax.set_xlabel(r'$\leftarrow$ west, into the cove (+)    velocity [cm s$^{-1}$]    out (-), east $\rightarrow$', fontsize=8)
                ax.set_title('lowest 3 m', fontsize=9)
            else:
                ax.set_ylim(0, None)
                ax.set_title(NAME[SN] + ', faces p = %d-%d' % tuple(R[SN][col + '_P']), fontsize=10)
        if j == 0:
            axs[0, 0].plot([], [], color='0.4', lw=2, label='resolved (1 m bins)')
            axs[0, 0].plot([], [], color='0.4', lw=2, ls='--', label='inferred log layer, grey band')
            axs[0, 0].fill_between([], [], facecolor='none', edgecolor='0.6', hatch='///', label='bin not full every hour:\naverages only higher-water hours')
        axs[0, j].legend(fontsize=7, loc='upper left')
    axs[0, 0].set_ylabel('height above the bed [m]')
    axs[1, 0].set_ylabel('height above the bed [m]')
    fig.suptitle('Velocity at the %s (two faces) vs height above the bed, U = sum Q_b / sum A_b in 1 m bins, hourly snapshots, %s, %s to %s'
                 % (COLNAME[col], GTAGEX, DS0, DS1), fontsize=10)
    fig.savefig(out_dir / fname, dpi=200, transparent=True)
    plt.close(fig)


plot('width', 'section_profiles.png', 'cross-section averaged')
for col in COLS:
    plot(col, col + '_profiles.png', 'velocity at the %s (two faces)' % COLNAME[col])
    plot_bed(col, col + '_profiles_bed.png')
print('\nsaved to ' + str(out_dir))
