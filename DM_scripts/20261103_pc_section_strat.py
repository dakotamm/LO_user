"""
Stratification at the three Penn Cove sections, pc_cp (off Coupeville), pc_lj
(middle) and pc_lp (mouth), of wb1_t0_xn11abbur00: the squared buoyancy
frequency N^2 and potential density sigma0, averaged over flood, ebb and all
hours, as cross-sections and profiles laid out like 20261103_pc_section_shear.py.

DENSITY. Salt and temp are at rho points, so the value at a section face is the
mean of the two rho cells either side (as the tef2 extraction does). ROMS temp
is potential temperature, so with TEOS-10 (gsw):
    SA = SA_from_SP(salt, p, lon, lat),  CT = CT_from_pt(SA, temp),  sigma0(SA, CT)
with p = -z_rho [dbar ~ m] and one lon/lat for the cove (the absolute-salinity
anomaly does not vary over 4 km).

N^2, hourly, within each face, between the centres of cells k and k+1 - the same
layers as the shear S in 20261103_pc_section_shear.py, so Ri = N^2 / S^2 can be
formed layer by layer:
    N^2 = -(g / rho0) * (sigma0[k+1] - sigma0[k]) / (z_rho[k+1] - z_rho[k])     [1/s^2]
Potential instead of in-situ density makes no difference over 25 m of water.
No N^2 is resolved below the bottom cell centre or above the top one (grey).

AVERAGING (area-weighted, as for shear): per element for the cross-sections,
weighted by layer (N^2) or cell (sigma0) thickness; per 1 m bin for the profiles,
each layer / cell split among the bins it overlaps. Bins not full every hour
are hatched.

FLOOD / EBB from the section-total transport Q(t) = int u dA (hourly):
flood Q > 0 (into the cove), ebb Q < 0; "all" = every hour.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc, hourly ocean_his
       SNAPSHOTS (u, salt, temp, z_rho, z_w); box indices as in
       20261103_pc_eulerian_u0u1u2.py.

LAYOUT sections are COLUMNS west to east (pc_cp | pc_lj | pc_lp); cross-section
panels look into the cove (west) with north at the right, each as wide as the
section really is.

OUTPUT LO_output/DM_outs/20261103_pc_section_strat/
    section_N2.png        cross-sections of mean N^2 with mean sigma0 contours; rows flood, ebb, all
    width_strat.png       whole-width profiles, rows N^2 | sigma0, lines flood / ebb / all
    [col]_strat.png       two-face column, depth below the surface; col = north, center, south
    [col]_strat_bed.png   the same column, height above the bed

run 20261103_pc_section_strat.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cmocean
import gsw

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
G, RHO0 = 9.81, 1025.0                    # gravity [m/s2], reference density [kg/m3]
LON, LAT = -122.69, 48.23                 # one position for the cove (for SA_from_SP)
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
out_dir = Ldir['LOo'] / 'DM_outs' / '20261103_pc_section_strat'
out_dir.mkdir(parents=True, exist_ok=True)


def isum(x):
    """Sum over elements (k, p) of a section."""
    return x.sum(axis=(-2, -1))


def bin_mean(X, dy, lo, hi, P):
    """Split each element (a cell or a layer between cell centres) among the bins by
    overlap, over the faces P only. lo < hi is its extent in the bin coordinate.
    Returns, per hour and bin, A = sum ov dy and XA = sum X ov dy."""
    X, lo, hi = X[:, :, P], lo[:, :, P], hi[:, :, P]
    A = np.zeros((len(X), NB))
    XA = np.zeros((len(X), NB))
    for b in range(NB):
        ova = np.clip(np.minimum(hi, EDGES[b + 1]) - np.maximum(lo, EDGES[b]), 0, None) * dy[None, None, P]   # (t, k, p) m2
        A[:, b] = isum(ova)
        XA[:, b] = isum(X * ova)
    return A, XA


def phase_mean(A, XA, MASK, amin):
    """Area-weighted mean per bin over the flood, ebb and all hours. Bins with mean
    area < amin are dropped; full = mean area / max area (for the hatching)."""
    out = {}
    for ph in PHASES:
        m = MASK[ph]
        Am = A[m].mean(axis=0)
        keep = Am >= amin
        out[ph] = dict(X=XA[m].sum(axis=0) / np.where(keep, A[m].sum(axis=0), np.nan),
                       full=np.where(keep, Am / np.maximum(A.max(axis=0), 1e-12), np.nan), frac=m.mean())
    return out


ds = xr.open_dataset(in_fn)
R = {}
for SN in FACES:
    XI_U, ETA = FACES[SN]['XI_U'], FACES[SN]['ETA']
    IRM, IRP = XI_U, XI_U + 1             # rho cells west (minus) and east (plus) of the face
    at_face = lambda vn: ds[vn].isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().astype(float).mean(axis=-1)

    # ------------------------------------------------------------------ load snapshots
    u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)   # (t, k, p) + into the cove
    salt, temp = at_face('salt'), at_face('temp')                          # (t, k, p) at the face
    zw, zr = at_face('z_w'), at_face('z_rho')                              # (t, w, p), (t, k, p)
    pn = ds.pn.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()
    dy = (1 / pn).mean(axis=-1)               # (p,) face width
    dz = np.diff(zw, axis=1)                  # (t, k, p) cell thickness
    Q = isum(u * dz * dy[None, None, :])      # (t,) section-total transport: sets flood / ebb
    MASK = {'flood': Q > 0, 'ebb': Q < 0, 'all': np.ones(len(Q), bool)}

    # ------------------------------------------------------------------ potential density (TEOS-10)
    SA = gsw.SA_from_SP(salt, -zr, LON, LAT)
    CT = gsw.CT_from_pt(SA, temp)
    sig = gsw.sigma0(SA, CT)                  # (t, k, p) [kg/m3 - 1000]

    # ------------------------------------------------------------------ N^2 between adjacent cell centres
    dzr = np.diff(zr, axis=1)                 # (t, k-1, p)
    N2 = -(G / RHO0) * np.diff(sig, axis=1) / dzr                         # (t, k-1, p) [1/s2]
    print('\n%s: mean N2 %.2e 1/s2, %.1f%% of layer-hours unstable (N2 < 0)' % (SN, N2.mean(), 100 * (N2 < 0).mean()))

    # ------------------------------------------------------------------ extents of cells and layers in both coordinates
    zeta, zbed = zw[:, -1:, :], zw[:, :1, :]
    EXT = {'cell': dict(d=(zeta - zw[:, 1:, :], zeta - zw[:, :-1, :]), h=(zw[:, :-1, :] - zbed, zw[:, 1:, :] - zbed)),
           'layer': dict(d=(zeta - zr[:, 1:, :], zeta - zr[:, :-1, :]), h=(zr[:, :-1, :] - zbed, zr[:, 1:, :] - zbed))}

    # ------------------------------------------------------------------ cross-sections: per element, per phase
    xs = {}
    for ph in PHASES:
        m = MASK[ph]
        xs[ph] = dict(N2=(N2[m] * dzr[m]).sum(axis=0) / dzr[m].sum(axis=0),      # (k-1, p) layer-thickness weighted
                      sig=(sig[m] * dz[m]).sum(axis=0) / dz[m].sum(axis=0),      # (k, p) cell-thickness weighted
                      zr=zr[m].mean(axis=0), zbed=zw[m, 0, :].mean(axis=0), zsurf=zw[m, -1, :].mean(axis=0))

    # ------------------------------------------------------------------ profiles: whole width, then each two-face column
    xc = np.cumsum(dy) - dy / 2               # (p,) face centres, m from the north end
    PROF = {'width': (np.arange(len(dy)), 'd')}
    for col, frac in COLS.items():
        P = np.sort(np.argsort(np.abs(xc - frac * dy.sum()))[:2])
        PROF[col] = (P, 'd')
        PROF[col + '_bed'] = (P, 'h')
    pr = {}
    for key, (P, c) in PROF.items():
        pr[key] = dict(P=P,
                       N2=phase_mean(*bin_mean(N2, dy, *EXT['layer'][c], P), MASK, 0.25 * dy.mean()),   # layers stop at cell centres
                       sig=phase_mean(*bin_mean(sig, dy, *EXT['cell'][c], P), MASK, dy.mean()))
        if c == 'h':
            pr[key]['zb'] = (zr[:, 0, P] - zw[:, 0, P]).mean()                 # bottom cell centre above the bed
    R[SN] = dict(xs=xs, prof=pr, dy=dy)
ds.close()


# ------------------------------------------------------------------ plot: cross-sections of N^2 with sigma0 contours
WR = [R[SN]['dy'].sum() for SN in FACES]  # column widths = real section widths, west to east
N2U = 1e-3                                # plotting unit of N^2 [1/s2]


def draw_xsec(ax, d, dy, vmax):
    """Each face a column of N^2 layers between its mean cell centres, grey where no N^2 is
    resolved, mean sigma0 contoured at the cell centres; north at right."""
    xe = np.concatenate([[0], np.cumsum(dy)]) / 1000
    for p in range(len(dy)):
        cs = ax.pcolormesh(xe[p:p + 2], d['zr'][:, p], d['N2'][:, p:p + 1] / N2U, cmap=cmocean.cm.dense, vmin=0, vmax=vmax, shading='flat')
        ax.fill_between(xe[p:p + 2], d['zbed'][p], d['zr'][0, p], color='0.8', lw=0)
        ax.fill_between(xe[p:p + 2], d['zr'][-1, p], d['zsurf'][p], color='0.8', lw=0)
    X = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), d['sig'].shape)
    lev = np.arange(np.floor(np.nanmin(d['sig']) * 4) / 4, np.nanmax(d['sig']) + 0.25, 0.25)
    cc = ax.contour(X, d['zr'], d['sig'], levels=lev, colors='w', linewidths=0.8)
    ax.clabel(cc, cc.levels[::2], fontsize=6, fmt='%.1f')
    ax.set_xlim(xe[-1], xe[0])           # reversed: north at right = looking west, into the cove
    ax.grid(True, alpha=0.3)
    return cs


vmax = np.nanpercentile(np.concatenate([R[SN]['xs'][ph]['N2'].ravel() for SN in FACES for ph in PHASES]), 99) / N2U
fig, axs = plt.subplots(3, 3, figsize=(14, 9), sharey=True, layout='constrained', gridspec_kw={'width_ratios': WR})
for i, ph in enumerate(PHASES):
    for j, SN in enumerate(FACES):
        cs = draw_xsec(axs[i, j], R[SN]['xs'][ph], R[SN]['dy'], vmax)
        axs[i, j].set_title('%s: %s (%.0f%% of hours)' % (NAME[SN], ph, 100 * R[SN]['prof']['width']['N2'][ph]['frac']), fontsize=9)
        if i == 2:
            axs[i, j].set_xlabel('distance from north shore [km]')
    axs[i, 0].set_ylabel('z [m]')
fig.colorbar(cs, ax=axs, location='bottom', shrink=0.5, aspect=40,
             label=r'mean N$^2$ [10$^{-3}$ s$^{-2}$]; white contours: mean $\sigma_0$ [kg m$^{-3}$], every 0.25; grey = no resolved N$^2$')
fig.suptitle('Stratification on the sections, west to east, each looking into the cove (west; north at right); hourly snapshots, %s, %s to %s'
             % (GTAGEX, DS0, DS1), fontsize=10)
fig.savefig(out_dir / 'section_N2.png', dpi=200, transparent=True)
plt.close(fig)


# ------------------------------------------------------------------ plot: profiles, rows N^2 | sigma0
def hatch_partial(ax, full):
    """Hatch the bins that are not full every hour (full < 0.99)."""
    for b in np.flatnonzero(full < 0.99):
        ax.axhspan(EDGES[b], EDGES[b + 1], facecolor='none', edgecolor='0.6', hatch='///', lw=0, zorder=0)


def plot_prof(key, fname, where, bed=False):
    fig, axs = plt.subplots(2, 3, figsize=(13, 10), sharey=True, layout='constrained')
    for j, SN in enumerate(FACES):
        pr = R[SN]['prof'][key + '_bed' if bed else key]
        sub = '' if key == 'width' else ', faces p = %d-%d' % tuple(pr['P'])
        for i, (fld, scale, lab) in enumerate([('N2', 1 / N2U, r'mean N$^2$ [10$^{-3}$ s$^{-2}$]'), ('sig', 1, r'mean $\sigma_0$ [kg m$^{-3}$]')]):
            ax = axs[i, j]
            hatch_partial(ax, pr[fld]['all']['full'])
            if bed and fld == 'N2':
                ax.axhspan(0, pr['zb'], color='0.88', zorder=0)
                ax.text(0.02, pr['zb'] / 2, 'no resolved N$^2$ below the bottom cell centre', transform=ax.get_yaxis_transform(),
                        fontsize=6, va='center', color='0.3')
            for ph in PHASES:
                ax.plot(scale * pr[fld][ph]['X'], DMID, color=COLOR[ph], lw=2, label='%s (%.0f%% of hours)' % (ph, 100 * pr[fld][ph]['frac']))
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(lab, fontsize=8)
            ax.set_title(NAME[SN] + sub + (': N$^2$' if fld == 'N2' else r': $\sigma_0$'), fontsize=9)
        axs[0, j].set_xlim(0, None)
    axs[0, 0].legend(fontsize=7, loc='lower right' if not bed else 'upper right')
    if not bed:
        axs[0, 0].invert_yaxis()
    for ax in axs[:, 0]:
        ax.set_ylabel('height above the bed [m]' if bed else 'depth below the surface [m]')
    fig.suptitle('Stratification, %s, area-weighted in 1 m bins (%s), hourly snapshots, %s, %s to %s'
                 % (where, 'height above the bed' if bed else 'depth below the moving surface', GTAGEX, DS0, DS1), fontsize=10)
    fig.savefig(out_dir / fname, dpi=200, transparent=True)
    plt.close(fig)


plot_prof('width', 'width_strat.png', 'whole width')
for col in COLS:
    plot_prof(col, col + '_strat.png', COLNAME[col] + ' (two faces)')
    plot_prof(col, col + '_strat_bed.png', COLNAME[col] + ' (two faces)', bed=True)
print('\nsaved to ' + str(out_dir))
