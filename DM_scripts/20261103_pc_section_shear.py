"""
Vertical shear S = du/dz at the three Penn Cove sections, pc_cp (off
Coupeville), pc_lj (middle) and pc_lp (mouth), of wb1_t0_xn11abbur00, plotted as
profiles in the same way as 20261103_pc_section_profiles.py: the whole width,
and the centre / north 1/3 / south 1/3 columns, for flood, ebb and all hours.

SHEAR, hourly, within each face (never across faces). Between the centres of
cells k and k+1 of a face,
    S = (u[k+1] - u[k]) / (z_rho[k+1] - z_rho[k])          [1/s], + = u more into the cove upward
which is the exact slope of u if u varies linearly between cell centres. Each
value belongs to the LAYER between the two centres, of thickness dzr =
z_rho[k+1] - z_rho[k] and area dy * dzr. There is no resolved shear below the
bottom cell centre (the bed enters only as a drag-law stress) or above the top
cell centre (the wind stress); those layers are left out, and the bed one is
drawn as a grey band.

BINS. As for velocity: each layer is split among the 1 m bins it overlaps,
by overlap. Shear is a property of the water, not a transport, so a bin and a
phase are AREA-weighted:
    S_b   = sum_t sum_c S ov dy  / sum_t sum_c ov dy         mean signed shear
    rms_b = sqrt( sum_t sum_c S^2 ov dy / sum_t sum_c ov dy ) rms shear
For piecewise-linear u the area-weighted mean of S over a bin is exactly the
mean du/dz across that bin.

FLOOD / EBB from the section-total transport Q(t) = int u dA (hourly):
flood Q > 0 (into the cove), ebb Q < 0; "all" = every hour.

ROWS of each figure
    top    mean signed S: flood, ebb, all hours. All-hours signed S is the shear
           of the residual (it mostly cancels between flood and ebb).
    bottom rms S: flood, ebb, all hours. Does not cancel; sets Ri = N^2 / S^2.

COORDINATES (as for velocity): depth below the moving surface for the whole
width and each column; height above the bed for each column. Bins that are not
full every hour (mean area / max area < 0.99) are hatched.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       hourly ocean_his SNAPSHOTS (on the hour); u at the face, z_rho and z_w
       as the mean of the two rho cells either side of the face.

SIGN  u is positive UP-ESTUARY = into Penn Cove (west). On the signed-shear
axes west (+) is at the LEFT, as on the velocity profiles; rms has no sign.

CROSS-SECTIONS. The same shear layers without binning: per element (layer
k+1/2 of face p), over the hours of a phase, weighted by layer thickness dzr,
    S = sum_t S dzr / sum_t dzr,   rms = sqrt( sum_t S^2 dzr / sum_t dzr )
drawn between the phase-mean cell-centre depths. The strips bed -> bottom cell
centre and top cell centre -> surface have no resolved shear and are grey.
Sections are COLUMNS west to east (pc_cp | pc_lj | pc_lp), each as wide as the
section really is, looking into the cove (west) with north at the right.

OUTPUT LO_output/DM_outs/20261103_pc_section_shear/
    section_shear_signed.png cross-sections of mean signed S; rows flood, ebb, all
    section_shear_rms.png    cross-sections of rms S; rows flood, ebb, all
    width_shear.png          whole width, depth below the surface
    [col]_shear.png          two-face column, depth below the surface; col = north, center, south
    [col]_shear_bed.png      the same column, height above the bed

run 20261103_pc_section_shear.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cmocean

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
out_dir = Ldir['LOo'] / 'DM_outs' / '20261103_pc_section_shear'
out_dir.mkdir(parents=True, exist_ok=True)


def isum(x):
    """Sum over elements (k, p) of a section."""
    return x.sum(axis=(-2, -1))


def bin_layers(S, dy, lo, hi, P):
    """Split each shear layer among the bins by overlap, over the faces P only.
    lo < hi is the layer's extent in the bin coordinate. Returns, per hour and bin,
    A = sum ov dy, SA = sum S ov dy, S2A = sum S^2 ov dy."""
    S, lo, hi = S[:, :, P], lo[:, :, P], hi[:, :, P]
    A = np.zeros((len(S), NB))
    SA = np.zeros((len(S), NB))
    S2A = np.zeros((len(S), NB))
    for b in range(NB):
        ova = np.clip(np.minimum(hi, EDGES[b + 1]) - np.maximum(lo, EDGES[b]), 0, None) * dy[None, None, P]   # (t, k, p) m2
        A[:, b] = isum(ova)
        SA[:, b] = isum(S * ova)
        S2A[:, b] = isum(S**2 * ova)
    return A, SA, S2A


def phase_profiles(A, SA, S2A, Q, amin):
    """Area-weighted mean and rms shear per bin over the flood, ebb and all hours.
    Bins with mean area < amin are dropped; full = mean area / max area."""
    MASK = {'flood': Q > 0, 'ebb': Q < 0, 'all': np.ones(len(Q), bool)}
    out = {}
    for ph in PHASES:
        m = MASK[ph]
        Am = A[m].mean(axis=0)
        keep = Am >= amin
        At = np.where(keep, A[m].sum(axis=0), np.nan)
        out[ph] = dict(S=SA[m].sum(axis=0) / At, rms=np.sqrt(S2A[m].sum(axis=0) / At),
                       full=np.where(keep, Am / np.maximum(A.max(axis=0), 1e-12), np.nan), frac=m.mean())
    return out


ds = xr.open_dataset(in_fn)
R = {}
for SN in FACES:
    XI_U, ETA = FACES[SN]['XI_U'], FACES[SN]['ETA']
    IRM, IRP = XI_U, XI_U + 1             # rho cells west (minus) and east (plus) of the face

    # ------------------------------------------------------------------ load snapshots
    u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)                 # (t, k, p)
    zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().mean(axis=-1)          # (t, w, p) at the face
    zr = ds.z_rho.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().mean(axis=-1)        # (t, k, p) cell centres
    pn = ds.pn.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()                         # (p, 2)
    dy = (1 / pn).mean(axis=-1)               # (p,) face width
    dz = np.diff(zw, axis=1)                  # (t, k, p) cell thickness
    Q = isum(u * dz * dy[None, None, :])      # (t,) section-total transport: sets flood / ebb
    h = -zw[0, 0, :]                          # (p,) depth at the face

    # ------------------------------------------------------------------ shear between adjacent cell centres
    dzr = np.diff(zr, axis=1)                 # (t, k-1, p) distance between centres k and k+1
    S = np.diff(u, axis=1) / dzr              # (t, k-1, p) [1/s], + = more into the cove upward

    # ------------------------------------------------------------------ where each shear layer sits
    zeta = zw[:, -1:, :]                      # free surface
    zbed = zw[:, :1, :]                       # bed
    d_lo, d_hi = zeta - zr[:, 1:, :], zeta - zr[:, :-1, :]      # depth below the surface of the layer's top, bottom
    h_lo, h_hi = zr[:, :-1, :] - zbed, zr[:, 1:, :] - zbed      # height above the bed of the layer's bottom, top
    assert d_hi.max() < EDGES[-1] and h_hi.max() < EDGES[-1]
    print('\n' + SN + ': rms S over the whole section %.4f 1/s, max |S| %.3f 1/s' % (np.sqrt(np.mean(S**2)), np.abs(S).max()))

    # ------------------------------------------------------------------ bin: whole width, then each two-face column
    AMIN = 0.25 * dy.mean()                   # bins covered by less than dy x 0.25 m on average are dropped
    ALL = np.arange(u.shape[2])
    R[SN] = {'width': phase_profiles(*bin_layers(S, dy, d_lo, d_hi, ALL), Q, AMIN)}
    xc = np.cumsum(dy) - dy / 2               # (p,) face centres, m from the north end
    for col, frac in COLS.items():
        P = np.sort(np.argsort(np.abs(xc - frac * dy.sum()))[:2])
        R[SN][col + '_P'] = P
        R[SN][col] = phase_profiles(*bin_layers(S, dy, d_lo, d_hi, P), Q, AMIN)            # depth below the surface
        R[SN][col + '_bed'] = phase_profiles(*bin_layers(S, dy, h_lo, h_hi, P), Q, AMIN)   # height above the bed
        R[SN][col + '_zb'] = (zr[:, 0, P] - zw[:, 0, P]).mean()                           # bottom cell centre above the bed
        print('  %-16s faces p = %d, %d (h %.1f, %.1f m), bottom cell centre %.2f m above the bed'
              % (COLNAME[col], P[0], P[1], h[P[0]], h[P[1]], R[SN][col + '_zb']))

    # ------------------------------------------------------------------ cross-section composites, per element
    MASK = {'flood': Q > 0, 'ebb': Q < 0, 'all': np.ones(len(Q), bool)}
    R[SN]['xsec'] = {}
    for ph in PHASES:
        m = MASK[ph]
        W = dzr[m]                            # (t, k-1, p) layer thickness weight
        R[SN]['xsec'][ph] = dict(S=(S[m] * W).sum(axis=0) / W.sum(axis=0),
                                 rms=np.sqrt((S[m]**2 * W).sum(axis=0) / W.sum(axis=0)),
                                 zr=zr[m].mean(axis=0),                   # (k, p) phase-mean cell centres = layer edges
                                 zbed=zw[m, 0, :].mean(axis=0), zsurf=zw[m, -1, :].mean(axis=0))
    R[SN]['dy'] = dy
ds.close()


# ------------------------------------------------------------------ plot
def hatch_partial(ax, full):
    """Hatch the bins that are not full every hour (full < 0.99), with the value written in."""
    for b in np.flatnonzero(full < 0.99):
        ax.axhspan(EDGES[b], EDGES[b + 1], facecolor='none', edgecolor='0.6', hatch='///', lw=0, zorder=0)
        ax.text(0.98, DMID[b], 'full %.0f%%' % (100 * full[b]), transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=6, color='0.35')


def plot(key, fname, where, bed=False):
    fig, axs = plt.subplots(2, 3, figsize=(13, 10), sharey=True, layout='constrained')
    for j, SN in enumerate(FACES):
        prof = R[SN][key + '_bed' if bed else key]
        sub = '' if key == 'width' else ', faces p = %d-%d' % tuple(R[SN][key + '_P'])
        for i, (fld, lab) in enumerate([('S', 'mean signed shear'), ('rms', 'rms shear')]):
            ax = axs[i, j]
            hatch_partial(ax, prof['all']['full'])
            if bed:
                zb = R[SN][key + '_zb']
                ax.axhspan(0, zb, color='0.88', zorder=0)
                ax.text(0.02, zb / 2, 'no resolved shear below the\nbottom cell centre (drag law)', transform=ax.get_yaxis_transform(),
                        fontsize=6, va='center', color='0.3')
            for ph in PHASES:
                ax.plot(1000 * prof[ph][fld], DMID, color=COLOR[ph], lw=2,
                        label='%s (%.0f%% of hours)' % (ph, 100 * prof[ph]['frac']))
            ax.axvline(0, color='k', lw=0.6)
            ax.grid(True, alpha=0.3)
            ax.set_title(NAME[SN] + sub + ': ' + lab, fontsize=9)
            if fld == 'S':
                ax.invert_xaxis()                 # west (+) at left, as on the velocity profiles
                ax.set_xlabel(r'$\leftarrow$ u more into the cove upward (+)    S [10$^{-3}$ s$^{-1}$]    (-) $\rightarrow$', fontsize=8)
            else:
                ax.set_xlim(0, None)
                ax.set_xlabel(r'rms S [10$^{-3}$ s$^{-1}$]', fontsize=8)
        axs[0, j].legend(fontsize=7, loc='lower left' if not bed else 'upper left')
    if not bed:
        axs[0, 0].invert_yaxis()                  # depth increases downward (shared y)
    for i in range(2):
        axs[i, 0].set_ylabel('height above the bed [m]' if bed else 'depth below the surface [m]')
    fig.suptitle('Vertical shear S = du/dz, %s, area-weighted in 1 m bins (%s), hourly snapshots, %s, %s to %s'
                 % (where, 'height above the bed' if bed else 'depth below the moving surface', GTAGEX, DS0, DS1), fontsize=10)
    fig.savefig(out_dir / fname, dpi=200, transparent=True)
    plt.close(fig)


def draw_xsec(ax, d, dy, fld, cmap, vmin, vmax, dlev):
    """One section: each face a column of shear layers between its cell centres; grey where
    there is no resolved shear (below the bottom centre, above the top centre)."""
    xe = np.concatenate([[0], np.cumsum(dy)]) / 1000                  # km, face edges from the north end
    for p in range(len(dy)):
        cs = ax.pcolormesh(xe[p:p + 2], d['zr'][:, p], 1000 * d[fld][:, p:p + 1], cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax.fill_between(xe[p:p + 2], d['zbed'][p], d['zr'][0, p], color='0.8', lw=0)
        ax.fill_between(xe[p:p + 2], d['zr'][-1, p], d['zsurf'][p], color='0.8', lw=0)
    X = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), d[fld].shape)
    Z = 0.5 * (d['zr'][1:] + d['zr'][:-1])                            # layer mid-depths
    lev = np.arange(np.floor(vmin / dlev) * dlev, vmax + dlev, dlev)
    ax.contour(X, Z, 1000 * d[fld], levels=lev[lev != 0], colors='0.3', linewidths=0.5)
    if vmin < 0:
        ax.contour(X, Z, 1000 * d[fld], levels=[0], colors='k', linewidths=1.2)
    ax.set_xlim(xe[-1], xe[0])           # reversed: north at right = looking west, into the cove
    ax.grid(True, alpha=0.3)
    return cs


WR = [R[SN]['dy'].sum() for SN in FACES]  # column widths = real section widths
for fld, cmap, sym, dlev, lab, fname in [
        ('S', cmocean.cm.balance, True, 5, r'mean signed S [10$^{-3}$ s$^{-1}$], + = u more into the cove upward (contours every 5)', 'section_shear_signed.png'),
        ('rms', cmocean.cm.amp, False, 10, r'rms S [10$^{-3}$ s$^{-1}$] (contours every 10)', 'section_shear_rms.png')]:
    allv = np.concatenate([1000 * R[SN]['xsec'][ph][fld].ravel() for SN in FACES for ph in PHASES])
    vmax = np.ceil(np.nanpercentile(np.abs(allv), 99))
    vmin = -vmax if sym else 0
    fig, axs = plt.subplots(3, 3, figsize=(14, 9), sharey=True, layout='constrained', gridspec_kw={'width_ratios': WR})
    for i, ph in enumerate(PHASES):
        for j, SN in enumerate(FACES):
            cs = draw_xsec(axs[i, j], R[SN]['xsec'][ph], R[SN]['dy'], fld, cmap, vmin, vmax, dlev)
            axs[i, j].set_title('%s: %s (%.0f%% of hours)' % (NAME[SN], ph, 100 * R[SN]['width'][ph]['frac']), fontsize=9)
            if i == 2:
                axs[i, j].set_xlabel('distance from north shore [km]')
        axs[i, 0].set_ylabel('z [m]')
    fig.colorbar(cs, ax=axs, location='bottom', shrink=0.5, aspect=40, label=lab)
    fig.suptitle('Vertical shear S = du/dz on the sections, west to east, each looking into the cove (west; north at right); grey = no resolved shear; hourly snapshots, %s, %s to %s'
                 % (GTAGEX, DS0, DS1), fontsize=10)
    fig.savefig(out_dir / fname, dpi=200, transparent=True)
    plt.close(fig)

plot('width', 'width_shear.png', 'whole width')
for col in COLS:
    plot(col, col + '_shear.png', COLNAME[col] + ' (two faces)')
    plot(col, col + '_shear_bed.png', COLNAME[col] + ' (two faces)', bed=True)
print('\nsaved to ' + str(out_dir))
