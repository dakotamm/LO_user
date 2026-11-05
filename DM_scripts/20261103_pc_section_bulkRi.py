"""
Bulk Richardson number of the water column at each face of the three Penn Cove
sections, pc_cp (off Coupeville), pc_lj (middle) and pc_lp (mouth), of
wb1_t0_xn11abbur00: can the current mix the whole column against its density
difference? (The estuarine Richardson number of Fischer 1972 does not apply: it
needs a local river discharge, and Penn Cove has essentially none - its
stratification is imported from Saratoga Passage.)

PER FACE, HOURLY
    Ri_B = g (drho / rho0) H / U^2
    drho = sigma0(bottom cell) - sigma0(top cell)    TEOS-10, salt/temp averaged onto the face
    H    = h + zeta                                  water depth
    U^2  = Uu^2 + Uv^2                               depth-mean u (normal to the section) and v
                                                     (along it, the mean of the 4 v points around
                                                     the face), dz-weighted, from the 3D fields

SUMMARIES per face, for flood / ebb / all hours
    Ri_B_bar = <g drho H / rho0> / <U^2>             ratio of averages (Ri_B itself is not averaged:
                                                     it runs to infinity at slack water)
    share of hours with Ri_B < 1 and < 1/4 (printed; it is ~0 everywhere)
    1st percentile of hourly Ri_B and Ri_D: the most mixable 1% of hours (plotted)
    Ri_D_bar = <g drho H / rho0> / <dU^2>            the same with the TOP-MINUS-BOTTOM velocity
                                                     difference, dU^2 = (u_top - u_bot)^2 + (v_top - v_bot)^2,
                                                     instead of the depth mean: the currents are
                                                     surface-intensified / two-layer, so the depth mean
                                                     understates the velocity that works against drho
    Ri_T = g <drho> <H> / (rho0 U_M2^2)              one value per face: the 2-year mean stratification
                                                     against the M2 tidal current, U_M2 = semi-major axis
                                                     of a utide fit to the depth-mean (u, v)
Below ~1 the current can mix the column; well below 1/4 it should be well mixed.
The bulk critical value is less sharp than the gradient 1/4, so both are marked.

FLOOD / EBB from the section-total transport Q(t) = int u dA (hourly):
flood Q > 0 (into the cove), ebb Q < 0; "all" = every hour.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc, hourly ocean_his
       SNAPSHOTS (u, v, salt, temp, z_rho, z_w); box indices as in
       20261103_pc_eulerian_u0u1u2.py; the 4 v points as in 20261103_pc_section_Ri.py.

LAYOUT sections are COLUMNS west to east (pc_cp | pc_lj | pc_lp), each as wide as
the section really is; x = distance from the north shore with north at the right
(looking into the cove, west), as on the cross-sections.

OUTPUT LO_output/DM_outs/20261103_pc_section_bulkRi/
    bulkRi.png    rows: Ri_B_bar and Ri_D_bar | their 1st percentiles | Ri_T, one column per section

CYCLE (-cycle springneap | tropeq | strat | wind | windhs | windls | cross | stratcalm; see pc_tidal_cycles.py): every phase is split into its high and low
third of the cycle (pc_tidal_cycles.py: terciles of the semidiurnal / diurnal
envelope of the mouth transport), e.g. 'flood, spring' and 'flood, neap'; high
half solid, low half dashed; figures get _springneap / _tropeq in their names
and go in the same folder. Default (none): the figures above, unchanged.

run 20261103_pc_section_bulkRi.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')        # one BLAS thread: utide is ~3x faster this way
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import gsw
import utide

from lo_tools import Lfun
import argparse
from pc_tidal_cycles import cycle_masks, expand_phases

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove; v is not flipped (+ = north)
# box indices of each section's u-faces (global i - 37, global j - 209), north to south;
# listed west to east (inner cove to mouth), which is the column order of the figure
FACES = {
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
G, RHO0 = 9.81, 1025.0                    # gravity [m/s2], reference density [kg/m3]
LON, LAT = -122.69, 48.23                 # one position for the cove (SA_from_SP, utide nodal corrections)
CONS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'MS4', 'M6', 'S1']   # as in 20261103_pc_section_tides.py
KW = dict(method='ols', conf_int='linear', constit=CONS, nodal=True, trend=False, verbose=False)
PHASES = ['flood', 'ebb', 'all']
COLOR = {'flood': '#D55E00', 'ebb': '#0072B2', 'all': 'k'}
parser = argparse.ArgumentParser()
parser.add_argument('-cycle', default='none', choices=['none', 'springneap', 'tropeq', 'strat', 'wind', 'windhs', 'windls', 'cross', 'stratcalm'])   # split each phase by the spring/neap or tropic/equatorial cycle
CYCLE = parser.parse_args().cycle
TAG = '' if CYCLE == 'none' else '_' + CYCLE  # added to every figure name
BASE_PHASES, BASE_COLOR = PHASES, COLOR
LS = {ph: '-' for ph in PHASES}           # line style per phase: spring / tropic solid, neap / equatorial dashed


Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261103_pc_section_bulkRi'
out_dir.mkdir(parents=True, exist_ok=True)


ds = xr.open_dataset(in_fn)
CM = cycle_masks(ds.ocean_time.values) if CYCLE != 'none' else None
T = ds.ocean_time.values
R = {}
for SN in FACES:
    XI_U, ETA = FACES[SN]['XI_U'], FACES[SN]['ETA']
    IRM, IRP = XI_U, XI_U + 1             # rho cells west (minus) and east (plus) of the face
    at_face = lambda vn: ds[vn].isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().astype(float).mean(axis=-1)

    # ------------------------------------------------------------------ load snapshots
    u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)              # (t, k, p) + into the cove
    v4 = [ds.v.isel(eta_v=ETA + dj, xi_v=[IRM, IRP]).to_numpy().astype(float) for dj in (-1, 0)]
    v = np.nan_to_num(np.concatenate(v4, axis=-1)).mean(axis=-1)                     # (t, k, p) along-section, + north; land = 0
    salt, temp = at_face('salt'), at_face('temp')
    zw, zr = at_face('z_w'), at_face('z_rho')
    pn = ds.pn.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()
    dy = (1 / pn).mean(axis=-1)               # (p,) face width
    dz = np.diff(zw, axis=1)                  # (t, k, p)
    Q = (u * dz * dy[None, None, :]).sum(axis=(1, 2))                               # (t,) section transport: sets flood / ebb
    MASK = {'flood': Q > 0, 'ebb': Q < 0, 'all': np.ones(len(Q), bool)}
    if CYCLE != 'none':
        PHASES, MASK, COLOR, LS = expand_phases(BASE_PHASES, MASK, BASE_COLOR, CYCLE, CM)

    # ------------------------------------------------------------------ the column: density difference, depth, depth-mean velocity
    SA = gsw.SA_from_SP(salt, -zr, LON, LAT)
    sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, temp))                                   # (t, k, p)
    drho = sig[:, 0, :] - sig[:, -1, :]       # (t, p) bottom cell minus top cell [kg/m3]
    H = zw[:, -1, :] - zw[:, 0, :]            # (t, p) water depth
    Uu = (u * dz).sum(axis=1) / H             # (t, p) depth-mean u
    Uv = (v * dz).sum(axis=1) / H             # (t, p) depth-mean v
    U2 = Uu**2 + Uv**2
    B = G * drho * H / RHO0                   # (t, p) g' H [m2/s2], the numerator
    RiB = B / np.maximum(U2, 1e-12)           # (t, p) hourly bulk Richardson number
    dU2 = (u[:, -1, :] - u[:, 0, :])**2 + (v[:, -1, :] - v[:, 0, :])**2              # (t, p) top minus bottom cell
    RiD = B / np.maximum(dU2, 1e-12)

    # ------------------------------------------------------------------ summaries per face, per phase
    st = {}
    for ph in PHASES:
        m = MASK[ph]
        st[ph] = dict(Ribar=B[m].mean(axis=0) / U2[m].mean(axis=0),
                      f1=(RiB[m] < 1).mean(axis=0), f025=(RiB[m] < 0.25).mean(axis=0), frac=m.mean(),
                      RiDbar=B[m].mean(axis=0) / dU2[m].mean(axis=0), fD1=(RiD[m] < 1).mean(axis=0),
                      p01=np.percentile(RiB[m], 1, axis=0), pD01=np.percentile(RiD[m], 1, axis=0))   # the most mixable 1% of hours
    Wd = H.mean(axis=0) * dy                  # (p,) face area, for a section-wide value
    print('\n%s: section-wide <g drho H/rho0>/<U^2> = %.2f; Ri_B < 1 in %.0f%% of face-hours, < 1/4 in %.0f%%'
          % (SN, (B.mean(axis=0) * Wd).sum() / (U2.mean(axis=0) * Wd).sum(), 100 * (RiB < 1).mean(), 100 * (RiB < 0.25).mean()))
    print('  with top-minus-bottom velocity: <g drho H/rho0>/<dU^2> = %.1f; Ri < 1 in %.1f%% of face-hours, < 1/4 in %.1f%%'
          % ((B.mean(axis=0) * Wd).sum() / (dU2.mean(axis=0) * Wd).sum(), 100 * (RiD < 1).mean(), 100 * (RiD < 0.25).mean()))

    # ------------------------------------------------------------------ Ri_T: mean stratification against the M2 tidal current
    UM2 = np.full(len(dy), np.nan)
    for p in range(len(dy)):
        c = utide.solve(T, Uu[:, p], Uv[:, p], lat=LAT, **KW)
        UM2[p] = c.Lsmaj[list(c.name).index('M2')]                                     # M2 semi-major axis [m/s]
    RiT = G * drho.mean(axis=0) * H.mean(axis=0) / (RHO0 * UM2**2)
    print('  U_M2 (depth-mean) %.1f-%.1f cm/s; Ri_T %.1f-%.1f' % (100 * UM2.min(), 100 * UM2.max(), RiT.min(), RiT.max()))

    R[SN] = dict(st=st, RiT=RiT, UM2=UM2, dy=dy, h=H.mean(axis=0), drho=drho.mean(axis=0))
ds.close()


# ------------------------------------------------------------------ plot
WR = [R[SN]['dy'].sum() for SN in FACES]  # column widths = real section widths, west to east
fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharey='row', layout='constrained', gridspec_kw={'width_ratios': WR})
for j, SN in enumerate(FACES):
    dy, st = R[SN]['dy'], R[SN]['st']
    x = (np.cumsum(dy) - dy / 2) / 1000   # face centres, km from the north shore
    xe = np.concatenate([[0], np.cumsum(dy)]) / 1000
    for ph in PHASES:
        axs[0, j].plot(x, st[ph]['Ribar'], color=COLOR[ph], ls=LS[ph], lw=2, marker='o', ms=4, label='%s (%.0f%% of hours)' % (ph, 100 * st[ph]['frac']))
        axs[0, j].plot(x, st[ph]['RiDbar'], color=COLOR[ph], lw=1 if CYCLE != 'none' else 1.5, ls='-.' if CYCLE == 'none' else LS[ph], marker='s', ms=3)
        axs[1, j].plot(x, st[ph]['p01'], color=COLOR[ph], ls=LS[ph], lw=2, marker='o', ms=4)
        axs[1, j].plot(x, st[ph]['pD01'], color=COLOR[ph], lw=1 if CYCLE != 'none' else 1.5, ls='-.' if CYCLE == 'none' else LS[ph], marker='s', ms=3)
    axs[2, j].plot(x, R[SN]['RiT'], color='k', lw=2, marker='o', ms=4)
    for i in [0, 1, 2]:
        axs[i, j].set_yscale('log')
        for r in [0.25, 1]:
            axs[i, j].axhline(r, color='0.3', lw=1, ls=':')
    for ax in axs[:, j]:
        ax.set_xlim(xe[-1], xe[0])           # reversed: north at right = looking west, into the cove
        ax.grid(True, alpha=0.3)
    axs[0, j].set_title(NAME[SN] + ': ratio of averages', fontsize=9)
    axs[1, j].set_title(NAME[SN] + ': 1st percentile of hourly Ri', fontsize=9)
    axs[2, j].set_title(NAME[SN] + r': Ri$_T$ = g<$\Delta\rho$><H> / ($\rho_0$U$_{M2}^2$)', fontsize=9)
    axs[2, j].set_xlabel('distance from north shore [km]')
axs[0, 0].set_ylabel(r'<g$\Delta\rho$H/$\rho_0$> / <U$^2$> or <$\Delta$U$^2$>')
axs[1, 0].set_ylabel('1st percentile of hourly Ri')
axs[2, 0].set_ylabel(r'Ri$_T$ (M2)')
axs[0, 0].plot([], [], color='0.4', lw=2, marker='o', ms=4, label='with depth-mean U')
axs[0, 0].plot([], [], color='0.4', lw=1.5 if CYCLE == 'none' else 1, ls='-.' if CYCLE == 'none' else '-', marker='s', ms=3, label=r'with top-bottom $\Delta$U')
axs[0, 0].legend(fontsize=7, loc='lower left')
fig.suptitle('Bulk Richardson number g($\\Delta\\rho/\\rho_0$)H/U$^2$ at each face ($\\Delta\\rho$ bottom minus top cell; U depth-mean, or $\\Delta$U top minus bottom, u and v); dotted: 1 and 1/4;\n'
             'sections west to east, north at right; hourly snapshots, %s, %s to %s' % (GTAGEX, DS0, DS1), fontsize=10)
fig.savefig(out_dir / ('bulkRi%s.png' % TAG), dpi=200, transparent=True)
plt.close(fig)
print('\nsaved to ' + str(out_dir))
