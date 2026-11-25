"""
Penn Cove: the baseline lunar day's peak flood, peak ebb, and the Godin-filtered residual at those
two hours. The two figures of 20261116_pc_peak_flood_ebb.py (pc_lj_peak_sections.png and
cove_peak_maps_all_all.png), for one day instead of all of 2024.

STEP 1  load pc_lj from the pc_cove box (hourly ocean_his) for the baseline day +- WIN, as
        20261116 STEP 1. WIN covers the Godin filter (35 h each side) at both peak hours.
STEP 2  half-cycles of the pc_lj section-total Q as 20261116 STEP 2 (MIN_RUN h runs absorbed); the
        baseline day (20261113_pc_candidate_days/picks.csv, 2024-08-31 05:00) = the 4 half-cycles
        from its first flood (F E F E); its greater flood / ebb = the one with the larger |Q| at its
        peak, and the peak hour = that half-cycle's max |Q| hour.
STEP 3  flood = u at the peak flood hour, ebb = u at the peak ebb hour; residual = Godin(u dz) /
        Godin(dz), dz-weighted over the two peak hours together (as 20261116 STEP 3).
STEP 4  section figure, as 20261116 STEP 4 (looking out of the cove, north LEFT, symbols per u = 0 region).
STEP 5  cove maps, as 20261116 STEPS 14-15: surface 1/3 | bottom 1/3 | depth average rows, flood | ebb |
        residual columns, colour = along-cove u, arrows = the full vector.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       LO_output/DM_outs/20261113_pc_candidate_days/picks.csv
OUTPUT LO_output/DM_outs/20261118_pc_baseline_peak_flood_ebb/
       pc_lj_baseline_peak_sections.png    STEP 4
       cove_baseline_peak_maps.png         STEP 5

run 20261118_pc_baseline_peak_flood_ebb.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'     # the box file
WIN = pd.Timedelta(days=3)                # either side of the baseline day, for the Godin filter
SIGN = -1                                 # section-positive (east) is out of the cove
XI_U, ETA = 23, np.arange(14, 6, -1)      # pc_lj u faces in the box, north to south
MIN_RUN = 3                               # half-cycles shorter than this [h] are slack jitter
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
DSYM = 40                                 # symbol diameter at the largest region transport [pt]
DMIN = 8                                  # smallest symbol diameter [pt]

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
pick_fn = Ldir['LOo'] / 'DM_outs' / '20261113_pc_candidate_days' / 'picks.csv'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261118_pc_baseline_peak_flood_ebb'
Lfun.make_dir(out_dir)

# ------------------------------------------------------------------ STEP 1: load pc_lj around the baseline day
picks = pd.read_csv(pick_fn, parse_dates=['t0', 't1']).set_index('case')
bt0, bt1 = picks.loc['baseline', 't0'], picks.loc['baseline', 't1']
ds = xr.open_dataset(box_fn).sel(ocean_time=slice(bt0 - WIN, bt1 + WIN))
ot = pd.DatetimeIndex(ds.ocean_time.values)
u = SIGN * ds['u'].isel(xi_u=XI_U, eta_u=ETA).values.astype(float)                   # (t, k, p) + into the cove
zw = ds['z_w'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)       # (t, w, p)
dy = (1 / ds['pn'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values).mean(axis=-1)  # (p,)
dz = np.diff(zw, axis=1)
Q = (u * dz * dy).sum(axis=(1, 2))                                                   # (t,) + = flood
print('%s  %d hours  %s to %s' % (box_fn.name, len(ot), ot[0], ot[-1]))

# ------------------------------------------------------------------ STEP 2: the baseline day's peak flood / ebb hours
def runs(x):
    """[(start, stop), ...] of constant x (stop exclusive)"""
    k = np.flatnonzero(np.diff(x) != 0) + 1
    return list(zip(np.r_[0, k], np.r_[k, len(x)]))


sg = np.sign(Q)
for a, b in runs(sg):
    if b - a < MIN_RUN:
        sg[a:b] = 0
sg = pd.Series(sg).replace(0, np.nan).ffill().bfill().to_numpy()
HC = runs(sg)[1:-1]
hc_sign = np.array([sg[a] for a, b in HC])
hc_pk = np.array([a + np.argmax(np.abs(Q[a:b])) for a, b in HC])
ia = np.flatnonzero((ot[[a for a, b in HC]] == bt0) & (hc_sign == 1))
assert len(ia) == 1, 'baseline t0 is not a flood start of this Q'
day = np.arange(ia[0], ia[0] + 4)                                    # F E F E
assert (hc_sign[day] == [1, -1, 1, -1]).all()
PKH = {}
for phase, sgn in [('flood', 1), ('ebb', -1)]:
    m = day[hc_sign[day] == sgn]
    PKH[phase] = hc_pk[m[np.argmax(np.abs(Q[hc_pk[m]]))]]
    print('%-5s peak %s  Q %+.0f m3/s  (the day\'s two: %s)'
          % (phase, ot[PKH[phase]], Q[PKH[phase]], ', '.join('%+.0f' % Q[hc_pk[i]] for i in m)))

# ------------------------------------------------------------------ STEP 3: flood / ebb / residual on the section
udz_res = zfun.lowpass(u * dz, f='godin')
dz_res = zfun.lowpass(dz, f='godin')
HRS = {'flood': np.array([PKH['flood']]), 'ebb': np.array([PKH['ebb']]), 'residual': np.array([PKH['flood'], PKH['ebb']])}
assert np.isfinite(dz_res[HRS['residual']]).all()
U = {'flood': u[PKH['flood']], 'ebb': u[PKH['ebb']],
     'residual': udz_res[HRS['residual']].sum(axis=0) / dz_res[HRS['residual']].sum(axis=0)}
DZ = {'flood': dz[PKH['flood']], 'ebb': dz[PKH['ebb']], 'residual': dz_res[HRS['residual']].mean(axis=0)}
ZW = {'flood': zw[PKH['flood']], 'ebb': zw[PKH['ebb']], 'residual': zw[HRS['residual']].mean(axis=0)}
for C in U:
    print('  %-9s section mean %+5.1f cm/s; elements %+5.1f to %+5.1f cm/s; inflow %+5.0f, outflow %+5.0f m3/s'
          % (C, 100 * (U[C] * DZ[C] * dy).sum() / (DZ[C] * dy).sum(), 100 * U[C].min(), 100 * U[C].max(),
             (np.clip(U[C], 0, None) * DZ[C] * dy).sum(), (np.clip(U[C], None, 0) * DZ[C] * dy).sum()))

# ------------------------------------------------------------------ STEP 4: section figure, looking out of the cove
xe = np.r_[0, np.cumsum(dy)] / 1000
CLASSES = ['flood', 'ebb', 'residual']
VMAX = {'flood': np.ceil(100 * max(np.abs(U[C]).max() for C in ['flood', 'ebb'])),
        'residual': np.ceil(100 * np.abs(U['residual']).max())}
VMAX['ebb'] = VMAX['flood']
QREF = max(np.abs(np.clip(U[C], lo, hi) * DZ[C] * dy).sum() for C in CLASSES for lo, hi in [(0, None), (None, 0)])
TTL = {'flood': 'Flood (peak, %s)' % ot[PKH['flood']].strftime('%m-%d %H:%M'),
       'ebb': 'Ebb (peak, %s)' % ot[PKH['ebb']].strftime('%m-%d %H:%M'),
       'residual': 'Godin residual (same 2 hours)'}

plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout='constrained', sharey=True)
for ax, C in zip(axs, CLASSES):
    uc = 100 * U[C]
    for j in range(len(dy)):
        CS = ax.pcolormesh(xe[j:j + 2], ZW[C][:, j], uc[:, j:j + 1], cmap='RdBu', vmin=-VMAX[C], vmax=VMAX[C])
    xc = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), uc.shape)
    zc = 0.5 * (ZW[C][1:] + ZW[C][:-1])
    ax.contour(xc, zc, uc, levels=[0], colors='k', linewidths=1.5)
    lab, nlab = ndimage.label(uc > 0)
    lab_o, nlab_o = ndimage.label(uc < 0)
    q = U[C] * DZ[C] * dy
    for L in [lab, np.where(lab_o > 0, lab_o + nlab, 0)]:
        for r in np.unique(L[L > 0]):
            m = L == r
            Qr, w = q[m].sum(), np.abs(q[m])
            x0, z0 = (xc[m] * w).sum() / w.sum(), (zc[m] * w).sum() / w.sum()
            d = max(DSYM * np.sqrt(abs(Qr) / QREF), DMIN)
            ax.scatter(x0, z0, s=d ** 2, facecolors='w', edgecolors='k', linewidths=1.2, zorder=3)
            ax.scatter(x0, z0, s=(0.3 * d) ** 2 if Qr > 0 else (0.5 * d) ** 2, marker='.' if Qr > 0 else 'x',
                       c='k', linewidths=1.2, zorder=4)
            ax.annotate('%+.0f' % Qr, (x0, z0), xytext=(0, -d / 2 - 2), textcoords='offset points',
                        ha='center', va='top', fontsize=10, fontweight='bold', zorder=4)
            print('  %-8s region %s: %3d cells, Q %+6.0f m3/s at x %.2f km, z %.1f m'
                  % (C, 'in ' if Qr > 0 else 'out', m.sum(), Qr, x0, z0))
    ax.set_xlim(xe[0], xe[-1])
    ax.set_title(TTL[C], fontweight='bold')
    ax.set_xlabel('distance across [km]   N ← | → S')
    if C == 'ebb':
        fig.colorbar(CS, ax=axs[:2], label='u [cm s$^{-1}$]   + into the cove', shrink=0.9)
    if C == 'residual':
        fig.colorbar(CS, ax=axs[2], label='u [cm s$^{-1}$]   + into the cove', shrink=0.9)
axs[0].set_ylabel('z [m]')
axs[0].set_ylim(top=np.ceil(zw[:, -1].max()))
fig.suptitle('pc_lj (Lovejoy) looking out of the cove (east), baseline day %s to %s UTC, %s, hourly ocean_his; blue / ⊙ = into the cove, red / ⊗ = out'
             % (bt0.strftime('%Y-%m-%d %H:%M'), bt1.strftime('%m-%d %H:%M'), GTAGEX), fontsize=12)
out_fn = out_dir / 'pc_lj_baseline_peak_sections.png'
fig.savefig(out_fn, dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % out_fn)

# ------------------------------------------------------------------ STEP 5: cove maps
MASK = ds['mask_rho'].values == 1
LON, LAT = ds['lon_rho'].values, ds['lat_rho'].values
cove = MASK.copy()
cove[:, 31:] = False                                     # the cove: rho columns <= 30
JJ, II = np.nonzero(cove)
MLAY = ['surface 1/3', 'bottom 1/3', 'depth average']
FR = [(0, 1 / 3), (2 / 3, 1), (0, 1)]
U_ = np.nan_to_num(ds['u'].values.astype(float))         # (t, k, eta_u, xi_u)
V_ = np.nan_to_num(ds['v'].values.astype(float))
ZWb = ds['z_w'].values
ds.close()
ur = np.zeros(U_.shape[:2] + ZWb.shape[2:])
vr = np.zeros(U_.shape[:2] + ZWb.shape[2:])
ur[..., 1:31] = 0.5 * (U_[..., 0:30] + U_[..., 1:31])    # wall faces count as 0
vr[:, :, 1:20, :] = 0.5 * (V_[:, :, 0:19, :] + V_[:, :, 1:20, :])
zwc = ZWb[:, :, JJ, II]
dwc = zwc[:, -1:, :] - zwc
dt_, db_ = dwc[:, 1:, :], dwc[:, :-1, :]
Hc = dwc[:, 0:1, :]
uc, vc = ur[:, :, JJ, II], vr[:, :, JJ, II]
Ux, Vy, Hl = [np.zeros((len(ot), len(MLAY), len(JJ))) for _ in range(3)]
for n, (f0, f1) in enumerate(FR):
    ov = np.clip(np.minimum(db_, f1 * Hc) - np.maximum(dt_, f0 * Hc), 0, None)
    Ux[:, n], Vy[:, n], Hl[:, n] = (uc * ov).sum(axis=1), (vc * ov).sum(axis=1), ov.sum(axis=1)
Ux_r, Vy_r, Hl_r = [zfun.lowpass(a, f='godin') for a in (Ux, Vy, Hl)]
al = lambda u, v: u * AXIS[0] + v * AXIS[1]              # along-cove, + toward the head


def to2d(a):
    out = np.full(LON.shape, np.nan)
    out[JJ, II] = a
    return out


M = {}
for n, L in enumerate(MLAY):
    M[L] = {}
    for C, (it, a, b, c) in {'flood': (HRS['flood'], Ux, Vy, Hl), 'ebb': (HRS['ebb'], Ux, Vy, Hl),
                             'residual': (HRS['residual'], Ux_r, Vy_r, Hl_r)}.items():
        hs = c[it, n].sum(axis=0)
        M[L][C] = (a[it, n].sum(axis=0) / hs, b[it, n].sum(axis=0) / hs)
VM = {'flood': np.ceil(100 * max(np.hypot(*M[L][C]).max() for L in MLAY for C in ['flood', 'ebb'])),
      'residual': np.ceil(100 * max(np.hypot(*M[L]['residual']).max() for L in MLAY))}
VM['ebb'] = VM['flood']

fig, axs = plt.subplots(3, 3, figsize=(16, 9.5), layout='constrained', sharex=True, sharey=True)
for i, L in enumerate(MLAY):
    for j, C in enumerate(CLASSES):
        ax = axs[i, j]
        ax.pcolormesh(LON, LAT, np.where(MASK, np.nan, 1.0), cmap='Greys', vmin=0, vmax=2.5, shading='nearest')
        for SN, XI in SECT_XI_U.items():
            jw = np.flatnonzero(MASK[:, XI] & MASK[:, XI + 1])
            lo_ = 0.5 * (LON[0, XI] + LON[0, XI + 1])
            ax.plot([lo_, lo_], [LAT[jw.min(), 0] - 0.0009, LAT[jw.max(), 0] + 0.0009], color='0.2',
                    lw=3 if SN == 'pc_lj' else 1.2)
        ax.set_xlim(LON[0, 0], LON[0, -1])
        ax.set_ylim(LAT[0, 0], LAT[-1, 0])
        ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
        ax.tick_params(labelsize=8)
        u_, v_ = M[L][C]
        CSm = ax.pcolormesh(LON, LAT, to2d(100 * al(u_, v_)), cmap='RdBu', vmin=-VM[C], vmax=VM[C], shading='nearest')
        ax.quiver(LON[JJ, II], LAT[JJ, II], u_, v_, scale=VM[C] / 100 * 12, scale_units='width', width=0.003, color='k')
        if i == 0:
            ax.set_title(TTL[C], fontweight='bold')
        if j == 0:
            ax.set_ylabel(L.upper(), fontweight='bold')
        if (i, C) in [(1, 'ebb'), (1, 'residual')]:
            fig.colorbar(CSm, ax=axs[:, :2] if C == 'ebb' else axs[:, 2], shrink=0.5,
                         label='along-cove u [cm s$^{-1}$], + into the cove' + (' (own scale)' if C == 'residual' else ''))
        print('  %-13s %-8s along-cove %+5.1f to %+5.1f cm/s' % (L, C, 100 * al(u_, v_).min(), 100 * al(u_, v_).max()))
fig.suptitle('Penn Cove, baseline day %s to %s UTC, %s, hourly ocean_his, at the pc_lj (thick line) peak flood / ebb hours (residual: Godin, same hours);\n'
             'velocity in the surface / bottom third of the local water depth and the depth average'
             % (bt0.strftime('%Y-%m-%d %H:%M'), bt1.strftime('%m-%d %H:%M'), GTAGEX), fontsize=12)
out_fn = out_dir / 'cove_baseline_peak_maps.png'
fig.savefig(out_fn, dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % out_fn)
