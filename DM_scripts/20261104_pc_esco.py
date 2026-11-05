"""
Eddy viscosity-shear covariance (ESCO; Dijkstra, Schuttelaars & Burchard 2017, JGR
Oceans 122, doi:10.1002/2016JC012379) at the three Penn Cove sections, pc_cp, pc_lj,
pc_lp, of wb1_t0_xn11abbur00: how much of the tidally averaged turbulent stress, which
drives the residual flow, comes from eddy viscosity and shear varying together over the
tide rather than from the mean viscosity acting on the mean shear.

DECOMPOSITION, at each face p and interior w-level k (the layer between cell centres
k-1 and k, where AKv and the shear coincide), hourly:
    tau  = A u_z                         turbulent stress / rho [m2/s2], A = AKv, u_z = du/dz
    <tau> = <A><u_z> + ESCO              < > = Godin tidal average, kept hourly
    ESCO = <A u_z> - <A><u_z>            the covariance part
u is the section-NORMAL velocity, + into the cove, so tau > 0 = the water above moves more
into the cove than the water below. The residual momentum forcing is the vertical
divergence d<tau>/dz = d(<A><u_z>)/dz + d(ESCO)/dz [m/s2]; both parts are shown.

INPUTS (hourly ocean_his snapshots; AKv at rho points put on the face as the mean of the
two rho cells either side, as salt is)
    pc_cove_[ds0]_[ds1].nc       u, z_rho, z_w
    pc_cove_mix_[ds0]_[ds1].nc   AKv
CAVEATS AKv is an instantaneous snapshot, so sub-hourly variability aliases into the
covariance; and AKv is the GLS closure only - numerical mixing (large in Penn Cove,
[[pc-tef-frozen-floor]]) is not in it, so this is the ESCO the closure produces.

CONDITIONS  the means are also taken over the hours of each class of pc_tidal_cycles.py
(spring / neap, tropic / equatorial, high / low strat, up- / down-cove stress), to test
whether ESCO scales with the tide, as tidal ESCO should.

OUTPUT LO_output/DM_outs/20261104_pc_esco/
    esco_sections.png     cross-sections (all hours): <A><u_z> | ESCO | ESCO share, sections west to east
    esco_profiles.png     section-averaged profiles of the stress terms and their divergence
    esco_conditions.png   section rms of mean-term and ESCO stress under each condition
    esco.p                time-mean fields per section and condition (cached)

run 20261104_pc_esco.py [-test True]
"""
import argparse
import pickle
from time import time as clock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cmocean

from lo_tools import Lfun, zfun
from pc_tidal_cycles import cycle_masks

parser = argparse.ArgumentParser()
parser.add_argument('-test', default=False, type=Lfun.boolean_string)   # True: first 60 days, separate cache
TEST = parser.parse_args().test

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove
FACES = {        # box indices of each section's u-faces (global i - 37, global j - 209), north to south; west to east
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
CONDS = ['all', 'spring', 'neap', 'tropic', 'equatorial', 'high strat', 'low strat', 'up-cove', 'down-cove']
EDGES = np.arange(0, 31, 1.0)             # depth-below-surface bins for the profiles [m]
DMID = 0.5 * (EDGES[1:] + EDGES[:-1])
BLK = 744

Ldir = Lfun.Lstart(gridname='wb1')
box = Ldir['LOo'] / 'extract' / GTAGEX / 'box'
vel_fn, mix_fn = box / ('pc_cove_%s_%s.nc' % (DS0, DS1)), box / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261104_pc_esco'
out_dir.mkdir(parents=True, exist_ok=True)
cache_fn = out_dir / ('esco%s.p' % ('_test' if TEST else ''))


def godin(x):
    """Godin tidal average on axis 0, hourly, NaN for 35 h at each end."""
    return zfun.lowpass(x, f='godin')


# ------------------------------------------------------------------ hourly fields, then ESCO (cached)
if cache_fn.exists():
    R = pickle.load(open(cache_fn, 'rb'))
    print('loaded ' + str(cache_fn))
else:
    vb, mx = xr.open_dataset(vel_fn), xr.open_dataset(mix_fn)
    assert np.array_equal(vb.ocean_time.values, mx.ocean_time.values)
    T = vb.ocean_time.values
    NT = 60 * 24 if TEST else len(T)
    T = T[:NT]
    N = vb.sizes['s_rho']
    H = {SN: {k: np.full((NT, N - 1, len(F['ETA'])), np.nan, np.float32) for k in ['A', 'uz', 'dw']} for SN, F in FACES.items()}
    ZR = {SN: np.zeros((N, len(F['ETA']))) for SN, F in FACES.items()}   # time-mean cell-centre z, for drawing
    tt0 = clock()
    for t0 in range(0, NT, BLK):
        tt = slice(t0, min(t0 + BLK, NT))
        u_b = vb.u.isel(ocean_time=tt).to_numpy().astype(float)
        zr_b = vb.z_rho.isel(ocean_time=tt).to_numpy()
        zw_b = vb.z_w.isel(ocean_time=tt).to_numpy()
        A_b = mx.AKv.isel(ocean_time=tt).to_numpy().astype(float)
        for SN, F in FACES.items():
            XI_U, ETA = F['XI_U'], F['ETA']
            face = lambda x: x[:, :, ETA][:, :, :, [XI_U, XI_U + 1]].mean(axis=-1)       # rho -> u-face, (t, lev, p)
            u = SIGN * u_b[:, :, ETA, XI_U]                                             # (t, k, p) + into the cove
            zr, zw, A = face(zr_b), face(zw_b), face(A_b)
            H[SN]['uz'][tt] = np.diff(u, axis=1) / np.diff(zr, axis=1)                 # (t, N-1, p) at interior w-levels
            H[SN]['A'][tt] = A[:, 1:-1, :]                                              # interior w-levels 1..N-1
            H[SN]['dw'][tt] = zw[:, -1:, :] - zw[:, 1:-1, :]                            # depth of each w-level below the surface
            ZR[SN] += zr.sum(axis=0)
        print('  hours %d-%d of %d, %.0f s' % (tt.start, tt.stop, NT, clock() - tt0))
    dy = {SN: (1 / vb.pn.values[F['ETA']][:, [F['XI_U'], F['XI_U'] + 1]]).mean(axis=-1) for SN, F in FACES.items()}
    h = {SN: vb.h.values[F['ETA']][:, [F['XI_U'], F['XI_U'] + 1]].mean(axis=-1) for SN, F in FACES.items()}
    vb.close(), mx.close()

    CM = cycle_masks(T)
    CM['all'] = np.ones(NT, bool)
    R = {'T': T, 'conds': CONDS}
    for SN in FACES:
        A, uz = H[SN]['A'].astype(float), H[SN]['uz'].astype(float)
        G_tau, G_A, G_uz = godin(A * uz), godin(A), godin(uz)    # < A u_z >, < A >, < u_z >
        mean_t = G_A * G_uz                                      # mean viscosity on the mean shear
        esco = G_tau - mean_t                                    # the covariance part
        ok = np.isfinite(G_A[:, 0, 0])                           # hours where the Godin filter is defined
        zr = ZR[SN] / NT                                         # (N, p) time-mean cell centres
        w = np.diff(zr, axis=0) * dy[SN][None, :]                # (N-1, p) layer areas, for section rms
        rms = lambda x: np.sqrt((x**2 * w[None]).sum(axis=(1, 2)) / w.sum())
        out = dict(dy=dy[SN], h=h[SN], zr=zr, dw=np.nanmean(H[SN]['dw'], axis=0),
                   rms_tau=rms(G_tau), rms_mean=rms(mean_t), rms_esco=rms(esco), ok=ok)
        for c in CONDS:
            m = CM[c] & ok
            out[c] = dict(tau=G_tau[m].mean(axis=0), mean=mean_t[m].mean(axis=0), esco=esco[m].mean(axis=0),
                          A=G_A[m].mean(axis=0), uz=G_uz[m].mean(axis=0), frac=m.mean())
        R[SN] = out
        a = out['all']
        print('%s: section rms of the time-mean stress [1e-6 m2/s2]: total %.2f, mean term %.2f, ESCO %.2f; corr(ESCO, total) %.2f'
              % (SN, *(1e6 * np.sqrt((x**2 * w).sum() / w.sum()) for x in (a['tau'], a['mean'], a['esco'])),
                 np.corrcoef(a['esco'].ravel(), a['tau'].ravel())[0, 1]))
    pickle.dump(R, open(cache_fn, 'wb'))
    print('saved ' + str(cache_fn))

SECTS = list(FACES)
TERMS = [('mean', r'mean term $\langle A\rangle\langle u_z\rangle$'), ('esco', r'ESCO $\langle A u_z\rangle - \langle A\rangle\langle u_z\rangle$'),
         ('tau', r'total $\langle A u_z\rangle$')]
TCOL = {'mean': '#0072B2', 'esco': '#D55E00', 'tau': 'k'}
U6 = 1e6                                  # plotting unit: 1e-6


def binned(x, dw, wt):
    """Area-weighted mean of x in the 1 m depth-below-surface bins (x, dw, wt: same shape)."""
    out = np.full(len(DMID), np.nan)
    for b in range(len(DMID)):
        m = (dw >= EDGES[b]) & (dw < EDGES[b + 1]) & np.isfinite(x)
        if wt[m].sum() > 0:
            out[b] = (x[m] * wt[m]).sum() / wt[m].sum()
    return out


# ------------------------------------------------------------------ figure 1: cross-sections, all hours
WR = [R[SN]['dy'].sum() for SN in SECTS]
vm = np.nanpercentile(np.concatenate([np.abs(R[SN]['all'][k]).ravel() for SN in SECTS for k in ['mean', 'esco', 'tau']]), 98) * U6
fig, axs = plt.subplots(3, 3, figsize=(14, 9), sharey=True, layout='constrained', gridspec_kw={'width_ratios': WR})
for j, SN in enumerate(SECTS):
    d, zr, dy, h = R[SN]['all'], R[SN]['zr'], R[SN]['dy'], R[SN]['h']
    xe = np.concatenate([[0], np.cumsum(dy)]) / 1000
    for i, (k, lab) in enumerate(TERMS):
        ax = axs[i, j]
        F = d[k] * U6
        for p in range(len(dy)):
            cs = ax.pcolormesh(xe[p:p + 2], zr[:, p], F[:, p:p + 1], cmap=cmocean.cm.balance, vmin=-vm, vmax=vm, shading='flat')
            ax.fill_between(xe[p:p + 2], -h[p], zr[0, p], color='0.8', lw=0)
            ax.fill_between(xe[p:p + 2], zr[-1, p], 0, color='0.8', lw=0)
        X = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), F.shape)
        ax.contour(X, 0.5 * (zr[1:] + zr[:-1]), F, levels=[0], colors='k', linewidths=0.8)
        ax.set_xlim(xe[-1], xe[0])       # reversed: north at right = looking west, into the cove
        ax.grid(True, alpha=0.3)
        ax.set_title('%s: %s' % (NAME[SN], lab), fontsize=9)
        if j == 0:
            ax.set_ylabel('z [m]')
        if i == 2:
            ax.set_xlabel('distance from north shore [km]')
fig.colorbar(cs, ax=axs, location='bottom', shrink=0.5, aspect=40,
             label=r'tidally averaged turbulent stress [10$^{-6}$ m$^2$ s$^{-2}$], + = water above moves more INTO the cove; grey = no resolved stress')
fig.suptitle('Eddy viscosity-shear covariance (ESCO, Dijkstra et al. 2017) at the Penn Cove sections, west to east, each looking into the cove (north at right); '
             '2-year mean of the Godin-filtered terms, %s' % GTAGEX, fontsize=10)
fig.savefig(out_dir / 'esco_sections.png', dpi=200, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ figure 2: section-averaged profiles of the stress and its divergence
fig, axs = plt.subplots(2, 3, figsize=(14, 10), sharey=True, layout='constrained')
for j, SN in enumerate(SECTS):
    d, zr, dy, dw = R[SN]['all'], R[SN]['zr'], R[SN]['dy'], R[SN]['dw']
    wt = np.diff(zr, axis=0) * dy[None, :]                      # layer areas at the w-levels
    for k, lab in TERMS:
        axs[0, j].plot(U6 * binned(d[k], dw, wt), DMID, color=TCOL[k], lw=2, label=lab)
        dv = np.diff(d[k], axis=0) / (dw[:-1] - dw[1:])           # d/dz between w-levels (z up = dw down)
        axs[1, j].plot(U6 * binned(dv, 0.5 * (dw[:-1] + dw[1:]), 0.5 * (wt[:-1] + wt[1:])), DMID, color=TCOL[k], lw=2, label=lab)
    axs[0, j].set_title('%s: stress' % NAME[SN], fontsize=10)
    axs[1, j].set_title('%s: its divergence (the push on the residual flow)' % NAME[SN], fontsize=10)
    axs[0, j].set_xlabel(r'stress [10$^{-6}$ m$^2$ s$^{-2}$], + = water above more into the cove', fontsize=8)
    axs[1, j].invert_xaxis()              # west (into the cove) at left, as on the velocity profiles
    axs[1, j].set_xlabel(r'$\leftarrow$ pushes into the cove (+)    d(stress)/dz [10$^{-6}$ m s$^{-2}$]    out (-) $\rightarrow$', fontsize=8)
for ax in axs.flat:
    ax.axvline(0, color='k', lw=0.6)
    ax.grid(True, alpha=0.3)
axs[0, 0].invert_yaxis()
for ax in axs[:, 0]:
    ax.set_ylabel('depth below the surface [m]')
axs[0, 0].legend(fontsize=8, loc='lower left')
fig.suptitle('Section-averaged ESCO and mean-viscosity stress (1 m depth bins), 2-year mean of the Godin-filtered terms, %s' % GTAGEX, fontsize=10)
fig.savefig(out_dir / 'esco_profiles.png', dpi=200, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ figure 3: under each condition
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout='constrained')
x = np.arange(len(CONDS))
for j, SN in enumerate(SECTS):
    w = np.diff(R[SN]['zr'], axis=0) * R[SN]['dy'][None, :]
    srms = lambda f: U6 * np.sqrt((f**2 * w).sum() / w.sum())
    rm = [srms(R[SN][c]['mean']) for c in CONDS]
    re = [srms(R[SN][c]['esco']) for c in CONDS]
    ax = axs[j]
    ax.bar(x - 0.2, rm, 0.4, color=TCOL['mean'], label='mean term')
    ax.bar(x + 0.2, re, 0.4, color=TCOL['esco'], label='ESCO')
    for xi, a_, b_ in zip(x, rm, re):
        ax.text(xi, max(a_, b_) * 1.02, '%.2f' % (b_ / a_), ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS, rotation=40, ha='right', fontsize=8)
    ax.set_title('%s (numbers: ESCO / mean term)' % NAME[SN], fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    print('%s: spring/neap ESCO %.2f, mean term %.2f; tropic/equatorial ESCO %.2f; high/low strat ESCO %.2f'
          % (SN, re[1] / re[2], rm[1] / rm[2], re[3] / re[4], re[5] / re[6]))
axs[0].set_ylabel(r'section rms of the time-mean stress [10$^{-6}$ m$^2$ s$^{-2}$]')
axs[0].legend(fontsize=8)
fig.suptitle('How strong is ESCO under each condition? Section rms of the time-mean stress terms, %s' % GTAGEX, fontsize=10)
fig.savefig(out_dir / 'esco_conditions.png', dpi=200, transparent=True)
plt.close(fig)
print('saved figures to ' + str(out_dir))
