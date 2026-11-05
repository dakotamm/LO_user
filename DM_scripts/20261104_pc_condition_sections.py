"""
One big figure: the section-normal velocity across pc_cp, pc_lj and pc_lp (west to
east) for the average flood, average ebb and residual, under every condition tested
with the -cycle option of the 20261103_pc_* scripts, so all can be seen at once.

ROWS (conditions; masks from pc_tidal_cycles.cycle_masks)
    all hours | spring, neap | tropic, equatorial (terciles within each month of the
    semidiurnal / diurnal envelope of the mouth transport) | high, low strat (record-wide
    thirds of drho) | up-cove, down-cove (along-cove wind stress beyond +-0.01 Pa) |
    northward, southward (cross-cove wind stress beyond +-0.01 Pa, + toward the north shore) |
    high / low strat within calm hours (both stress components within +-0.01 Pa) |
    up / down-cove within high strat, up / down-cove within low strat
COLUMNS  flood | ebb | residual, each with the three sections west to east, each as wide
    as the section really is, looking into the cove (west) so north is at the RIGHT
SALINITY  also composited per cell and stored as R[...]['S'] (area-weighted, sum_t s A / sum_t A, salt
    averaged onto the face from the two rho cells either side), for 20261105_pc_conceptual_map.py.
AVERAGE  as 20261103_pc_flood_ebb_sections.py (SRC = 'his', AVG = 'transport'):
    flood = hours when the section's own total Q > 0, ebb Q < 0, residual = all hours of
    the condition; U = sum_t q / sum_t A per cell, drawn on the composite's mean interfaces
COLOR    ONE scale for every panel (flood, ebb and residual, every row), so a color means
    the same speed anywhere on the page. + (red) = into the cove.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (hourly ocean_his snapshots)
OUTPUT LO_output/DM_outs/20261104_pc_condition_sections/condition_sections.png
       and condition_sections.p (the composites, for replotting)

run 20261104_pc_condition_sections.py
"""
import pickle
from time import time as clock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cmocean

from lo_tools import Lfun
from pc_tidal_cycles import cycle_masks

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove
FACES = {
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
SECTS = list(FACES)                       # west to east
CONDS = ['all', 'spring', 'neap', 'tropic', 'equatorial', 'high strat', 'low strat', 'up-cove', 'down-cove',
         'northward', 'southward', 'high strat, calm', 'low strat, calm',
         'up-cove, high strat', 'down-cove, high strat', 'up-cove, low strat', 'down-cove, low strat']
PHASES = ['flood', 'ebb', 'residual']
BLK = 744

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261104_pc_condition_sections'
out_dir.mkdir(parents=True, exist_ok=True)
cache_fn = out_dir / 'condition_sections.p'

# ------------------------------------------------------------------ composites (cached)
if cache_fn.exists():
    R = pickle.load(open(cache_fn, 'rb'))
    print('loaded ' + str(cache_fn))
else:
    ds = xr.open_dataset(in_fn)
    T = ds.ocean_time.values
    CM = cycle_masks(T)
    CM['all'] = np.ones(len(T), bool)
    pn = ds.pn.values
    DY = {SN: (1 / pn[F['ETA']][:, [F['XI_U'], F['XI_U'] + 1]]).mean(axis=-1) for SN, F in FACES.items()}
    # running sums per section, condition and phase: sum q, sum A, sum dz, hours, sum Q
    NK = ds.sizes['s_rho']
    acc = {(SN, c, ph): dict(qA=0, A=0, dz=0, n=0, Q=0, sA=0) for SN in SECTS for c in CONDS for ph in PHASES}
    tt0 = clock()
    for t0 in range(0, len(T), BLK):
        tt = slice(t0, min(t0 + BLK, len(T)))
        u_b = ds.u.isel(ocean_time=tt).to_numpy().astype(float)
        zw_b = ds.z_w.isel(ocean_time=tt).to_numpy()
        s_b = ds.salt.isel(ocean_time=tt).to_numpy().astype(float)
        for SN, F in FACES.items():
            XI_U, ETA = F['XI_U'], F['ETA']
            u = SIGN * u_b[:, :, ETA, XI_U]                                        # (t, k, p)
            zw = zw_b[:, :, ETA][:, :, :, [XI_U, XI_U + 1]].mean(axis=-1)          # (t, k+1, p)
            salt = s_b[:, :, ETA][:, :, :, [XI_U, XI_U + 1]].mean(axis=-1)         # (t, k, p) at the face
            dz = np.diff(zw, axis=1)
            A = dz * DY[SN][None, None, :]
            q = u * A
            Q = q.sum(axis=(1, 2))
            PH = {'flood': Q > 0, 'ebb': Q < 0, 'residual': np.ones(len(Q), bool)}
            for c in CONDS:
                for ph in PHASES:
                    m = PH[ph] & CM[c][tt]
                    a = acc[(SN, c, ph)]
                    a['qA'] = a['qA'] + q[m].sum(axis=0)
                    a['A'] = a['A'] + A[m].sum(axis=0)
                    a['dz'] = a['dz'] + dz[m].sum(axis=0)
                    a['sA'] = a['sA'] + (salt * A)[m].sum(axis=0)
                    a['n'] += m.sum()
                    a['Q'] += Q[m].sum()
        print('  hours %d-%d of %d, %.0f s' % (tt.start, tt.stop, len(T), clock() - tt0))
    h = {SN: ds.h.values[F['ETA']][:, [F['XI_U'], F['XI_U'] + 1]].mean(axis=-1) for SN, F in FACES.items()}   # bed depth at each face
    ds.close()
    R = {'dy': DY, 'n_all': len(T)}
    for (SN, c, ph), a in acc.items():
        zw = np.concatenate([-h[SN][None, :], -h[SN][None, :] + np.cumsum(a['dz'] / a['n'], axis=0)], axis=0)
        R[(SN, c, ph)] = dict(U=a['qA'] / a['A'], zw=zw, frac=a['n'] / len(T), Q=a['Q'] / a['n'], S=a['sA'] / a['A'])
    pickle.dump(R, open(cache_fn, 'wb'))
    print('saved ' + str(cache_fn))

# ------------------------------------------------------------------ plot
W = [R['dy'][SN].sum() / 1000 for SN in SECTS]                      # section widths [km]
GAP = 0.35
wr = W + [GAP] + W + [GAP] + W
COLS = [(ph, SN) for ph in PHASES for SN in SECTS]
ci = [0, 1, 2, 4, 5, 6, 8, 9, 10]                                 # gridspec columns (gaps at 3, 7)
vm = np.nanpercentile(np.concatenate([np.abs(R[(SN, c, ph)]['U']).ravel() for SN in SECTS for c in CONDS for ph in PHASES]), 99) * 100   # one scale for all

fig = plt.figure(figsize=(26, 1.75 * len(CONDS) + 2.5), layout='constrained')
gs = fig.add_gridspec(len(CONDS), len(wr), width_ratios=wr)
for i, c in enumerate(CONDS):
    for (ph, SN), j in zip(COLS, ci):
        ax = fig.add_subplot(gs[i, j])
        d = R[(SN, c, ph)]
        U, zw, dy = 100 * d['U'], d['zw'], R['dy'][SN]
        xe = np.concatenate([[0], np.cumsum(dy)]) / 1000
        for p in range(len(dy)):
            cs = ax.pcolormesh(xe[p:p + 2], zw[:, p], U[:, p:p + 1], cmap=cmocean.cm.balance, vmin=-vm, vmax=vm, shading='flat')
        X = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), U.shape)
        ax.contour(X, 0.5 * (zw[1:] + zw[:-1]), U, levels=[0], colors='k', linewidths=0.8)
        ax.set_xlim(xe[-1], xe[0])       # reversed: north at right = looking west, into the cove
        ax.set_ylim(-28, 0)
        ax.text(0.03, 0.04, 'Q %+.0f' % d['Q'], transform=ax.transAxes, fontsize=6, color='0.25')
        ax.tick_params(labelsize=6, length=2)
        if j != 0:
            ax.set_yticklabels([])
        if i != len(CONDS) - 1:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_title('%s\n%s' % (ph.upper(), SN) if SN == 'pc_lj' else SN, fontsize=9)
        if j == 0:
            ax.set_ylabel('%s\n(%.0f%% of hours)\nz [m]' % (c, 100 * R[(SN, c, 'residual')]['frac']), fontsize=8)
        if i == len(CONDS) - 1 and SN == 'pc_lj':
            ax.set_xlabel('km from north shore (north at right)', fontsize=7)
fig.colorbar(cs, ax=fig.axes, location='bottom', shrink=0.4, aspect=50, pad=0.01,
             label='section-normal velocity [cm s$^{-1}$], + into the cove (black: 0); one scale for every panel')
fig.suptitle('Penn Cove section-normal velocity under each condition: average flood | average ebb | residual, sections west to east '
             '(each looking into the cove, north at right); Q = mean section transport [m$^3$ s$^{-1}$]; hourly snapshots, transport-weighted, %s, %s to %s'
             % (GTAGEX, DS0, DS1), fontsize=11)
fn = out_dir / 'condition_sections.png'
fig.savefig(fn, dpi=150, transparent=True)
plt.close(fig)
print('saved ' + str(fn))
