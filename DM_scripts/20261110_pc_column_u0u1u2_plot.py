"""
Plots of the per-column u = u0 + u1 + u2 decomposition of 20261110_pc_column_u0u1u2.py,
every u-face column of Penn Cove, head to mouth.

FIGURES (LO_output/DM_outs/20261110_pc_column_u0u1u2/)
    alongcove_u0u1u2.png   x = distance west of the mouth (head LEFT, mouth RIGHT):
        (a) u0 mean and std (subtidal section-mean velocity)
        (b) rms of time-mean u1, and its lateral / layered parts
        (c) rms of u2, and its slab / lateral / layered parts
        (d) rms(time-mean u1) / rms(u2): mean exchange relative to the tide
    grid_u1_mean.png, grid_u_max_flood.png, grid_u_max_ebb.png, grid_u2_max_flood.png,
    grid_u2_max_ebb.png
        the field on all 30 columns, head (top left) to mouth (bottom right), reading
        order; each looking INTO the cove (west), north at RIGHT, as 20261103; one km scale
        and one color scale for all panels, so widths and speeds compare. pc_cp, pc_lj and
        pc_lp titled in their colors. Max flood / ebb on the mouth-Q clock.
alongcove_u0u1u2.csv     the numbers in (a)-(d), one row per column

run 20261110_pc_column_u0u1u2_plot.py
"""
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lo_tools import Lfun

SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2'}
SNAME = {v: k for k, v in SECT_XI_U.items()}
PCOL = {'total': 'k', 'slab': 'tab:green', 'lateral': 'tab:blue', 'layered': 'tab:orange'}

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_column_u0u1u2'
with open(out_dir / 'column_u0u1u2.p', 'rb') as f:
    D = pickle.load(f)
R = D['R']
cols = np.array(sorted(R))
S = pd.DataFrame([dict(col=c, **R[c]['S']) for c in cols])
S['ratio_u1m_u2'] = S.u1m_rms / S.u2_rms
S.to_csv(out_dir / 'alongcove_u0u1u2.csv', index=False, float_format='%.4g')
x = S.x_km.values

# ------------------------------------------------------------------ along-cove curves
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, layout='constrained')
axs = axs.ravel()
ax = axs[0]
ax.fill_between(x, 1e3 * (S.u0_mean - S.u0_std), 1e3 * (S.u0_mean + S.u0_std), color='0.85', label='$\\pm$1 std')
ax.plot(x, 1e3 * S.u0_mean, 'k-', label='mean')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('$u_0$ [mm s$^{-1}$]', fontsize=9)
ax.set_title('(a) $u_0$: subtidal section-mean velocity', fontsize=9, loc='left', pad=14)
ax = axs[1]
for k, nm in [('u1m_rms', 'total'), ('u1m_lat', 'lateral'), ('u1m_lay', 'layered')]:
    ax.plot(x, 100 * S[k], color=PCOL[nm], label=nm)
ax.set_ylabel('rms [cm s$^{-1}$]', fontsize=9)
ax.set_title('(b) time-mean $u_1$ (the mean exchange)', fontsize=9, loc='left', pad=14)
ax = axs[2]
for k, nm in [('u2_rms', 'total'), ('u2_slab', 'slab'), ('u2_lat', 'lateral'), ('u2_lay', 'layered')]:
    ax.plot(x, 100 * S[k], color=PCOL[nm], label=nm)
ax.set_ylabel('rms [cm s$^{-1}$]', fontsize=9)
ax.set_title('(c) $u_2$ (tidal, hourly)', fontsize=9, loc='left', pad=14)
ax = axs[3]
ax.plot(x, S.ratio_u1m_u2, 'k-')
ax.set_ylabel('rms($\\overline{u_1}$) / rms($u_2$)', fontsize=9)
ax.set_title('(d) mean exchange relative to the tide', fontsize=9, loc='left', pad=14)
for n, ax in enumerate(axs):
    for SN, XI_U in SECT_XI_U.items():
        ax.axvline(x[cols == XI_U][0], color=SCOL[SN], lw=1, alpha=0.6, zorder=0)
        if n < 2:
            ax.text(x[cols == XI_U][0], 1.01, SN, color=SCOL[SN], fontsize=7, ha='center', va='bottom', transform=ax.get_xaxis_transform())
    ax.grid(True, color='0.9', lw=0.5)
    ax.tick_params(labelsize=8)
    if n < 3:
        ax.legend(fontsize=7, frameon=False)
axs[0].invert_xaxis()
for ax in axs[2:]:
    ax.set_xlabel('distance west of the mouth [km]   (head left, mouth right)', fontsize=8)
fig.suptitle('$u = u_0 + u_1 + u_2$ (MacCready 2011 Sec. 2) at every u-face column of Penn Cove, %s, 2024-2025 hourly; + into the cove\n'
             'lateral = each face\'s depth mean; layered = departure from it; slab = section mean of $u_2$' % D['gtagex'], fontsize=10)
fig.savefig(out_dir / 'alongcove_u0u1u2.png', dpi=200, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ section grids
WMAX = max(R[c]['dy'].sum() for c in cols) / 1000


def field(c, key):
    if key == 'u1_mean':
        return R[c]['u1m']
    term, ph = key.split('_', 1)
    return R[c]['comp'][ph.replace('_', ' ')][term]


for key, lab in [('u1_mean', 'time-mean $u_1$ (Eq. 2.4)'), ('u_max_flood', '$u$ at max flood'), ('u_max_ebb', '$u$ at max ebb'),
                 ('u2_max_flood', '$u_2$ at max flood'), ('u2_max_ebb', '$u_2$ at max ebb')]:
    vm = 100 * max(np.abs(field(c, key)).max() for c in cols)
    fig, axs = plt.subplots(5, 6, figsize=(16, 11), sharex=True, sharey=True, layout='constrained')
    for ax, c in zip(axs.ravel(), cols):
        dy, h, z0 = R[c]['dy'], R[c]['h'], R[c]['z0']
        xc = np.cumsum(dy) / 1000 - dy / 2000                 # face centres, km from the north end
        xe = np.concatenate(([0], np.cumsum(dy) / 1000))
        X = np.broadcast_to(xc[None, :], z0.shape)
        fld = 100 * field(c, key)
        cs = ax.pcolormesh(X, z0, fld, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='nearest')
        if np.ptp(fld) > 0 and fld.min() < 0 < fld.max():
            ax.contour(X, z0, fld, levels=[0], colors='k', linewidths=0.7)
        ax.fill_between(np.repeat(xe, 2)[1:-1], np.repeat(-h, 2), -30, color='0.6')
        ax.set_xlim(WMAX, 0)                                  # reversed: north at right = looking into the cove
        ax.set_ylim(-28, 0)
        ax.set_title('col %d, %.1f km%s' % (c, R[c]['S']['x_km'], (' - ' + SNAME[c]) if c in SNAME else ''),
                     fontsize=8, color=SCOL.get(SNAME.get(c), 'k'))
        ax.tick_params(labelsize=7)
    for ax in axs[-1]:
        ax.set_xlabel('km from the north shore', fontsize=7)
    for ax in axs[:, 0]:
        ax.set_ylabel('z [m]', fontsize=7)
    fig.colorbar(cs, ax=axs, shrink=0.6, label=lab + ' [cm s$^{-1}$], + into the cove')
    clock_ = ('\nmax flood / ebb = the peak-|Q| hour of each flood / ebb of the MOUTH transport (%d / %d hours), 2024-2025'
              % (D['PF'].sum(), D['PE'].sum())) if 'max' in key else ', 2024-2025'
    fig.suptitle('%s at every u-face column of Penn Cove, %s: head (top left) to mouth (bottom right), each looking into the cove (west), north at right%s'
                 % (lab, D['gtagex'], clock_), fontsize=10)
    fig.savefig(out_dir / ('grid_%s.png' % key), dpi=200, transparent=True)
    plt.close(fig)
    print('saved grid_%s.png' % key)
print('saved to ' + str(out_dir))
