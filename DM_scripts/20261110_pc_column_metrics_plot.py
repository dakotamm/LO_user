"""
Along-cove variation of the Penn Cove flow metrics, head to mouth, from the 30 u-face
columns of 20261110_pc_column_metrics.py: the section analyses of pc_cp, pc_lj and
pc_lp (marked) extended to every column.

PANELS (x = distance west of the mouth; head at LEFT, mouth at RIGHT, as the sections
are laid out west to east)
    geometry        time-mean section area A and mean depth A / width
    tidal Q         rms of Q - <Q> (<> = Godin), and the PRISM EXPECTATION: rms of
                    dVland/dt - <dVland/dt>, Vland the volume anomaly landward of the
                    column, differentiated spectrally (a centred difference over 2 h
                    would damp M2 by 4%)
    tidal U         the same divided by the hourly section area: section-mean tidal
                    velocity, measured and prism-expected
    u95             area-weighted 95th percentile of |u|, mean over the hours
    exchange        Qx/G, LC/G (lateral), VC/G (layered): ratios of sums over the hours
    shear           sqrt(mean Srms^2)
    drho, fcrit     mean bottom - top sigma0; mean share of the layer area with Ri < 1/4
    two-layer share and DIVIDE depth (median, IQR): runs in Uprof (width-mean u on depth
                    below the moving surface), rules of 20261028_pc_mouth_interface.py
                    (|U| >= 1 cm/s, runs >= 2 m, same-sign neighbours merged), divide =
                    zero crossing between the runs, two-layer hours only; with the mean
                    water depth A / width for reference
FLOOD / EBB on ONE clock for every column: the sign of the hourly Q at the mouth (col 30),
as 20261103_pc_alongchannel_salt.py (the cove floods and ebbs together). Solid = all
hours, dashed = flood, dotted = ebb.

INPUT  LO_output/DM_outs/20261110_pc_column_metrics/column_metrics.nc
OUTPUT LO_output/DM_outs/20261110_pc_column_metrics/
    column_metrics_alongcove.png
    column_metrics_alongcove.csv   the plotted numbers, one row per column

run 20261110_pc_column_metrics_plot.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2'}
UMIN, HMIN = 0.01, 2.0                    # run rules of 20261028_pc_mouth_interface.py
PH_LS = {'all': '-', 'flood': '--', 'ebb': ':'}

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_column_metrics'
d = xr.open_dataset(out_dir / 'column_metrics.nc')
x = d.x_km.values
cols = d.col.values
NT, NC = d.Q.shape
DG = d.depth.values
DZG = DG[1] - DG[0]
lp = lambda a: zfun.lowpass(a, f='godin')

# ------------------------------------------------------------------ tidal transport: measured vs prism (continuity)
Q, A = d.Q.values, d.A.values
V = d.Vland.values
om = 2 * np.pi * np.fft.rfftfreq(NT, d=3600.0)
dVdt = np.fft.irfft(1j * om[:, None] * np.fft.rfft(V - V.mean(axis=0), axis=0), n=NT, axis=0)
Qt, Pt = Q - lp(Q), dVdt - lp(dVdt)
ok = np.isfinite(Qt[:, 0])
rms = lambda a: np.sqrt(np.nanmean(a[ok]**2, axis=0))
R = pd.DataFrame({'col': cols, 'x_km': x, 'A': A.mean(axis=0), 'H': A.mean(axis=0) / d.width.values,
                  'Qt_rms': rms(Qt), 'Qprism_rms': rms(Pt), 'Ut_rms': rms(Qt / A), 'Uprism_rms': rms(Pt / A),
                  'r_Q_prism': [np.corrcoef(Qt[ok, c], Pt[ok, c])[0, 1] for c in range(NC)]})

# ------------------------------------------------------------------ hourly metrics by phase (mouth-Q clock)
Qm = Q[:, cols == 30][:, 0]
PH = {'all': np.ones(NT, bool), 'flood': Qm > 0, 'ebb': Qm < 0}
for ph, m in PH.items():
    G = d.G.values[m].sum(axis=0)
    for k in ['Qx', 'LC', 'VC']:
        R['%s_G_%s' % (k, ph)] = d[k].values[m].sum(axis=0) / G
    R['u95_' + ph] = d.u95.values[m].mean(axis=0)
    R['Srms_' + ph] = np.sqrt((d.Srms.values[m]**2).mean(axis=0))
    R['drho_' + ph] = d.drho.values[m].mean(axis=0)
    R['fcrit_' + ph] = d.fcrit.values[m].mean(axis=0)


# ------------------------------------------------------------------ two-layer divide per column and hour (copied from 20261028_pc_mouth_interface.py)
def runs_of(U):
    """[[sign, top_index, bottom_index, thickness, mean U], ...] after the 1 cm/s / 2 m rules"""
    sg = np.where(np.isfinite(U) & (np.abs(U) >= UMIN), np.sign(U), 0)
    rr = []
    for n, s_ in enumerate(sg):
        if s_ == 0:
            continue
        if rr and rr[-1][0] == s_ and rr[-1][2] == n - 1:
            rr[-1][2] = n
        else:
            rr.append([s_, n, n])
    rr = [r for r in rr if (r[2] - r[1] + 1) * DZG >= HMIN]
    merged = []
    for r in rr:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    return [[s_, a, b, (b - a + 1) * DZG, np.nanmean(U[a:b + 1])] for s_, a, b in merged]


def divide(U, r):
    """zero crossing between the upper run r[0] and the lower run r[1]"""
    a, b = r[0][2], r[1][1]
    seg = U[a:b + 1]
    dd = DG[a:b + 1]
    k = np.flatnonzero(np.sign(seg[:-1]) != np.sign(seg[1:]))
    if len(k):
        k = k[0]
        return dd[k] + (dd[k + 1] - dd[k]) * seg[k] / (seg[k] - seg[k + 1])
    return 0.5 * (dd[0] + dd[-1])


NL = np.zeros((NT, NC), int)
DIV = np.full((NT, NC), np.nan)
UP = d.Uprof.values
for c in range(NC):
    for t in range(NT):
        r = runs_of(UP[t, c])
        NL[t, c] = min(len(r), 3)
        if len(r) == 2:
            DIV[t, c] = divide(UP[t, c], r)
    print('divide: col %d done' % cols[c])
for ph, m in PH.items():
    R['twolayer_' + ph] = (NL[m] == 2).mean(axis=0)
    R['div_med_' + ph] = np.nanmedian(DIV[m], axis=0)
R['div_q25'], R['div_q75'] = np.nanpercentile(DIV, 25, axis=0), np.nanpercentile(DIV, 75, axis=0)
R.to_csv(out_dir / 'column_metrics_alongcove.csv', index=False, float_format='%.4g')

# ------------------------------------------------------------------ plot
fig, axs = plt.subplots(5, 2, figsize=(12, 14), sharex=True, layout='constrained')
axs = axs.ravel()


def phased(ax, key, scale=1, color='k', label=None):
    for ph, ls in PH_LS.items():
        ax.plot(x, scale * R['%s_%s' % (key, ph)], ls=ls, color=color, lw=1.5 if ph == 'all' else 1,
                label=(label + ' ' if label else '') + ph if (label or color == 'k') else None)


ax = axs[0]
ax.plot(x, R.A / 1e3, 'k-', label='area A')
ax.set_ylabel('section area [10$^3$ m$^2$]', fontsize=8)
ax2 = ax.twinx()
ax2.plot(x, R.H, color='0.5', ls='--')
ax2.set_ylabel('mean depth A / width [m]', fontsize=8, color='0.4')

ax = axs[1]
ax.plot(x, R.Qt_rms, 'k-', label='measured: Q - <Q>')
ax.plot(x, R.Qprism_rms, color='tab:red', ls='--', label='prism: dV$_{land}$/dt - <.>')
ax.set_ylabel('rms tidal transport [m$^3$ s$^{-1}$]', fontsize=8)

ax = axs[2]
ax.plot(x, 100 * R.Ut_rms, 'k-', label='measured (Q - <Q>) / A')
ax.plot(x, 100 * R.Uprism_rms, color='tab:red', ls='--', label='prism / A')
ax.set_ylabel('rms section-mean\ntidal velocity [cm s$^{-1}$]', fontsize=8)

ax = axs[3]
phased(ax, 'u95', 100)
ax.set_ylabel('u95 = 95th pctl |u| [cm s$^{-1}$]', fontsize=8)

ax = axs[4]
for k, col, nm in [('Qx', 'k', 'all counterflow Qx'), ('LC', 'tab:blue', 'lateral LC'), ('VC', 'tab:orange', 'layered VC')]:
    phased(ax, k + '_G', 1, col, nm)
ax.set_ylabel('share of gross transport', fontsize=8)

ax = axs[5]
phased(ax, 'Srms', 1)
ax.set_ylabel('rms shear S [s$^{-1}$]', fontsize=8)

ax = axs[6]
phased(ax, 'drho', 1)
ax.set_ylabel(r'$\Delta\sigma_0$ bottom - top [kg m$^{-3}$]', fontsize=8)

ax = axs[7]
phased(ax, 'fcrit', 1)
ax.set_ylabel('share of area with Ri < 1/4', fontsize=8)

ax = axs[8]
phased(ax, 'twolayer', 100)
ax.set_ylabel('two-layer hours [%]', fontsize=8)

ax = axs[9]
ax.fill_between(x, R.div_q25, R.div_q75, color='0.85', label='IQR (all)')
phased(ax, 'div_med', 1)
ax.plot(x, R.H, color='tab:brown', lw=1, label='mean water depth')
ax.invert_yaxis()
ax.set_ylabel('divide depth below\nthe surface, median [m]', fontsize=8)

LET = 'abcdefghij'
for n, ax in enumerate(axs):
    for SN, XI_U in SECT_XI_U.items():
        ax.axvline(x[cols == XI_U][0], color=SCOL[SN], lw=1, ls='-', alpha=0.6, zorder=0)
        if n < 2:
            ax.text(x[cols == XI_U][0], 1.01, SN, color=SCOL[SN], fontsize=7, ha='center', va='bottom', transform=ax.get_xaxis_transform())
    ax.text(0.01, 0.97, '(%s)' % LET[n], transform=ax.transAxes, fontsize=9, va='top')
    ax.grid(True, color='0.9', lw=0.5)
    ax.tick_params(labelsize=7)
    if n in (1, 2, 4, 9) or n == 3:
        ax.legend(fontsize=6.5, frameon=False, loc='best', ncol=2 if n == 4 else 1)
axs[0].invert_xaxis()                     # head (west) at left, mouth at right
for ax in axs[-2:]:
    ax.set_xlabel('distance west of the mouth [km]   (head left, mouth right)', fontsize=8)
fig.suptitle('Penn Cove along-cove flow metrics, every u-face column, 2024-2025 hourly snapshots, %s\n'
             'flood / ebb = sign of mouth Q; + into the cove' % GTAGEX, fontsize=10)
fig.savefig(out_dir / 'column_metrics_alongcove.png', dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(out_dir / 'column_metrics_alongcove.png'))
