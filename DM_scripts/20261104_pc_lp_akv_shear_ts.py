"""
Time series of the eddy viscosity and the vertical shear at the Penn Cove mouth
section pc_lp, wb1_t0_xn11abbur00, hourly 2024-2025.

EDDY VISCOSITY  AKv [m2/s] from the GLS closure (ocean_his snapshots, pc_cove_mix box),
    on the w-levels at rho points; put on the section's u-faces as the mean of the two
    rho cells either side. Interior w-level k (1..N-1) sits between cell centres k-1 and k:
    the same layer as the shear there, so the two are averaged over the same layers.
    Each hour: area-weighted section mean over the interior w-levels, weight = dy * (z_rho[k]
    - z_rho[k-1]). The bed and surface w-levels (boundary values) are left out, as the shear is.
SHEAR  Srms [1/s], the hourly rms full shear sqrt((du/dz)^2 + (dv/dz)^2) over the same
    layers, from 20261104_pc_flow_metrics.py (flow_metrics.nc, sect = pc_lp).

PANELS  a, b: the two years, hourly (grey) and daily mean (black); AKv on a log axis.
        c: a 15-day zoom (ZOOM), hourly, AKv (left, log) and shear (right) on the same axes;
        d: the section transport Q under it, for tidal phase.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_mix_[ds0]_[ds1].nc (AKv)
       LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (z_rho, for the layer weights)
       LO_output/DM_outs/20261104_pc_flow_metrics/flow_metrics.nc (Srms, Q)
OUTPUT LO_output/DM_outs/20261104_pc_lp_akv_shear_ts/
    pc_lp_akv_shear_ts.png
    pc_lp_akv.npz          the hourly section-mean AKv (cached; delete to recompute)

run 20261104_pc_lp_akv_shear_ts.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
XI_U, ETA = 30, np.arange(19, 7, -1)      # pc_lp u-faces in the box (global i 67, j 228..217), north to south
IRM, IRP = XI_U, XI_U + 1                 # rho cells either side of the face
ZOOM = ('2024-08-01', '2024-08-16')
BLK = 744

Ldir = Lfun.Lstart(gridname='wb1')
box = Ldir['LOo'] / 'extract' / GTAGEX / 'box'
mix_fn = box / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
vel_fn = box / ('pc_cove_%s_%s.nc' % (DS0, DS1))
fm_fn = Ldir['LOo'] / 'DM_outs' / '20261104_pc_flow_metrics' / 'flow_metrics.nc'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261104_pc_lp_akv_shear_ts'
out_dir.mkdir(parents=True, exist_ok=True)
cache_fn = out_dir / 'pc_lp_akv.npz'

# ------------------------------------------------------------------ hourly section-mean AKv (cached)
if cache_fn.exists():
    z = np.load(cache_fn)
    T, akv = pd.DatetimeIndex(z['t']), z['akv']
else:
    mx, vb = xr.open_dataset(mix_fn), xr.open_dataset(vel_fn)
    T = pd.DatetimeIndex(mx.ocean_time.values)
    assert np.array_equal(mx.ocean_time.values, vb.ocean_time.values)
    dy = (1 / vb.pn.values[ETA][:, [IRM, IRP]]).mean(axis=-1)            # (p,) face width
    akv = np.full(len(T), np.nan)
    for t0 in range(0, len(T), BLK):
        tt = slice(t0, min(t0 + BLK, len(T)))
        A = mx.AKv.isel(ocean_time=tt, eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().astype(float).mean(axis=-1)   # (t, w, p)
        zr = vb.z_rho.isel(ocean_time=tt, eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().mean(axis=-1)              # (t, k, p)
        w = np.diff(zr, axis=1) * dy[None, None, :]                    # (t, k-1, p) layer area between cell centres
        akv[tt] = (A[:, 1:-1, :] * w).sum(axis=(1, 2)) / w.sum(axis=(1, 2))   # interior w-levels 1..N-1
        print('  hours %d-%d of %d' % (tt.start, tt.stop, len(T)))
    mx.close(), vb.close()
    np.savez(cache_fn, t=T.values, akv=akv)

fm = xr.open_dataset(fm_fn).sel(sect='pc_lp')
assert np.array_equal(fm.time.values, T.values)
S, Q = fm.Srms.values, fm.Q.values
day = T.floor('D')
dA = pd.Series(akv, index=T).groupby(day).mean()
dS = pd.Series(S, index=T).groupby(day).mean()
print('AKv: median %.1e, 5-95%% %.1e-%.1e m2/s; Srms median %.3f 1/s; corr(daily log AKv, daily Srms) %.2f'
      % (np.median(akv), *np.percentile(akv, [5, 95]), np.median(S), np.corrcoef(np.log10(dA), dS)[0, 1]))

# ------------------------------------------------------------------ plot
fig = plt.figure(figsize=(14, 11), layout='constrained')
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.4, 0.5])
ax = fig.add_subplot(gs[0])
ax.plot(T, akv, color='0.7', lw=0.3, label='hourly')
ax.plot(dA.index, dA, color='k', lw=1, label='daily mean')
ax.set_yscale('log')
ax.set_ylabel('AKv [m$^2$ s$^{-1}$]')
ax.set_title('a  eddy viscosity AKv (GLS), pc_lp section mean over the interior w-levels', fontsize=10, loc='left')
ax.axvspan(pd.Timestamp(ZOOM[0]), pd.Timestamp(ZOOM[1]), color='#F7DFB0', lw=0, zorder=0, label='zoom (c, d)')   # opaque: alpha turns solid on the transparent background
ax.legend(fontsize=8, loc='upper left', ncol=3)
ax1 = fig.add_subplot(gs[1], sharex=ax)
ax1.plot(T, 1000 * S, color='0.7', lw=0.3)
ax1.plot(dS.index, 1000 * dS, color='k', lw=1)
ax1.axvspan(pd.Timestamp(ZOOM[0]), pd.Timestamp(ZOOM[1]), color='#F7DFB0', lw=0, zorder=0)
ax1.set_ylabel('rms shear [10$^{-3}$ s$^{-1}$]')
ax1.set_title('b  rms full shear (du/dz, dv/dz), pc_lp, same layers', fontsize=10, loc='left')
ax1.set_xlim(T[0], T[-1])
z = (T >= ZOOM[0]) & (T < ZOOM[1])
CA, CS = 'k', '#0072B2'                   # AKv, shear
ax2 = fig.add_subplot(gs[2])
ax2.plot(T[z], akv[z], color=CA, lw=1.2, label='AKv (left, log)')
ax2.set_yscale('log')
ax2.set_ylabel('AKv [m$^2$ s$^{-1}$]', color=CA)
ax2.set_title('c  AKv and rms shear on the same axes, hourly, %s to %s' % ZOOM, fontsize=10, loc='left')
t2 = ax2.twinx()
t2.plot(T[z], 1000 * S[z], color=CS, lw=1.2, label='rms shear (right)')
t2.set_ylabel('rms shear [10$^{-3}$ s$^{-1}$]', color=CS)
t2.tick_params(axis='y', labelcolor=CS)
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = t2.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left', ncol=2)
ax3 = fig.add_subplot(gs[3], sharex=ax2)
ax3.fill_between(T[z], 0, Q[z], where=Q[z] > 0, color='#E69F00', lw=0, label='flood (into the cove)')
ax3.fill_between(T[z], 0, Q[z], where=Q[z] < 0, color='#56B4E9', lw=0, label='ebb')
ax3.axhline(0, color='k', lw=0.5)
ax3.set_ylabel('Q [m$^3$ s$^{-1}$]', fontsize=8)
ax3.set_title('d  section transport at pc_lp, + into the cove', fontsize=10, loc='left')
ax3.legend(fontsize=7, loc='upper right', ncol=2)
for a in [ax, ax1, ax2, ax3]:
    a.grid(True, alpha=0.3)
fig.suptitle('Eddy viscosity and shear at the Penn Cove mouth (pc_lp), hourly snapshots, %s, %s to %s' % (GTAGEX, DS0, DS1), fontsize=11)
fig.savefig(out_dir / 'pc_lp_akv_shear_ts.png', dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(out_dir / 'pc_lp_akv_shear_ts.png'))
