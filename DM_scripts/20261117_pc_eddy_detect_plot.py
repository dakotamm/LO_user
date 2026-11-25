"""
Figures for 20261117_pc_eddy_detect.py (eddy WHEN / WHERE / HOW STRONG, per hour and 2 m depth bin).

when.png      (a) daily share of hours present (in a kept event) by depth through the record, (b) daily median U_max of
              the present hours, (c) monthly presence by depth (both years pooled)
where.png     one map per depth bin: record-mean zeta / f, centres of the present hours (density), the median centre
              and a circle of the median r_max
strength.png  (a) median u_th(r) of the present hours, one line per bin, (b-e) by depth: % present, U_max, r_max,
              zeta_c / f (median and 25-75% of the present hours)

INPUT  LO_output/DM_outs/20261117_pc_eddy_detect/eddy_hourly.nc
OUTPUT LO_output/DM_outs/20261117_pc_eddy_detect/*.png

run 20261117_pc_eddy_detect_plot.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

Ldir = Lfun.Lstart()
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_eddy_detect'
d = xr.open_dataset(out_dir / 'eddy_hourly.nc')
E = d.attrs['edges_m']
zc = d.z.values
ok = np.isfinite(d.present.values).any(axis=0)                     # bins with an inner-cove search region
zc, E = zc[ok], E[:ok.sum() + 1]
d = d.isel(z=np.flatnonzero(ok))
tt = pd.DatetimeIndex(d.time.values)
pe = d.present_ev.values == 1                                       # (t, z)
GT = d.attrs['gtagex']
plt.rcParams.update({'font.size': 12})
grid = dict(color='lightgray', linestyle='--', alpha=0.5)


def save(fig, name):
    fig.savefig(out_dir / name, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % (out_dir / name))


# ------------------------------------------------------------------ when.png
P = pd.DataFrame(pe.astype(float), index=tt)
Pd = P.resample('D').mean()
Um = pd.DataFrame(np.where(pe, 100 * d.U_max.values, np.nan), index=tt).resample('D').median()
Pm = 100 * P.groupby(tt.month).mean()
fig = plt.figure(figsize=(18, 13), layout='constrained')
gs = fig.add_gridspec(3, 2, width_ratios=[4, 1], height_ratios=[1, 1, 1])
ax = fig.add_subplot(gs[0, :])
im = ax.pcolormesh(Pd.index, zc, 100 * Pd.values.T, cmap='Greens', vmin=0, vmax=100, shading='nearest')
fig.colorbar(im, ax=ax, label='% of the day present')
ax.set_title('(a) When: share of each day in a kept eddy event', loc='left', fontweight='bold')
ax2 = fig.add_subplot(gs[1, :], sharex=ax)
im = ax2.pcolormesh(Um.index, zc, Um.values.T, cmap='Purples', vmin=0, vmax=np.nanpercentile(Um.values, 98), shading='nearest')
fig.colorbar(im, ax=ax2, label='U$_{max}$ [cm s$^{-1}$]')
ax2.set_title('(b) How strong: daily median U$_{max}$ (swirl speed at r$_{max}$) of the present hours', loc='left', fontweight='bold')
ax3 = fig.add_subplot(gs[2, 0])
im = ax3.pcolormesh(Pm.index, zc, Pm.values.T, cmap='Greens', vmin=0, vmax=100, shading='nearest')
for m in Pm.index:
    for j, z in enumerate(zc):
        ax3.text(m, z, '%.0f' % Pm.loc[m, j], ha='center', va='center', fontsize=8,
                 color='w' if Pm.loc[m, j] > 60 else 'k')
ax3.set_xticks(range(1, 13), ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
fig.colorbar(im, ax=ax3, label='% of hours present')
ax3.set_title('(c) Monthly presence by depth (2024-25 pooled)', loc='left', fontweight='bold')
ax4 = fig.add_subplot(gs[2, 1])
ax4.barh(zc, 100 * pe.mean(axis=0), height=0.8 * np.diff(E), color='#228833')
ax4.set_xlabel('% of hours present')
ax4.set_title('(d) All hours', loc='left', fontweight='bold')
ax4.grid(**grid)
for a in [ax, ax2, ax3, ax4]:
    a.set_ylim(E[-1], 0)
    a.set_ylabel('depth below the surface [m]')
fig.suptitle('Penn Cove inner-cove eddy (counterclockwise), %s, Godin-filtered, 2 m depth bins' % GT, fontsize=14)
save(fig, 'when.png')

# ------------------------------------------------------------------ where.png
nb = len(zc)
ncol = 4
nrow = int(np.ceil(nb / ncol))
zf = d.zeta_mean.values / d.f_psi.values[None]
lim = np.nanpercentile(np.abs(zf), 98)
lon_r, lat_r = d.lon_rho.values, d.lat_rho.values
fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow), layout='constrained', squeeze=False)
for j in range(nb):
    ax = axs.flat[j]
    ax.pcolormesh(lon_r, lat_r, np.where(d.mask_bin.values[j] == 1, np.nan, 0.0), cmap='Greys', vmin=-1, vmax=1.5, shading='nearest')
    im = ax.pcolormesh(d.lon_psi, d.lat_psi, zf[j], cmap='RdBu_r', vmin=-lim, vmax=lim, shading='nearest')
    p = pe[:, j]
    if p.any():
        lc, la = d.lon_c.values[p, j], d.lat_c.values[p, j]
        C = pd.DataFrame({'lon': lc, 'lat': la}).value_counts(normalize=True)
        ax.scatter(C.index.get_level_values('lon'), C.index.get_level_values('lat'), s=2000 * C.values, facecolor='none',
                   edgecolor='k', lw=1)
        lo0, la0, rm = np.median(lc), np.median(la), np.median(d.r_max.values[p, j])
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(lo0 + rm * np.cos(th) / (111.32e3 * np.cos(np.deg2rad(la0))), la0 + rm * np.sin(th) / 111.32e3, '-',
                color='#EE7733', lw=2)
        ax.plot(lo0, la0, '+', color='#EE7733', ms=12, mew=2)
    ax.axvline(d.attrs['lon_pc_cp'], color='k', ls='--', lw=0.8)
    ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
    ax.set_xlim(lon_r.min(), d.attrs['lon_pc_cp'] + 0.01)
    ax.set_title('%.0f-%.0f m: present %.0f%% of hours' % (E[j], E[j + 1], 100 * p.mean()), fontsize=11)
    ax.tick_params(labelsize=8)
for ax in axs.flat[nb:]:
    ax.set_visible(False)
fig.colorbar(im, ax=axs, label='record-mean $\\zeta$ / f', shrink=0.6)
fig.suptitle('Where: record-mean $\\zeta$/f, centres of the present hours (circle area = share of hours), '
             'median centre and median r$_{max}$ (orange); dashed = pc_cp', fontsize=13)
save(fig, 'where.png')

# ------------------------------------------------------------------ strength.png
fig, axs = plt.subplots(1, 5, figsize=(22, 7), layout='constrained', gridspec_kw={'width_ratios': [1.6, 1, 1, 1, 1]})
cm = plt.get_cmap('viridis')
for j in range(nb):
    p = pe[:, j]
    if p.sum() > 0:
        axs[0].plot(d.r.values, 100 * np.median(d.u_th.values[p, j], axis=0), '-o', color=cm(j / max(nb - 1, 1)), lw=2, ms=5,
                    label='%.0f-%.0f m' % (E[j], E[j + 1]))
axs[0].axhline(0, color='0.4', lw=0.8)
axs[0].set_xlabel('distance from the centre [m]')
axs[0].set_ylabel('u$_\\theta$ [cm s$^{-1}$] (+ = counterclockwise)')
axs[0].set_title('(a) Swirl profile, median of present hours', loc='left', fontweight='bold')
axs[0].legend(fontsize=9)
axs[0].grid(**grid)
axs[1].barh(zc, 100 * pe.mean(axis=0), height=0.8 * np.diff(E), color='#228833')
axs[1].set_xlabel('% of hours present')
axs[1].set_title('(b) Presence', loc='left', fontweight='bold')
for ax, vn, sc, lab, tl in [(axs[2], 'U_max', 100, 'U$_{max}$ [cm s$^{-1}$]', '(c) Strength'),
                            (axs[3], 'r_max', 1, 'r$_{max}$ [m]', '(d) Radius'),
                            (axs[4], 'zeta_c_f', 1, '$\\zeta_c$ / f (disk mean at the centre)', '(e) Rossby number')]:
    x = np.where(pe, sc * d[vn].values, np.nan)
    q = np.nanpercentile(x, [25, 50, 75], axis=0)
    ax.fill_betweenx(zc, q[0], q[2], color='#0077BB', alpha=0.25, lw=0)
    ax.plot(q[1], zc, '-o', color='#0077BB', lw=2, ms=5)
    ax.set_xlabel(lab)
    ax.set_title(tl, loc='left', fontweight='bold')
for ax in axs[1:]:
    ax.set_ylim(E[-1], 0)
    ax.grid(**grid)
axs[1].set_ylabel('depth below the surface [m]')
fig.suptitle('How strong and where vertically: present hours only (median, 25-75%%), %s' % GT, fontsize=13)
save(fig, 'strength.png')
