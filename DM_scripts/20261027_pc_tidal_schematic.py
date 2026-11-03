"""
Schematic of the tidal flow in Penn Cove, drawn from the real fields so arrow
lengths are true amplitudes (band fits from 20261027_tidal_ellipses_windowed.py,
depth-averaged, wb1_t0_xn11abbur00 2024-2025).

  a  semidiurnal: double-headed flood/ebb arrows along each ellipse's major
     axis, dark = spring, light = neap (same scale)
  b  diurnal: the same, dark = tropic, light = equatorial
  c  residual (time-mean) flow and the inner-cove counter-clockwise cell
  d  the mouth in depth: a snapshot at the moment the surface flows in hardest,
     for M2, K1 and the daily wind cycle S1 (2-year fits by layer from
     20261027_pc_diurnal_S1.py), so opposite flow at depth shows as reversed
     arrows; plus the M2 ellipse rotation near the surface and the bed

Output: LO_output/DM_outs/20261027_pc_tidal_schematic/pc_tidal_schematic.png

  python 20261027_pc_tidal_schematic.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse
from matplotlib.path import Path as MplPath

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
ell_dir = Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses'
B = xr.open_dataset(ell_dir / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_pc_tidal_schematic'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
C2D, C2L = '#0072B2', '#9CC9E8' # semidiurnal dark / light
C1D, C1L = '#D55E00', '#F2B48A' # diurnal dark / light
CRES = '#009E73'
LAG = {'D2': (3.2, 3.0), 'D1': (7.0, 5.4)} # h before HW of max flood, h after HW of max ebb (amplitude-weighted cove mean)

lon, lat = B.lon_rho.values, B.lat_rho.values
plon, plat = pfun.get_plon_plat(lon, lat)
wb = pd.read_pickle(sect_dir / 'wb.p')
pth = MplPath(np.column_stack([wb.x.values, wb.y.values]))
show = (B.mask_rho.values == 1) & pth.contains_points(np.column_stack([lon.ravel(), lat.ravel()])).reshape(lon.shape)
LON_LP = pd.read_pickle(sect_dir / 'pc_lp.p').x.mean()
imouth = np.flatnonzero(lon[0] < LON_LP)[-1]
jm = np.flatnonzero(show[:, imouth])
z0m = np.nanmean(B.z0_rho.values[:, jm, imouth], axis=1)
aa = [plon.min(), plon.max(), plat.min(), plat.max()]
lat0 = 0.5 * (aa[2] + aa[3])
asp = (aa[3] - aa[2]) / ((aa[1] - aa[0]) * np.cos(np.deg2rad(lat0)))
M_LON = 111.32e3 * np.cos(np.deg2rad(lat0))
M_LAT = 111.32e3
dy_m = np.nanmedian(np.diff(lat[:, 0])) * M_LAT
STEP = 3
jj, ii = np.meshgrid(np.arange(1, lon.shape[0], STEP), np.arange(1, lon.shape[1], STEP), indexing='ij')
sel = show[jj, ii]
JJ, II = jj[sel], ii[sel]

def get(vn, band, b):
    return B['bar_' + vn].sel(band=band, bin=b).values

Lmax = max(np.nanmax(get('Lsmaj', bd, b)[JJ, II]) for bd, b in [('D2', 'spring'), ('D1', 'tropic')])
SC = 0.62 * STEP * dy_m / Lmax # metres of half-arrow per (m/s)

def tide_arrows(ax, band, b, color, lw):
    L, th = get('Lsmaj', band, b), np.deg2rad(get('theta', band, b))
    for j, i in zip(JJ, II):
        if not np.isfinite(L[j, i]):
            continue
        h = L[j, i] * SC
        dxl, dyl = h * np.cos(th[j, i]) / M_LON, h * np.sin(th[j, i]) / M_LAT
        ax.add_patch(FancyArrowPatch((lon[j, i] - dxl, lat[j, i] - dyl), (lon[j, i] + dxl, lat[j, i] + dyl),
                                     arrowstyle='<|-|>', mutation_scale=7, color=color, lw=lw, zorder=4, shrinkA=0, shrinkB=0))

def base(ax):
    ax.pcolormesh(plon, plat, np.where(show, 1.0, np.nan), cmap='Greys', vmin=0, vmax=12, shading='flat', zorder=0)
    pfun.add_coast(ax, color='0.4', linewidth=0.8)
    ax.axis(aa)
    pfun.dar(ax)
    ax.set_xticks([])
    ax.set_yticks([])

def ref_arrow(ax, cms, label):
    h = cms / 100 * SC
    x0, y0 = aa[0] + 0.06 * (aa[1] - aa[0]), aa[2] + 0.07 * (aa[3] - aa[2])
    ax.add_patch(FancyArrowPatch((x0, y0), (x0 + 2 * h / M_LON, y0), arrowstyle='<|-|>', mutation_scale=7, color='k', lw=1.2))
    ax.text(x0 + h / M_LON, y0 + 0.025 * (aa[3] - aa[2]), label, ha='center', va='bottom', fontsize=8)

def flood_ebb_labels(ax, band, col):
    xm, ym = lon[0, imouth] + 0.002, 48.2445
    ax.annotate('', xy=(xm - 0.012, ym), xytext=(xm, ym), arrowprops=dict(arrowstyle='-|>', color=col, lw=2.5))
    ax.text(xm - 0.006, ym + 0.0015, 'FLOOD (in)\n%.0f h before HW' % LAG[band][0], ha='center', va='bottom', fontsize=8, color=col, fontweight='bold')
    ym2 = 48.2205
    ax.annotate('', xy=(xm, ym2), xytext=(xm - 0.012, ym2), arrowprops=dict(arrowstyle='-|>', color=col, lw=2.5))
    ax.text(xm - 0.006, ym2 - 0.0015, 'EBB (out)\n%.0f h after HW' % LAG[band][1], ha='center', va='top', fontsize=8, color=col, fontweight='bold')

fig = plt.figure(figsize=(15, 2 * 7.2 * asp + 1.5), layout='constrained')
axd = fig.subplot_mosaic([['a', 'b'], ['c', 'd']])

ax = axd['a']
base(ax)
tide_arrows(ax, 'D2', 'spring', C2D, 1.6)
tide_arrows(ax, 'D2', 'neap', C2L, 1.0)
ref_arrow(ax, 5, '5 cm/s peak')
flood_ebb_labels(ax, 'D2', C2D)
ax.set_title('a  Semidiurnal tide (twice a day): back-and-forth along the cove', loc='left', fontsize=11, fontweight='bold')
ax.text(0.01, 0.97, 'dark = spring: current x1.6, keeping pace with the tide (x1.6)\nlight = neap\nstrongest along the north side of the outer cove;\ninner cove and southern lobe barely move\nslack at high and low water',
        transform=ax.transAxes, fontsize=8.5, va='top', bbox=dict(fc='white', ec='0.7', alpha=0.9))

ax = axd['b']
base(ax)
tide_arrows(ax, 'D1', 'tropic', C1D, 1.6)
tide_arrows(ax, 'D1', 'equatorial', C1L, 1.0)
ref_arrow(ax, 5, '5 cm/s peak')
flood_ebb_labels(ax, 'D1', C1D)
ax.set_title('b  Diurnal tide (once a day): reaches into the southern lobe', loc='left', fontsize=11, fontweight='bold')
ax.text(0.01, 0.97, 'dark = tropic (moon far from the equator), light = equatorial\ntide x2.5 but current only x1.8: the diurnal current grows less than its forcing\non tropic days the inner cove is diurnal-dominated (D1 > D2)\narrows turn to follow the lobe: sloshing in and out of it\nthis band also holds the daily wind-driven current (S1), see d',
        transform=ax.transAxes, fontsize=8.5, va='top', bbox=dict(fc='white', ec='0.7', alpha=0.9))

ax = axd['c']
base(ax)
u, v = B.bar_umean.sel(bin='all').values, B.bar_vmean.sel(bin='all').values
j2, i2 = np.meshgrid(np.arange(0, lon.shape[0], 2), np.arange(0, lon.shape[1], 2), indexing='ij')
s2 = show[j2, i2] & np.isfinite(u[j2, i2])
q = ax.quiver(lon[j2, i2][s2], lat[j2, i2][s2], u[j2, i2][s2], v[j2, i2][s2], color=CRES, scale=0.03 / 0.35, scale_units='inches', width=0.004, zorder=4)
ax.quiverkey(q, 0.12, 0.07, 0.02, '2 cm/s', labelpos='N', coordinates='axes', fontproperties={'size': 8})
xc, yc, r = -122.708, 48.226, 650.0
ax.add_patch(Ellipse((xc, yc), 2 * r / M_LON, 2 * r / M_LAT, fill=False, ec=CRES, lw=2, ls='--', zorder=5))
ax.annotate('', xy=(xc - r / M_LON * 0.05, yc - r / M_LAT), xytext=(xc - r / M_LON * 0.4, yc - r / M_LAT * 0.92),
            arrowprops=dict(arrowstyle='-|>', color=CRES, lw=2.5), zorder=6)
ax.text(xc, yc + r / M_LAT + 0.0006, 'counter-clockwise cell', ha='center', fontsize=8.5, color=CRES, fontweight='bold')
ax.text(-122.672, 48.2385, 'IN along the north side', fontsize=8.5, color=CRES, fontweight='bold', ha='center', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1), zorder=7)
ax.text(-122.672, 48.2238, 'OUT along the south side', fontsize=8.5, color=CRES, fontweight='bold', ha='center', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1), zorder=7)
ax.set_title('c  What the tide leaves behind (2-year mean flow): NOT tidal', loc='left', fontsize=11, fontweight='bold')
ax.text(0.01, 0.97, '~2 cm/s, as large as the tidal currents\nsame pattern at spring and neap (strength +5%)\nset mostly by stratification and the inflow, not the tide',
        transform=ax.transAxes, fontsize=8.5, va='top', bbox=dict(fc='white', ec='0.7', alpha=0.9))

ax = axd['d']
# 2-year constituent fits by layer at the mouth (20261027_pc_diurnal_S1.py): section-mean amplitude and
# amplitude-weighted phase of the eastward-oriented major axis; snapshot = the moment the SURFACE flows in hardest
PR = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261027_pc_diurnal_S1' / 'profiles.csv')
PR = PR[PR.sect == 'mouth']
SNAP = [('M2', 'semidiurnal tide (M2)', C2D, 0.6), ('K1', 'diurnal tide (K1)', C1D, 0.0), ('S1', 'daily wind cycle (S1, solar day)', '#CC79A7', -0.6)]
for con, lab, col, off in SNAP:
    q = PR[PR.con == con]
    A = q.groupby('k').A.mean().values
    g = np.angle((q.A * np.exp(1j * np.deg2rad(q.g))).groupby(q.k).sum().values)
    vin = 100 * A * np.cos(g - g[-1]) # + = into the cove, at the time of the surface's strongest inflow
    for zt in [-0.5, -3, -6, -9, -12, -15, -18, -21]:
        a = np.interp(zt, z0m, vin)
        ax.add_patch(FancyArrowPatch((0, zt + off), (a, zt + off), arrowstyle='-|>', mutation_scale=10, color=col, lw=2.2, shrinkA=0, shrinkB=0))
ax.axvline(0, color='0.4', lw=1)
ax.axhline(0, color='0.4', lw=1)
ax.fill_between([-22, 16], -23, z0m.min() - 0.3, color='0.85')
ax.set_xlim(-22, 16)
ax.set_ylim(-23, 4.5)
ax.set_xlabel('<- OUT of the cove          current [cm/s]          INTO the cove ->')
ax.set_ylabel('depth [m]')
for n, (con, lab, col, off) in enumerate(SNAP):
    ax.text(-21.5, 3.7 - 1.15 * n, lab, color=col, fontsize=9, fontweight='bold', va='top')
def rot_icon(x, y, ccw, col, txt, above=False):
    ax.add_patch(Ellipse((x, y), 3.0, 1.0, fill=False, ec=col, lw=1.8))
    sgn = 1 if ccw else -1
    ax.annotate('', xy=(x + 0.2 * sgn * -1, y + 0.5), xytext=(x + 0.9 * sgn, y + 0.42), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8))
    ax.text(x - 4.5, y + (1.0 if above else -1.3), txt, va='bottom' if above else 'top', fontsize=8.5)
rot_icon(11.0, -4.0, False, '0.2', 'M2 surface: clockwise in summer\n(stratified), counter-cw in winter')
rot_icon(11.0, -20.5, True, '0.2', 'M2 near bed: counter-clockwise\nall year (bottom friction)', above=True)
ax.set_title('d  At the mouth: the moment the surface flows IN hardest', loc='left', fontsize=11, fontweight='bold')
ax.text(0.01, 0.62, 'semidiurnal: the whole column moves together,\nweaker toward the bed\ndiurnal tide and daily wind: the surface and the\ndeep water flow OPPOSITE ways, crossing near 10 m\nthe daily wind cycle drives the biggest once-a-day\ncurrent at the surface (8 cm/s)',
        transform=ax.transAxes, fontsize=8.5, va='top', bbox=dict(fc='white', ec='0.7', alpha=0.9))
ax.grid(color='lightgray', linestyle='--', alpha=0.5)

fig.suptitle('Tidal flow in Penn Cove (model %s, 2024-2025): arrows are true peak currents' % GTX, fontsize=12)
fn = out_dir / 'pc_tidal_schematic.png'
fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved ' + str(fn))
