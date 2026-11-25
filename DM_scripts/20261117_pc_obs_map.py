"""
Map of the observations used in 20261117_pc_runs_timeseries.py, the model points they are compared with, and the other
stations nearby that are not used. Markers match the time series figure.
LEFT   Whidbey Basin: + Ecology core stations (2012-2025 in ecology_nc) and every KC Whidbey Basin station.
RIGHT  Penn Cove and Saratoga Passage off the mouth, cas7 bathymetry, wb1 coastline, cas7 cove cells used by the
       cove script (red cell outlines). Labels inside the zoom box are drawn only on the right.
POSITIONS  mean cast position per station from LO_output/obs (kc_whidbeyBasin/ctd 2022-2025, ecology_nc/ctd 2012-2025,
       nceiSalish/bottle_ctd 2014-2024 within 1 km of WOAC station 4); raft from the raft_main.nc attributes.
OUT    LO_output/DM_outs/20261117_pc_obs_map/pc_obs_map.png
    python 20261117_pc_obs_map.py
"""

import pickle
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='cas7', tag='t1', ex_name='x11ab')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_obs_map'
Lfun.make_dir(out_dir)
obs = Ldir['LOo'] / 'obs'

MODEL_PTS = {'pc_west': (-122.70878, 48.22710), 'M5': (-122.582462, 48.245981)}
USED = {'PENNCOVEWEST', 'SARATOGARP'}
AX_WIDE = [-122.80, -122.25, 47.98, 48.36]
AX_ZOOM = [-122.75, -122.53, 48.205, 48.265]


def read(src, otype, years):
    fns = [obs / src / otype / f'{y}.p' for y in years]
    return pd.concat([pd.read_pickle(f) for f in fns if f.exists()])


def station_means(d):
    return d.groupby('name')[['lon', 'lat']].mean()


kc = station_means(read('kc_whidbeyBasin', 'ctd', range(2022, 2026)))
ec = station_means(read('ecology_nc', 'ctd', range(2012, 2026)))
wo = read('nceiSalish', 'bottle_ctd', range(2014, 2025))
wo = wo[np.hypot((wo['lon'] + 122.554) * np.cos(np.deg2rad(48.242)) * 111, (wo['lat'] - 48.242) * 111) < 1]
wo_pos = (wo['lon'].mean(), wo['lat'].mean())
with xr.open_dataset(obs / 'pcRaft' / 'moor' / 'raft_main.nc') as rf:
    raft_pos = (float(rf.attrs['lon']), float(rf.attrs['lat']))

# ---- cas7 grid, cove cells (same rule as the cove script), wb1 coast ----
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, mask, h = g.lon_rho.values, g.lat_rho.values, g.mask_rho.values, g.h.values
lonp, latp = g.lon_psi.values, g.lat_psi.values
g.close()
wet = (mask == 1) & (lon < -122.652) & (lat > 48.20) & (lat < 48.26)
lab, _ = label(wet)
js, is_ = np.unravel_index(np.argmin((lon - MODEL_PTS['pc_west'][0])**2 + (lat - MODEL_PTS['pc_west'][1])**2),
                           lon.shape)
cove = lab == lab[js, is_]
d = pickle.load(open(Ldir['LOo'] / 'DM_outs' / '20260806_hypoxia' /
                     'hypoxia_lowpass_wb1_t0_xn11abbur00_2024.01.01_2025.12.31.p', 'rb'))

# label offsets (dlon, dlat, ha, va) in the zoom panel, where labels would collide
OFF = {'KC PENNCOVEWEST': (-0.003, 0.0, 'right', 'center'), 'mussel raft': (0.002, -0.003, 'left', 'top'),
       'pc_west': (0.0, 0.003, 'center', 'bottom'), 'M5': (0.0, 0.003, 'center', 'bottom'),
       'WOAC stn 4': (0.004, 0.003, 'left', 'bottom'), 'KC SARATOGARP': (0.0, -0.003, 'center', 'top')}
inside = lambda x, y, A: A[0] < x < A[1] and A[2] < y < A[3]


def label_pt(ax, x, y, txt, zoom, dflt, **kw):
    """Label a point; in the overview, skip anything inside the zoom box (it is labelled on the right)."""
    if not zoom and inside(x, y, AX_ZOOM):
        return
    dx, dy, ha, va = OFF.get(txt, dflt) if zoom else dflt
    ax.text(x + dx, y + dy, txt, ha=ha, va=va, **kw)


fig, axs = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={'width_ratios': [1, 1.35]})
for ax, AX in zip(axs, [AX_WIDE, AX_ZOOM]):
    zoom = AX is AX_ZOOM
    sel = (lon[0] > AX[0] - 0.05) & (lon[0] < AX[1] + 0.05)
    selj = (lat[:, 0] > AX[2] - 0.05) & (lat[:, 0] < AX[3] + 0.05)
    J, I = np.where(selj)[0], np.where(sel)[0]
    jj, ii = slice(J.min(), J.max()), slice(I.min(), I.max())
    cs = ax.pcolormesh(lonp[jj, ii], latp[jj, ii], np.where(mask == 1, h, np.nan)[jj, ii][1:, 1:],
                       cmap='Blues', vmin=0, vmax=200 if AX is AX_WIDE else 60)
    fig.colorbar(cs, ax=ax, label='cas7 h [m]', shrink=0.8)
    ax.contour(d['lon'], d['lat'], d['water'].astype(float), [0.5], colors='0.3', linewidths=0.5)
    if zoom:   # each cas7 cove cell outlined on its psi corners
        for j, i in zip(*np.where(cove)):
            ax.plot([lonp[j-1, i-1], lonp[j-1, i], lonp[j, i], lonp[j, i-1], lonp[j-1, i-1]],
                    [latp[j-1, i-1], latp[j-1, i], latp[j, i], latp[j, i-1], latp[j-1, i-1]], '-', color='tab:red',
                    lw=0.6, alpha=0.7)
    ax.axis(AX)
    ax.set_aspect(1 / np.cos(np.deg2rad(48.2)))
    ax.set_xlabel('Longitude')

    # not used: other KC Whidbey stations, Ecology stations
    for nm, r in kc.iterrows():
        if nm not in USED and AX[0] < r.lon < AX[1] and AX[2] < r.lat < AX[3]:
            ax.plot(r.lon, r.lat, 'o', mfc='none', mec='0.4', ms=7)
            label_pt(ax, r.lon, r.lat, f'KC {nm}', zoom, (0, 0.003, 'center', 'bottom'), fontsize=7, color='0.35')
    for nm, r in ec.iterrows():
        if AX[0] < r.lon < AX[1] and AX[2] < r.lat < AX[3]:
            ax.plot(r.lon, r.lat, 'D', mfc='none', mec='0.4', ms=6)
            label_pt(ax, r.lon, r.lat, f'Ecology {nm}', zoom, (0, 0.006, 'center', 'bottom'), fontsize=7,
                     color='0.35')
    # used
    for nm in USED:
        r = kc.loc[nm]
        ax.plot(r.lon, r.lat, 'o', color='k', ms=7, zorder=6)
        label_pt(ax, r.lon, r.lat, f'KC {nm}', zoom, (0, -0.003, 'center', 'top'), fontsize=8, fontweight='bold')
    ax.plot(*wo_pos, 's', mfc='none', mec='k', mew=1.5, ms=8, zorder=6)
    label_pt(ax, *wo_pos, 'WOAC stn 4', zoom, (0.004, 0.003, 'left', 'bottom'), fontsize=8, fontweight='bold')
    ax.plot(*raft_pos, 'o', color='0.45', ms=6, zorder=6)
    label_pt(ax, *raft_pos, 'mussel raft', zoom, (0, -0.003, 'center', 'top'), fontsize=8, fontweight='bold',
             color='0.3')
    for nm, (x, y) in MODEL_PTS.items():
        ax.plot(x, y, '*', color='gold', mec='k', ms=15, zorder=7)
        label_pt(ax, x, y, nm, zoom, (0, 0.003, 'center', 'bottom'), fontsize=9, fontweight='bold',
                 color='darkgoldenrod')
axs[0].plot([AX_ZOOM[0], AX_ZOOM[1], AX_ZOOM[1], AX_ZOOM[0], AX_ZOOM[0]],
            [AX_ZOOM[2], AX_ZOOM[2], AX_ZOOM[3], AX_ZOOM[3], AX_ZOOM[2]], '-', color='k', lw=1)
axs[0].set_ylabel('Latitude')
axs[0].set_title('Whidbey Basin (box = right panel)', fontsize=10)
axs[1].set_title('Penn Cove and Saratoga Passage; red = cas7 cove cells', fontsize=10)
h_ = [plt.Line2D([], [], marker='*', color='gold', mec='k', ms=12, ls='', label='model points'),
      plt.Line2D([], [], marker='o', color='k', ms=7, ls='', label='KC casts, used'),
      plt.Line2D([], [], marker='s', mfc='none', mec='k', mew=1.5, ms=8, ls='', label='WOAC bottles, used'),
      plt.Line2D([], [], marker='o', color='0.45', ms=6, ls='', label='raft sondes 1 & 7 m, used'),
      plt.Line2D([], [], marker='o', mfc='none', mec='0.4', ms=7, ls='', label='KC, not used'),
      plt.Line2D([], [], marker='D', mfc='none', mec='0.4', ms=6, ls='', label='Ecology, not used'),
      plt.Line2D([], [], color='tab:red', lw=1, label='cas7 cove cells (cove script)')]
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.legend(handles=h_, loc='lower center', ncol=7, fontsize=9, frameon=False)
fig_fn = out_dir / 'pc_obs_map.png'
fig.savefig(fig_fn, dpi=200, transparent=True)
plt.close(fig)
print(f'saved: {fig_fn}')
print('KC:', kc.round(4).to_dict('index'))
print('WOAC 4:', np.round(wo_pos, 4), ' raft:', raft_pos)
