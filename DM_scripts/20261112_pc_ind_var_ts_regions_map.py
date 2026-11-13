"""
Map of the two regions 20261112_pc_ind_var_ts.py averages wind stress over:
    Penn Cove        STEP 1: pc_cove_mix box, rho columns xi_rho <= 30, water
    Saratoga Passage STEP 2: tef2 segment pc_lp_p (between pc_lp, sp_mid, skagit_sp)
plus the bounding section lines and the sp_stress box outline.

Output: LO_output/DM_outs/20261112_pc_ind_var_ts/regions_map.png

run 20261112_pc_ind_var_ts_regions_map.py
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SP_AA = [-122.668235, -122.465294, 48.180630, 48.303326]      # sp_stress box
C_PC, C_SP = '#2a78d6', '#eb6834'

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
out_dir = Ldir['LOo'] / 'DM_outs' / '20261112_pc_ind_var_ts'
Lfun.make_dir(out_dir)

G = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = G.lon_rho.values
lat = G.lat_rho.values
water = G.mask_rho.values == 1

# Saratoga: segment cells on the full grid
ji = np.array(pickle.load(open(Ldir['LOo'] / 'extract' / 'tef2' / 'seg_info_dict_wb1_pc1_trapsN00.p', 'rb'))['pc_lp_p']['ji_list'])
sp = np.zeros(water.shape, dtype=bool)
sp[ji[:, 0], ji[:, 1]] = True

# Penn Cove: pc_cove_mix box, xi_rho <= 30, placed on the full grid by the box corner
dm = xr.open_dataset(Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_mix_' + DS0 + '_' + DS1 + '.nc'))
i0 = np.argmin(np.abs(lon[0, :] - dm.lon_rho.values[0, 0]))
j0 = np.argmin(np.abs(lat[:, 0] - dm.lat_rho.values[0, 0]))
cove_b = dm.mask_rho.values == 1
cove_b[:, 31:] = False
pc = np.zeros(water.shape, dtype=bool)
pc[j0:j0 + cove_b.shape[0], i0:i0 + cove_b.shape[1]] = cove_b

# 0 land, 1 other water, 2 Penn Cove, 3 Saratoga
cls = np.where(water, 1, 0)
cls[pc] = 2
cls[sp] = 3
cmap = ListedColormap(['#d9d8d2', '#f4f3ee', C_PC, C_SP])

fig, ax = plt.subplots(figsize=(8, 7))
ax.pcolormesh(G.lon_psi.values, G.lat_psi.values, cls[1:-1, 1:-1], cmap=cmap, vmin=-0.5, vmax=3.5)

for sn in ['pc_lp', 'sp_mid', 'skagit_sp']:
    df = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (sn + '.p'))
    ax.plot(df.x, df.y, color='k', lw=2)
    ax.text(df.x.values[-1], df.y.values[-1], ' ' + sn, fontsize=9, va='center')

ax.add_patch(Rectangle((SP_AA[0], SP_AA[2]), SP_AA[1] - SP_AA[0], SP_AA[3] - SP_AA[2],
                       fill=False, ls='--', lw=1, ec='0.3'))

ax.legend(handles=[Patch(fc=C_PC, label='Penn Cove (STEP 1): %d cells' % pc.sum()),
                   Patch(fc=C_SP, label='Saratoga Passage = pc_lp_p (STEP 2): %d cells' % sp.sum()),
                   Patch(fc='none', ec='0.3', ls='--', label='sp_stress box')],
          loc='lower left', fontsize=9)

ax.set_xlim(-122.76, -122.42)
ax.set_ylim(48.16, 48.33)
ax.set_aspect(1 / np.cos(np.radians(48.25)))
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('Wind stress averaging regions, wb1')

fig.savefig(out_dir / 'regions_map.png', dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved', out_dir / 'regions_map.png')
