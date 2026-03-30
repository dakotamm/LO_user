"""
Diagnostic: check whether the section faces fully bound the segment volume.

For each cell (j,i) in the segment, compute the net Huon/Hvom flux through
all four faces. Sum these to get the true dV/dt from ROMS. Then compare to:
(a) the section-only flux (from the extraction)
(b) the volume change from history files

This tells us EXACTLY where the missing flux is.

To run:
run budget_diagnostic2 -gtx cas7_trapsV00_meV00 -ctag c0 -riv trapsV00 -0 2017.01.01 -1 2017.01.22

"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
import tef_fun

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

from lo_tools import extract_argfun as exfun
Ldir = exfun.intro()

from tef2_avg_fun import get_avg_fn_list

sect_gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
riv_gctag = Ldir['gridname'] + '_' + Ldir['riv']
date_str = '_' + Ldir['ds0'] + '_' + Ldir['ds1']

# get budget_functions
pth = Ldir['LO'] / 'extract' / 'tef2'
upth = Ldir['LOu'] / 'extract' / 'tef2'
if (upth / 'budget_functions.py').is_file():
    bfun = Lfun.module_from_file('budget_functions', upth / 'budget_functions.py')
else:
    bfun = Lfun.module_from_file('budget_functions', pth / 'budget_functions.py')

which_vol = 'Penn Cove'

dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'

# Get section/segment info
sntup_list, sect_base_list, outer_sns_list = bfun.get_sntup_list(sect_gctag, which_vol)
print('Section tuples:', sntup_list)
print('Outer sns excluded:', outer_sns_list)

dir2 = Ldir['LOo'] / 'extract' / 'tef2'
seg_info_dict_fn = dir2 / ('seg_info_dict_' + sect_gctag + '_' + Ldir['riv'] + '.p')
seg_info_dict = pd.read_pickle(seg_info_dict_fn)

sect_df_fn = dir2 / ('sect_df_' + sect_gctag + '.p')
sect_df = pd.read_pickle(sect_df_fn)
sn_list = list(sect_df.sn)

# find valid segments (same logic as tracer_budget)
sns_list = []
for snb in sect_base_list:
    for sn in sn_list:
        if snb in sn:
            for pm in ['_p','_m']:
                sns = sn + pm
                if (sns not in outer_sns_list) and (sns not in sns_list):
                    sns_list.append(sns)

print('\nValid segment-sides (sns_list):', sns_list)

good_seg_key_list = []
for sk in seg_info_dict.keys():
    this_sns_list = seg_info_dict[sk]['sns_list']
    check_list = [item for item in this_sns_list if item in sns_list]
    if len(check_list) >= 1:
        good_seg_key_list.append(sk)

print('Good segment keys:', good_seg_key_list)

# Collect ALL grid cells in the volume
all_j = []
all_i = []
for sk in good_seg_key_list:
    ji_list = seg_info_dict[sk]['ji_list']
    for ji in ji_list:
        all_j.append(ji[0])
        all_i.append(ji[1])
all_j = np.array(all_j, dtype=int)
all_i = np.array(all_i, dtype=int)
print(f'\nTotal cells in volume: {len(all_j)}')

# Make a 2D mask of the volume cells
# Get grid info from first avg file
fn_list = get_avg_fn_list(Ldir, Ldir['ds0'], Ldir['ds1'])
fn0 = fn_list[0]
G, S, T = zrfun.get_basic_info(fn0)
M, L = G['M'], G['L']

vol_mask = np.zeros((M, L), dtype=bool)
vol_mask[all_j, all_i] = True
print(f'Volume mask shape: {vol_mask.shape}, cells in mask: {vol_mask.sum()}')

# ===== For a single avg file, compute flux through ALL boundary faces =====
# Pick a file in the middle (skip first few for spinup)
test_fn = fn_list[min(12, len(fn_list)-1)]
print(f'\nTest file: {test_fn}')

ds = xr.open_dataset(test_fn)
Huon = ds.Huon.values.squeeze()  # (z, eta_u, xi_u)
Hvom = ds.Hvom.values.squeeze()  # (z, eta_v, xi_v)
NZ = Huon.shape[0]
print(f'Huon shape: {Huon.shape}')
print(f'Hvom shape: {Hvom.shape}')

# For each cell (j,i) in the volume, the net flux is:
#   Huon(j, i+1) - Huon(j, i) + Hvom(j+1, i) - Hvom(j, i)
# where Huon is on the u-grid: Huon(j, i) is the u-face between rho(j,i-1) and rho(j,i)
# So flux INTO rho-cell (j,i) through its east face = Huon(j, i+1)
#                                   through west face = -Huon(j, i)
#                                   through north face = Hvom(j+1, i) -- wait, need to check ROMS convention

# ROMS u-grid indexing: u(j, i) is between rho(j, i-1) and rho(j, i), i.e., the WEST face of rho(j,i)
# So flux INTO rho(j,i) = Huon(j, i) - Huon(j, i+1) IF Huon is positive in the +x direction
# Wait, let me be more careful:
# Huon(j, i) = flux through the face at the LEFT/WEST side of rho(j, i)
#   Positive Huon(j, i) means flux from rho(j, i-1) to rho(j, i), i.e., INTO rho(j,i) from west
# Huon(j, i+1) = flux through the face at the RIGHT/EAST side of rho(j, i)  
#   Positive Huon(j, i+1) means flux from rho(j, i) to rho(j, i+1), i.e., OUT of rho(j,i) to east
#
# Similarly for v-grid:
# Hvom(j, i) = flux through the face at the SOUTH side of rho(j, i)
#   Positive Hvom(j, i) means flux from rho(j-1, i) to rho(j, i), i.e., INTO rho(j,i) from south
# Hvom(j+1, i) = flux through the face at the NORTH side of rho(j, i)
#   Positive Hvom(j+1, i) means flux from rho(j, i) to rho(j+1, i), i.e., OUT of rho(j,i) to north
#
# Net flux INTO cell (j, i) at each z-level:
#   Huon(z, j, i) - Huon(z, j, i+1) + Hvom(z, j, i) - Hvom(z, j+1, i)

# Sum over all cells in the volume to get net flux through ALL boundary faces
# (interior faces cancel because each interior face is shared by two cells)
total_flux_all_faces = 0.0
for k in range(len(all_j)):
    j, i = all_j[k], all_i[k]
    # West face: Huon(:, j, i) INTO cell
    if i < Huon.shape[2]:
        total_flux_all_faces += np.nansum(Huon[:, j, i])
    # East face: -Huon(:, j, i+1) OUT of cell
    if (i+1) < Huon.shape[2]:
        total_flux_all_faces -= np.nansum(Huon[:, j, i+1])
    # South face: Hvom(:, j, i) INTO cell
    if j < Hvom.shape[1]:
        total_flux_all_faces += np.nansum(Hvom[:, j, i])
    # North face: -Hvom(:, j+1, i) OUT of cell
    if (j+1) < Hvom.shape[1]:
        total_flux_all_faces -= np.nansum(Hvom[:, j+1, i])

print(f'\nFlux through ALL boundary faces (from Huon/Hvom): {total_flux_all_faces:.4f} m3/s')

# ===== Compare to section-only flux =====
in_dir = dir0 / ('extractions_avg_' + Ldir['ds0'] + '_' + Ldir['ds1'])
# Find the time index corresponding to our test file
# The extraction concatenates all files, so we need to find which index
# corresponds to our test file. We'll use time index 12 to match test_fn = fn_list[12].
test_idx = min(12, len(fn_list)-1)

sect_flux = 0.0
for tup in sntup_list:
    sn = tup[0]
    sgn = tup[1]
    ds_sect = xr.open_dataset(in_dir / (sn + '.nc'))
    q = ds_sect['q'].values  # (time, z, p)
    qnet_at_t = np.nansum(q[test_idx, :, :]) * sgn
    sect_flux += qnet_at_t
    print(f'Section {sn}: qnet at t={test_idx} = {np.nansum(q[test_idx,:,:]):.4f}, with sgn={sgn} -> {qnet_at_t:.4f}')
    ds_sect.close()

print(f'\nFlux through SECTION faces only: {sect_flux:.4f} m3/s')
print(f'Flux through ALL boundary faces: {total_flux_all_faces:.4f} m3/s')
print(f'Missing flux (all - section): {total_flux_all_faces - sect_flux:.4f} m3/s')
print(f'|missing|/|all|: {abs(total_flux_all_faces - sect_flux)/(abs(total_flux_all_faces)+1e-10):.4f}')

# ===== Also print section geometry info =====
print('\n--- Section geometry ---')
for tup in sntup_list:
    sn = tup[0]
    this_sect = sect_df[sect_df.sn == sn]
    n_u = len(this_sect[this_sect.uv == 'u'])
    n_v = len(this_sect[this_sect.uv == 'v'])
    print(f'Section {sn}: {n_u} u-faces + {n_v} v-faces = {len(this_sect)} total faces')
    print(f'  j range: {this_sect.j.min()} to {this_sect.j.max()}')
    print(f'  i range: {this_sect.i.min()} to {this_sect.i.max()}')

# ===== Print segment info =====
print('\n--- Segment info ---')
for sk in good_seg_key_list:
    ji_list = seg_info_dict[sk]['ji_list']
    sns = seg_info_dict[sk]['sns_list']
    rivs = seg_info_dict[sk]['riv_list']
    js = [ji[0] for ji in ji_list]
    Is = [ji[1] for ji in ji_list]
    print(f'Segment {sk}: {len(ji_list)} cells, j=[{min(js)},{max(js)}], i=[{min(Is)},{max(Is)}]')
    print(f'  sns_list: {sns}')
    print(f'  riv_list: {rivs}')

# ===== Visualize =====
pfun.start_plot(figsize=(12, 10))
fig, ax = plt.subplots(1, 1)

# Plot volume mask
ax.pcolormesh(vol_mask, cmap='Blues', alpha=0.5)
ax.set_title('Penn Cove: segment cells (blue) + section faces (red/green)')

# Plot section faces
for tup in sntup_list:
    sn = tup[0]
    this_sect = sect_df[sect_df.sn == sn]
    for _, row in this_sect.iterrows():
        if row.uv == 'u':
            # u-face at (j, i): vertical line between rho(j,i-1) and rho(j,i)
            ax.plot([row.i, row.i], [row.j, row.j+1], 'r-', linewidth=2)
        else:
            # v-face at (j, i): horizontal line between rho(j-1,i) and rho(j,i)
            ax.plot([row.i, row.i+1], [row.j, row.j], 'g-', linewidth=2)

# Zoom to region of interest
j_min = all_j.min() - 3
j_max = all_j.max() + 3
i_min = all_i.min() - 3
i_max = all_i.max() + 3
ax.set_xlim(i_min, i_max)
ax.set_ylim(j_min, j_max)
ax.set_aspect('equal')
ax.set_xlabel('i (xi)')
ax.set_ylabel('j (eta)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_fig = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2' / 'budget_diagnostic2.png'
plt.savefig(out_fig)
print(f'\nFigure saved to {out_fig}')
plt.show()
pfun.end_plot()

ds.close()
