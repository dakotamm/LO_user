"""
Quick comparison of gridded bathymetry between two wb1 grid files.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from lo_tools import plotting_functions as pfun

f1 = '/Users/dakotamascarenas/LO_output/pgrid/wb1/grid_m10_r01_s02_x02.nc'
f2 = '/Users/dakotamascarenas/LO_data/grids/wb1/grid.nc'

ds1 = xr.open_dataset(f1)
ds2 = xr.open_dataset(f2)

# Mask land for display (h where mask_rho == 0 set to NaN)
def masked_h(ds):
    h = ds['h'].values.astype(float)
    m = ds['mask_rho'].values
    h_m = np.where(m == 1, h, np.nan)
    return h_m

h1 = masked_h(ds1)
h2 = masked_h(ds2)
z1 = -h1
z2 = -h2
diff = z1 - z2  # positive => file1 shallower than file2

# Land mask = 1 where either grid considers it land
m1 = ds1['mask_rho'].values
m2 = ds2['mask_rho'].values
land = ((m1 == 0) | (m2 == 0)).astype(float)
land = np.where(land == 1, 1.0, np.nan)

# Cells where land/ocean classification differs
# 1 -> file1 ocean, file2 land  (file1 added ocean / file2 added land)
# 2 -> file1 land,  file2 ocean
mask_diff = np.full(m1.shape, np.nan)
mask_diff[(m1 == 1) & (m2 == 0)] = 1
mask_diff[(m1 == 0) & (m2 == 1)] = 2

lon = ds1['lon_rho'].values
lat = ds1['lat_rho'].values
plon, plat = pfun.get_plon_plat(lon, lat)

vmin = np.nanmin([np.nanmin(z1), np.nanmin(z2)])
vmax = np.nanmax([np.nanmax(z1), np.nanmax(z2)])

dmax = np.nanmax(np.abs(diff))

fig, axs = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)

pcm0 = axs[0].pcolormesh(plon, plat, z1, vmin=vmin, vmax=vmax, cmap='viridis')
axs[0].set_title('pgrid/wb1/grid_m10_r01_s02_x02.nc')
fig.colorbar(pcm0, ax=axs[0], label='z [m]')

pcm1 = axs[1].pcolormesh(plon, plat, z2, vmin=vmin, vmax=vmax, cmap='viridis')
axs[1].set_title('LO_data/grids/wb1/grid.nc')
fig.colorbar(pcm1, ax=axs[1], label='z [m]')

pcm2 = axs[2].pcolormesh(plon, plat, diff, vmin=-dmax, vmax=dmax, cmap='RdBu_r')
axs[2].set_title('Δz = file1 − file2\n(red: file1 shallower, blue: file1 deeper)')
cb2 = fig.colorbar(pcm2, ax=axs[2], label='Δz [m]')
cb2.ax.text(1.05, 1.02, 'file1 shallower', transform=cb2.ax.transAxes,
            ha='left', va='bottom', fontsize=9, color='darkred')
cb2.ax.text(1.05, -0.02, 'file1 deeper', transform=cb2.ax.transAxes,
            ha='left', va='top', fontsize=9, color='darkblue')

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
land_cmap = ListedColormap(['lightgray'])
mask_diff_cmap = ListedColormap(['magenta', 'cyan'])

for ax in axs:
    ax.pcolormesh(plon, plat, land, cmap=land_cmap, shading='auto', zorder=2)
    pfun.dar(ax)
    ax.set_xlim(plon.min(), plon.max())
    ax.set_ylim(plat.min(), plat.max())
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

# Show land/ocean mask differences only on the second (file2) panel
axs[1].pcolormesh(plon, plat, mask_diff, cmap=mask_diff_cmap,
                  vmin=1, vmax=2, shading='auto', zorder=3)

# Legend for mask differences (on the file2 panel)
legend_handles = [
    Patch(facecolor='magenta', edgecolor='k', label='file1 ocean, file2 land'),
    Patch(facecolor='cyan',    edgecolor='k', label='file1 land, file2 ocean'),
]
axs[1].legend(handles=legend_handles, loc='lower left', fontsize=8, framealpha=0.9)

# Print quick stats
print(f'z1: min={np.nanmin(z1):.3f}, max={np.nanmax(z1):.3f}, mean={np.nanmean(z1):.3f}')
print(f'z2: min={np.nanmin(z2):.3f}, max={np.nanmax(z2):.3f}, mean={np.nanmean(z2):.3f}')
print(f'Δz: min={np.nanmin(diff):.3f}, max={np.nanmax(diff):.3f}, '
      f'mean={np.nanmean(diff):.3f}, |max|={dmax:.3f}')
print(f'mask_rho equal: {np.array_equal(m1, m2)}')
print(f'cells file1 ocean / file2 land: {int(np.sum((m1==1)&(m2==0)))}')
print(f'cells file1 land  / file2 ocean: {int(np.sum((m1==0)&(m2==1)))}')

out_path = '/Users/dakotamascarenas/Desktop/pltz/20260421_compare_wb1_bathy.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'saved: {out_path}')

plt.show()
