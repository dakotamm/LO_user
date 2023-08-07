"""
This is a one-off piece of code to create a 0025 history file for the
updated ROMS, including bio variables, from an old ROMS file.

This allows you to run the new ROMS, with bio, using "-start_type continuation"
starting the day after f_str.
"""

import xarray as xr
from lo_tools import Lfun, zrfun
from pathlib import Path

Ldir = Lfun.Lstart()

in_dir0 = Path('/pgdat1/parker/LO_roms')

f_str = 'f2023.04.30'

in_fn = in_dir0 / 'cas6_v0_live' / f_str / 'ocean_his_0025.nc'

out_dir = Ldir['roms_out'] / 'cas6_traps2_x2b' / f_str
Lfun.make_dir(out_dir)
out_fn = out_dir / 'ocean_his_0025.nc'

ds0 = xr.open_dataset(in_fn, decode_times=False)

ds1 = xr.Dataset()

ot_vec = ds0.ocean_time.values
ds1['ocean_time'] = (('ocean_time',), [ot_vec[0]])
ds1['ocean_time'].attrs['units'] = Lfun.roms_time_units

roms_names = ['zeta', 'ubar', 'vbar', 'temp', 'salt', 'u', 'v']

bvn_list = ['NO3', 'NH4', 'chlorophyll', 'phytoplankton', 'zooplankton',
        'LdetritusN', 'SdetritusN', 'LdetritusC', 'SdetritusC',
        'TIC', 'alkalinity', 'oxygen']

for vn in roms_names + bvn_list:
    vinfo = zrfun.get_varinfo(vn)
    if vn in ds0.data_vars:
        ds1[vn] = (('ocean_time',) + ds0[vn].dims[1:], ds0[vn].values)
    else:
        ds1[vn] = (('ocean_time',) + ds0['salt'].dims[1:], 0 * ds0['salt'].values)
    ds1[vn].attrs['units'] = vinfo['units']
    ds1[vn].attrs['long_name'] = vinfo['long_name']
    
ds0.close()
        
# and save to NetCDF
Enc_dict = {vn:zrfun.enc_dict for vn in ds1.data_vars}
ds1.to_netcdf(out_fn, encoding=Enc_dict)
ds1.close()


