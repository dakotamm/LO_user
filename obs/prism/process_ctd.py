"""
Code to process the PRISM Salish Cruise CTD data for Puget Sound.

To process the downcast CTD subset (T, S, sigma-t, CTD-O2) of the full Salish
Cruise CTD data set, compiled by Dakota Mascarenas from the NANOOS NVS data
archive. We consider only T and S here.

Initial author date: 2026/06/15

Written by: Dakota Mascarenas

Most recent update: 2026/06/15

NOTE: Salish Cruise CTD timestamps are research-vessel logs and are taken to be
in UTC.

"""

import pandas as pd
import numpy as np
import gsw

from lo_tools import Lfun, obs_functions
Ldir = Lfun.Lstart()


# source location
source = 'prism'
otype = 'ctd'
in_dir0 = Ldir['data'] / 'obs' / source
year_list = range(1998,2019)

# output location
out_dir = Ldir['LOo'] / 'obs' / source / otype
Lfun.make_dir(out_dir)

# Load big data set (lat/lon already in signed decimal degrees).
big_df_raw = pd.read_excel(in_dir0 / 'SalishCruise_downcast_CTDdata_121998to092018_TSstO2subset.xlsx', sheet_name='all data')

# Build a single time column from the separate date/time columns.
big_df_use = big_df_raw.copy()
big_df_use['time'] = pd.to_datetime(
    big_df_use['YEAR'].astype(int).astype(str) + '-'
    + big_df_use['MONTH'].astype(int).astype(str) + '-'
    + big_df_use['DAY'].astype(int).astype(str) + ' '
    + big_df_use['TIME'].astype(str), errors='coerce')
# Ensure times are timezone-aware UTC (raw data taken to be in UTC).
big_df_use['time'] = big_df_use['time'].dt.tz_localize('UTC')

# Build a station name (station numbers are sequential within each cruise; some
# rows are missing a station number, so fall back to 'NA').
stn_str = big_df_use['STN'].apply(lambda s: str(int(s)) if pd.notna(s) else 'NA')
big_df_use['name'] = big_df_use['CRUISE'].astype(str) + '_' + stn_str

# Create unique cast IDs (cid). Each cast (profile) has a single timestamp, so
# identify casts by time; this is robust to missing station numbers.
big_df_use['cid'] = np.nan
big_df_use['unique_date_location'] = big_df_use['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
c = 0
for pid in big_df_use['unique_date_location'].unique(): # profile ID is unique identifier
    big_df_use.loc[big_df_use['unique_date_location'] == pid, 'cid'] = c
    c+=1

# Create dictionary for important variable and column names.
v_dict = {'TEMP':'IT', 'SAL':'SP'}
v_dict['cid'] = 'cid'
v_dict['time'] = 'time'
v_dict['LAT'] = 'lat'
v_dict['LONG'] = 'lon'
v_dict['DEPTH'] = 'z' # will be converted to negative later in script
v_dict['name'] = 'name'
v_dict['CRUISE'] = 'cruise'

# Loop through to rename variables and columns, clean the dataset, and produce output dataframes.
df0 = big_df_use.copy()
for year in year_list:
    ys = str(year)
    print('\n'+ys)
    out_fn = out_dir / (ys + '.p')
    info_out_fn = out_dir / ('info_' + ys + '.p')
    t = pd.DatetimeIndex(df0['time'])
    df1 = df0.loc[t.year==year,:].copy()
    # select and rename variables
    df = pd.DataFrame()
    for v in df1.columns:
        if v in v_dict.keys():
            if len(v_dict[v]) > 0:
                df[v_dict[v]] = df1[v]
    # a little more cleaning up
    df = df.dropna(axis=0, how='all') # drop rows with no good data
    df = df[df.time.notna()] # drop rows with bad time
    df = df.reset_index(drop=True)
    df['z'] = df['z']*-1 # IMPORTANT!!!!!! - from above!
    SP = df.SP.to_numpy()
    IT = df.IT.to_numpy()
    z= df.z.to_numpy()
    lon = df.lon.to_numpy()
    lat = df.lat.to_numpy()
    # do the gsw conversions
    p = gsw.p_from_z(z, lat)
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    CT = gsw.CT_from_t(SA, IT, p)
    # add the results to the DataFrame
    df['SA'] = SA
    df['CT'] = CT
    # retain only selected variables
    cols = ['cid', 'time', 'lat', 'lon', 'z', 'cruise', 'name',
        'CT', 'SA']
    this_cols = [item for item in cols if item in df.columns]
    df = df[this_cols]
    # save
    print(' - processed %d casts' % ( len(df.cid.unique()) ))
    if len(df) > 0:
        # Save the data
        df.to_pickle(out_fn)
        info_df = obs_functions.make_info_df(df)
        info_df.to_pickle(info_out_fn)
