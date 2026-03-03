"""
Code to process Penn Cove mussel raft sonde data as discussed in Roberts & Carrington (2023): https://doi.org/10.1016/j.jembe.2023.151927

This specificically CONVERTS TO PANDAS DATAFRAMES, similar to the process for CTD and Bottle casts.

Data received via email on 2026/01/02 from Emily Carrington to Dakota Mascarenas.

Initial author date: 2026/02/24

Finalized for public use: XXXX/XX/XX

Written by: Dakota Mascarenas

"""

#### STILL NEEDS UTC CONVERSION

import pandas as pd
import numpy as np
import gsw
import xarray as xr

from lo_tools import Lfun, obs_functions
Ldir = Lfun.Lstart()

# source location
source = 'pcRaft'
otype = 'sonde'
in_dir0 = Ldir['data'] / 'obs' / source 
year_list = range(2014,2020)

# output locations for both output types
out_dir_df = Ldir['LOo'] / 'obs' / source / otype 
out_dir_nc = Ldir['LOo'] / 'obs' / source / otype / 'nc'
Lfun.make_dir(out_dir_df)
Lfun.make_dir(out_dir_nc, clean=True)

# Load big data set and stations.
big_df = pd.read_excel(in_dir0/ 'PennCove-mussel-raft-sonde-data-2014-2019.xlsx', sheet_name='cleaned_data')
big_df['time'] = pd.DatetimeIndex(big_df['TIMESTAMP'])
big_df = big_df[['time', 'depth', 'temp', 'sal', 'chl', 'pH', 'DO']]

# Load in July 2017 high resolution sampling (10 minute frequency to be selected only on even hour marks); also not loading the 1m and 7m redundant loggers.
little_df = pd.DataFrame()
for depth in [0.5, 2, 3, 4, 5, 6]:
    little_df_temp = pd.read_csv(in_dir0/ ('B8_' + str(depth) + 'm.csv'))
    little_df_temp['depth'] = depth
    little_df_temp = little_df_temp.set_axis(['#', 'TIME', 'temp', 'depth'], axis=1)
    little_df = pd.concat([little_df, little_df_temp])
little_df['time'] = pd.DatetimeIndex(little_df['TIME'])
little_df = little_df[['time', 'depth', 'temp']]

# Combine dataframes.
big_df = pd.concat([big_df, little_df])

# Create dictionary for important variable and column names.
v_dict = {'chl':'Chl (ug -L)',
          'DO':'DO (mg -L)',
          'pH':'PH',
          'sal':'SP', 
          'temp':'IT' 
          } # not dealing with light/PAR right now...
v_dict_use = {}
for v in v_dict.keys():
    if len(v_dict[v]) > 0:
        v_dict_use[v] = v_dict[v]
v_list = np.array(list(v_dict_use.keys())) #redundant but fine
        
# Clean column names and add coordinates.
big_df_use = big_df.copy()
big_df_use['lat'] = 48.220861
big_df_use['lon'] = -122.705667

# Select data occurring only at whole hours.
big_df_use = big_df_use[big_df_use['time'].dt.minute == 0]


# Create unique cast IDs (cid). NOTE: Not the same instrument, just concurrent collection.
big_df_use['cid'] = np.nan
big_df_use = big_df_use.copy()
c = 0
for pid in big_df_use['time'].unique(): # profile ID is unique identifier
    big_df_use.loc[big_df_use['time'] == pid, 'cid'] = c
    c+=1
    

# Rename some columns in variable dictionary.
v_dict['cid'] = 'cid'
v_dict['time'] = 'time'
v_dict['lat'] = 'lat'
v_dict['lon'] = 'lon'
v_dict['depth'] = 'z' # will be converted to negative later in script

# Loop through to rename variables and columns, clean the dataset, and produce output dataframes.
df0 = big_df_use.copy()
for year in year_list:
    ys = str(year)
    print('\n'+ys)
    out_fn_df = out_dir_df / (ys + '.p')
    info_out_fn_df = out_dir_df / ('info_' + ys + '.p')
    out_fn_nc = out_dir_nc / (ys + '.nc')
    t = pd.DatetimeIndex(df0.time)
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
    rho = gsw.rho(SA,CT,p)
    # unit conversions
    if 'DO (mg -L)' in df.columns:
        df['DO (uM)'] = (1000/32) * df['DO (mg -L)']
    if 'NO3 (mg -L)' in df.columns:
        df['NO3 (uM)'] = (1000/62) * df['NO3 (mg -L)']
    if 'Chl (ug -L)' in df.columns:
        df['Chl (mg m-3)'] = df['Chl (ug -L)']
    # retain only selected variables             
    df['cruise'] = ''
    df['name'] = ''
    cols = ['cid', 'time', 'lat', 'lon', 'z', 'cruise', 'name',
        'CT', 'SA', 'DO (uM)',
        'PH', 'Chl (mg m-3)']
    this_cols = [item for item in cols if item in df.columns]
    df = df[this_cols]
    df = df[~df.duplicated(subset=["time", "z"], keep=False)]
    # save
    print(' - processed %d casts' % ( len(df.cid.unique()) ))
    if len(df) > 0:
        # Save the data
        df.to_pickle(out_fn_df)
        info_df = obs_functions.make_info_df(df)
        info_df.to_pickle(info_out_fn_df)
        ds = (
            df.set_index(["time", "z"])
                [["SA", "CT", "DO (uM)", "PH", "Chl (mg m-3)"]]
                .to_xarray()
        )
        ds.attrs["lat"] = float(df["lat"].iloc[0])
        ds.attrs["lon"] = float(df["lon"].iloc[0])
        ds['SA'].attrs={'units':'g kg-1', 'long_name':'Absolute Salinity'}
        ds['CT'].attrs={'units':'degC', 'long_name':'Conservative Temperature'}
        ds['DO (uM)'].attrs={'units':'uM', 'long_name':'Dissolved Oxygen'}
        ds['PH'].attrs={'units':'NBS scale', 'long_name':'pH'}
        ds['Chl (mg m-3)'].attrs={'units':'mg m-3', 'long_name':'Chlorophyll'}
        ds.to_netcdf(out_fn_nc)
