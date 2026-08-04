"""
Code to process King County Water Quality CTD data at Point Jefferson.

To process data received via email from Taylor Martin, King County, to Dakota Mascarenas on 2024/05/02.

Station information downloaded by Dakota Mascarenas from: https://data.kingcounty.gov/Environment-Waste-Management/WLRD-Sites/wbhs-bbzf

Initial author date: 2024/05/09

Finalized for group use: 2025/09/05

Written by: Dakota Mascarenas

Most recent update: 2026/06/23

NOTE: TaylorQuality and TaylorNote columns are considered insofar as to filter to only TaylorQuality = 'ok'.

NOTE: Variables here are the field/CTD measurements ('Salinity, Field', 'Dissolved Oxygen, Field') plus 'Temperature' (which has no bottle equivalent); the bottle 'Salinity'/'Dissolved Oxygen' are excluded.

NOTE: INSTRUMENT TRANSITION (per King County methods, his_methods_email - TM to DM 20250117.pdf): the Seabird CTD came into use ~October 1998. 'Salinity, Field' and 'Dissolved Oxygen, Field' only exist from 1998-10-20 onward (before that, field salinity is absent and field DO was a handheld YSI-type sonde; temperature was an alcohol/mercury thermometer). Because CT/SA require salinity, pre-Oct-1998 casts (thermometer temperature, no CTD salinity) yield NaN CT/SA, so the CTD product effectively begins ~Oct 1998. Further CTD caveats from the same email: Seabird 43 DO sensor added July 2004; DO calibration approach changed Feb 2005 and again Jan 2010; depths before 6/1/2010 may be biased deep (~0.7 m near surface) and before Jan 2010 by +/- 0.4 m (no barometric offset).

NOTE: TIMES IN UTC

"""

import pandas as pd
import numpy as np
import gsw
import sys

from lo_tools import Lfun, obs_functions
Ldir = Lfun.Lstart()

# source location
source = 'kc_pointJefferson'
otype = 'ctd'
in_dir0 = Ldir['data'] / 'obs' / source
year_list = range(1970,2025)

# output location
out_dir = Ldir['LOo'] / 'obs' / source / otype
Lfun.make_dir(out_dir)

# Load big data set and stations.
big_df_raw = pd.read_csv(in_dir0/ 'bottle_DO_data_for_Dakota.csv', low_memory=False) #confusingly named but fine
sta_df = pd.read_csv(in_dir0 / 'WLRD_Sites_March2024.csv')

# Merge station data.
big_df = big_df_raw.merge(sta_df[['Locator','Latitude', 'Longitude']], on = 'Locator', how='left')

# Clean dataframe based on QC.
big_df_use0 = big_df[big_df['TaylorQuality'] == 'ok']

# Create dictionary and filter for important variable and column names.
cols_all = big_df_use0['ParmDisplayName'].unique()
v_dict = {}
v_dict = {col:'' for col in cols_all}
v_dict['Temperature'] = 'IT'
v_dict['Salinity, Field'] = 'SP'
v_dict['Dissolved Oxygen, Field'] = 'DO (mg -L)'
v_dict_use = {}
for v in v_dict.keys():
    if len(v_dict[v]) > 0:
        v_dict_use[v] = v_dict[v]
v_list = np.array(list(v_dict_use.keys()))
big_df_use1 = big_df_use0[big_df_use0['ParmDisplayName'].isin(v_list)]

# Clean dataframe more.
big_df_use2 = big_df_use1[['CollectDateTime', 'Depth', 'ParmDisplayName', 'Value','Latitude', 'Longitude', 'Locator']]
big_df_use5 = big_df_use2.pivot_table(index = ['CollectDateTime', 'Depth','Latitude', 'Longitude', 'Locator'],
                                      columns = 'ParmDisplayName', values = 'Value').reset_index()
big_df_use6 = big_df_use5.copy()
# Parse to timezone-aware UTC. Raw 'CollectDateTime' carries a 'Z' suffix, so use to_datetime
# with utc=True to robustly handle already-aware (Z) values; newer pandas parses 'Z' as
# tz-aware, which broke the prior DatetimeIndex + tz_localize('UTC') approach.
big_df_use6['time'] = pd.to_datetime(big_df_use6['CollectDateTime'], utc=True)

# Create unique cast IDs (cid).
big_df_use7 = big_df_use6.copy()
big_df_use7['unique_date_location'] = big_df_use7['Locator'] + big_df_use7['CollectDateTime']
c = 0
for pid in big_df_use7['unique_date_location'].unique(): # profile ID is unique identifier
    big_df_use7.loc[big_df_use7['unique_date_location'] == pid, 'cid'] = c
    c+=1
    
# Rename some columns in variable dictionary.
v_dict['cid'] = 'cid'
v_dict['time'] = 'time'
v_dict['Latitude'] = 'lat'
v_dict['Longitude'] = 'lon'
v_dict['Depth'] = 'z' # will be converted to negative later in script
v_dict['Locator'] = 'name'

# Loop through to rename variables and columns, clean the dataset, and produce output dataframes.
df0 = big_df_use7.copy()
for year in year_list:
    ys = str(year)
    print('\n'+ys)
    out_fn = out_dir / (ys + '.p')
    info_out_fn = out_dir / ('info_' + ys + '.p')
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
    # add the results to the dataframe
    df['SA'] = SA
    df['CT'] = CT
    rho = gsw.rho(SA,CT,p)
    # unit conversions
    if 'DO (mg -L)' in df.columns:
        df['DO (uM)'] = (1000/32) * df['DO (mg -L)']
    if 'NH4 (mg -L)' in df.columns:
        df['NH4 (uM)'] = (1000/14) * df['NH4 (mg -L)']  # N atomic weight
    if 'NO3 (mg -L)' in df.columns:
        df['NO3 (uM)'] = (1000/14) * df['NO3 (mg -L)']  # N atomic weight
    if 'SiO4 (mg -L)' in df.columns:
        df['SiO4 (uM)'] = (1000/28.0855) * df['SiO4 (mg -L)']  # Si atomic weight
    if 'PO4 (mg -L)' in df.columns:
        df['PO4 (uM)'] = (1000/30.973762) * df['PO4 (mg -L)']  # P atomic weight
    if 'Chl (ug -L)' in df.columns:
        df['Chl (mg m-3)'] = df['Chl (ug -L)']
    for vn in ['TA','DIC']:
        if (vn+' (umol -kg)') in df.columns:
            df[vn+' (uM)'] = (rho/1000) * df[vn+' (umol -kg)']
    # retain only selected variables
    df['cruise'] = ''
    cols = ['cid', 'time', 'lat', 'lon', 'z', 'cruise', 'name',
        'CT', 'SA', 'DO (uM)',
        'NO3 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'SiO4 (uM)',
        'TA (uM)', 'DIC (uM)', 'Chl (mg m-3)']
    this_cols = [item for item in cols if item in df.columns]
    df = df[this_cols]
    # save
    print(' - processed %d casts' % ( len(df.cid.unique()) ))
    if len(df) > 0:
        # Save the data
        df.to_pickle(out_fn)
        info_df = obs_functions.make_info_df(df)
        info_df.to_pickle(info_out_fn)