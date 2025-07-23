"""
Code to process the King County Water Quality data for Puget Sound.

5/9/2024

To process data received via email from Taylor Martin 5/2/2024.

sites: https://data.kingcounty.gov/Environment-Waste-Management/WLRD-Sites/wbhs-bbzf

Note: TaylorQuality and TaylorNote columns are considered insofar as to filter to only TaylorQuality = 'ok'.

NOTE: "field" data and temperature are from CTD. Here, considering just bottle FOR NOW (and temperature concurrently).

TIMES IN UTC

"""

import pandas as pd
import numpy as np
import gsw
import sys

from lo_tools import Lfun, obs_functions
Ldir = Lfun.Lstart()

# BOTTLE
source = 'kc_point_jefferson'
otype = 'bottle'
in_dir0 = Ldir['data'] / 'obs' / source
year_list = range(1970,2025)

# output location
out_dir = Ldir['LOo'] / 'obs' / source / otype
Lfun.make_dir(out_dir)


# %%

# Load big data set and stations.

big_df_raw = pd.read_csv(in_dir0/ 'bottle_DO_data_for_Dakota.csv')

sta_df = pd.read_csv(in_dir0 / 'WLRD_Sites_March2024.csv')

# %%

big_df = big_df_raw.merge(sta_df[['Locator','Latitude', 'Longitude']], on = 'Locator', how='left')


# %%

big_df_use0 = big_df[big_df['TaylorQuality'] == 'ok']


# %% 

cols_all = big_df_use0['ParmDisplayName'].unique()

v_dict = {}

v_dict = {col:'' for col in cols_all}

# %%

v_dict['Temperature'] = 'IT' #NEED TO COVERT TO CONS TEMP if necessary???

v_dict['Salinity'] = 'SP' #NEED TO CONVERT TO ABS SALINITY if necessary???

v_dict['Dissolved Oxygen'] = 'DO (mg -L)' #NEED TO CONVERT TO micromolar

# v_dict['Nitrite + Nitrate Nitrogen'] = 'NO3 (mg -L)' #measured together assuming a 0 NO2 (add that column later), NEED TO CONVERT to micromolar

# v_dict['Ammonia Nitrogen'] = 'NH4 (mg -L)' #NEED TO CONVERT to micromolar

# v_dict['Total Phosphorus'] = 'PO4 (mg -L)' #NEED TO CONVERT to micromolar - OR should I use ORTHOPHOSPHATE PHOSPHORUS??

# v_dict['Silica'] = 'SiO4 (mg -L)' #NEED TO CONVERT to micromolar, I think this is correct but not sure***

# #v_dict['Total Alkalinity'] = 'TA (umol -kg)' #need to convert to micromolar!!!

# #v_dict['Dissolved Inorganic Carbon'] = 'DIC (umol -kg)'

# v_dict['Chlorophyll a'] = 'Chl (mg -3)' #no need to convert

# %%

v_dict_use = {}

for v in v_dict.keys():
    if len(v_dict[v]) > 0:
        v_dict_use[v] = v_dict[v]
        
v_list = np.array(list(v_dict_use.keys()))
        
# %%

big_df_use1 = big_df_use0[big_df_use0['ParmDisplayName'].isin(v_list)]

# %%

big_df_use2 = big_df_use1[['CollectDateTime', 'Depth', 'ParmDisplayName', 'Value','Latitude', 'Longitude', 'Locator']]


# %%

# replicates = big_df_use2['Replicates'].dropna().unique()

# big_df_use3 = big_df_use2[~big_df_use2['Sample ID'].isin(replicates)]

# %%

#big_df_use4 = big_df_use3[['Profile ID', 'Collect DateTime', 'Depth (m)', 'Parameter', 'Value', 'Latitude', 'Longitude']]

# %%

big_df_use5 = big_df_use2.pivot_table(index = ['CollectDateTime', 'Depth','Latitude', 'Longitude', 'Locator'],
                                      columns = 'ParmDisplayName', values = 'Value').reset_index()

# %%

big_df_use6 = big_df_use5.copy()

big_df_use6['time'] = pd.DatetimeIndex(big_df_use6['CollectDateTime'])

# %%


# start_date = pd.Timestamp('2025-01-01')

# mask = (big_df_use6['time'] >= start_date)


# big_df_use6.loc[mask, 'time'] -= pd.DateOffset(years=100)

# big_df_use6['cid'] = np.nan


# %%

big_df_use7 = big_df_use6.copy()


big_df_use7['unique_date_location'] = big_df_use7['Locator'] + big_df_use7['CollectDateTime']

c = 0

for pid in big_df_use7['unique_date_location'].unique(): # profile ID is unique identifier
    
    big_df_use7.loc[big_df_use7['unique_date_location'] == pid, 'cid'] = c
    
    c+=1
    
# %%

v_dict['cid'] = 'cid'

v_dict['time'] = 'time'

v_dict['Latitude'] = 'lat'

v_dict['Longitude'] = 'lon'

v_dict['Depth'] = 'z' #convert to negative

v_dict['Locator'] = 'name'


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
                
    # missing data is -999
   # df[df==-999] = np.nan
    
    # a little more cleaning up
    df = df.dropna(axis=0, how='all') # drop rows with no good data
    df = df[df.time.notna()] # drop rows with bad time
    df = df.reset_index(drop=True)
    
    df['z'] = df['z']*-1 # IMPORTANT!!!!!!

    SP = df.SP.to_numpy()
    IT = df.IT.to_numpy()
    z= df.z.to_numpy()
    lon = df.lon.to_numpy()
    lat = df.lat.to_numpy()
    
    p = gsw.p_from_z(z, lat) # does this make sense????

    # - do the conversions
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    CT = gsw.CT_from_t(SA, IT, p)
    # - add the results to the DataFrame
    df['SA'] = SA
    df['CT'] = CT
    rho = gsw.rho(SA,CT,p)

    if 'DO (mg -L)' in df.columns:
        df['DO (uM)'] = (1000/32) * df['DO (mg -L)']
    if 'NH4 (mg -L)' in df.columns:
        df['NH4 (uM)'] = (1000/18) * df['NH4 (mg -L)']
    if 'NO3 (mg -L)' in df.columns:
        df['NO3 (uM)'] = (1000/62) * df['NO3 (mg -L)']
    if 'SiO4 (mg -L)' in df.columns:
        df['SiO4 (uM)'] = (1000/92) * df['SiO4 (mg -L)']
    if 'PO4 (mg -L)' in df.columns:
        df['PO4 (uM)'] = (1000/95) * df['PO4 (mg -L)']
        
    if 'Chl (ug -L)' in df.columns:
        df['Chl (mg m-3)'] = df['Chl (ug -L)']
        
    for vn in ['TA','DIC']:
        if (vn+' (umol -kg)') in df.columns:
            df[vn+' (uM)'] = (rho/1000) * df[vn+' (umol -kg)']
                        
    df['cruise'] = ''

        
    # (3) retain only selected variables
    cols = ['cid', 'time', 'lat', 'lon', 'z', 'cruise', 'name',
        'CT', 'SA', 'DO (uM)',
        'NO3 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'SiO4 (uM)',
        'TA (uM)', 'DIC (uM)', 'Chl (mg m-3)']
    this_cols = [item for item in cols if item in df.columns]
    df = df[this_cols]

    print(' - processed %d casts' % ( len(df.cid.unique()) ))
        
    # Renumber cid to be increasing from zero in steps of one.
    #df = obs_functions.renumber_cid(df)
    
    if len(df) > 0:
        # Save the data
        df.to_pickle(out_fn)
        info_df = obs_functions.make_info_df(df)
        info_df.to_pickle(info_out_fn)