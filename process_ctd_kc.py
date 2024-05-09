"""
Code to process the King County Water Quality data for Puget Sound.

IN PROGRESS DM 7/13/2023 - restarted effort on 3/29/2024

some notes from https://green2.kingcounty.gov/marine/Monitoring/OffshoreCTD: "Light Transmission data prior to May 19, 2014 were referenced to air. After this date, all Light Transmission data are referenced to water. To convert the pre-May 19, 2014 data to ‘referenced to water’, multiply the values by 1.095."

***any modifications done to temperature???

***local times or UTC?

sites: https://data.kingcounty.gov/Environment-Waste-Management/WLRD-Sites/wbhs-bbzf

big data set: https://data.kingcounty.gov/Environment-Waste-Management/Water-Quality/vwmt-pvjw
- from 1965 to 2023

whidbey bottle (***I THINK THIS IS JUST A FILTERED BY AREA VERSION OF THE BIG DATA SET): https://data.kingcounty.gov/Environment-Waste-Management/Whidbey-Bottle-Data/vuu8-t6kc

whidbey CTD (additional detail just for whidbey - not sure where the rest of CTD casts are): https://data.kingcounty.gov/Environment-Waste-Management/Whidbey-Basin-CTD-Casts/uz4m-4d96


Received KC CTD dataset 4/5/2024 from Greg Ikeda (KC) direct - KC QCed

"""

import pandas as pd
import numpy as np
import gsw
import sys

import glob

from lo_tools import Lfun, obs_functions
Ldir = Lfun.Lstart()

# BOTTLE
source = 'kc'
otype = 'ctd'
in_dir0 = Ldir['data'] / 'obs' / source 
year_list = range(1998,2025)

# output location
out_dir = Ldir['LOo'] / 'obs' / source / otype
Lfun.make_dir(out_dir)


# %%

# Load big data set and stations.

fn = glob.glob(str(in_dir0) + '/' + otype + '/*.csv')

# %%

big_df_raw = pd.DataFrame()

for f in fn:

    
    raw = pd.read_csv(f, encoding='cp1252')
    
    if 'ï»¿Locator' in raw.columns:
        
        raw = raw.rename(columns={'ï»¿Locator':'Locator'})
    
    if big_df_raw.empty:
        
        big_df_raw = raw

        
    else:
        
        big_df_raw = pd.concat([big_df_raw, raw])
    

# %%

sta_df = pd.read_csv(in_dir0 / 'WLRD_Sites_March2024.csv')

# %%

big_df = big_df_raw.merge(sta_df[['Locator','Latitude', 'Longitude']], on = 'Locator', how='left')


# %%

big_df_use0 = big_df[big_df['Updown'] == 'Down']


# %% 
#cols_all = big_df_use0['Parameter'].unique()

v_dict = {'Chlorophyll, Field (mg/m^3)':'Chl (mg m-3)',
          #'Density field':'rho',
          'Dissolved Oxygen, Field (mg/l ws=2)': 'DO (mg -L)',
          'Nitrite + Nitrate Nitrogen, Field (mg/L)': 'NO3 (mg -L)', #measured together assuming a 0 NO2 (add that column later), NEED TO CONVERT to micromolar
          'Salinity, Field (PSS)':'SP', #NEED TO CONVERT TO ABS SALINITY if necessary???
          'Sample Temperature, Field (deg C)':'IT' #NEED TO COVERT TO CONS TEMP if necessary???
          } # not dealing with light/PAR right now...

# %%

v_dict_use = {}

for v in v_dict.keys():
    if len(v_dict[v]) > 0:
        v_dict_use[v] = v_dict[v]
        
v_list = np.array(list(v_dict_use.keys()))
        
# %%

# big_df_use1 = big_df_use0[big_df_use0['Parameter'].isin(v_list)]

# # %%

# big_df_use2 = big_df_use1[['Sample ID','Profile ID', 'Collect DateTime', 'Depth (m)', 'Parameter', 'Value', 'Replicates', 'Replicate Of', 'Latitude', 'Longitude', 'Locator']]


# # %%

# replicates = big_df_use2['Replicates'].dropna().unique()

# big_df_use3 = big_df_use2[~big_df_use2['Sample ID'].isin(replicates)]

# # %%

# big_df_use4 = big_df_use3[['Profile ID', 'Collect DateTime', 'Depth (m)', 'Parameter', 'Value', 'Latitude', 'Longitude', 'Locator']]

# # %%

# big_df_use5 = big_df_use4.pivot_table(index = ['Profile ID', 'Collect DateTime', 'Depth (m)', 'Latitude', 'Longitude', 'Locator'],
#                                       columns = 'Parameter', values = 'Value').reset_index()

# %%

big_df_use6 = big_df_use0.copy()

big_df_use6['time'] = pd.DatetimeIndex(big_df_use6['Sampledate'])

big_df_use6['cid'] = np.nan

# %%

big_df_use7 = big_df_use6.copy()


big_df_use7['unique_date_location'] = big_df_use7['Locator'] + (big_df_use7['time'].dt.year).astype(str) + (big_df_use7['time'].dt.month).astype(str) + (big_df_use7['time'].dt.day).astype(str)

# %%
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
        'NO3 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'SiO4 (uM)', #removed NO2...
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