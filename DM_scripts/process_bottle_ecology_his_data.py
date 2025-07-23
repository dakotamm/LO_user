"""
Code to process the Ecology historical monitoring data for Puget Sound.

7/10/2024

To process data received via records request on 5/31/2024.

"""

import pandas as pd
import numpy as np
import gsw
import sys

from lo_tools import Lfun, obs_functions
Ldir = Lfun.Lstart()

# BOTTLE
source = 'ecology_his'
otype = 'bottle' # NOT SURE IF THIS IS TRUE
in_dir0 = Ldir['data'] / 'obs' / source
year_list = range(1973,1999)

# output location
out_dir = Ldir['LOo'] / 'obs' / source / otype
Lfun.make_dir(out_dir)


# %%

# Load big data set and stations.

big_df_raw0 = pd.read_excel(in_dir0/ 'Aug1973toOct1989CTDandDiscrete.xlsx', parse_dates=['Date'], skiprows=[0])

big_df_raw1 = pd.read_excel(in_dir0/ 'Nov1989toDec1998Discrete.xlsx', parse_dates=['Date'], skiprows=[0])

#ctd_raw = pd.read_excel(in_dir0/ 'Nov1989toDec1998CTDprofiles.xlsx', parse_dates=['Date']) ### don't actually need this for my purposes right now...

sta_df = pd.read_excel(in_dir0 / 'ParkerMacCreadyCoreStationInfoFeb2018.xlsx') #taken from apogee: dat1/parker/LO_data/ecology


# %%

xx = sta_df['Long_NAD83 (deg / dec_min)'].values
yy = sta_df['Lat_NAD83 (deg / dec_min)'].values
lon = [-(float(x.split()[0]) + float(x.split()[1])/60) for x in xx]
lat = [(float(y.split()[0]) + float(y.split()[1])/60) for y in yy]
sta_df['lon'] = lon
sta_df['lat'] = lat

# %%

# actually I'm just going to leave out the CTD DO stuff because I don't need it right now but I can see why it might be necessary later!!! noted 7/12/2024

#big_df_raw1 = big_df_raw1.drop(columns=['DO'])

#big_df_raw1 = big_df_raw1.merge(ctd_raw[['Date','Station', 'DepthInterval', 'Temp', 'Salinity', 'DO']], on = ['Date', 'Station', 'DepthInterval'], how='left')


# %%

big_df_raw0 = big_df_raw0.rename(columns = {'NO2+NO3':'NO2+NO3 (total size fraction)', 'NO2':'NO2 (total size fraction)', 'NH4':'NH4 (total size fraction)', 'TP':'TP (total size fraction)', 'PO4':'PO4 (total size fraction)', 'TOC':'TOC (total size fraction)'})

big_df_raw1 = big_df_raw1.rename(columns = {'NO2+NO3':'NO2+NO3 (dissolved size fraction)', 'NO2':'NO2 (dissolved size fraction)', 'NH4':'NH4 (dissolved size fraction)', 'PO4':'PO4 (dissolved size fraction)', 'NO2+NO3.1':'NO2+NO3 (total size fraction)', 'NO2.1':'NO2 (total size fraction)', 'NH4.1':'NH4 (total size fraction)', 'PO4.1':'PO4 (total size fraction)', 'TP':'TP (total size fraction)'})


# %%


big_df_raw = pd.concat([big_df_raw0, big_df_raw1])


# %%

big_df = big_df_raw.merge(sta_df[['Station','lat', 'lon']], on = 'Station', how='left')

# %%

data_original_names = ['Temp', 'Salinity', 'DO', 'Chla', 'NO2 (total size fraction)', 'NO2+NO3 (total size fraction)', 'NH4 (total size fraction)', 'PO4 (total size fraction)']

data_new_names = ['IT', 'SP', 'DO (mg -L)', 'Chl (ug -L)', 'NO2 (mg -L)', 'NO2+NO3 (mg -L)', 'NH4 (mg -L)', 'PO4 (mg -L)']

# %%

big_df_use = big_df.copy()

big_df_use['cid'] = np.nan

big_df_use['unique_date_location'] = big_df_use['Station'] + big_df_use['Date'].dt.strftime('%Y-%m-%d')

c = 0

for pid in big_df_use['unique_date_location'].unique(): # profile ID is unique identifier
    
    big_df_use.loc[big_df_use['unique_date_location'] == pid, 'cid'] = c
    
    c+=1

# %%

v_dict = dict(zip(data_original_names, data_new_names))


v_dict['cid'] = 'cid'

v_dict['Date'] = 'time'

v_dict['lat'] = 'lat'

v_dict['lon'] = 'lon'

v_dict['DepthInterval'] = 'z' #convert to negative

v_dict['Station'] = 'name'


# %%


df0 = big_df_use.copy()

for year in year_list:
    
    ys = str(year)
    print('\n'+ys)

    out_fn = out_dir / (ys + '.p')
    info_out_fn = out_dir / ('info_' + ys + '.p')
    
    t = pd.DatetimeIndex(df0['Date'])
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
        
    if 'NO2+NO3 (mg -L)' in df.columns:
        df['NO3 (uM)'] = (1000/62) * (df['NO2+NO3 (mg -L)'] - df['NO2 (mg -L)'])
        
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