"""
Code to process the King County Water Quality data for Puget Sound.

IN PROGRESS DM 7/13/2023

some notes from https://green2.kingcounty.gov/marine/Monitoring/OffshoreCTD: "Light Transmission data prior to May 19, 2014 were referenced to air. After this date, all Light Transmission data are referenced to water. To convert the pre-May 19, 2014 data to ‘referenced to water’, multiply the values by 1.095."

***any modifications done to temperature???

***local times or UTC?

sites: https://data.kingcounty.gov/Environment-Waste-Management/WLRD-Sites/wbhs-bbzf

big data set: https://data.kingcounty.gov/Environment-Waste-Management/Water-Quality/vwmt-pvjw
- from 1965 to 2023

whidbey bottle (***I THINK THIS IS JUST A FILTERED BY AREA VERSION OF THE BIG DATA SET): https://data.kingcounty.gov/Environment-Waste-Management/Whidbey-Bottle-Data/vuu8-t6kc

whidbey CTD (additional detail just for whidbey - not sure where the rest of CTD casts are): https://data.kingcounty.gov/Environment-Waste-Management/Whidbey-Basin-CTD-Casts/uz4m-4d96

"""

import pandas as pd
import numpy as np
import gsw
import sys

from lo_tools import Lfun, obs_functions
Ldir = Lfun.Lstart()

# BOTTLE
source = 'kc'
otype = 'bottle'
in_dir0 = Ldir['data'] / 'obs' / source
year_list = range(1963,2024)

# output location
out_dir = Ldir['LOo'] / 'obs' / source / otype
Lfun.make_dir(out_dir)


# %%

# Load big data set and stations.

big_df_raw = pd.read_csv(in_dir0 / otype / 'Water_Quality_July2023.csv')

sta_df = pd.read_csv(in_dir0 / 'WLRD_Sites_July2023.csv')

# %%

big_df = big_df_raw.merge(sta_df[['Locator','Latitude', 'Longitude']], on = 'Locator', how='left')


# %%

big_df_use0 = big_df[big_df['Site Type'] == 'Marine Offshore']


# %% 

cols_all = big_df_use0['Parameter'].unique()

v_dict = {}

v_dict = {col:'' for col in cols_all}

# %% 

v_dict['Temperature'] = 'CT' #NOT SURE IF CONSERVATIVE OR NOT

v_dict['Salinity'] = 'SP' #NEED TO CONVERT TO ABS SALINITY if necessary???

v_dict['Dissolved Oxygen'] = 'DO (mg -L)' #NEED TO CONVERT TO micromolar

v_dict['Nitrite + Nitrate Nitrogen'] = 'NO3 (mg -L)' #measured together assuming a 0 NO2 (add that column later), NEED TO CONVERT to micromolar

v_dict['Ammonia Nitrogen'] = 'NH4 (mg -L)' #NEED TO CONVERT to micromolar

v_dict['Total Phosphorus'] = 'PO4 (mg -L)' #NEED TO CONVERT to micromolar - OR should I use ORTHOPHOSPHATE PHOSPHORUS??

v_dict['Silica'] = 'SiO4 (mg -L)' #NEED TO CONVERT to micromolar, I think this is correct but not sure***

v_dict['Total Alkalinity'] = 'TA (umol -kg)' #need to convert to micromolar!!!

v_dict['Dissolved Inorganic Carbon'] = 'DIC (umol -kg)'

v_dict['Chlorophyll a'] = 'Chl (ug -L)' #need to convert to mg/m^3

# %%

v_dict_use = {}

for v in v_dict.keys():
    if len(v_dict[v]) > 0:
        v_dict_use[v] = v_dict[v]
        
v_list = np.array(list(v_dict_use.keys()))
        
# %%

big_df_use1 = big_df_use0[big_df_use0['Parameter'].isin(v_list)]

# %%

big_df_use2 = big_df_use1[['Sample ID','Profile ID', 'Collect DateTime', 'Depth (m)', 'Parameter', 'Value', 'Replicates', 'Replicate Of', 'Latitude', 'Longitude']]


# %%

replicates = big_df_use2['Replicates'].dropna().unique()

big_df_use3 = big_df_use2[~big_df_use2['Sample ID'].isin(replicates)]

# %%

big_df_use4 = big_df_use3[['Profile ID', 'Collect DateTime', 'Depth (m)', 'Parameter', 'Value', 'Latitude', 'Longitude']]

# %%

big_df_use5 = big_df_use4.pivot_table(index = ['Profile ID', 'Collect DateTime', 'Depth (m)', 'Latitude', 'Longitude'],
                                      columns = 'Parameter', values = 'Value').reset_index()

# %%

big_df_use6 = big_df_use5.copy()

big_df_use6['time'] = pd.DatetimeIndex(big_df_use6['Collect DateTime'])

big_df_use6['cid'] = np.nan

big_df_use6['']



# %%

for year in year_list:
    
    ys = str(year)
    print('\n'+ys)

    out_fn = out_dir / (ys + '.p')
    info_out_fn = out_dir / ('info_' + ys + '.p')
    
    



# %%

load_data = True
for year in year_list:
    ys = str(year)
    print('\n'+ys)
    
    # name output files
    out_fn = out_dir / (ys + '.p')
    info_out_fn = out_dir / ('info_' + ys + '.p')
    
    if (year in range(2006,2018)) and load_data:
        in_fn =  in_dir0 / 'bottle_2006_2017_fixed.p'
        df0 = pd.read_pickle(in_fn)
        # This code was used to generate the text used in v_dict above.
        if False:
            for v in df0.columns:
                print("\'%s\':\'\'," % (v))
        # select and rename variables
        df1 = pd.DataFrame()
        for v in df0.columns:
            if v in v_dict.keys():
                if len(v_dict[v]) > 0:
                    df1[v_dict[v]] = df0[v]
        load_data = False # only load the first time










# %%




# This is a dict of all the columns after the initial reading.
# We add values to a key for any variable we want to save
v_dict = {
    'time':'time',
    'record':'',
    'EXPOCODE':'',
    'CRUISE_ID':'cruise',
    'DATE_LOCAL':'',
    'TIME_LOCAL':'',
    'LONGITUDE_DEC':'lon',
    'LATITUDE_DEC':'lat',
    'STATION_NO':'name',
    'NISKIN_NO':'',
    'CTDPRS_DBAR':'P (dbar)',
    'CTDTMP_DEG_C_ITS90':'IT', # in situ temperature, deg C
    'CTDTMP_FLAG_W':'',
    'CTDSAL_PSS78':'SP',
    'CTDSAL_FLAG_W':'',
    'SIGMATHETA_KG_M3':'',
    'CTDOXY_UMOL_KG_ADJ':'DO (umol/kg)',
    'CTDOXY_UMOL_KG':'',
    'CTDOXY_MG_L_1':'',
    'CTDOXY_MG_L_2':'',
    'CTDOXY_FLAG_W':'',
    'OXYGEN_UMOL_KG':'',
    'OXYGEN_MG_L_1':'',
    'OXYGEN_MG_L_2':'',
    'OXYGEN_MG_L_3':'',
    'OXYGEN_FLAG_W':'',
    'TA_UMOL_KG':'TA (umol/kg)',
    'DIC_UMOL_KG':'DIC (umol/kg)',
    'TA_FLAG_W':'',
    'DIC_FLAG_W':'',
    'NITRATE_UMOL_KG':'',
    'NITRATE_UMOL_L':'NO3 (uM)',
    'NITRITE_UMOL_KG':'',
    'NITRITE_UMOL_L':'NO2 (uM)',
    'AMMONIA_UMOL_KG':'',
    'AMMONIUM_UMOL_L':'NH4 (uM)',
    'PHOSPHATE_UMOL_KG':'',
    'PHOSPHATE_UMOL_L':'PO4 (uM)',
    'SILICATE_UMOL_KG':'',
    'SILICATE_UMOL_L':'Si4 (uM)',
    'NUTRIENTS_FLAG_W':'',
}

load_data = True
for year in year_list:
    ys = str(year)
    print('\n'+ys)
    
    # name output files
    out_fn = out_dir / (ys + '.p')
    info_out_fn = out_dir / ('info_' + ys + '.p')
    
    if year in range(2008,2019):
        in_fn =  in_dir0 / 'SalishCruise_dataPackage_2008to2018_06-29-2021.csv'
        df0 = pd.read_csv(in_fn, parse_dates={'time':['DATE_UTC', 'TIME_UTC']})
        load_data = False # only load the first time (not used?)
    
    # select one year
    t = pd.DatetimeIndex(df0.time)
    df1 = df0.loc[t.year==year,:].copy()
    
    # select and rename variables
    df = pd.DataFrame()
    for v in df1.columns:
        if v in v_dict.keys():
            if len(v_dict[v]) > 0:
                df[v_dict[v]] = df1[v]
                
    # missing data is -999
    df[df==-999] = np.nan
    
    # a little more cleaning up
    df = df.dropna(axis=0, how='all') # drop rows with no good data
    df = df[df.time.notna()] # drop rows with bad time
    df = df.reset_index(drop=True)
    
    # Now proceed with the processing to get a single DataFrame for the year.
    
    # add the "cid" (cast ID) column
    #
    # Note that we will save the field "name" for station number, since this dataset has
    # repeat stations which is helpful for plotting sections. Then we will generate our own
    # cid, a unique one for each cast, being careful to keep them unique for the collection
    # of cruises in this year, even though a station may be repeated on all cruises.
    #
    # We will also save the field "cruise" as a convenient way to select a collection of
    # casts.
    df['cid'] = np.nan
    cid = 0
    for cruise in df.cruise.unique():
        for name in df.name.unique():
            df.loc[(df.name==name) & (df.cruise==cruise),'cid'] = cid
            cid += 1
    for cid in df.cid.unique():
        # Check that there are not two different casts associated with the same station
        # by looking for large time differences. Pretty ad hoc, but it works.
        time_diff = df[df.cid==cid].time.values[-1] - df[df.cid==cid].time.values[0]
        time_diff = pd.to_timedelta(time_diff)
        if time_diff.days > 1 or time_diff.days < -1:
            cruise = df[df.cid==cid].cruise.values[0]
            name = df[df.cid==cid].name.values[0]
            print('Cruise: %s, Station %s has time diff of %d days' % (cruise, str(name), time_diff.days))
            # copy in just the first cast at this repeated station
            dff = df[df.cid==cid].copy()
            dfft = dff.time.values
            Dfft = pd.to_timedelta(dfft - dfft[0])
            dff = dff[Dfft.days==0]
            print('  - length of df before removing repeat cast at this station: %d' % (len(df)))
            df = df[df.cid != cid]
            df = pd.concat((df,dff))
            print('  - length of df before removing repeat cast at this station: %d' % (len(df)))
        # Force certain fields to be the same throughout the cast.
        df.loc[df.cid==cid,'lon'] = df[df.cid==cid].lon.values[0]
        df.loc[df.cid==cid,'lat'] = df[df.cid==cid].lat.values[0]
        df.loc[df.cid==cid,'time'] = df[df.cid==cid].time.values[0]
                    
    # Next make derived quantities and do unit conversions

    # (1) Create CT, SA, and z
    # - pull out variables
    SP = df.SP.to_numpy()
    IT = df.IT.to_numpy()
    p = df['P (dbar)'].to_numpy()
    lon = df.lon.to_numpy()
    lat = df.lat.to_numpy()
    # - do the conversions
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    CT = gsw.CT_from_t(SA, IT, p)
    z = gsw.z_from_p(p, lat)
    # - add the results to the DataFrame
    df['SA'] = SA
    df['CT'] = CT
    df['z'] = z
    rho = gsw.rho(SA,CT,p)

    # (2) units
    if 'DO (mg/L)' in df.columns:
        df['DO (uM)'] = (1000/32) * df['DO (mg/L)']
    for vn in ['DO','TA','DIC']:
        if (vn+' (umol/kg)') in df.columns:
            df[vn+' (uM)'] = (rho/1000) * df[vn+' (umol/kg)']
        
    # (3) retain only selected variables
    cols = ['cid', 'cruise', 'time', 'lat', 'lon', 'name', 'z',
        'CT', 'SA', 'DO (uM)',
        'NO3 (uM)', 'NO2 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'SiO4 (uM)',
        'TA (uM)', 'DIC (uM)']
    this_cols = [item for item in cols if item in df.columns]
    df = df[this_cols]
        
    print(' - processed %d casts' % ( len(df.cid.unique()) ))
        
    # Renumber cid to be increasing from zero in steps of one.
    df = obs_functions.renumber_cid(df)
    
    if len(df) > 0:
        # Save the data
        df.to_pickle(out_fn)
        info_df = obs_functions.make_info_df(df)
        info_df.to_pickle(info_out_fn)
