"""
Functions used by Dakota!

Created 2023/12/14.

"""

import sys
import pandas as pd
import xarray as xr
import numpy as np

from lo_tools import Lfun, zfun, zrfun #original LO tools
import extract_argfun_DM as exfun
import cast_functions_DM as cfun
from lo_tools import plotting_functions as pfun #original LO tools
import tef_fun_DM as tfun
import pickle

from time import time as Time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from scipy.spatial import KDTree

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import matplotlib.colors as cm

import math


import itertools

import copy

from pathlib import PosixPath

from datetime import timedelta


from dateutil.relativedelta import relativedelta

import matplotlib.path as mpth
import datetime

from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages

import seaborn as sns

import scipy.stats as stats

import gsw



# %%

def getPolyData(Ldir, poly_list, source_list=['ecology_nc', 'nceiSalish', 'dfo1', 'collias'], otype_list=['ctd', 'bottle'], year_list=np.arange(1930, 2022)):

    fng = Ldir['grid'] / 'grid.nc'
    dsg = xr.open_dataset(fng)
    x = dsg.lon_rho.values
    y = dsg.lat_rho.values
    m = dsg.mask_rho.values
    xp, yp = pfun.get_plon_plat(x,y)
    h = dsg.h.values
    h[m==0] = np.nan
    
    path_dict = dict()
    xxyy_dict = dict()
    for poly in poly_list:
        # polygon
        fnp = Ldir['LOo'] / 'section_lines' / (poly+'.p')
        p = pd.read_pickle(fnp)
        xx = p.x.to_numpy()
        yy = p.y.to_numpy()
        xxyy = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
        path = mpth.Path(xxyy)
        # store in dicts
        path_dict[poly] = path
        xxyy_dict[poly] = xxyy

    # observations
    ii = 0
    for year in year_list:
        for source in source_list:
            for otype in otype_list:
                odir = Ldir['LOo'] / 'obs' / source / otype
                try:
                    if ii == 0:
                        odf = pd.read_pickle( odir / (str(year) + '.p'))
                        if 'ecology' in source_list:
                            if source == 'ecology' and otype == 'bottle': #keep an eye on this for calculating confidence intervals!!!
                                odf['DO (uM)'] == np.nan
                        if 'kc_pointJefferson' in source_list:
                            if source == 'kc_pointJefferson' and otype == 'bottle': #keep an eye on this for calculating confidence intervals!!!
                                odf['CT'] == np.nan    
                        if 'kc_his' in source_list:
                            if source == 'kc_his' and otype == 'bottle': #keep an eye on this for calculating confidence intervals!!!
                                odf['CT'] == np.nan    
                        odf['source'] = source
                        odf['otype'] = otype
                        # print(odf.columns)
                    else:
                        this_odf = pd.read_pickle( odir / (str(year) + '.p'))
                        if 'ecology' in source_list:
                            if source == 'ecology' and otype == 'bottle':
                                this_odf['DO (uM)'] == np.nan
                        if 'kc_pointJefferson' in source_list:
                            if source == 'kc_pointJefferson' and otype == 'bottle': #keep an eye on this for calculating confidence intervals!!!
                                odf['CT'] == np.nan   
                        if 'kc_his' in source_list:
                            if source == 'kc_his' and otype == 'bottle': #keep an eye on this for calculating confidence intervals!!!
                                odf['CT'] == np.nan
                        this_odf['cid'] = this_odf['cid'] + odf['cid'].max() + 1
                        this_odf['source'] = source
                        this_odf['otype'] = otype
                        # print(this_odf.columns)
                        odf = pd.concat((odf,this_odf),ignore_index=True)
                    ii += 1
                except FileNotFoundError:
                    pass
                
        print(str(year))

    # if True:
    #     # limit time range
    #     ti = pd.DatetimeIndex(odf.time)
    #     mo = ti.month
    #     mo_mask = mo==0 # initialize all false
    #     for imo in [9,10,11]:
    #         mo_mask = mo_mask | (mo==imo)
    #     odf = odf.loc[mo_mask,:]
        
    # get lon lat of (remaining) obs
    ox = odf.lon.to_numpy()
    oy = odf.lat.to_numpy()
    oxoy = np.concatenate((ox.reshape(-1,1),oy.reshape(-1,1)), axis=1)


    # get all profiles inside each polygon
    odf_dict = dict()
    for poly in poly_list:
        path = path_dict[poly]
        oisin = path.contains_points(oxoy)
        odfin = odf.loc[oisin,:]
        odf_dict[poly] = odfin.copy()
        
    return odf_dict, path_dict

# %%

def getPathDict(Ldir,poly_list):
    
    path_dict = dict()

    for poly in poly_list:
        # polygon
        fnp = Ldir['LOo'] / 'section_lines' / (poly+'.p')
        p = pd.read_pickle(fnp)
        xx = p.x.to_numpy()
        yy = p.y.to_numpy()
        xxyy = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
        path = mpth.Path(xxyy)
        # store in dicts
        path_dict[poly] = path
        path = path_dict[poly]
        
    return path_dict
    

# %%

def mann_kendall(V, alpha=0.05):
    '''Mann Kendall Test (adapted from original Matlab function)
       Performs original Mann-Kendall test of the null hypothesis of trend absence in the vector V, against the alternative of trend.
       The result of the test is returned in reject_null:
       reject_null = True indicates a rejection of the null hypothesis at the alpha significance level. 
       reject_null = False indicates a failure to reject the null hypothesis at the alpha significance level.

       INPUTS:
       V = time series [vector]
       alpha =  significance level of the test [scalar] (i.e. for 95% confidence, alpha=0.05)
       OUTPUTS:
       reject_null = True/False (True: reject the null hypothesis) (False: insufficient evidence to reject the null hypothesis)
       p_value = p-value of the test
       
       From Original Matlab Help Documentation:
       The significance level of a test is a threshold of probability a agreed to before the test is conducted. 
       A typical value of alpha is 0.05. If the p-value of a test is less than alpha,        
       the test rejects the null hypothesis. If the p-value is greater than alpha, there is insufficient evidence 
       to reject the null hypothesis. 
       The p-value of a test is the probability, under the null hypothesis, of obtaining a value
       of the test statistic as extreme or more extreme than the value computed from
       the sample.
       
       References 
       Mann, H. B. (1945), Nonparametric tests against trend, Econometrica, 13, 245-259.
       Kendall, M. G. (1975), Rank Correlation Methods, Griffin, London.
       
       Original written by Simone Fatichi - simonef@dicea.unifi.it
       Copyright 2009
       Date: 2009/10/03
       modified: E.I. (1/12/2012)
       modified and converted to python: Steven Pestana - spestana@uw.edu (10/17/2019)
       '''

    V = np.reshape(V, (len(V), 1))
    alpha = alpha/2
    n = len(V)
    S = 0

    for i in range(0, n-1):
        for j in range(i+1, n):
            if V[j]>V[i]:
                S = S+1
            if V[j]<V[i]:
                S = S-1

    VarS = (n*(n-1)*(2*n+5))/18
    StdS = np.sqrt(VarS)
    # Ties are not considered

    # Kendall tau correction coefficient
    Kendall_Tau = S/(n*(n-1)/2)
    if S>=0:
        if S==0:
             Z = 0
        else:
            Z = ((S-1)/StdS)
    else:
        Z = (S+1)/StdS

    Zalpha = stats.norm.ppf(1-alpha,0,1)
    p_value = 2*(1-stats.norm.cdf(abs(Z), 0, 1)) #Two-tailed test p-value

    reject_null = abs(Z) > Zalpha # reject null hypothesis only if abs(Z) > Zalpha
    
    return reject_null, p_value, Z
    
# %%
    
def dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list):
    
    for key in odf_dict.keys():
        
        odf_dict[key] = (odf_dict[key]
                          .assign(
                              datetime=(lambda x: pd.to_datetime(x['time'], utc=True)),
                              year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                              month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                              # season=(lambda x: pd.cut(x['month'],
                              #                         bins=[0,3,7,11,12],
                              #                         labels=['winter', 'grow', 'loDO', 'winter'], ordered=False)),
                              DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                              # NO3_uM=(lambda x: x['NO3 (uM)']),
                              # Chl_mg_m3=(lambda x: x['Chl (mg m-3)']),
                              date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())),
                              segment=(lambda x: key),
                              decade=(lambda x: pd.cut(x['year'],
                                                      bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                                                      labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
                                  )
                          )
        
        odf_dict[key].loc[odf_dict[key]['month'].isin([1,2,3,12]), 'season'] = 'winter'
        
        odf_dict[key].loc[odf_dict[key]['month'].isin([4,5,6,7]), 'season'] = 'grow'
        
        odf_dict[key].loc[odf_dict[key]['month'].isin([8,9,10,11]), 'season'] = 'loDO'


        
        for var in var_list:
            
            if var not in odf_dict[key].columns:
                
                odf_dict[key][var] = np.nan
                
        odf_dict[key] = pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'decade', 'name'],
                                              value_vars=var_list, var_name='var', value_name = 'val')
        

    odf = pd.concat(odf_dict.values(), ignore_index=True)


    odf['source_type'] = odf['source'] + '_' + odf['otype']


    odf = odf.dropna()
    
    odf = odf.assign(
        ix=(lambda x: x['lon'].apply(lambda x: zfun.find_nearest_ind(lon_1D, x))),
        iy=(lambda x: x['lat'].apply(lambda x: zfun.find_nearest_ind(lat_1D, x)))
    )


    odf['h'] = odf.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)


    odf['yearday'] = odf['datetime'].dt.dayofyear


    odf = odf[odf['val'] >0]


    max_depths_dict = dict()

    ox = lon
    oy = lat
    oxoy = np.concatenate((ox.reshape(-1,1),oy.reshape(-1,1)), axis=1)


    for poly in poly_list:

        path = path_dict[poly]
        
        oisin = path.contains_points(oxoy)
        
        this_depths = depths.flatten()[oisin]
        
        max_depth = np.nanmax(this_depths)
        
        max_depths_dict[poly] = max_depth.copy()
        


    for basin in basin_list:
        
        odf.loc[odf['segment'] == basin, 'min_segment_h'] = -max_depths_dict[basin]
    
    return odf

# %%

# FOR SHORT & LONG TIME HISTORIES - v bespoke(created 9/6/2024)

## more info: this deals with cleaning the odf for the five polygons for 60+ year time histories and then the 20-year ecology sites and bins into two depths

def longShortClean(odf):

    short_exclude_sites = ['BUD002', 'QMH002', 'PMA001', 'OCH014', 'DYE004', 'SUZ001', 'HLM001', 'PNN001', 'PSS010', 'TOT002', 'TOT001', 'HND001','ELD001', 'ELD002', 'CSE002', 'CSE001', 'HCB010', 'SKG003','HCB006', 'HCB008', 'HCB009', 'CMB006', 'EAG001', 'HCB013', 'POD007']
    
    big_basin_list = ['mb', 'wb', 'ss', 'hc']
    
    long_site_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']
    
    
    short_mask_ecology = (odf['segment'].isin(big_basin_list)) & (odf['source'] == 'ecology_nc') & (~odf['name'].isin(short_exclude_sites)) & (odf['year'] >= 1998)
    
    short_mask_point_jefferson = (odf['segment'] == 'mb') & (odf['name'] =='KSBP01') & (odf['year'] >= 1998)
    
    long_mask = (odf['segment'].isin(long_site_list))
    
    
    odf.loc[short_mask_ecology, 'short_long'] = 'short'
    
    odf.loc[short_mask_point_jefferson, 'short_long'] = 'short'
    
    odf.loc[long_mask, 'short_long'] = 'long'
    
    
    odf = odf[odf['short_long'] != 'nan']
    
    
    short_site_list = odf[odf['short_long'] == 'short']['name'].unique().tolist()
    
    
    
    long_deep_non_lc_nso_mask = (odf['z'] < 0.8*odf['min_segment_h']) & (odf['segment'] != 'lynch_cove_mid') & (odf['segment'] != 'near_seattle_offshore') & (odf['short_long'] == 'long')
    
    long_deep_lc_mask = (odf['z'] < 0.4*odf['min_segment_h']) & (odf['segment'] == 'lynch_cove_mid') & (odf['short_long'] == 'long')
    
    long_deep_nso_mask = (odf['z'] < 0.75*odf['min_segment_h']) & (odf['segment'] == 'near_seattle_offshore') & (odf['short_long'] == 'long') #CHANGED 5/21/2024
    
    
    short_deep_mask = (odf['z'] < 0.8*odf['h']) & (odf['short_long'] == 'short')
    
    surf_mask = (odf['z'] >= -5)
    
    
    odf.loc[surf_mask, 'surf_deep'] = 'surf'
    
    odf.loc[long_deep_non_lc_nso_mask, 'surf_deep'] = 'deep'
    
    odf.loc[long_deep_lc_mask, 'surf_deep'] = 'deep'
    
    odf.loc[long_deep_nso_mask, 'surf_deep'] = 'deep'
    
    odf.loc[short_deep_mask, 'surf_deep'] = 'deep'
    
    
    odf.loc[odf['short_long'] == 'short', 'site'] = odf[odf['short_long'] == 'short']['name']
    
    odf.loc[odf['short_long'] == 'long', 'site'] = odf[odf['short_long'] == 'long']['segment']
    
    print('done')
    
    
    temp = odf.groupby(['site','cid']).min(numeric_only=True).reset_index()
    
    cid_exclude = temp[(temp['site'].isin(['HCB005', 'HCB007', 'lynch_cove_mid'])) & (temp['z'] < -50)]['cid']
    
    odf = odf[~odf['cid'].isin(cid_exclude)]
    
    
    
    temp0 = odf[odf['surf_deep'] != 'nan']
    
    # DM addition 20250801
    temp0.loc[(temp0['var'] == 'CT') & (temp0['val'] >27)] = np.nan


    odf_depth_mean = temp0.groupby(['site','surf_deep', 'year', 'season', 'var','cid']).mean(numeric_only=True).reset_index().dropna() #####
    
    # DM addition 20250812
    odf_depth_mean.loc[(odf_depth_mean['site'] == 'near_seattle_offshore') & (odf_depth_mean['var'] == 'SA') & (odf_depth_mean['surf_deep'] == 'deep') & (odf_depth_mean['val'] < 29)] = np.nan
    odf_depth_mean = odf_depth_mean.dropna()


    cid_deep = odf_depth_mean.loc[odf_depth_mean['surf_deep'] == 'deep', 'cid']


    odf_depth_mean_deep = odf_depth_mean[odf_depth_mean['cid'].isin(cid_deep)]


    odf_calc = odf_depth_mean_deep.pivot(index = ['site', 'year', 'month', 'season','date_ordinal','cid'], columns = ['surf_deep', 'var'], values ='val')

    odf_calc.columns = odf_calc.columns.to_flat_index().map('_'.join)

    odf_calc = odf_calc.reset_index()


    odf_calc['surf_dens'] = gsw.density.sigma0(odf_calc['surf_SA'], odf_calc['surf_CT'])

    odf_calc['deep_dens'] = gsw.density.sigma0(odf_calc['deep_SA'], odf_calc['deep_CT'])


    odf_calc['strat_sigma'] = odf_calc['deep_dens'] - odf_calc['surf_dens']


    A_0 = 5.80818 #all in umol/kg, from Gordon & Garcia (1992)

    A_1 = 3.20684

    A_2 = 4.11890

    A_3 = 4.93845

    A_4 = 1.01567

    A_5 = 1.41575

    B_0 = -7.01211e-3

    B_1 = -7.25958e-3

    B_2 = -7.93334e-3

    B_3 = -5.54491e-3

    C_0 = -1.32412e-7


    odf_calc['surf_T_s'] = np.log((298.15 - odf_calc['surf_CT'])/(273.15 + odf_calc['surf_CT']))

    odf_calc['surf_C_o_*'] = np.exp(A_0 + A_1*odf_calc['surf_T_s'] + A_2*odf_calc['surf_T_s']**2 + A_3*odf_calc['surf_T_s']**3 + A_4*odf_calc['surf_T_s']**4 + A_5*odf_calc['surf_T_s']**5 + 
                           odf_calc['surf_SA']*(B_0 + B_1*odf_calc['surf_T_s'] + B_2*odf_calc['surf_T_s']**2 + B_3*odf_calc['surf_T_s']**3) + C_0*odf_calc['surf_SA']**2)

    odf_calc['surf_DO_sol'] =  odf_calc['surf_C_o_*']*(odf_calc['surf_dens']/1000 + 1)*32/1000


    odf_calc['deep_T_s'] = np.log((298.15 - odf_calc['deep_CT'])/(273.15 + odf_calc['deep_CT']))

    odf_calc['deep_C_o_*'] = np.exp(A_0 + A_1*odf_calc['deep_T_s'] + A_2*odf_calc['deep_T_s']**2 + A_3*odf_calc['deep_T_s']**3 + A_4*odf_calc['deep_T_s']**4 + A_5*odf_calc['deep_T_s']**5 + 
                           odf_calc['deep_SA']*(B_0 + B_1*odf_calc['deep_T_s'] + B_2*odf_calc['deep_T_s']**2 + B_3*odf_calc['deep_T_s']**3) + C_0*odf_calc['deep_SA']**2)

    odf_calc['deep_DO_sol'] =  odf_calc['deep_C_o_*']*(odf_calc['deep_dens']/1000 + 1)*32/1000



    odf_calc_long = pd.melt(odf_calc, id_vars = ['site', 'year', 'month', 'season', 'date_ordinal','cid'], value_vars=['strat_sigma', 'surf_DO_sol', 'deep_DO_sol'], var_name='var', value_name='val')


    odf_depth_mean_deep_DO = odf_depth_mean[(odf_depth_mean['var'] == 'DO_mg_L') & (odf_depth_mean['surf_deep'] == 'deep')]
    
    
    odf_depth_mean_deep_DO['year_fudge'] = odf_depth_mean_deep_DO['year']
    
    odf_depth_mean_deep_DO.loc[odf_depth_mean_deep_DO['month'] == 12, 'year_fudge'] = odf_depth_mean_deep_DO['year'] + 1
    



    odf_depth_mean_deep_DO_q50 = odf_depth_mean_deep_DO[['site', 'year_fudge', 'season','val']].groupby(['site', 'year_fudge', 'season']).quantile(0.5)

    odf_depth_mean_deep_DO_q50 = odf_depth_mean_deep_DO_q50.rename(columns={'val':'deep_DO_q50'})
        

    # odf_depth_mean_deep_DO_q75 = odf_depth_mean_deep_DO[['site', 'year', 'summer_non_summer','val']].groupby(['site', 'year', 'summer_non_summer']).quantile(0.75)

    # odf_depth_mean_deep_DO_q75 = odf_depth_mean_deep_DO_q75.rename(columns={'val':'deep_DO_q75'})

    # odf_depth_mean_deep_DO_q25 = odf_depth_mean_deep_DO[['site', 'year', 'summer_non_summer','val']].groupby(['site', 'year', 'summer_non_summer']).quantile(0.25)

    # odf_depth_mean_deep_DO_q25 = odf_depth_mean_deep_DO_q25.rename(columns={'val':'deep_DO_q25'})


    odf_depth_mean_deep_DO_percentiles = pd.merge(odf_depth_mean_deep_DO, odf_depth_mean_deep_DO_q50, how='left', on=['site','season','year_fudge'])

    # odf_depth_mean_deep_DO_percentiles = pd.merge(odf_depth_mean_deep_DO_percentiles, odf_depth_mean_deep_DO_q75, how='left', on=['site','summer_non_summer','year'])

    # odf_depth_mean_deep_DO_percentiles = pd.merge(odf_depth_mean_deep_DO_percentiles, odf_depth_mean_deep_DO_q25, how='left', on=['site','summer_non_summer','year'])


    site_list = short_site_list + long_site_list
    
    return odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list

# %%

def seasonalDepthAverageDF(odf_depth_mean, odf_calc_long):
    
    odf_working = odf_depth_mean.copy()
    
    odf_working['year_fudge'] = odf_working['year'] # this is to make seasons chronological because winter overlaps calendar years!
    
    odf_working.loc[odf_working['month'] == 12, 'year_fudge'] = odf_working['year'] + 1
    
    seasonal_counts_0 = (odf_working
                          .dropna()
                          .groupby(['site','year_fudge','surf_deep', 'season', 'var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )

    odf_use= odf_working.groupby(['site', 'surf_deep', 'season', 'year_fudge','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})


    odf_use.columns = odf_use.columns.to_flat_index().map('_'.join)

    odf_use = odf_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


    odf_use = (odf_use
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )


    odf_use = pd.merge(odf_use, seasonal_counts_0, how='left', on=['site','surf_deep', 'season', 'year_fudge','var'])


    odf_use = odf_use[odf_use['cid_count'] >1] #redundant but fine (see note line 234)

    odf_use['val_ci95hi'] = odf_use['val_mean'] + 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])

    odf_use['val_ci95lo'] = odf_use['val_mean'] - 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])
    
    odf_use['val'] = odf_use['val_mean']
    
    odf_use['year'] = odf_use['year_fudge']
    
    
    
    odf_calc_working = odf_calc_long.copy()
    
    odf_calc_working['year_fudge'] = odf_calc_working['year'] # this is to make seasons chronological because winter overlaps calendar years!
    
    odf_calc_working.loc[odf_calc_working['month'] == 12, 'year_fudge'] = odf_calc_working['year'] + 1
    
    
    seasonal_counts_1 = (odf_calc_working
                          .dropna()
                          .groupby(['site','year_fudge', 'season', 'var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )

    odf_calc_use= odf_calc_working.groupby(['site', 'season', 'year_fudge','var']).agg({'val':['mean', 'std'], 'date_ordinal':['mean']})


    odf_calc_use.columns = odf_calc_use.columns.to_flat_index().map('_'.join)

    odf_calc_use = odf_calc_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


    odf_calc_use = (odf_calc_use
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )


    odf_calc_use = pd.merge(odf_calc_use, seasonal_counts_1, how='left', on=['site','year_fudge', 'season', 'var'])


    odf_calc_use = odf_calc_use[odf_calc_use['cid_count'] >1] #redundant but fine (see note line 234)

    odf_calc_use['val_ci95hi'] = odf_calc_use['val_mean'] + 1.96*odf_calc_use['val_std']/np.sqrt(odf_calc_use['cid_count'])

    odf_calc_use['val_ci95lo'] = odf_calc_use['val_mean'] - 1.96*odf_calc_use['val_std']/np.sqrt(odf_calc_use['cid_count'])
    
    odf_calc_use['val'] = odf_calc_use['val_mean']

    odf_calc_use['year'] = odf_calc_use['year_fudge']


    
    return odf_use, odf_calc_use

# %%

def annualDepthAverageDF(odf_depth_mean, odf_calc_long):
    
    annual_counts_0 = (odf_depth_mean
                          .dropna()
                          .groupby(['site','year', 'surf_deep','var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )

    odf_use= odf_depth_mean.groupby(['site', 'year','surf_deep', 'var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})


    odf_use.columns = odf_use.columns.to_flat_index().map('_'.join)

    odf_use = odf_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


    odf_use = (odf_use
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )


    odf_use = pd.merge(odf_use, annual_counts_0, how='left', on=['site','year','surf_deep','var'])


    odf_use = odf_use[odf_use['cid_count'] >1] #redundant but fine (see note line 234)

    odf_use['val_ci95hi'] = odf_use['val_mean'] + 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])

    odf_use['val_ci95lo'] = odf_use['val_mean'] - 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])
    
    odf_use['val'] = odf_use['val_mean']
    
    
    annual_counts_1 = (odf_calc_long
                          .dropna()
                          .groupby(['site','year', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )

    odf_calc_use= odf_calc_long.groupby(['site', 'surf_deep', 'year','var']).agg({'val':['mean', 'std'], 'date_ordinal':['mean']})


    odf_calc_use.columns = odf_calc_use.columns.to_flat_index().map('_'.join)

    odf_calc_use = odf_calc_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


    odf_calc_use = (odf_calc_use
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )


    odf_calc_use = pd.merge(odf_calc_use, annual_counts_1, how='left', on=['site','year', 'surf_deep', 'var'])


    odf_calc_use = odf_calc_use[odf_calc_use['cid_count'] >1] #redundant but fine (see note line 234)

    odf_calc_use['val_ci95hi'] = odf_calc_use['val_mean'] + 1.96*odf_calc_use['val_std']/np.sqrt(odf_calc_use['cid_count'])

    odf_calc_use['val_ci95lo'] = odf_calc_use['val_mean'] - 1.96*odf_calc_use['val_std']/np.sqrt(odf_calc_use['cid_count'])
    
    odf_calc_use['val'] = odf_calc_use['val_mean']
    
    return odf_use

# %%

def monthlyDepthAverageDF(odf_depth_mean): #NO STD DEVIATION

    odf_use= odf_depth_mean.groupby(['site', 'year','month', 'surf_deep', 'var']).agg({'val':['mean'], 'z':['mean'], 'date_ordinal':['mean']})
    
    odf_use.columns = odf_use.columns.to_flat_index().map('_'.join)
    
    odf_use = odf_use.reset_index().dropna()
    
    odf_use = (odf_use
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )

    odf_use['val'] = odf_use['val_mean']
    
    return odf_use



# %%


def annualAverageDF(odf):
    
    annual_counts = (odf
                          .dropna()
                          .groupby(['site','year', 'var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )

    odf_use= odf.groupby(['site', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})


    odf_use.columns = odf_use.columns.to_flat_index().map('_'.join)

    odf_use = odf_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


    odf_use = (odf_use
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )


    odf_use = pd.merge(odf_use, annual_counts, how='left', on=['site','year','var'])


    odf_use = odf_use[odf_use['cid_count'] >1] #redundant but fine (see note line 234)

    odf_use['val_ci95hi'] = odf_use['val_mean'] + 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])

    odf_use['val_ci95lo'] = odf_use['val_mean'] - 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])
    
    odf_use['val'] = odf_use['val_mean']
    
    return odf_use

# %%


def buildStatsDF(odf_use, site_list, odf_calc_use=None, odf_depth_mean_deep_DO_percentiles=None, alpha=0.05,  deep_DO_q_list = ['deep_DO_q50'], season_list = ['allyear', 'winter', 'grow', 'loDO'], stat_list = ['mk_ts'], depth_list=['surf', 'deep']):
    
    all_stats_filt = pd.DataFrame()
    
    for deep_DO_q in deep_DO_q_list:
        
        if deep_DO_q != 'all':
                                    
            if odf_depth_mean_deep_DO_percentiles is not None:

                odf_depth_mean_deep_DO_less_than_percentile = odf_depth_mean_deep_DO_percentiles[odf_depth_mean_deep_DO_percentiles['val'] <= odf_depth_mean_deep_DO_percentiles[deep_DO_q]]
    
                cid_deep_DO_less_than_percentile = odf_depth_mean_deep_DO_less_than_percentile['cid']
                
        odf_use = (odf_use
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
        
        
        for depth in depth_list:
            
            if depth == 'deep': #picking one of the two depth options with a odf_calc associated, else nothing happens
                
                if odf_calc_use is not None:
        
                    odf_calc_use = (odf_calc_use
                                  .dropna()
                                  .assign(
                                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                                          )
                                  )
        
                    for site in site_list:
                        
                        for season in season_list:
                            
                            for var in odf_calc_use['var'].unique():
                                
                                if season == 'allyear':
                                    
                                    mask = (odf_calc_use['site'] == site) & (odf_calc_use['var'] == var)
                                    
                                else:
                                    
                                    mask = (odf_calc_use['site'] == site) & (odf_calc_use['season'] == season) & (odf_calc_use['var'] == var)
                                
                                plot_df = odf_calc_use[mask]
                                
                                x = plot_df['date_ordinal']
                                
                                x_plot = plot_df['datetime']
                                
                                y = plot_df['val']
                                
                                for stat in stat_list:
                                    
                                    plot_df = odf_calc_use[mask]
                                    
                                    if stat == 'linreg':
                                        
                                        plot_df['stat'] = stat
                                
                                        result = stats.linregress(x, y)
                                        
                                        B1 = result.slope
                                        
                                        B0 = result.intercept
                                        
                                        plot_df['B1'] = B1

                                        plot_df['B0'] = B0
                                        
                                        sB1 = result.stderr
                                        
                                        n = len(x)
                                        
                                        plot_df['n'] = n 
                                        
                                        dof = n-2
                                        
                                        t = stats.t.ppf(1-alpha/2, dof)
                                        
                                        high_sB1 = B1 + t * sB1
                                        
                                        low_sB1 = B1 - t * sB1
                                        
                                        plot_df['p'] = result.pvalue
                                        
                                        slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
                                        
                                        slope_datetime_s_hi = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                                        
                                        slope_datetime_s_lo = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                                
                                        plot_df['slope_datetime'] = slope_datetime #per year
                                        
                                        plot_df['slope_datetime_s_hi'] = slope_datetime_s_hi #per year
                                        
                                        plot_df['slope_datetime_s_lo'] = slope_datetime_s_lo #per year
                                                                                
                                        plot_df_concat = plot_df[['site','stat','var', 'p', 'n', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo', 'B1', 'B0']].head(1) #slope_datetime_unc_cent, slope_datetime_s
                                        
                                        plot_df_concat['deep_DO_q'] = deep_DO_q
                                        
                                        plot_df_concat['season'] = season
                            
                                        all_stats_filt = pd.concat([all_stats_filt, plot_df_concat])
                                
                                    elif stat == 'mk_ts':
                                        
                                        plot_df['stat'] = stat
                                        
                                        reject_null, p_value, Z = mann_kendall(y, alpha) #dfun
                                                    
                                        plot_df['p'] = p_value
                                        
                                        n = len(x)
                                        
                                        plot_df['n'] = n
                                                    
                                        result = stats.theilslopes(y,x,alpha=alpha)
                                
                                        B1 = result.slope
                                
                                        B0 = result.intercept
                                        
                                        plot_df['B1'] = B1

                                        plot_df['B0'] = B0
                                        
                                        high_sB1 = result.high_slope
                                        
                                        low_sB1 = result.low_slope
                    
                                        slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
                                
                                        plot_df['slope_datetime'] = slope_datetime #per year
                                        
                                        slope_datetime_s_hi = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                                        
                                        slope_datetime_s_lo = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                                        
                                        plot_df['slope_datetime_s_hi'] = slope_datetime_s_hi #per year
                                        
                                        plot_df['slope_datetime_s_lo'] = slope_datetime_s_lo #per year
                                        
                                        plot_df_concat = plot_df[['site','stat','var', 'p', 'n', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo', 'B1', 'B0']].head(1) #slope_datetime_unc_cent, slope_datetime_s
                                        
                                        plot_df_concat['deep_DO_q'] = deep_DO_q
                                        
                                        plot_df_concat['season'] = season
                            
                                        all_stats_filt = pd.concat([all_stats_filt, plot_df_concat])
                
                
            for site in site_list:
                
                for season in season_list:
                
                    for var in odf_use['var'].unique():
                    
                        if depth == 'all':
                            
                            if season == 'allyear':
                                
                                if (var == 'DO_mg_L') and (deep_DO_q != 'all'):
                                    
                                    mask = (odf_use['cid'].isin(cid_deep_DO_less_than_percentile)) & (odf_use['site'] == site) & (odf_use['var'] == var)
                                
                                else:
                                    
                                    mask = (odf_use['site'] == site) & (odf_use['var'] == var)
                            
                            else:
                                
                                if (var == 'DO_mg_L') and (deep_DO_q != 'all'):
                                    
                                    mask = (odf_use['cid'].isin(cid_deep_DO_less_than_percentile)) & (odf_use['site'] == site) & (odf_use['season'] == season) & (odf_use['var'] == var)
                                
                                else:
                                    
                                    mask = (odf_use['site'] == site) & (odf_use['season'] == season) & (odf_use['var'] == var)
                            
                        else:
                            
                            
                            if season == 'allyear':
                                
                                if (var == 'DO_mg_L') and (deep_DO_q != 'all'):
                                    
                                    mask = (odf_use['cid'].isin(cid_deep_DO_less_than_percentile)) & (odf_use['site'] == site) & (odf_use['surf_deep'] == depth) & (odf_use['var'] == var)
                                
                                else:
                                    
                                    mask = (odf_use['site'] == site) & (odf_use['surf_deep'] == depth) & (odf_use['var'] == var)
                                
                            else:
                            
                                if (var == 'DO_mg_L') and (deep_DO_q != 'all'):
                                    
                                    mask = (odf_use['cid'].isin(cid_deep_DO_less_than_percentile)) & (odf_use['site'] == site) & (odf_use['season'] == season) & (odf_use['surf_deep'] == depth) & (odf_use['var'] == var)
                                
                                else:
                                    
                                    mask = (odf_use['site'] == site) & (odf_use['season'] == season) & (odf_use['surf_deep'] == depth) & (odf_use['var'] == var)
                                    
                                    
                            
                        plot_df = odf_use[mask]
                        
                        x = plot_df['date_ordinal']
                        
                        x_plot = plot_df['datetime']
                        
                        y = plot_df['val']
                        
                        for stat in stat_list:
                            
                            plot_df = odf_use[mask]
                            
                            if stat == 'linreg':
                                
                                plot_df['stat'] = stat
                        
                                result = stats.linregress(x, y)
                                
                                B1 = result.slope
                                
                                B0 = result.intercept
                                
                                plot_df['B1'] = B1

                                plot_df['B0'] = B0
                                
                                sB1 = result.stderr
                                
                                n = len(x)
                                
                                plot_df['n'] = n
                                
                                dof = n-2
                                
                                t = stats.t.ppf(1-alpha/2, dof)
                                
                                high_sB1 = B1 + t * sB1
                                
                                low_sB1 = B1 - t * sB1
                                
                                plot_df['p'] = result.pvalue
                                
                                slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
                                
                                slope_datetime_s_hi = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                                
                                slope_datetime_s_lo = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                        
                                plot_df['slope_datetime'] = slope_datetime #per year
                                
                                plot_df['slope_datetime_s_hi'] = slope_datetime_s_hi #per year
                                
                                plot_df['slope_datetime_s_lo'] = slope_datetime_s_lo #per year
                    
                                all_stats_filt = pd.concat([all_stats_filt, plot_df_concat])
                                                                
                                if depth != 'all':
                                                            
                                    plot_df['var'] = plot_df['surf_deep'] + '_' + plot_df['var']
                                                                
                                plot_df_concat = plot_df[['site','stat','var', 'p', 'n', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo', 'B1', 'B0']].head(1) #slope_datetime_unc_cent, slope_datetime_s
                                
                                plot_df_concat['deep_DO_q'] = deep_DO_q
                                
                                plot_df_concat['season'] = season
                    
                                all_stats_filt = pd.concat([all_stats_filt, plot_df_concat])
                        
                            elif stat == 'mk_ts':
                                
                                plot_df['stat'] = stat
                                
                                reject_null, p_value, Z = mann_kendall(y, alpha) #dfun
                                            
                                plot_df['p'] = p_value
                                
                                n = len(x)
                                
                                plot_df['n'] = n
                        
                        
                                result = stats.theilslopes(y,x,alpha=alpha)
                        
                                B1 = result.slope
                        
                                B0 = result.intercept
                                
                                plot_df['B1'] = B1

                                plot_df['B0'] = B0
                                
                                high_sB1 = result.high_slope
                                
                                low_sB1 = result.low_slope
        
                                slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
                        
                                plot_df['slope_datetime'] = slope_datetime #per year
                                
                                slope_datetime_s_hi = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                                
                                slope_datetime_s_lo = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                                
                                plot_df['slope_datetime_s_hi'] = slope_datetime_s_hi #per year
                                
                                plot_df['slope_datetime_s_lo'] = slope_datetime_s_lo #per year
                                
                                if depth != 'all':
                                                            
                                    plot_df['var'] = plot_df['surf_deep'] + '_' + plot_df['var']
                                                                                                        
                                plot_df_concat = plot_df[['site','stat','var', 'p', 'n', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo', 'B1', 'B0']].head(1) #slope_datetime_unc_cent, slope_datetime_s
                                
                                plot_df_concat['deep_DO_q'] = deep_DO_q
                                
                                plot_df_concat['season'] = season
                    
                                all_stats_filt = pd.concat([all_stats_filt, plot_df_concat])
                        
                        
    return all_stats_filt

# %%

def calcSeriesAvgs(odf_depth_mean, odf_depth_mean_deep_DO_percentiles, deep_DO_q = 'deep_DO_q50', filter_out = False):
    
    # note 9/6/2024: build in flexibility to not deal with DO percentiles and seasons
    
    # to work with the longShortDF build...
    
    # sloppy edits 9/2/2025 to make the filter a flag - so messy - just using the odf_use_seasonal_CTSA gives the unfiltered DO values too
    
    if filter_out:
        

        odf_depth_mean_deep_DO_less_than_median = odf_depth_mean_deep_DO_percentiles[odf_depth_mean_deep_DO_percentiles['val'] <= odf_depth_mean_deep_DO_percentiles[deep_DO_q]]
        
    else:
        
        odf_depth_mean_deep_DO_less_than_median = odf_depth_mean_deep_DO_percentiles.copy()
        
    
    cid_deep_DO_less_than_median= odf_depth_mean_deep_DO_less_than_median['cid']





    series_counts_seasonal_CTSA = (odf_depth_mean
                          .dropna()
                          #.set_index('datetime')
                          .groupby(['site', 'season', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )
    
    
    odf_use_seasonal_CTSA = odf_depth_mean.groupby(['site', 'surf_deep', 'season','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
    
    
    odf_use_seasonal_CTSA.columns = odf_use_seasonal_CTSA.columns.to_flat_index().map('_'.join)
    
    odf_use_seasonal_CTSA = odf_use_seasonal_CTSA.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!
    
    
    odf_use_seasonal_CTSA = (odf_use_seasonal_CTSA
                      # .drop(columns=['date_ordinal_std'])
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      #.reset_index() 
                      .dropna()
                      .assign(
                              #segment=(lambda x: key),
                              # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                              # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                              # season=(lambda x: pd.cut(x['month'],
                              #                          bins=[0,3,6,9,12],
                              #                          labels=['winter', 'spring', 'summer', 'fall'])),
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    
    
    odf_use_seasonal_CTSA = pd.merge(odf_use_seasonal_CTSA, series_counts_seasonal_CTSA, how='left', on=['site','surf_deep','season','var'])
    
    odf_use_seasonal_CTSA = odf_use_seasonal_CTSA[odf_use_seasonal_CTSA['cid_count'] >1] #redundant but fine (see note line 234)
    
    odf_use_seasonal_CTSA['val_ci95hi'] = odf_use_seasonal_CTSA['val_mean'] + 1.96*odf_use_seasonal_CTSA['val_std']/np.sqrt(odf_use_seasonal_CTSA['cid_count'])
    
    odf_use_seasonal_CTSA['val_ci95lo'] = odf_use_seasonal_CTSA['val_mean'] - 1.96*odf_use_seasonal_CTSA['val_std']/np.sqrt(odf_use_seasonal_CTSA['cid_count'])




    series_counts_seasonal_DO = (odf_depth_mean[odf_depth_mean['cid'].isin(cid_deep_DO_less_than_median)]
                          .dropna()
                          #.set_index('datetime')
                          .groupby(['site','season', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )
    
    
    odf_use_seasonal_DO = odf_depth_mean[odf_depth_mean['cid'].isin(cid_deep_DO_less_than_median)].groupby(['site', 'surf_deep', 'season','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
    
    
    odf_use_seasonal_DO.columns = odf_use_seasonal_DO.columns.to_flat_index().map('_'.join)
    
    odf_use_seasonal_DO = odf_use_seasonal_DO.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!
    
    
    odf_use_seasonal_DO = (odf_use_seasonal_DO
                      # .drop(columns=['date_ordinal_std'])
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      #.reset_index() 
                      .dropna()
                      .assign(
                              #segment=(lambda x: key),
                              # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                              # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                              # season=(lambda x: pd.cut(x['month'],
                              #                          bins=[0,3,6,9,12],
                              #                          labels=['winter', 'spring', 'summer', 'fall'])),
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    
    
    odf_use_seasonal_DO = pd.merge(odf_use_seasonal_DO, series_counts_seasonal_DO, how='left', on=['site','surf_deep','season','var'])
    
    odf_use_seasonal_DO = odf_use_seasonal_DO[odf_use_seasonal_DO['cid_count'] >1] #redundant but fine (see note line 234)
    
    odf_use_seasonal_DO['val_ci95hi'] = odf_use_seasonal_DO['val_mean'] + 1.96*odf_use_seasonal_DO['val_std']/np.sqrt(odf_use_seasonal_DO['cid_count'])
    
    odf_use_seasonal_DO['val_ci95lo'] = odf_use_seasonal_DO['val_mean'] - 1.96*odf_use_seasonal_DO['val_std']/np.sqrt(odf_use_seasonal_DO['cid_count'])
    



    
    series_counts_annual_CTSA = (odf_depth_mean
                          .dropna()
                          #.set_index('datetime')
                          .groupby(['site', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )
    
    
    odf_use_annual_CTSA = odf_depth_mean.groupby(['site', 'surf_deep','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
    
    
    odf_use_annual_CTSA.columns = odf_use_annual_CTSA.columns.to_flat_index().map('_'.join)
    
    odf_use_annual_CTSA = odf_use_annual_CTSA.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!
    
    
    odf_use_annual_CTSA = (odf_use_annual_CTSA
                      # .drop(columns=['date_ordinal_std'])
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      #.reset_index() 
                      .dropna()
                      .assign(
                              #segment=(lambda x: key),
                              # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                              # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                              # season=(lambda x: pd.cut(x['month'],
                              #                          bins=[0,3,6,9,12],
                              #                          labels=['winter', 'spring', 'summer', 'fall'])),
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    
    
    odf_use_annual_CTSA = pd.merge(odf_use_annual_CTSA, series_counts_annual_CTSA, how='left', on=['site','surf_deep','var'])
    
    odf_use_annual_CTSA = odf_use_annual_CTSA[odf_use_annual_CTSA['cid_count'] >1] #redundant but fine (see note line 234)
    
    odf_use_annual_CTSA['val_ci95hi'] = odf_use_annual_CTSA['val_mean'] + 1.96*odf_use_annual_CTSA['val_std']/np.sqrt(odf_use_annual_CTSA['cid_count'])
    
    odf_use_annual_CTSA['val_ci95lo'] = odf_use_annual_CTSA['val_mean'] - 1.96*odf_use_annual_CTSA['val_std']/np.sqrt(odf_use_annual_CTSA['cid_count'])
    



    
    series_counts_annual_DO = (odf_depth_mean[odf_depth_mean['cid'].isin(cid_deep_DO_less_than_median)]
                          .dropna()
                          #.set_index('datetime')
                          .groupby(['site', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )
    
    
    odf_use_annual_DO = odf_depth_mean[odf_depth_mean['cid'].isin(cid_deep_DO_less_than_median)].groupby(['site', 'surf_deep','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
    
    
    odf_use_annual_DO.columns = odf_use_annual_DO.columns.to_flat_index().map('_'.join)
    
    odf_use_annual_DO = odf_use_annual_DO.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!
    
    
    odf_use_annual_DO = (odf_use_annual_DO
                      # .drop(columns=['date_ordinal_std'])
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      #.reset_index() 
                      .dropna()
                      .assign(
                              #segment=(lambda x: key),
                              # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                              # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                              # season=(lambda x: pd.cut(x['month'],
                              #                          bins=[0,3,6,9,12],
                              #                          labels=['winter', 'spring', 'summer', 'fall'])),
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    
    
    odf_use_annual_DO = pd.merge(odf_use_annual_DO, series_counts_annual_DO, how='left', on=['site','surf_deep','var'])
    
    odf_use_annual_DO = odf_use_annual_DO[odf_use_annual_DO['cid_count'] >1] #redundant but fine (see note line 234)
    
    odf_use_annual_DO['val_ci95hi'] = odf_use_annual_DO['val_mean'] + 1.96*odf_use_annual_DO['val_std']/np.sqrt(odf_use_annual_DO['cid_count'])
    
    odf_use_annual_DO['val_ci95lo'] = odf_use_annual_DO['val_mean'] - 1.96*odf_use_annual_DO['val_std']/np.sqrt(odf_use_annual_DO['cid_count'])
    
    
    return odf_use_seasonal_DO, odf_use_seasonal_CTSA, odf_use_annual_DO, odf_use_annual_CTSA