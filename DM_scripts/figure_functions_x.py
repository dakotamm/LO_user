#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepared for publication: 2026/01/07

Author: Dakota Mascarenas

Plotting code for: "Century-Scale Changes in Dissolved Oxygen, Temperature, and Salinity in Puget Sound" (Mascarenas et al., in review; submitted 2026/01/09 to Estuaries & Coasts)

This script provides functions for data analysis and figure plotting in corresponding manuscript. Please reach out to the author at dakotamm@uw.edu for any questions.

"""

# import modules
import numpy as np
import pandas as pd
import scipy.stats as stats
import gsw

# get unique cast locations for all casts in Puget Sound given locations on LiveOcean grid (MacCready et al., 2021; see text for more details and citations)
def get_cast_locations(df):
    df['ix_iy'] = df['ix'].astype(str).apply(lambda x: x.zfill(4)) + '_' + df['iy'].astype(str).apply(lambda x: x.zfill(4))
    cast_locations = df.groupby(['ix_iy']).first().reset_index()
    return cast_locations

# plot coast; adapted from Parker MacCready's public LO github repository (https://github.com/parkermac/LO/blob/main/lo_tools/lo_tools/plotting_functions.py)
def add_coast(ax, df_directory, color='k', linewidth=0.5):
    '''
    adapted from (https://github.com/parkermac/LO/blob/main/lo_tools/lo_tools/plotting_functions.py)
    '''
    fn = df_directory + '/coast_pnw.p'
    C = pd.read_pickle(fn)
    ax.plot(C['lon'].values, C['lat'].values, '-', color=color, linewidth=linewidth)

# make plot aspect ratio locally Cartesian; adapted from Parker MacCready's public LO github repository (https://github.com/parkermac/LO/blob/main/lo_tools/lo_tools/plotting_functions.py)
def dar(ax):
    '''
    adapted from (https://github.com/parkermac/LO/blob/main/lo_tools/lo_tools/plotting_functions.py)
    '''
    yl = ax.get_ylim()
    yav = (yl[0] + yl[1])/2
    ax.set_aspect(1/np.cos(np.pi*yav/180))


# apply DO filtering to casts in the bottom 50th percentile of annual seasonal bottom DO values
def filter_DO(site_depth_avg_var_DF):
    df_deep_DO = site_depth_avg_var_DF[(site_depth_avg_var_DF['var'] == 'DO_mg_L') & (site_depth_avg_var_DF['surf_deep'] == 'deep')]
    df_deep_DO['year_adjusted'] = df_deep_DO['year']
    df_deep_DO.loc[df_deep_DO['month'] == 12, 'year_adjusted'] = df_deep_DO['year'] + 1 #since trimesters do not evenly bisect one calendar year, this incorporates december into the following calendar year !!!!!*****
    df_deep_DO_q50 = df_deep_DO[['site', 'year_adjusted', 'season','val']].groupby(['site', 'year_adjusted', 'season']).quantile(0.5)
    df_deep_DO_q50 = df_deep_DO_q50.rename(columns={'val':'deep_DO_q50'})
    df_deep_DO_w_q50 = pd.merge(df_deep_DO, df_deep_DO_q50, how='left', on=['site','season','year_adjusted'])
    df_deep_DO_leq_q50 = df_deep_DO_w_q50[df_deep_DO_w_q50['val'] <= df_deep_DO_w_q50['deep_DO_q50']]
    cid_DO_leq_q50 = df_deep_DO_leq_q50['cid'].unique()
    filter_DO_DF = site_depth_avg_var_DF[(site_depth_avg_var_DF['var'] == 'DO_mg_L') & (site_depth_avg_var_DF['cid'].isin(cid_DO_leq_q50))]
    return filter_DO_DF

# calculate Theil-Sen slopes with confidence intervals for given time series and values and specified alpha
def calc_ts_slopes(working_df, alpha):
    x = working_df['date_ordinal'].copy()
    x_working = working_df['datetime'].copy()
    y = working_df['val'].copy()
    result = stats.theilslopes(y,x,alpha=alpha)
    B1 = result.slope
    B0 = result.intercept
    concat_df = working_df.head(1).copy()
    concat_df['B1'] = B1
    concat_df['B0'] = B0
    high_sB1 = result.high_slope
    low_sB1 = result.low_slope
    slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_working.max().year - x_working.min().year)
    concat_df['slope_datetime'] = slope_datetime #per year
    slope_datetime_s_hi = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_working.max().year - x_working.min().year)
    slope_datetime_s_lo = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_working.max().year - x_working.min().year)
    concat_df['slope_datetime_s_hi'] = slope_datetime_s_hi #per year
    concat_df['slope_datetime_s_lo'] = slope_datetime_s_lo #per year
    working_df_concat = concat_df[['site', 'season', 'surf_deep', 'var', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo', 'B1', 'B0']]
    return working_df_concat

# calculate Theil-Sen slopes for DO, absolute salinity, conservative temperature, and calculated DO saturation
def calc_slopes_var(alpha, site_depth_avg_var_DF, filter_DO_DF=None, DO_sat_DF=None):
    slope_DF = pd.DataFrame()
    if filter_DO_DF is not None:
        CT_SA_DF = site_depth_avg_var_DF[site_depth_avg_var_DF['var'].isin(['CT','SA'])]
        if DO_sat_DF is not None:
            big_df = pd.concat([CT_SA_DF, filter_DO_DF, DO_sat_DF])
        else:
            big_df = pd.concat([CT_SA_DF, filter_DO_DF])
    else:
        if DO_sat_DF is not None:
            big_df = pd.concat([site_depth_avg_var_DF, DO_sat_DF])
        else:
            big_df = site_depth_avg_var_DF
    for site in big_df['site'].unique():
        for season in big_df['season'].unique():
            for var in big_df['var'].unique():
                for depth in big_df['surf_deep'].unique():
                    working_df = big_df[(big_df['site'] == site) & (big_df['season'] == season) & (big_df['var'] == var) & (big_df['surf_deep'] == depth)]
                    #working_df['var'] = working_df['surf_deep'] + '_' + working_df['var']
                    working_df_concat = calc_ts_slopes(working_df, alpha)
                    slope_DF = pd.concat([slope_DF, working_df_concat])    
    slope_DF.loc[slope_DF['site'] == 'point_jefferson', 'site_label'] = 'PJ' #labels for plotting
    slope_DF.loc[slope_DF['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'
    slope_DF.loc[slope_DF['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'
    slope_DF.loc[slope_DF['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'
    slope_DF.loc[slope_DF['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'
    slope_DF.loc[slope_DF['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'
    slope_DF.loc[slope_DF['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'
    slope_DF.loc[slope_DF['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'
    slope_DF.loc[slope_DF['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'
    slope_DF.loc[slope_DF['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'
    slope_DF.loc[slope_DF['site'] == 'point_jefferson', 'site_num'] = 1
    slope_DF.loc[slope_DF['site'] == 'near_seattle_offshore', 'site_num'] = 2
    slope_DF.loc[slope_DF['site'] == 'carr_inlet_mid', 'site_num'] = 3
    slope_DF.loc[slope_DF['site'] == 'saratoga_passage_mid', 'site_num'] = 4
    slope_DF.loc[slope_DF['site'] == 'lynch_cove_mid', 'site_num'] = 5
    slope_DF.loc[slope_DF['season'] == 'grow', 'season_label'] = 'Spring (Apr-Jul)'
    slope_DF.loc[slope_DF['season'] == 'loDO', 'season_label'] = 'Low-DO (Aug-Nov)'
    slope_DF.loc[slope_DF['season'] == 'winter', 'season_label'] = 'Winter (Dec-Mar)'
    slope_DF.loc[slope_DF['surf_deep'] == 'surf', 'depth_label'] = 'Surface'
    slope_DF.loc[slope_DF['surf_deep'] == 'deep', 'depth_label'] = 'Bottom'
    slope_DF.loc[slope_DF['var'] == 'CT', 'var_label'] = '[°C]'
    slope_DF.loc[slope_DF['var'] == 'SA', 'var_label'] = '[g/kg]'
    slope_DF.loc[slope_DF['var'] == 'DO_mg_L', 'var_label'] = '[mg/L]'
    slope_DF.loc[slope_DF['var'] == 'DO_sol', 'var_label'] = '[mg/L]'
    return slope_DF

# calculate time series average values for each season
def calc_seasonal_series_means(site_depth_avg_var_DF, filter_DO_DF=None):
    if filter_DO_DF is not None:
        CT_SA_DF = site_depth_avg_var_DF[site_depth_avg_var_DF['var'].isin(['CT','SA'])]
        big_df = pd.concat([CT_SA_DF, filter_DO_DF])
    else:
        big_df = site_depth_avg_var_DF
    series_counts = (big_df
                     .dropna()
                     .groupby(['site', 'season', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                     .reset_index()
                     .rename(columns={'cid':'cid_count'})
                     )
    means_DF = big_df.groupby(['site', 'surf_deep', 'season','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
    means_DF.columns = means_DF.columns.to_flat_index().map('_'.join)
    means_DF = means_DF.reset_index().dropna() 
    means_DF = (means_DF
                    .rename(columns={'date_ordinal_mean':'date_ordinal'})
                    .dropna()
                    .assign(
                            datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                            )
                    )
    means_DF = pd.merge(means_DF, series_counts, how='left', on=['site','surf_deep','season','var'])
    means_DF = means_DF[means_DF['cid_count'] >1] #redundant but fine (see note line 234)
    means_DF['val_ci95hi'] = means_DF['val_mean'] + 1.96*means_DF['val_std']/np.sqrt(means_DF['cid_count'])
    means_DF['val_ci95lo'] = means_DF['val_mean'] - 1.96*means_DF['val_std']/np.sqrt(means_DF['cid_count'])
    means_DF.loc[means_DF['site'] == 'point_jefferson', 'site_label'] = 'PJ' #labels for plotting
    means_DF.loc[means_DF['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'
    means_DF.loc[means_DF['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'
    means_DF.loc[means_DF['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'
    means_DF.loc[means_DF['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'
    means_DF.loc[means_DF['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'
    means_DF.loc[means_DF['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'
    means_DF.loc[means_DF['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'
    means_DF.loc[means_DF['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'
    means_DF.loc[means_DF['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'
    means_DF.loc[means_DF['site'] == 'point_jefferson', 'site_num'] = 1
    means_DF.loc[means_DF['site'] == 'near_seattle_offshore', 'site_num'] = 2
    means_DF.loc[means_DF['site'] == 'carr_inlet_mid', 'site_num'] = 3
    means_DF.loc[means_DF['site'] == 'saratoga_passage_mid', 'site_num'] = 4
    means_DF.loc[means_DF['site'] == 'lynch_cove_mid', 'site_num'] = 5
    means_DF.loc[means_DF['season'] == 'grow', 'season_label'] = 'Spring (Apr-Jul)'
    means_DF.loc[means_DF['season'] == 'loDO', 'season_label'] = 'Low-DO (Aug-Nov)'
    means_DF.loc[means_DF['season'] == 'winter', 'season_label'] = 'Winter (Dec-Mar)'
    means_DF.loc[means_DF['surf_deep'] == 'surf', 'depth_label'] = 'Surface'
    means_DF.loc[means_DF['surf_deep'] == 'deep', 'depth_label'] = 'Bottom'
    means_DF.loc[means_DF['var'] == 'CT', 'var_label'] = '[°C]'
    means_DF.loc[means_DF['var'] == 'SA', 'var_label'] = '[g/kg]'
    means_DF.loc[means_DF['var'] == 'DO_mg_L', 'var_label'] = '[mg/L]'
    return means_DF

    
# calculate DO saturation using individual cast, depth-binned conservative temperature and absolute salinity; NOTE: this only uses casts that go to bottom water to avoid oversampling in surface waters
def calc_DO_sat(site_depth_avg_var_DF):
    cid_deep = site_depth_avg_var_DF.loc[site_depth_avg_var_DF['surf_deep'] == 'deep', 'cid'] #find unique casts that exceed the threshold depth for bottom water depth-binning
    df_deep = site_depth_avg_var_DF[site_depth_avg_var_DF['cid'].isin(cid_deep)] #filter dataframe to bottom water casts
    df_calc = df_deep.pivot(index = ['site', 'year', 'month', 'season','date_ordinal','cid'], columns = ['surf_deep', 'var'], values ='val')
    df_calc.columns = df_calc.columns.to_flat_index().map('_'.join)
    df_calc = df_calc.reset_index()
    df_calc['surf_dens'] = gsw.density.sigma0(df_calc['surf_SA'], df_calc['surf_CT'])
    df_calc['deep_dens'] = gsw.density.sigma0(df_calc['deep_SA'], df_calc['deep_CT'])
    A_0 = 5.80818 #all in umol/kg, from Garcia & Gordon (1992)
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
    df_calc['surf_T_s'] = np.log((298.15 - df_calc['surf_CT'])/(273.15 + df_calc['surf_CT']))
    df_calc['surf_C_o_*'] = np.exp(A_0 + A_1*df_calc['surf_T_s'] + A_2*df_calc['surf_T_s']**2 + A_3*df_calc['surf_T_s']**3 + A_4*df_calc['surf_T_s']**4 + A_5*df_calc['surf_T_s']**5 + 
                           df_calc['surf_SA']*(B_0 + B_1*df_calc['surf_T_s'] + B_2*df_calc['surf_T_s']**2 + B_3*df_calc['surf_T_s']**3) + C_0*df_calc['surf_SA']**2)
    df_calc['surf_DO_sat'] =  df_calc['surf_C_o_*']*(df_calc['surf_dens']/1000 + 1)*32/1000
    df_calc['deep_T_s'] = np.log((298.15 - df_calc['deep_CT'])/(273.15 + df_calc['deep_CT']))
    df_calc['deep_C_o_*'] = np.exp(A_0 + A_1*df_calc['deep_T_s'] + A_2*df_calc['deep_T_s']**2 + A_3*df_calc['deep_T_s']**3 + A_4*df_calc['deep_T_s']**4 + A_5*df_calc['deep_T_s']**5 + 
                           df_calc['deep_SA']*(B_0 + B_1*df_calc['deep_T_s'] + B_2*df_calc['deep_T_s']**2 + B_3*df_calc['deep_T_s']**3) + C_0*df_calc['deep_SA']**2)
    df_calc['deep_DO_sat'] =  df_calc['deep_C_o_*']*(df_calc['deep_dens']/1000 + 1)*32/1000
    DO_sat_DF = pd.melt(df_calc, id_vars = ['site', 'year', 'month', 'season', 'date_ordinal','cid'], value_vars=['surf_DO_sat', 'deep_DO_sat'], var_name='var', value_name='val')
    DO_sat_DF.loc[DO_sat_DF['var'] == 'surf_DO_sat', 'surf_deep'] = 'surf' # match formatting
    DO_sat_DF.loc[DO_sat_DF['var'] == 'deep_DO_sat', 'surf_deep'] = 'deep'
    DO_sat_DF.loc[DO_sat_DF['var'] == 'surf_DO_sat', 'var'] = 'DO_sat'
    DO_sat_DF.loc[DO_sat_DF['var'] == 'deep_DO_sat', 'var'] = 'DO_sat'
    DO_sat_DF = DO_sat_DF.dropna().assign(datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))
    return DO_sat_DF



# clean specific air temperature dataset for Figure 10; averages daily maximum and minimum for monthly average
def get_clean_temps(temp_temp_df):
    temp_df = temp_temp_df.copy()
    temp_df['site'] = 'seatac'
    temp_df['datetime'] = pd.to_datetime(temp_df['DATE'])
    temp_df['year'] = pd.DatetimeIndex(temp_df['datetime']).year
    temp_df['month'] = pd.DatetimeIndex(temp_df['datetime']).month
    temp_df['year_month'] = temp_df['year'].astype(str) + '_' + temp_df['month'].astype(str).apply(lambda x: x.zfill(2))
    temp_df['date_ordinal'] = temp_df['datetime'].apply(lambda x: x.toordinal())
    temp_df.loc[temp_df['month'].isin([4,5,6,7]), 'season'] = 'grow'
    temp_df.loc[temp_df['month'].isin([8,9,10,11]), 'season'] = 'loDO'
    temp_df.loc[temp_df['month'].isin([12,1,2,3]), 'season'] = 'winter'
    temp_df['TAVG'] = temp_df[['TMAX', 'TMIN']].mean(axis=1)
    temp_df['year_season'] = temp_df['year'].astype(str) + '_' + temp_df['season']
    temp_df = pd.melt(temp_df, id_vars =['site', 'STATION', 'NAME', 'DATE', 'datetime', 'year', 'month', 'year_month', 'date_ordinal', 'season', 'year_season'], value_vars=['PRCP', 'TMAX', 'TMIN', 'TSUN', 'TAVG'], var_name='var', value_name='val')
    return temp_df

# calculate monthly average air temperatures for specific dataset for Figure 10 with 95% confidence intervals
def get_monthly_temps(temp_df):
    monthly_counts = (temp_df
                          .dropna()
                          .groupby(['site','year_month', 'var']).agg({'val' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'val':'val_count'})
                          )
    temp_monthly_avg_df = temp_df[['site', 'datetime', 'date_ordinal', 'year', 'month', 'year_month', 'season', 'year_season', 'var', 'val']].groupby(['site', 'year','month','year_month','season', 'year_season', 'var']).agg({'val':['mean', 'std'], 'date_ordinal':['mean']})
    temp_monthly_avg_df.columns = temp_monthly_avg_df.columns.to_flat_index().map('_'.join)
    temp_monthly_avg_df = temp_monthly_avg_df.reset_index().dropna()
    temp_monthly_avg_df = (temp_monthly_avg_df
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))
                      )
    temp_monthly_avg_df = pd.merge(temp_monthly_avg_df, monthly_counts, how='left', on=['site', 'year_month', 'var'])
    temp_monthly_avg_df = temp_monthly_avg_df[temp_monthly_avg_df['val_count'] >1] #redundant but sanity check
    temp_monthly_avg_df['val_ci95hi'] = temp_monthly_avg_df['val_mean'] + 1.96*temp_monthly_avg_df['val_std']/np.sqrt(temp_monthly_avg_df['val_count'])
    temp_monthly_avg_df['val_ci95lo'] = temp_monthly_avg_df['val_mean'] - 1.96*temp_monthly_avg_df['val_std']/np.sqrt(temp_monthly_avg_df['val_count'])
    temp_monthly_avg_df['val'] = temp_monthly_avg_df['val_mean']
    temp_monthly_DF = temp_monthly_avg_df.copy()
    return temp_monthly_DF

# calculate annual average air temperatures for specific dataset for Figure 10 with 95% confidence intervals
def get_annual_temps(temp_df):
    annual_counts = (temp_df
                          .dropna()
                          .groupby(['site','year', 'var']).agg({'val' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'val':'val_count'})
                          )
    temp_annual_avg_df = temp_df[['site', 'datetime', 'date_ordinal', 'year','var','val']].groupby(['site', 'year', 'var']).agg({'val':['mean', 'std'], 'date_ordinal':['mean']})
    temp_annual_avg_df.columns = temp_annual_avg_df.columns.to_flat_index().map('_'.join)
    temp_annual_avg_df = temp_annual_avg_df.reset_index().dropna()
    temp_annual_avg_df = (temp_annual_avg_df
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))
                      )
    temp_annual_avg_df = pd.merge(temp_annual_avg_df, annual_counts, how='left', on=['site', 'year', 'var'])
    temp_annual_avg_df = temp_annual_avg_df[temp_annual_avg_df['val_count'] >1] #redundant but sanity check
    temp_annual_avg_df['val_ci95hi'] = temp_annual_avg_df['val_mean'] + 1.96*temp_annual_avg_df['val_std']/np.sqrt(temp_annual_avg_df['val_count'])
    temp_annual_avg_df['val_ci95lo'] = temp_annual_avg_df['val_mean'] - 1.96*temp_annual_avg_df['val_std']/np.sqrt(temp_annual_avg_df['val_count'])
    temp_annual_avg_df['val'] = temp_annual_avg_df['val_mean']
    temp_annual_DF = temp_annual_avg_df.copy()
    return temp_annual_DF

# calculate depth-binned, yearly seasonal average DO, absolute salinity, conservative temperature, and calculated DO saturation with 95% confidence intervals
def get_seasonal_vars(site_depth_avg_var_DF, filter_DO_DF=None):
    if filter_DO_DF is not None:
        CT_SA_DF = site_depth_avg_var_DF[site_depth_avg_var_DF['var'].isin(['CT','SA'])]
        big_df = pd.concat([CT_SA_DF, filter_DO_DF])
    else:
        big_df = site_depth_avg_var_DF
    seasonal_counts = (big_df
                          .dropna()
                          .groupby(['site','year','surf_deep', 'season', 'var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )
    seasonal_avg_df = big_df.groupby(['site', 'surf_deep', 'season', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
    seasonal_avg_df.columns = seasonal_avg_df.columns.to_flat_index().map('_'.join)
    seasonal_avg_df = seasonal_avg_df.reset_index().dropna()
    seasonal_avg_df = (seasonal_avg_df
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))
                      )
    seasonal_avg_df = pd.merge(seasonal_avg_df, seasonal_counts, how='left', on=['site','surf_deep', 'season', 'year','var'])
    seasonal_avg_df = seasonal_avg_df[seasonal_avg_df['cid_count'] >1] #redundant but sanity check
    seasonal_avg_df['val_ci95hi'] = seasonal_avg_df['val_mean'] + 1.96*seasonal_avg_df['val_std']/np.sqrt(seasonal_avg_df['cid_count'])
    seasonal_avg_df['val_ci95lo'] = seasonal_avg_df['val_mean'] - 1.96*seasonal_avg_df['val_std']/np.sqrt(seasonal_avg_df['cid_count'])
    seasonal_avg_df['val'] = seasonal_avg_df['val_mean']
    seasonal_DF = seasonal_avg_df.copy()
    return seasonal_DF