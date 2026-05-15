"""
Code to explore an initial condition for LiveOcean. This is focused
just on observations, with the goal of coming up with reasonable values
for all tracers by basin/depth.
"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
import matplotlib.pyplot as plt
import matplotlib.path as mpth
import xarray as xr
import numpy as np
import pandas as pd
import datetime

from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages

import seaborn as sns

import scipy.stats as stats



# %%
Ldir = Lfun.Lstart(gridname='cas7')

# grid
fng = Ldir['grid'] / 'grid.nc'
dsg = xr.open_dataset(fng)
x = dsg.lon_rho.values
y = dsg.lat_rho.values
m = dsg.mask_rho.values
xp, yp = pfun.get_plon_plat(x,y)
h = dsg.h.values
h[m==0] = np.nan

# polygons
basin_list = ['sog_n', 'sog_s', 'soj','sji','mb', 'wb', 'hc', 'ss']
#basin_list = ['ps', 'mb','hc', 'wb', 'ss']
# c_list = ['r','b','g','c'] # colors to associate with basins
# c_dict = dict(zip(basin_list,c_list))

path_dict = dict()
xxyy_dict = dict()
for basin in basin_list:
    # polygon
    fnp = Ldir['LOo'] / 'section_lines' / (basin+'.p')
    p = pd.read_pickle(fnp)
    xx = p.x.to_numpy()
    yy = p.y.to_numpy()
    xxyy = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
    path = mpth.Path(xxyy)
    # store in dicts
    path_dict[basin] = path
    xxyy_dict[basin] = xxyy

# observations
source_list = ['ecology', 'nceiSalish', 'dfo1', 'collias']
otype_list = ['ctd', 'bottle']
year_list = np.arange(1930, 2022)
ii = 0
for year in year_list:
    for source in source_list:
        for otype in otype_list:
            odir = Ldir['LOo'] / 'obs' / source / otype
            try:
                if ii == 0:
                    odf = pd.read_pickle( odir / (str(year) + '.p'))
                    if source == 'ecology' and otype == 'bottle':
                        odf['DO (uM)'] == np.nan
                    # print(odf.columns)
                else:
                    this_odf = pd.read_pickle( odir / (str(year) + '.p'))
                    if source == 'ecology' and otype == 'bottle':
                        this_odf['DO (uM)'] == np.nan
                    this_odf['cid'] = this_odf['cid'] + odf['cid'].max() + 1
                    # print(this_odf.columns)
                    odf = pd.concat((odf,this_odf),ignore_index=True)
                ii += 1
            except FileNotFoundError:
                pass

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
for basin in basin_list:
    path = path_dict[basin]
    oisin = path.contains_points(oxoy)
    odfin = odf.loc[oisin,:]
    odf_dict[basin] = odfin.copy()
    

# %%

depth_div_0 = -15
depth_div_1 = -35

#year_div = 2010

for key in odf_dict.keys():
    
    odf_dict[key] = (odf_dict[key]
                     .assign(
                         datetime=(lambda x: pd.to_datetime(x['time'])),
                         depth_bool=(lambda x: pd.cut(x['z'], 
                                                      bins=[x['z'].min()-1, depth_div_1, depth_div_0, 0],
                                                      labels= ['deep', 'mid', 'shallow'])),
                         year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                         month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                         season=(lambda x: pd.cut(x['month'],
                                                  bins=[0,3,6,9,12],
                                                  labels=['winter', 'spring', 'summer', 'fall'])),
                         DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                         # period_label=(lambda x: pd.cut(x['year'], 
                         #                              bins=[x['year'].min()-1, year_div-1, x['year'].max()],
                         #                              labels= ['pre', 'post'])),
                         date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())),
                         segment=(lambda x: key)
                             )
                     )


# %%

# full_avgs_dict = dict()

# for key in odf_dict.keys():
    
#     full_avgs_dict[key] = (odf_dict[key]
#                       .set_index('datetime')
#                       .groupby([pd.Grouper(freq='M')]).mean()
#                       .drop(columns =['lat','lon','cid'])
#                       .assign(season=(lambda x: pd.cut(x['month'],
#                                                bins=[0,3,6,9,12],
#                                                labels=['winter', 'spring', 'summer', 'fall'])),
#                               # period_label=(lambda x: pd.cut(x['year'], 
#                               #                              bins=[x['year'].min()-1, year_div-1, x['year'].max()],
#                               #                              labels= ['pre', 'post'])),
#                               depth_bool=(lambda x: 'full_depth'),
#                               segment=(lambda x: key)
#                               )
#                       .reset_index()
#                       )



# %%

var_list = ['SA', 'CT', 'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO_mg_L', 'DO (uM)']


depth_avgs_dict = dict()

for key in odf_dict.keys():
    
    depth_avgs_dict[key] = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_bool', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                      .set_index('datetime')
                      .groupby([pd.Grouper(freq='M'), 'depth_bool', 'var']).agg(['mean', 'count', 'std'])
                      .drop(columns =['lat','lon','cid', 'z','year', 'month', 'date_ordinal'])
                      )
    
    depth_avgs_dict[key].columns = depth_avgs_dict[key].columns.to_flat_index().map('_'.join)
    
    depth_avgs_dict[key] = (depth_avgs_dict[key]
                      .reset_index() 
                      .assign(
                              # period_label=(lambda x: pd.cut(x['year'], 
                              #                              bins=[x['year'].min()-1, year_div-1, x['year'].max()],
                              #                              labels= ['pre', 'post'])),
                              segment=(lambda x: key),
                              year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                              month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                              season=(lambda x: pd.cut(x['month'],
                                                       bins=[0,3,6,9,12],
                                                       labels=['winter', 'spring', 'summer', 'fall'])),
                              date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal()))
                              )
                      )
        
# %%

depth_avgs_df = pd.concat(depth_avgs_dict.values(), ignore_index=True)

depth_avgs_df['val_ci95hi'] = depth_avgs_df['val_mean'] + 1.96*depth_avgs_df['val_std']/np.sqrt(depth_avgs_df['val_count'])

depth_avgs_df['val_ci95lo'] = depth_avgs_df['val_mean'] - 1.96*depth_avgs_df['val_std']/np.sqrt(depth_avgs_df['val_count'])

# %%
shallow_avgs_df = depth_avgs_df[depth_avgs_df['depth_bool'] == 'shallow']

mid_avgs_df = depth_avgs_df[depth_avgs_df['depth_bool'] == 'mid']

deep_avgs_df  = depth_avgs_df[depth_avgs_df['depth_bool'] == 'deep']

# %%

 
for basin in basin_list:
    
    for var in var_list:
        
        plot_df = deep_avgs_df[(deep_avgs_df['segment'] == basin) & (deep_avgs_df['var'] == var) & (deep_avgs_df['val_count'] > 1)]
        
        if var == 'DO_mg_L':
            
            plot_df = plot_df[plot_df['val_mean'] <20]
            
        elif var == 'DO (uM)':
            
            plot_df = plot_df[plot_df['val_mean'] < 20*1000/32]
    
        fig, ax = plt.subplots(figsize=(10,5))
                
        for idx in plot_df.index:
            
            plt.vlines(plot_df.loc[idx, 'datetime'], plot_df.loc[idx, 'val_ci95lo'], plot_df.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
        plt.scatter(plot_df['datetime'], plot_df['val_mean'], color='k', sizes=[5])
        
        plt.title(basin + ' ' + var + ' >35m deep')
        
        ax.set_xlabel('Date')
        
        ax.set_ylabel(var)
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + var.replace(' ','').replace('(','').replace(')','') + '_deep_ci.png', dpi=500)
        
    
# %%

for basin in basin_list:

    plot_df = deep_avgs_df[(deep_avgs_df['segment'] == basin) & (deep_avgs_df['val_count'] > 1)]
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
    
    plot_df_SA = plot_df[plot_df['var'] == 'SA']
    
    plot_df_CT = plot_df[plot_df['var'] == 'CT']
    
    plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
    
    for idx in plot_df_SA.index:
        
        ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    for idx in plot_df_CT.index:
        
        ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
    
    for idx in plot_df_DO.index:
        
        ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    
    ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
    
    ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
    
    ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
    
    ax2.set_xlabel('Date')
    
    ax0.set_ylabel('SA [psu]')
    
    ax1.set_ylabel('CT [deg C]')
    
    ax2.set_ylabel('DO [mg/L]')
    
    ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax0.set_ylim([26,36])
    
    ax1.set_ylim([2,16])
    
    ax2.set_ylim([0,14])
    
    ax0.set_title(basin + ' >35m deep')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_deep_ci.png', dpi=500)

# %%
    
for basin in basin_list:
    
    for season in ['winter', 'spring', 'summer', 'fall']:
        
        plot_df = deep_avgs_df[(deep_avgs_df['segment'] == basin) & (deep_avgs_df['val_count'] > 1) & (deep_avgs_df['season'] == season)]
        
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
        
        plot_df_SA = plot_df[plot_df['var'] == 'SA']
        
        plot_df_CT = plot_df[plot_df['var'] == 'CT']
        
        plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
        
        for idx in plot_df_SA.index:
            
            ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        for idx in plot_df_CT.index:
            
            ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
        
        for idx in plot_df_DO.index:
            
            ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        
        ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
        
        ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
        
        ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
        
        ax2.set_xlabel('Date')
        
        ax0.set_ylabel('SA [psu]')
        
        ax1.set_ylabel('CT [deg C]')
        
        ax2.set_ylabel('DO [mg/L]')
        
        ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax0.set_ylim([26,36])
        
        ax1.set_ylim([2,16])
        
        ax2.set_ylim([0,14])
        
        ax0.set_title(basin + ' ' + season + ' >35m deep')
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + season + '_SA_CT_DO_deep_ci.png', dpi=500)
        

# %%

 
for basin in basin_list:
    
    for var in var_list:
        
        plot_df = mid_avgs_df[(mid_avgs_df['segment'] == basin) & (mid_avgs_df['var'] == var) & (mid_avgs_df['val_count'] > 1)]
        
        if var == 'DO_mg_L':
            
            plot_df = plot_df[plot_df['val_mean'] <20]
            
        elif var == 'DO (uM)':
            
            plot_df = plot_df[plot_df['val_mean'] < 20*1000/32]
    
        fig, ax = plt.subplots(figsize=(10,5))
                
        for idx in plot_df.index:
            
            plt.vlines(plot_df.loc[idx, 'datetime'], plot_df.loc[idx, 'val_ci95lo'], plot_df.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
        plt.scatter(plot_df['datetime'], plot_df['val_mean'], color='k', sizes=[5])
        
        plt.title(basin + ' ' + var + ' 15-35m deep')
        
        ax.set_xlabel('Date')
        
        ax.set_ylabel(var)
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + var.replace(' ','').replace('(','').replace(')','') + '_mid_ci.png', dpi=500)
        
    
# %%

for basin in basin_list:

    plot_df = mid_avgs_df[(mid_avgs_df['segment'] == basin) & (mid_avgs_df['val_count'] > 1)]
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
    
    plot_df_SA = plot_df[plot_df['var'] == 'SA']
    
    plot_df_CT = plot_df[plot_df['var'] == 'CT']
    
    plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
    
    for idx in plot_df_SA.index:
        
        ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    for idx in plot_df_CT.index:
        
        ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
    
    for idx in plot_df_DO.index:
        
        ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    
    ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
    
    ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
    
    ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
    
    ax2.set_xlabel('Date')
    
    ax0.set_ylabel('SA [psu]')
    
    ax1.set_ylabel('CT [deg C]')
    
    ax2.set_ylabel('DO [mg/L]')
    
    ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax0.set_ylim([26,36])
    
    ax1.set_ylim([2,16])
    
    ax2.set_ylim([0,14])
    
    ax0.set_title(basin + ' 15-35m deep')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_mid_ci.png', dpi=500)

# %%
    
for basin in basin_list:
    
    for season in ['winter', 'spring', 'summer', 'fall']:
        
        plot_df = mid_avgs_df[(mid_avgs_df['segment'] == basin) & (mid_avgs_df['val_count'] > 1) & (mid_avgs_df['season'] == season)]
        
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
        
        plot_df_SA = plot_df[plot_df['var'] == 'SA']
        
        plot_df_CT = plot_df[plot_df['var'] == 'CT']
        
        plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
        
        for idx in plot_df_SA.index:
            
            ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        for idx in plot_df_CT.index:
            
            ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
        
        for idx in plot_df_DO.index:
            
            ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        
        ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
        
        ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
        
        ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
        
        ax2.set_xlabel('Date')
        
        ax0.set_ylabel('SA [psu]')
        
        ax1.set_ylabel('CT [deg C]')
        
        ax2.set_ylabel('DO [mg/L]')
        
        ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax0.set_ylim([26,36])
        
        ax1.set_ylim([2,16])
        
        ax2.set_ylim([0,14])
        
        ax0.set_title(basin + ' ' + season + ' 15-35m deep')
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + season + '_SA_CT_DO_mid_ci.png', dpi=500)
        
# %%

for basin in basin_list:
    
    for var in var_list:
        
        plot_df = shallow_avgs_df[(shallow_avgs_df['segment'] == basin) & (shallow_avgs_df['var'] == var) & (shallow_avgs_df['val_count'] > 1)]
        
        if var == 'DO_mg_L':
            
            plot_df = plot_df[plot_df['val_mean'] <20]
            
        elif var == 'DO (uM)':
            
            plot_df = plot_df[plot_df['val_mean'] < 20*1000/32]
    
        fig, ax = plt.subplots(figsize=(10,5))
                
        for idx in plot_df.index:
            
            plt.vlines(plot_df.loc[idx, 'datetime'], plot_df.loc[idx, 'val_ci95lo'], plot_df.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
        plt.scatter(plot_df['datetime'], plot_df['val_mean'], color='k', sizes=[5])
        
        plt.title(basin + ' ' + var + ' <15m deep')
        
        ax.set_xlabel('Date')
        
        ax.set_ylabel(var)
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + var.replace(' ','').replace('(','').replace(')','') + '_shallow_ci.png', dpi=500)
        
    
# %%

for basin in basin_list:

    plot_df = shallow_avgs_df[(shallow_avgs_df['segment'] == basin) & (shallow_avgs_df['val_count'] > 1)]
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
    
    plot_df_SA = plot_df[plot_df['var'] == 'SA']
    
    plot_df_CT = plot_df[plot_df['var'] == 'CT']
    
    plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
    
    for idx in plot_df_SA.index:
        
        ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    for idx in plot_df_CT.index:
        
        ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
    
    for idx in plot_df_DO.index:
        
        ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    
    ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
    
    ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
    
    ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
    
    ax2.set_xlabel('Date')
    
    ax0.set_ylabel('SA [psu]')
    
    ax1.set_ylabel('CT [deg C]')
    
    ax2.set_ylabel('DO [mg/L]')
    
    ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax0.set_ylim([26,36])
    
    ax1.set_ylim([2,16])
    
    ax2.set_ylim([0,14])
    
    ax0.set_title(basin + ' <15m deep')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_shallow_ci.png', dpi=500)

# %%
    
for basin in basin_list:
    
    for season in ['winter', 'spring', 'summer', 'fall']:
        
        plot_df = shallow_avgs_df[(shallow_avgs_df['segment'] == basin) & (shallow_avgs_df['val_count'] > 1) & (shallow_avgs_df['season'] == season)]
        
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
        
        plot_df_SA = plot_df[plot_df['var'] == 'SA']
        
        plot_df_CT = plot_df[plot_df['var'] == 'CT']
        
        plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
        
        for idx in plot_df_SA.index:
            
            ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        for idx in plot_df_CT.index:
            
            ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
        
        for idx in plot_df_DO.index:
            
            ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        
        ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
        
        ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
        
        ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
        
        ax2.set_xlabel('Date')
        
        ax0.set_ylabel('SA [psu]')
        
        ax1.set_ylabel('CT [deg C]')
        
        ax2.set_ylabel('DO [mg/L]')
        
        ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax0.set_ylim([26,36])
        
        ax1.set_ylim([2,16])
        
        ax2.set_ylim([0,14])
        
        ax0.set_title(basin + ' ' + season + ' <15m deep')
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + season + '_SA_CT_DO_shallow_ci.png', dpi=500)
        