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
#basin_list = ['sog_n', 'sog_s', 'soj','sji','mb', 'wb', 'hc', 'ss']
basin_list = ['ps', 'mb','hc', 'wb', 'ss']
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
source_list = ['ecology', 'nceiSalish']
otype_list = ['ctd', 'bottle']
year_list = np.arange(1999, 2022)
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

depth_div = -35

year_div = 2010

for key in odf_dict.keys():
    
    odf_dict[key] = (odf_dict[key]
                     .assign(
                         datetime=(lambda x: pd.to_datetime(x['time'])),
                         depth_bool=(lambda x: pd.cut(x['z'], 
                                                      bins=[x['z'].min()-1, depth_div, 0],
                                                      labels= ['deep', 'shallow'])),
                         year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                         month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                         season=(lambda x: pd.cut(x['month'],
                                                  bins=[0,3,6,9,12],
                                                  labels=['winter', 'spring', 'summer', 'fall'])),
                         DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                         period_label=(lambda x: pd.cut(x['year'], 
                                                      bins=[x['year'].min()-1, year_div-1, x['year'].max()],
                                                      labels= ['pre', 'post'])),
                         date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())),
                         segment=(lambda x: key)
                             )
                     )


# %%

full_avgs_dict = dict()

for key in odf_dict.keys():
    
    full_avgs_dict[key] = (odf_dict[key]
                      .set_index('datetime')
                      .groupby([pd.Grouper(freq='M')]).mean()
                      .drop(columns =['lat','lon','cid'])
                      .assign(season=(lambda x: pd.cut(x['month'],
                                               bins=[0,3,6,9,12],
                                               labels=['winter', 'spring', 'summer', 'fall'])),
                              period_label=(lambda x: pd.cut(x['year'], 
                                                           bins=[x['year'].min()-1, year_div-1, x['year'].max()],
                                                           labels= ['pre', 'post'])),
                              depth_bool=(lambda x: 'full_depth'),
                              segment=(lambda x: key)
                              )
                      .reset_index()
                      )



# %%

depth_avgs_dict = dict()

for key in odf_dict.keys():
    
    depth_avgs_dict[key] = (odf_dict[key]
                      .set_index('datetime')
                      .groupby([pd.Grouper(freq='M'),'depth_bool']).mean()
                      .drop(columns =['lat','lon','cid'])
                      .assign(season=(lambda x: pd.cut(x['month'],
                                               bins=[0,3,6,9,12],
                                               labels=['winter', 'spring', 'summer', 'fall'])),
                              period_label=(lambda x: pd.cut(x['year'], 
                                                           bins=[x['year'].min()-1, year_div-1, x['year'].max()],
                                                           labels= ['pre', 'post'])),
                              segment=(lambda x: key)
                              )
                      .reset_index()
                      )

# %%

full_avgs_df = pd.concat(full_avgs_dict.values(), ignore_index=True)

depth_avgs_df = pd.concat(depth_avgs_dict.values(), ignore_index=True)

# %%

avgs_df = pd.concat([full_avgs_df, depth_avgs_df], ignore_index=True)
    
# %%

avgs_df_copy0 = (avgs_df.copy()
                .assign(period_label=(lambda x: 'full_period'))
                )

avgs_df_temp0 = pd.concat([avgs_df, avgs_df_copy0], ignore_index=True)


avgs_df_copy1 = (avgs_df_temp0.copy()
                .assign(season=(lambda x: 'all_year'))
                )


avgs_df_extend = pd.concat([avgs_df_temp0, avgs_df_copy1], ignore_index=True)


# %%



pre_color = '#2d4159'
post_color = '#ff715b'

deep_color = '#2c456b'
shallow_color = '#83aff0'

winter_color = '#1126a5'
spring_color = '#a4d13a'
summer_color = '#fd3f41'
fall_color = '#680e03'
# %%

# for segment in avgs_df['segment'].unique():
    
#     for season in 
    
#     for depth in avgs_df['depth_bool'].unique():
        
        
    
#     plot_df = avgs_df[(avgs_df['segment'] == segment) & (avgs_df['depth_bool'] =)]
    
#     fig, ax = plt.subplots(figsize=(10,5))
    
#     plt.scatter(plot_df['datetime'], plot_df['DO_mg_L'], color = 'k')
    
#     ax.set_xlabel('Date')
    
#     ax.set_ylabel('DO [mg/L]')
    
#     ax.set_ylim(0,15)
    
#     ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
    
#     plt.grid(color='lightgray', alpha=0.5, linestyle='--')
    
#     fig.tight_layout()
    
#     plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + segment + '_month_avg_DO.png')
    # COME BACK LATER
    


# %%


for key in odf_dict.keys():
    
    plot_df = full_avgs_dict[key].copy()
    

    fig, ax = plt.subplots(figsize=(10,5))
    
    plt.scatter(plot_df['datetime'], plot_df['DO_mg_L'], color = 'k')
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('DO [mg/L]')
    
    ax.set_ylim(0,15)
    
    ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
    
    plt.grid(color='lightgray', alpha=0.5, linestyle='--')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + key+ '_month_avg_DO.png')


    plot_df = depth_avgs_dict[key].copy()

    fig, ax = plt.subplots(figsize=(10,5))
    
    plt.scatter(plot_df[plot_df['depth_bool'] == 'deep']['datetime'], plot_df[plot_df['depth_bool'] == 'deep']['DO_mg_L'], color = deep_color, label='Deep [<35m]')
    
    plt.scatter(plot_df[plot_df['depth_bool'] == 'shallow']['datetime'], plot_df[plot_df['depth_bool'] == 'shallow']['DO_mg_L'], color = shallow_color, label='Shallow [>=35m]')
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('DO [mg/L]')
    
    ax.set_ylim(0,15)
    
    ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
    
    ax.legend(loc='best')
    
    plt.grid(color='lightgray', alpha=0.5, linestyle='--')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + key +'_month_avg_DO_deep_shallow.png')
    
    for season in ['winter', 'spring', 'summer','fall']:
        
        if season == 'winter':
            
            c = winter_color
        
        elif season == 'spring':
            
            c= spring_color
            
        elif season == 'summer':
            
            c= summer_color
        
        elif season == 'fall':
            
            c= fall_color
        
        plot_df = full_avgs_dict[key][avgs_dict[key]['season'] == season].copy()
        
        fig, ax = plt.subplots(figsize=(10,5))
        
        plt.scatter(plot_df['datetime'], plot_df['DO_mg_L'], color = c)
        
        ax.set_xlabel('Date')
        
        ax.set_ylabel('DO [mg/L]')
        
        ax.set_ylim(0,15)
        
        ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        
        plt.grid(color='lightgray', alpha=0.5, linestyle='--')
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + key+ '_' + season + '_month_avg_DO.png')


        plot_df = depth_avgs_dict[key][depth_avgs_dict[key]['season'] == season].copy()

        fig, ax = plt.subplots(figsize=(10,5))
        
        plt.scatter(plot_df[plot_df['depth_bool'] == 'deep']['datetime'], plot_df[plot_df['depth_bool'] == 'deep']['DO_mg_L'], color = deep_color, label='Deep [<35m]')
        
        plt.scatter(plot_df[plot_df['depth_bool'] == 'shallow']['datetime'], plot_df[plot_df['depth_bool'] == 'shallow']['DO_mg_L'], color = shallow_color, label='Shallow [>=35m]')
        
        ax.set_xlabel('Date')
        
        ax.set_ylabel('DO [mg/L]')
        
        ax.set_ylim(0,15)
        
        ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        
        ax.legend(loc='best')
        
        plt.grid(color='lightgray', alpha=0.5, linestyle='--')
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + key + '_' + season +'_month_avg_DO_deep_shallow.png')



# %%

for key in odf_dict.keys():
    
    stat_df = full_avgs_dict[key].copy()
    
    fig, ax = plt.subplots(figsize=(5,5))

    sns.boxplot(data=stat_df, x="period_label", y="DO_mg_L", hue="period_label", palette ={'pre':pre_color,'post':post_color})
    
    ax.set_xticklabels(['Pre-2010', '2010-On'])
    
    ax.set_ylabel('DO [mg/L]')
    
    ax.set_ylim(0,15)
    
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5, zorder=3)
    
    ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/box_plots_' + key +'_DO_avgs.png')
    
    
    stat_df = depth_avgs_dict[key].copy()

    fig, ax = plt.subplots(figsize=(5,5))

    sns.boxplot(data=stat_df, x="period_label", y="DO_mg_L", hue="depth_bool", palette ={'shallow':shallow_color,'deep':deep_color})
    
    ax.set_xticklabels(['Pre-2010', '2010-On'])
    
    ax.set_ylabel('DO [mg/L]')
    
    ax.set_ylim(0,15)
    
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5, zorder=3)
    
    ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/box_plots_' + key + '_depths_DO_avgs.png')
    
    
    
    for season in ['winter', 'spring', 'summer','fall']:
        
        
        plot_df = full_avgs_dict[key][full_avgs_dict[key]['season'] == season].copy()
        
        fig, ax = plt.subplots(figsize=(5,5))

        sns.boxplot(data=stat_df, x="period_label", y="DO_mg_L", hue="period_label", palette ={'pre':pre_color,'post':post_color})
        
        ax.set_xticklabels(['Pre-2010', '2010-On'])
        
        ax.set_ylabel('DO [mg/L]')
        
        ax.set_ylim(0,15)
        
        #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5, zorder=3)
        
        ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/box_plots_' + key + '_' + season +'_DO_avgs.png')
        
        
        plot_df = depth_avgs_dict[key][depth_avgs_dict[key]['season'] == season].copy()

        fig, ax = plt.subplots(figsize=(5,5))

        sns.boxplot(data=stat_df, x="period_label", y="DO_mg_L", hue="depth_bool", palette ={'shallow':shallow_color,'deep':deep_color})
        
        ax.set_xticklabels(['Pre-2010', '2010-On'])
        
        ax.set_ylabel('DO [mg/L]')
        
        ax.set_ylim(0,15)
        
        #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5, zorder=3)
        
        ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/box_plots_' + key + '_' + season + '_depths_DO_avgs.png')


# %%



alpha = 0.05
conf = 1 - alpha

conf_upper = 1-alpha/2
conf_lower = alpha/2

z_alpha_upper = stats.norm.ppf(conf_upper)
z_alpha_lower = stats.norm.ppf(conf_lower)


mu_0 = 0

# %%

n_pre_dict = dict()

n_post_dict = dict()

X_pre_dict = dict()

X_post_dict = dict()

s_pre_dict = dict()

s_post_dict = dict()

z_test_dict = dict()

p_value_dict = dict()


for key in odf_dict.keys():
    
    n_pre_dict[key] = dict()

    n_post_dict[key] = dict()

    X_pre_dict[key] = dict()

    X_post_dict[key] = dict()

    s_pre_dict[key] = dict()

    s_post_dict[key] = dict()

    z_test_dict[key] = dict()
    
    p_value_dict[key] = dict()
    
    
    n_pre_dict[key]['all_year'] = dict()

    n_post_dict[key]['all_year'] = dict()

    X_pre_dict[key]['all_year'] = dict()

    X_post_dict[key]['all_year'] = dict()

    s_pre_dict[key]['all_year'] = dict()

    s_post_dict[key]['all_year'] = dict()

    z_test_dict[key]['all_year'] = dict()
    
    p_value_dict[key]['all_year'] = dict()

    
    
    stat_df = full_avgs_dict[key].copy()

    pre = stat_df[stat_df['period_label'] == 'pre']
    
    post = stat_df[stat_df['period_label'] == 'post']


    n_1 = pre['year'].count()
    n_2 = post['year'].count()
    

    X_1 = pre['DO_mg_L'].mean()
    X_2 = post['DO_mg_L'].mean()
    
    s_1 = pre['DO_mg_L'].std(ddof=1)
    s_2 = post['DO_mg_L'].std(ddof=1)
    
    sigma_prime = np.sqrt(s_1**2/n_1 + s_2**2/n_2)


    z_score = (X_2 - X_1 - mu_0)/sigma_prime

    p_value = 1 - stats.norm.cdf(z_score)
    
    
    n_pre_dict[key]['all_year']['full_depth'] = n_1

    n_post_dict[key]['all_year']['full_depth'] = n_2

    X_pre_dict[key]['all_year']['full_depth'] = X_1

    X_post_dict[key]['all_year']['full_depth'] = X_2

    s_pre_dict[key]['all_year']['full_depth'] = s_1

    s_post_dict[key]['all_year']['full_depth'] = s_2

    z_test_dict[key]['all_year']['full_depth'] = z_score
    
    p_value_dict[key]['all_year']['full_depth'] = p_value
    
    z = np.linspace(-4, 4, num=160) * sigma_prime
    
    plt.figure(figsize=(10,7))
    # Plot the z-distribution here
    plt.plot(z, stats.norm.pdf(z, 0, sigma_prime), label='Null PDF: ($\overline{X}_2 - \overline{X}_1$) = 0')
    
    plt.axvline(z_alpha_upper*sigma_prime, color='black', linestyle='-', label='$z_{a}$')
    plt.axvline(z_alpha_lower*sigma_prime, color='black', linestyle='-') # , label='$z_{a}$')
    shade_upper = np.linspace(z_alpha_upper*sigma_prime, np.max(z), 10)
    shade_lower = np.linspace(np.min(z), z_alpha_lower*sigma_prime, 10)
    
    plt.fill_between(shade_upper, stats.norm.pdf(shade_upper, 0, sigma_prime) ,  color='k', alpha=0.5, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
    plt.fill_between(shade_lower, stats.norm.pdf(shade_lower, 0, sigma_prime) ,  color='k', alpha=0.5) #, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
    
    plt.axvline(z_score*sigma_prime, color='red', linestyle='-', label='z-test')
    plt.xlabel('($\overline{X}_2 - \overline{X}_1$) [cfs]')
    plt.ylabel('PDF')
    #plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.ylim(bottom = 0)
    plt.legend(loc='upper right');
    
    plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/z_test_pre_post_2010_'+ key + '_DO.png')
    
    
    for depth in ['shallow', 'deep']:
        
        stat_df = depth_avgs_dict[key][depth_avgs_dict[key]['depth_bool'] == depth].copy()

        pre = stat_df[stat_df['period_label'] == 'pre']
        
        post = stat_df[stat_df['period_label'] == 'post']


        n_1 = pre['year'].count()
        n_2 = post['year'].count()
        

        X_1 = pre['DO_mg_L'].mean()
        X_2 = post['DO_mg_L'].mean()
        
        s_1 = pre['DO_mg_L'].std(ddof=1)
        s_2 = post['DO_mg_L'].std(ddof=1)
        
        sigma_prime = np.sqrt(s_1**2/n_1 + s_2**2/n_2)


        z_score = (X_2 - X_1 - mu_0)/sigma_prime

        p_value = 1 - stats.norm.cdf(z_score)
        
        
        n_pre_dict[key]['all_year'][depth] = n_1

        n_post_dict[key]['all_year'][depth] = n_2

        X_pre_dict[key]['all_year'][depth] = X_1

        X_post_dict[key]['all_year'][depth] = X_2

        s_pre_dict[key]['all_year'][depth] = s_1

        s_post_dict[key]['all_year'][depth] = s_2

        z_test_dict[key]['all_year'][depth] = z_score
        
        p_value_dict[key]['all_year'][depth] = p_value



        z = np.linspace(-4, 4, num=160) * sigma_prime
        
        plt.figure(figsize=(10,7))
        # Plot the z-distribution here
        plt.plot(z, stats.norm.pdf(z, 0, sigma_prime), label='Null PDF: ($\overline{X}_2 - \overline{X}_1$) = 0')
        
        plt.axvline(z_alpha_upper*sigma_prime, color='black', linestyle='-', label='$z_{a}$')
        plt.axvline(z_alpha_lower*sigma_prime, color='black', linestyle='-') # , label='$z_{a}$')
        shade_upper = np.linspace(z_alpha_upper*sigma_prime, np.max(z), 10)
        shade_lower = np.linspace(np.min(z), z_alpha_lower*sigma_prime, 10)
        
        plt.fill_between(shade_upper, stats.norm.pdf(shade_upper, 0, sigma_prime) ,  color='k', alpha=0.5, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
        plt.fill_between(shade_lower, stats.norm.pdf(shade_lower, 0, sigma_prime) ,  color='k', alpha=0.5) #, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
        
        plt.axvline(z_score*sigma_prime, color='red', linestyle='-', label='z-test')
        plt.xlabel('($\overline{X}_2 - \overline{X}_1$) [cfs]')
        plt.ylabel('PDF')
        # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.ylim(bottom = 0)
        plt.legend(loc='upper right');
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/z_test_pre_post_2010_'+ key + '_' + depth + '_DO.png')
        
        
        
        
    for season in ['winter', 'spring', 'summer','fall']:
        
        stat_df = full_avgs_dict[key][full_avgs_dict[key]['season'] == season].copy()

        n_pre_dict[key][season] = dict()

        n_post_dict[key][season] = dict()

        X_pre_dict[key][season] = dict()

        X_post_dict[key][season] = dict()

        s_pre_dict[key][season] = dict()

        s_post_dict[key][season] = dict()

        z_test_dict[key][season] = dict()
        
        p_value_dict[key][season]= dict()

        
        pre = stat_df[stat_df['period_label'] == 'pre']
        
        post = stat_df[stat_df['period_label'] == 'post']
        
        
        n_1 = pre['year'].count()
        n_2 = post['year'].count()
        

        X_1 = pre['DO_mg_L'].mean()
        X_2 = post['DO_mg_L'].mean()
        
        s_1 = pre['DO_mg_L'].std(ddof=1)
        s_2 = post['DO_mg_L'].std(ddof=1)
        
        sigma_prime = np.sqrt(s_1**2/n_1 + s_2**2/n_2)
        
        
        z_score = (X_2 - X_1 - mu_0)/sigma_prime
        
        p_value = 1 - stats.norm.cdf(z_score)
        
        
        n_pre_dict[key][season]['full_depth'] = n_1
        
        n_post_dict[key][season]['full_depth'] = n_2
        
        X_pre_dict[key][season]['full_depth'] = X_1
        
        X_post_dict[key][season]['full_depth'] = X_2
        
        s_pre_dict[key][season]['full_depth'] = s_1
        
        s_post_dict[key][season]['full_depth'] = s_2
        
        z_test_dict[key][season]['full_depth'] = z_score
        
        p_value_dict[key][season]['full_depth'] = p_value
        
        z = np.linspace(-4, 4, num=160) * sigma_prime
        
        plt.figure(figsize=(10,7))
        # Plot the z-distribution here
        plt.plot(z, stats.norm.pdf(z, 0, sigma_prime), label='Null PDF: ($\overline{X}_2 - \overline{X}_1$) = 0')
        
        plt.axvline(z_alpha_upper*sigma_prime, color='black', linestyle='-', label='$z_{a}$')
        plt.axvline(z_alpha_lower*sigma_prime, color='black', linestyle='-') # , label='$z_{a}$')
        shade_upper = np.linspace(z_alpha_upper*sigma_prime, np.max(z), 10)
        shade_lower = np.linspace(np.min(z), z_alpha_lower*sigma_prime, 10)
        
        plt.fill_between(shade_upper, stats.norm.pdf(shade_upper, 0, sigma_prime) ,  color='k', alpha=0.5, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
        plt.fill_between(shade_lower, stats.norm.pdf(shade_lower, 0, sigma_prime) ,  color='k', alpha=0.5) #, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
        
        plt.axvline(z_score*sigma_prime, color='red', linestyle='-', label='z-test')
        plt.xlabel('($\overline{X}_2 - \overline{X}_1$) [cfs]')
        plt.ylabel('PDF')
        #plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        #plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.ylim(bottom = 0)
        plt.legend(loc='upper right');
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/z_test_pre_post_2010_'+ key + '_' + season + '_DO.png')
        
        
        for depth in ['shallow', 'deep']:
            
            stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == depth) & (depth_avgs_dict[key]['season'] == season)].copy()

            pre = stat_df[stat_df['period_label'] == 'pre']
            
            post = stat_df[stat_df['period_label'] == 'post']


            n_1 = pre['year'].count()
            n_2 = post['year'].count()
            

            X_1 = pre['DO_mg_L'].mean()
            X_2 = post['DO_mg_L'].mean()
            
            s_1 = pre['DO_mg_L'].std(ddof=1)
            s_2 = post['DO_mg_L'].std(ddof=1)
            
            sigma_prime = np.sqrt(s_1**2/n_1 + s_2**2/n_2)


            z_score = (X_2 - X_1 - mu_0)/sigma_prime

            p_value = 1 - stats.norm.cdf(z_score)
            
            
            n_pre_dict[key][season][depth] = n_1

            n_post_dict[key][season][depth] = n_2

            X_pre_dict[key][season][depth] = X_1

            X_post_dict[key][season][depth] = X_2

            s_pre_dict[key][season][depth] = s_1

            s_post_dict[key][season][depth] = s_2

            z_test_dict[key][season][depth] = z_score
            
            p_value_dict[key][season][depth] = p_value



            z = np.linspace(-4, 4, num=160) * sigma_prime
            
            plt.figure(figsize=(10,7))
            # Plot the z-distribution here
            plt.plot(z, stats.norm.pdf(z, 0, sigma_prime), label='Null PDF: ($\overline{X}_2 - \overline{X}_1$) = 0')
            
            plt.axvline(z_alpha_upper*sigma_prime, color='black', linestyle='-', label='$z_{a}$')
            plt.axvline(z_alpha_lower*sigma_prime, color='black', linestyle='-') # , label='$z_{a}$')
            shade_upper = np.linspace(z_alpha_upper*sigma_prime, np.max(z), 10)
            shade_lower = np.linspace(np.min(z), z_alpha_lower*sigma_prime, 10)
            
            plt.fill_between(shade_upper, stats.norm.pdf(shade_upper, 0, sigma_prime) ,  color='k', alpha=0.5, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
            plt.fill_between(shade_lower, stats.norm.pdf(shade_lower, 0, sigma_prime) ,  color='k', alpha=0.5) #, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
            
            plt.axvline(z_score*sigma_prime, color='red', linestyle='-', label='z-test')
            plt.xlabel('($\overline{X}_2 - \overline{X}_1$) [cfs]')
            plt.ylabel('PDF')
            # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.ylim(bottom = 0)
            plt.legend(loc='upper right');
            
            plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/z_test_pre_post_2010_'+ key + '_' + season + '_' + depth + '_DO.png')
        
        
# %%

for key in ['ps']:
    
    for depth in ['deep']:
        
        stat_df = depth_avgs_dict[key][depth_avgs_dict[key]['depth_bool'] == depth].copy()

        pre = stat_df[stat_df['period_label'] == 'pre']
        
        post = stat_df[stat_df['period_label'] == 'post']


        n_1 = pre['year'].count()
        n_2 = post['year'].count()
        

        X_1 = pre['DO_mg_L'].mean()
        X_2 = post['DO_mg_L'].mean()
        
        s_1 = pre['DO_mg_L'].std(ddof=1)
        s_2 = post['DO_mg_L'].std(ddof=1)
        
        sigma_prime = np.sqrt(s_1**2/n_1 + s_2**2/n_2)


        z_score = (X_2 - X_1 - mu_0)/sigma_prime

        p_value = 1 - stats.norm.cdf(z_score)
        
        
        # n_pre_dict[key]['all_year'][depth] = n_1

        # n_post_dict[key]['all_year'][depth] = n_2

        # X_pre_dict[key]['all_year'][depth] = X_1

        # X_post_dict[key]['all_year'][depth] = X_2

        # s_pre_dict[key]['all_year'][depth] = s_1

        # s_post_dict[key]['all_year'][depth] = s_2

        # z_test_dict[key]['all_year'][depth] = z_score
        
        # p_value_dict[key]['all_year'][depth] = p_value



        z = np.linspace(-4, 4, num=160) * sigma_prime
        
        plt.figure(figsize=(10,6))
        # Plot the z-distribution here
        plt.plot(z, stats.norm.pdf(z, 0, sigma_prime), label='Null PDF:\n($\overline{X}_{2010-On}-\overline{X}_{Pre-2010}$)=0')
        
        plt.axvline(z_alpha_upper*sigma_prime, color='black', linestyle='-', label='$z_{a}$')
        plt.axvline(z_alpha_lower*sigma_prime, color='black', linestyle='-') # , label='$z_{a}$')
        shade_upper = np.linspace(z_alpha_upper*sigma_prime, np.max(z), 10)
        shade_lower = np.linspace(np.min(z), z_alpha_lower*sigma_prime, 10)
        
        plt.fill_between(shade_upper, stats.norm.pdf(shade_upper, 0, sigma_prime) ,  color='k', alpha=0.5, label='rejection region for\nalpha={}'.format(np.round(1-conf,2)))
        plt.fill_between(shade_lower, stats.norm.pdf(shade_lower, 0, sigma_prime) ,  color='k', alpha=0.5) #, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
        
        plt.axvline(z_score*sigma_prime, color='red', linestyle='-', label='z-test\n(for observed mean diff.)')
        plt.xlabel('($\overline{X}_{2010-On} - \overline{X}_{Pre-2010}$)')
        plt.ylabel('Probability Density Function')
        # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.ylim(bottom = 0)
        plt.legend(loc='upper right');
        
        plt.title('Puget Sound [DO] - Deep (>= 35m) - All Months')
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/z_test_pre_post_2010_'+ key + '_' + depth + '_DO_present.png', dpi=500)

            
            
# %%

r_dict = dict()

for key in ['ps']:
    
    for depth in ['deep']:
        
        stat_df = depth_avgs_dict[key][depth_avgs_dict[key]['depth_bool'] == depth].copy()
            
        x_temp = stat_df['date_ordinal']
        y_temp = stat_df['DO_mg_L'].to_numpy()
        
        mask = ~np.isnan(x_temp) & ~np.isnan(y_temp)
        
        x = x_temp[mask]
        
        y = y_temp[mask]

        
        B1, B0, r, p, sB1 = stats.linregress(x, y)
        
        alpha = 0.05

        n = len(x)
        
        dof = n - 2
        
        t = stats.t.ppf(1-alpha/2, dof)
        
        B1_upper = B1 + t * sB1
        B1_lower = B1 - t * sB1
        
        B0_upper = y.mean() - B1_upper*x.mean()
        B0_lower = y.mean() - B1_lower*x.mean()
        
        p_x = np.linspace(x.min(),x.max(),100)
        
        p_y = B0 + B1*p_x
        
        sst_x = np.sum((x - np.mean(x))**2)
        
        s = sB1 * np.sqrt(sst_x)
        
        sigma_ep = np.sqrt( s**2 * (1 + 1/n + ( ( n*(p_x-x.mean())**2 ) / ( n*np.sum(x**2) - np.sum(x)**2 ) ) ) )
        
        n_p = len(p_x)
        
        dof = n_p - 2
        
        t_p = stats.t.ppf(1-alpha/2, dof)
        
        p_y_lower = p_y - t_p * sigma_ep
        p_y_upper = p_y + t_p * sigma_ep

        fig, ax = plt.subplots(figsize=(10,6))

        ax.scatter(x, y, c='gray', alpha=0.9, label='original data')
        
        ax.plot([x.min(), x.max()], [y.mean(), y.mean()] , '--k', label='null hypothesis')
        
        #plt.axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
        
        ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-r', label='least-squares linear regression')
        
        ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , ':r', label='slope confidence limit')
        ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , ':r')
        
        ax.plot(p_x, p_y_upper, ':b', label='prediction interval')
        ax.plot(p_x, p_y_lower, ':b')
        
        plt.legend(loc='lower left') #, bbox_to_anchor=(1, 0.5));
        
        
        ax.set_xlabel('Date')
        ax.set_ylabel('DO [mg/L]')
        
        ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        labels = [datetime.date(1999,1,1), datetime.date(2000,1,1), datetime.date(2001,1,1), datetime.date(2002,1,1), datetime.date(2003,1,1),
              datetime.date(2004,1,1), datetime.date(2005,1,1), 
              datetime.date(2006,1,1), datetime.date(2007,1,1), 
              datetime.date(2008,1,1),  datetime.date(2009,1,1), 
              datetime.date(2010,1,1),  datetime.date(2011,1,1), 
              datetime.date(2012,1,1),  datetime.date(2013,1,1), 
              datetime.date(2014,1,1),  datetime.date(2015,1,1), 
              datetime.date(2016,1,1), datetime.date(2017,1,1), 
              datetime.date(2018,1,1), datetime.date(2019,1,1), 
              datetime.date(2020,1,1)]

        new_labels = [datetime.date.toordinal(item) for item in labels]

        ax.set_xticks(new_labels)


        ax.set_xticklabels(['1999', '','2001','','2003','','2005','','2007',
                    '','2009','','2011','','2013','','2015','','2017','','2019',
                    ''], rotation=0,
                           fontdict={'horizontalalignment':'center'})
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        plt.title('Puget Sound [DO] - Deep (>= 35m) - All Months')
                
        ax.set_ylim(0,10)
        
        ax.text(0.9, 0.9, '$r^2 = {}$'.format(np.round(r**2,2)), horizontalalignment='center', fontsize=14,
         verticalalignment='center', transform=ax.transAxes)

        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/reg_trend_'+ key + '_' + depth + '_DO_present.png', dpi=500)

# %%

for key in ['ps']:
    
    for depth in ['deep']:
        
        stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == depth) & (depth_avgs_dict[key]['period_label'] == 'post')].copy()
        
            
        x_temp = stat_df['date_ordinal']
        y_temp = stat_df['DO_mg_L'].to_numpy()
        
        mask = ~np.isnan(x_temp) & ~np.isnan(y_temp)
        
        x = x_temp[mask]
        
        y = y_temp[mask]

        
        B1, B0, r, p, sB1 = stats.linregress(x, y)
        
        alpha = 0.05

        n = len(x)
        
        dof = n - 2
        
        t = stats.t.ppf(1-alpha/2, dof)
        
        B1_upper = B1 + t * sB1
        B1_lower = B1 - t * sB1
        
        B0_upper = y.mean() - B1_upper*x.mean()
        B0_lower = y.mean() - B1_lower*x.mean()
        
        p_x = np.linspace(x.min(),x.max(),100)
        
        p_y = B0 + B1*p_x
        
        sst_x = np.sum((x - np.mean(x))**2)
        
        s = sB1 * np.sqrt(sst_x)
        
        sigma_ep = np.sqrt( s**2 * (1 + 1/n + ( ( n*(p_x-x.mean())**2 ) / ( n*np.sum(x**2) - np.sum(x)**2 ) ) ) )
        
        n_p = len(p_x)
        
        dof = n_p - 2
        
        t_p = stats.t.ppf(1-alpha/2, dof)
        
        p_y_lower = p_y - t_p * sigma_ep
        p_y_upper = p_y + t_p * sigma_ep

        fig, ax = plt.subplots(figsize=(6,6))

        ax.scatter(x, y, c='gray', alpha=0.9, label='original data')
        
        ax.plot([x.min(), x.max()], [y.mean(), y.mean()] , '--k', label='null hypothesis')
        
        #plt.axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
        
        ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-r', label='least-squares linear regression')
        
        ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , ':r', label='slope confidence limit')
        ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , ':r')
        
        ax.plot(p_x, p_y_upper, ':b', label='prediction interval')
        ax.plot(p_x, p_y_lower, ':b')
        
        plt.legend(loc='lower left') #, bbox_to_anchor=(1, 0.5));
        
        
        ax.set_xlabel('Date')
        ax.set_ylabel('DO [mg/L]')
        
        ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        labels = [
              datetime.date(2010,1,1),  datetime.date(2011,1,1), 
              datetime.date(2012,1,1),  datetime.date(2013,1,1), 
              datetime.date(2014,1,1),  datetime.date(2015,1,1), 
              datetime.date(2016,1,1), datetime.date(2017,1,1), 
              datetime.date(2018,1,1), datetime.date(2019,1,1), 
              datetime.date(2020,1,1)]

        new_labels = [datetime.date.toordinal(item) for item in labels]

        ax.set_xticks(new_labels)


        ax.set_xticklabels(['2010','','2012','','2014','','2016','','2018','',
                    '2020'], rotation=0,
                           fontdict={'horizontalalignment':'center'})
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        plt.title('Puget Sound [DO] - 2010-On - Deep (>= 35m) - All Months')
                
        ax.set_ylim(0,10)
        
        ax.text(0.9, 0.9, '$r^2 = {}$'.format(np.round(r**2,2)), horizontalalignment='center', fontsize=14,
         verticalalignment='center', transform=ax.transAxes)

        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/reg_trend_'+ key + '_' + depth + '_post_2010_DO_present.png', dpi=500)



# %%

# okay now time trend analysis

r_dict = dict()

for key in odf_dict.keys():
    
    stat_df = full_avgs_dict[key].copy()
    
    r_dict[key] = dict()
    
    r_dict[key]['all_year'] = dict() 

    x_temp = stat_df['date_ordinal']
    y_temp = stat_df['DO_mg_L'].to_numpy()
    
    mask = ~np.isnan(x_temp) & ~np.isnan(y_temp)
    
    x = x_temp[mask]
    
    y = y_temp[mask]

    
    B1, B0, r, p, sB1 = stats.linregress(x, y)
    
    alpha = 0.05

    n = len(x)
    
    dof = n - 2
    
    t = stats.t.ppf(1-alpha/2, dof)
    
    B1_upper = B1 + t * sB1
    B1_lower = B1 - t * sB1
    
    B0_upper = y.mean() - B1_upper*x.mean()
    B0_lower = y.mean() - B1_lower*x.mean()
    
    p_x = np.linspace(x.min(),x.max(),100)
    
    p_y = B0 + B1*p_x
    
    sst_x = np.sum((x - np.mean(x))**2)
    
    s = sB1 * np.sqrt(sst_x)
    
    sigma_ep = np.sqrt( s**2 * (1 + 1/n + ( ( n*(p_x-x.mean())**2 ) / ( n*np.sum(x**2) - np.sum(x)**2 ) ) ) )
    
    n_p = len(p_x)
    
    dof = n_p - 2
    
    t_p = stats.t.ppf(1-alpha/2, dof)
    
    p_y_lower = p_y - t_p * sigma_ep
    p_y_upper = p_y + t_p * sigma_ep

    fig, ax = plt.subplots(figsize=(12,8))

    ax.scatter(x, y, c='k', label='Original Data')
    
    ax.plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
    
    plt.axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
    
    ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-r', label='Least Squares Linear Regression Model')
    
    ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
    ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
    
    ax.plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
    ax.plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    
    ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia

    ax.set_xlabel('Date')
    ax.set_ylabel('DO [mg/L]')
    
    labels = [datetime.date(1999,1,1), datetime.date(2000,1,1), datetime.date(2001,1,1), datetime.date(2002,1,1), datetime.date(2003,1,1),
          datetime.date(2004,1,1), datetime.date(2005,1,1), 
          datetime.date(2006,1,1), datetime.date(2007,1,1), 
          datetime.date(2008,1,1),  datetime.date(2009,1,1), 
          datetime.date(2010,1,1),  datetime.date(2011,1,1), 
          datetime.date(2012,1,1),  datetime.date(2013,1,1), 
          datetime.date(2014,1,1),  datetime.date(2015,1,1), 
          datetime.date(2016,1,1), datetime.date(2017,1,1), 
          datetime.date(2018,1,1), datetime.date(2019,1,1), 
          datetime.date(2020,1,1)]

    new_labels = [datetime.date.toordinal(item) for item in labels]

    ax.set_xticks(new_labels)


    ax.set_xticklabels(['1999', '2000','2001','2002','2003','2004','2005','2006','2007',
                '2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
                '2020'], rotation=0,
                       fontdict={'horizontalalignment':'center'})
    
    plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax.set_ylim(0,15)
    
    ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r**2,2)), horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes)


    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/reg_trend_'+ key + '_DO.png')
    
    r_dict[key]['all_year']['full_depth'] = r

    
    for depth in ['shallow','deep']:
        
        stat_df = depth_avgs_dict[key][depth_avgs_dict[key]['depth_bool'] == depth].copy()
            
        x_temp = stat_df['date_ordinal']
        y_temp = stat_df['DO_mg_L'].to_numpy()
        
        mask = ~np.isnan(x_temp) & ~np.isnan(y_temp)
        
        x = x_temp[mask]
        
        y = y_temp[mask]

        
        B1, B0, r, p, sB1 = stats.linregress(x, y)
        
        alpha = 0.05

        n = len(x)
        
        dof = n - 2
        
        t = stats.t.ppf(1-alpha/2, dof)
        
        B1_upper = B1 + t * sB1
        B1_lower = B1 - t * sB1
        
        B0_upper = y.mean() - B1_upper*x.mean()
        B0_lower = y.mean() - B1_lower*x.mean()
        
        p_x = np.linspace(x.min(),x.max(),100)
        
        p_y = B0 + B1*p_x
        
        sst_x = np.sum((x - np.mean(x))**2)
        
        s = sB1 * np.sqrt(sst_x)
        
        sigma_ep = np.sqrt( s**2 * (1 + 1/n + ( ( n*(p_x-x.mean())**2 ) / ( n*np.sum(x**2) - np.sum(x)**2 ) ) ) )
        
        n_p = len(p_x)
        
        dof = n_p - 2
        
        t_p = stats.t.ppf(1-alpha/2, dof)
        
        p_y_lower = p_y - t_p * sigma_ep
        p_y_upper = p_y + t_p * sigma_ep

        fig, ax = plt.subplots(figsize=(12,8))

        ax.scatter(x, y, c='k', label='Original Data')
        
        ax.plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
        
        plt.axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
        
        ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-r', label='Least Squares Linear Regression Model')
        
        ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
        ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
        
        ax.plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
        ax.plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')
        
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
        
        
        ax.set_xlabel('Date')
        ax.set_ylabel('DO [mg/L]')
        
        ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        labels = [datetime.date(1999,1,1), datetime.date(2000,1,1), datetime.date(2001,1,1), datetime.date(2002,1,1), datetime.date(2003,1,1),
              datetime.date(2004,1,1), datetime.date(2005,1,1), 
              datetime.date(2006,1,1), datetime.date(2007,1,1), 
              datetime.date(2008,1,1),  datetime.date(2009,1,1), 
              datetime.date(2010,1,1),  datetime.date(2011,1,1), 
              datetime.date(2012,1,1),  datetime.date(2013,1,1), 
              datetime.date(2014,1,1),  datetime.date(2015,1,1), 
              datetime.date(2016,1,1), datetime.date(2017,1,1), 
              datetime.date(2018,1,1), datetime.date(2019,1,1), 
              datetime.date(2020,1,1)]

        new_labels = [datetime.date.toordinal(item) for item in labels]

        ax.set_xticks(new_labels)


        ax.set_xticklabels(['1999', '2000','2001','2002','2003','2004','2005','2006','2007',
                    '2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
                    '2020'], rotation=0,
                           fontdict={'horizontalalignment':'center'})
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax.set_ylim(0,15)
        
        ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r**2,2)), horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes)

        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/reg_trend_'+ key + '_' + depth + '_DO.png')

        r_dict[key]['all_year'][depth] = r
        
    
    # for season in ['winter','spring','summer','fall']:
        
    #     r_dict[key][season] = dict()
        
    #     stat_df = full_avgs_dict[key][full_avgs_dict[key]['season'] == season].copy()
   
    #     x_temp = stat_df['date_ordinal']
    #     y_temp = stat_df['DO_mg_L'].to_numpy()
        
    #     mask = ~np.isnan(x_temp) & ~np.isnan(y_temp)
        
    #     x = x_temp[mask]
        
    #     y = y_temp[mask]

        
    #     B1, B0, r, p, sB1 = stats.linregress(x, y)  
        
    #     alpha = 0.05

    #     n = len(x)
        
    #     dof = n - 2
        
    #     t = stats.t.ppf(1-alpha/2, dof)
        
    #     B1_upper = B1 + t * sB1
    #     B1_lower = B1 - t * sB1
        
    #     B0_upper = y.mean() - B1_upper*x.mean()
    #     B0_lower = y.mean() - B1_lower*x.mean()
         
    #     p_x = np.linspace(x.min(),x.max(),100)
        
    #     p_y = B0 + B1*p_x
        
    #     sst_x = np.sum((x - np.mean(x))**2)
        
    #     s = sB1 * np.sqrt(sst_x)
        
    #     sigma_ep = np.sqrt( s**2 * (1 + 1/n + ( ( n*(p_x-x.mean())**2 ) / ( n*np.sum(x**2) - np.sum(x)**2 ) ) ) )
        
    #     n_p = len(p_x)
        
    #     dof = n_p - 2
        
    #     t_p = stats.t.ppf(1-alpha/2, dof)
        
    #     p_y_lower = p_y - t_p * sigma_ep
    #     p_y_upper = p_y + t_p * sigma_ep
        

    #     fig, ax = plt.subplots(figsize=(16,6))

    #     ax.scatter(x, y, c='k', label='Original Data')
        
    #     ax.plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
        
    #     plt.axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
        
    #     ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-r', label='Least Squares Linear Regression Model')
        
    #     ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
    #     ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
        
    #     ax.plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
    #     ax.plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')
        
    #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
        
    #     ax.set_xlabel('Date')
    #     ax.set_ylabel('DO [mg/L]')
        
    #     ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
    #     labels = [datetime.date(1999,1,1), datetime.date(2000,1,1), datetime.date(2001,1,1), datetime.date(2002,1,1), datetime.date(2003,1,1),
    #           datetime.date(2004,1,1), datetime.date(2005,1,1), 
    #           datetime.date(2006,1,1), datetime.date(2007,1,1), 
    #           datetime.date(2008,1,1),  datetime.date(2009,1,1), 
    #           datetime.date(2010,1,1),  datetime.date(2011,1,1), 
    #           datetime.date(2012,1,1),  datetime.date(2013,1,1), 
    #           datetime.date(2014,1,1),  datetime.date(2015,1,1), 
    #           datetime.date(2016,1,1), datetime.date(2017,1,1), 
    #           datetime.date(2018,1,1), datetime.date(2019,1,1), 
    #           datetime.date(2020,1,1)]

    #     new_labels = [datetime.date.toordinal(item) for item in labels]

    #     ax.set_xticks(new_labels)


    #     ax.set_xticklabels(['1999', '2000','2001','2002','2003','2004','2005','2006','2007',
    #                 '2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
    #                 '2020'], rotation=0,
    #                        fontdict={'horizontalalignment':'center'})
        
    #     plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
    #     ax.set_ylim(0,15)
        
    #     ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r**2,2)), horizontalalignment='center',
    #      verticalalignment='center', transform=ax.transAxes)

    #     fig.tight_layout()
        
    #     plt.savefig('/Users/dakotamascarenas/Desktop/pltz/reg_trend_'+ key + '_' + season + '_DO.png')

    #     r_dict[key][season]['full_depth'] = r

    
    #     for depth in ['shallow','deep']:
            
    #         stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == depth) & (depth_avgs_dict[key]['season'] == season)].copy()

            
    #         x_temp = stat_df['date_ordinal']

    #         y_temp = stat_df['DO_mg_L'].to_numpy()
            
    #         mask = ~np.isnan(x_temp) & ~np.isnan(y_temp)
            
    #         x = x_temp[mask]
            
    #         y = y_temp[mask]

            
    #         B1, B0, r, p, sB1 = stats.linregress(x, y)

    #         alpha = 0.05

    #         n = len(x)
            
    #         dof = n - 2
            
    #         t = stats.t.ppf(1-alpha/2, dof)
            
    #         B1_upper = B1 + t * sB1
    #         B1_lower = B1 - t * sB1
            
    #         B0_upper = y.mean() - B1_upper*x.mean()
    #         B0_lower = y.mean() - B1_lower*x.mean()
            
    #         p_x = np.linspace(x.min(),x.max(),100)
            
    #         p_y = B0 + B1*p_x
            
    #         sst_x = np.sum((x - np.mean(x))**2)
            
    #         s = sB1 * np.sqrt(sst_x)
            
    #         sigma_ep = np.sqrt( s**2 * (1 + 1/n + ( ( n*(p_x-x.mean())**2 ) / ( n*np.sum(x**2) - np.sum(x)**2 ) ) ) )
            
    #         n_p = len(p_x)
            
    #         dof = n_p - 2
            
    #         t_p = stats.t.ppf(1-alpha/2, dof)
            
    #         p_y_lower = p_y - t_p * sigma_ep
    #         p_y_upper = p_y + t_p * sigma_ep

    #         fig, ax = plt.subplots(figsize=(16,6))

    #         ax.scatter(x, y, c='k', label='Original Data')
            
    #         ax.plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
            
    #         plt.axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
            
    #         ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-r', label='Least Squares Linear Regression Model')
            
    #         ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
    #         ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            
    #         ax.plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
    #         ax.plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')
            
    #         plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
            
    #         ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
            
    #         ax.set_xlabel('Date')
    #         ax.set_ylabel('DO [mg/L]')
            
    #         #ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
    #         labels = [datetime.date(1999,1,1), datetime.date(2000,1,1), datetime.date(2001,1,1), datetime.date(2002,1,1), datetime.date(2003,1,1),
    #               datetime.date(2004,1,1), datetime.date(2005,1,1), 
    #               datetime.date(2006,1,1), datetime.date(2007,1,1), 
    #               datetime.date(2008,1,1),  datetime.date(2009,1,1), 
    #               datetime.date(2010,1,1),  datetime.date(2011,1,1), 
    #               datetime.date(2012,1,1),  datetime.date(2013,1,1), 
    #               datetime.date(2014,1,1),  datetime.date(2015,1,1), 
    #               datetime.date(2016,1,1), datetime.date(2017,1,1), 
    #               datetime.date(2018,1,1), datetime.date(2019,1,1), 
    #               datetime.date(2020,1,1)]

    #         new_labels = [datetime.date.toordinal(item) for item in labels]

    #         ax.set_xticks(new_labels)


    #         ax.set_xticklabels(['1999', '2000','2001','2002','2003','2004','2005','2006','2007',
    #                     '2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
    #                     '2020'], rotation=0,
    #                            fontdict={'horizontalalignment':'center'})
            
    #         plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
    #         ax.set_ylim(0,15)
            
    #         ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r**2,2)), horizontalalignment='center',
    #          verticalalignment='center', transform=ax.transAxes)

    #         fig.tight_layout()
            
    #         plt.savefig('/Users/dakotamascarenas/Desktop/pltz/reg_trend_'+ key + '_' + season + '_' + depth + '_DO.png')

    #         r_dict[key][season][depth] = r

# %%

# okay now onto trying to do a poly fit with r

r_poly_dict = dict()

for key in odf_dict.keys():
    
    stat_df = full_avgs_dict[key].copy()
    
    r_poly_dict[key] = dict()
    
    r_poly_dict[key]['all_year'] = dict() 

    x_temp = stat_df['date_ordinal']
    y_temp = stat_df['DO_mg_L'].to_numpy()
    
    mask = ~np.isnan(x_temp) & ~np.isnan(y_temp)
    
    x = x_temp[mask]
    
    y = y_temp[mask]

    
    coeffs = np.polyfit(x,y,3)
    
    p = np.poly1d(coeffs)
    
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    r_2 = ssreg / sstot
    
    # alpha = 0.05

    # n = len(x)
    
    # dof = n - 2
    
    # t = stats.t.ppf(1-alpha/2, dof)
    
    # B1_upper = B1 + t * sB1
    # B1_lower = B1 - t * sB1
    
    # B0_upper = y.mean() - B1_upper*x.mean()
    # B0_lower = y.mean() - B1_lower*x.mean()
    
    # p_x = np.linspace(x.min(),x.max(),100)
    
    # p_y = B0 + B1*p_x
    
    # sst_x = np.sum((x - np.mean(x))**2)
    
    # s = sB1 * np.sqrt(sst_x)
    
    # sigma_ep = np.sqrt( s**2 * (1 + 1/n + ( ( n*(p_x-x.mean())**2 ) / ( n*np.sum(x**2) - np.sum(x)**2 ) ) ) )
    
    # n_p = len(p_x)
    
    # dof = n_p - 2
    
    # t_p = stats.t.ppf(1-alpha/2, dof)
    
    # p_y_lower = p_y - t_p * sigma_ep
    # p_y_upper = p_y + t_p * sigma_ep

    fig, ax = plt.subplots(figsize=(9,6))

    ax.scatter(x, y, c='k', label='Original Data')
    
    #ax.plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
    
    #plt.axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
    
    #ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-r', label='Least Squares Linear Regression Model')
    
    # ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
    # ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
    
    # ax.plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
    # ax.plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')
    
    ax.plot(x, p(x), '-', label='second-order polyfit')
    
    ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r_2,2)), horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    
    ax.set_xlabel('Date Ordinal')
    ax.set_ylabel('DO [mg/L]')
    
    #ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia

    
    plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    #ax.set_ylim(bottom = 0)
    


    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/polyfit_'+ key + '_DO.png')
    
    r_poly_dict[key]['all_year']['full_depth'] = r_2

##### FILL IN REST

# %%

### LIN REG with TEMP, NO3, strat***

# use strat = delta deep - shallow

top_sal = avgs_df[avgs_df['depth_bool'] == 'shallow']['SA'].to_numpy()

bot_sal = avgs_df[avgs_df['depth_bool'] == 'deep']['SA'].to_numpy()

sal_diff = bot_sal - top_sal

avgs_df['sal_diff'] = np.nan

avgs_df.loc[avgs_df['depth_bool'] == 'full_depth', 'sal_diff'] = sal_diff

avgs_df.loc[avgs_df['depth_bool'] == 'shallow', 'sal_diff'] = sal_diff

avgs_df.loc[avgs_df['depth_bool'] == 'deep', 'sal_diff'] = sal_diff

# %%


for key in odf_dict.keys():
    
    for depth in ['shallow','deep']:
        
        for season in ['winter','spring','summer','fall']:
    
            stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == depth) & (depth_avgs_dict[key]['season'] == season)].copy()
        
            x_temp =  stat_df['CT'].to_numpy()
            y_temp = stat_df['DO_mg_L'].to_numpy()
            
            mask = ~np.isnan(x_temp) & ~np.isnan(y_temp)
            
            x = x_temp[mask]
            
            y = y_temp[mask]
        
            B1, B0, r, p, sB1 = stats.linregress(x, y)
            
            alpha = 0.05
        
            n = len(x)
            
            dof = n - 2
            
            t = stats.t.ppf(1-alpha/2, dof)
            
            B1_upper = B1 + t * sB1
            B1_lower = B1 - t * sB1
            
            B0_upper = y.mean() - B1_upper*x.mean()
            B0_lower = y.mean() - B1_lower*x.mean()
            
            p_x = np.linspace(x.min(),x.max(),100)
            
            p_y = B0 + B1*p_x
            
            sst_x = np.sum((x - np.mean(x))**2)
            
            s = sB1 * np.sqrt(sst_x)
            
            sigma_ep = np.sqrt( s**2 * (1 + 1/n + ( ( n*(p_x-x.mean())**2 ) / ( n*np.sum(x**2) - np.sum(x)**2 ) ) ) )
            
            n_p = len(p_x)
            
            dof = n_p - 2
            
            t_p = stats.t.ppf(1-alpha/2, dof)
            
            p_y_lower = p_y - t_p * sigma_ep
            p_y_upper = p_y + t_p * sigma_ep
        
            fig, ax = plt.subplots(figsize=(9,6))
        
            ax.scatter(x, y, c='k', label='Original Data')
            
            #ax.plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
            
            #plt.axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
            
            ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-r', label='Least Squares Linear Regression Model')
            
            # ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            # ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            
            ax.plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
            ax.plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')
            
            #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
            
            ax.set_xlabel('CT [deg C]')
            ax.set_ylabel('DO [mg/L]')
            
            r_2 = r**2
            
            # ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        
            
            plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            # ax.set_ylim(bottom = 0)
            
            ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r_2,2)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
        
            fig.tight_layout()
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/temp_trend_'+ key + '_'  + season + '_' + depth +'_DO.png')
    
# %%

for key in odf_dict.keys():
    
    for depth in ['shallow','deep']:
        
        for season in ['winter','spring','summer','fall']:
    
            stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == depth) & (depth_avgs_dict[key]['season'] == season)].copy()
        
    
            x_temp = stat_df['NO3 (uM)']
            y_temp = stat_df['DO_mg_L'].to_numpy()
            
            mask = ~np.isnan(x_temp) & ~np.isnan(y_temp)
            
            x = x_temp[mask]
            
            y = y_temp[mask]
        
            B1, B0, r, p, sB1 = stats.linregress(x, y)
            
            alpha = 0.05
        
            n = len(x)
            
            dof = n - 2
            
            t = stats.t.ppf(1-alpha/2, dof)
            
            B1_upper = B1 + t * sB1
            B1_lower = B1 - t * sB1
            
            B0_upper = y.mean() - B1_upper*x.mean()
            B0_lower = y.mean() - B1_lower*x.mean()
            
            p_x = np.linspace(x.min(),x.max(),100)
            
            p_y = B0 + B1*p_x
            
            sst_x = np.sum((x - np.mean(x))**2)
            
            s = sB1 * np.sqrt(sst_x)
            
            sigma_ep = np.sqrt( s**2 * (1 + 1/n + ( ( n*(p_x-x.mean())**2 ) / ( n*np.sum(x**2) - np.sum(x)**2 ) ) ) )
            
            n_p = len(p_x)
            
            dof = n_p - 2
            
            t_p = stats.t.ppf(1-alpha/2, dof)
            
            p_y_lower = p_y - t_p * sigma_ep
            p_y_upper = p_y + t_p * sigma_ep
        
            fig, ax = plt.subplots(figsize=(9,6))
        
            ax.scatter(x, y, c='k', label='Original Data')
            
            # ax.plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
            
            # plt.axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
            
            ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-r', label='Least Squares Linear Regression Model')
            
            # ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            # ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            
            ax.plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
            ax.plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')
            
            #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
            
            ax.set_xlabel('NO3 [uM]')
            ax.set_ylabel('DO [mg/L]')
            
            r_2 = r**2
            
            # ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        
            
            plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            # ax.set_ylim(bottom = 0)
            
            ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r_2,2)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
        
            fig.tight_layout()
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/NO3_trend_'+ key + '_' + season + '_' + depth + '_DO.png')

# %%

for key in odf_dict.keys():
    
    stat_df = full_avgs_dict[key].copy()

    x_temp = avgs_df[(avgs_df['segment'] == key) & (avgs_df['depth_bool'] == 'full_depth')]['sal_diff'].to_numpy()
    y_temp = stat_df['DO_mg_L'].to_numpy()
    
    mask = ~np.isnan(x_temp) & ~np.isnan(y_temp)
    
    x = x_temp[mask]
    
    y = y_temp[mask]

    B1, B0, r, p, sB1 = stats.linregress(x, y)
    
    alpha = 0.05

    n = len(x)
    
    dof = n - 2
    
    t = stats.t.ppf(1-alpha/2, dof)
    
    B1_upper = B1 + t * sB1
    B1_lower = B1 - t * sB1
    
    B0_upper = y.mean() - B1_upper*x.mean()
    B0_lower = y.mean() - B1_lower*x.mean()
    
    p_x = np.linspace(x.min(),x.max(),100)
    
    p_y = B0 + B1*p_x
    
    sst_x = np.sum((x - np.mean(x))**2)
    
    s = sB1 * np.sqrt(sst_x)
    
    sigma_ep = np.sqrt( s**2 * (1 + 1/n + ( ( n*(p_x-x.mean())**2 ) / ( n*np.sum(x**2) - np.sum(x)**2 ) ) ) )
    
    n_p = len(p_x)
    
    dof = n_p - 2
    
    t_p = stats.t.ppf(1-alpha/2, dof)
    
    p_y_lower = p_y - t_p * sigma_ep
    p_y_upper = p_y + t_p * sigma_ep

    fig, ax = plt.subplots(figsize=(9,6))

    ax.scatter(x, y, c='k', label='Original Data')
    
    ax.plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
    
    plt.axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
    
    ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-r', label='Least Squares Linear Regression Model')
    
    ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
    ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
    
    ax.plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
    ax.plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    
    ax.set_xlabel('NO3 [uM]')
    ax.set_ylabel('DO [mg/L]')
    
    r_2 = r**2
    
    # ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia

    
    plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    # ax.set_ylim(bottom = 0)
    
    ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r_2,2)), horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes)

    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/saldiff_trend_'+ key + '_DO.png')


