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

from scipy.interpolate import interp1d


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

for key in ['ps']:
    
    for season in ['all_year']:
        
        for depth in ['deep']:
            
            plot_df = depth_avgs_dict[key][depth_avgs_dict[key]['depth_bool'] == depth].copy()

            fig, ax = plt.subplots(figsize=(10,5))
            
            plt.scatter(plot_df['datetime'], plot_df['DO_mg_L'], color = 'k')
            
            ax.set_xlabel('Date')
            
            ax.set_ylabel('DO [mg/L]')
            
            ax.set_ylim(0,15)
            
            ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4, label = 'hypoxia [DO <2 mg/L]') #fill hypoxia
            
            plt.grid(color='lightgray', alpha=0.5, linestyle='--')
            
            plt.legend(loc='upper right')
            
            plt.title('Puget Sound [DO] Time Series')
            
            fig.tight_layout()
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + key+ '_month_avg_DO_.png', dpi=500)
            
# %%

for key in ['ps']:
    
    for season in ['all_year']:
        
        for depth in ['deep']:
            
            plot_df = depth_avgs_dict[key][depth_avgs_dict[key]['depth_bool'] == depth].copy()
            

            fig, ax = plt.subplots(figsize=(10,5))
            
            plt.scatter(plot_df['datetime'], plot_df['DO_mg_L'], color = 'k')
            
            ax.set_xlabel('Date')
            
            ax.set_ylabel('DO [mg/L]')
            
            ax.set_ylim(0,15)
            
            ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4, label = 'hypoxia [DO <2 mg/L]') #fill hypoxia
            
            plt.grid(color='lightgray', alpha=0.5, linestyle='--')
            
            plt.legend(loc='upper right')
            
            plt.title('Puget Sound [DO] Time Series')
            
            fig.tight_layout()
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + key+ '_month_avg_DO_.png', dpi=500)


# %%

depth_avgs_df = pd.concat(depth_avgs_dict.values(), ignore_index=True)

deep_avgs_df  = depth_avgs_df[depth_avgs_df['depth_bool'] == 'deep']

deep_avgs_df_temp = (deep_avgs_df.copy()
                .assign(season=(lambda x: 'all_year'))
                )

deep_avgs_df = pd.concat([deep_avgs_df, deep_avgs_df_temp], ignore_index=True)

# %%


plot_df = deep_avgs_df[deep_avgs_df['season'] == 'all_year']

g = sns.relplot(data=plot_df, x='datetime', y='DO_mg_L', col='segment', height=5, aspect=0.7, color='k', style='segment') #, title='Full Year [DO] Time Series By Region')

c=0

basins = ['All Puget Sound', 'Main Basin', 'Hood Canal', 'Whidbey Basin', 'South Sound']

for ax in g.axes.flatten():
    
    ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4, label = 'hypoxia [DO <2 mg/L]') #fill hypoxia
    
    ax.grid(color='lightgray', alpha=0.5, linestyle='--') 
    
    ax.set_title(basins[c])
    
    ax.set_ylim(0,15)
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('DO [mg/L]')
    
    c+=1
        
g._legend.remove()



plt.savefig('/Users/dakotamascarenas/Desktop/pltz/all_segs_avg_DO_.png', dpi=500)


# %%

winter_color = '#1126a5'
spring_color = '#a4d13a'
summer_color = '#fd3f41'
fall_color = '#680e03'

# %%

plot_df = deep_avgs_df[deep_avgs_df['segment'] == 'ps']

g = sns.relplot(data=plot_df, x='datetime', y='DO_mg_L', col='season', col_order=['all_year','winter','spring','summer','fall'], height=5, aspect=0.7, hue='season', hue_order=['all_year','winter','spring','summer','fall'], palette=['k', winter_color, spring_color, summer_color, fall_color])

c=0

seasons = ['All Seasons', 'Winter', 'Spring', 'Summer', 'Fall']

for ax in g.axes.flatten():
    
    ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4, label = 'hypoxia [DO <2 mg/L]') #fill hypoxia
    
    ax.grid(color='lightgray', alpha=0.5, linestyle='--') 
    
    ax.set_title(seasons[c])
    
    ax.set_ylim(0,15)
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('DO [mg/L]')
    
    c+=1
    
#g.fig.suptitle('Puget Sound [DO] Time Series By Season')

g._legend.remove()


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/all_seasons_ps_avg_DO_.png', dpi=500)

# %%

alpha = 0.05
conf = 1 - alpha

conf_upper = 1-alpha/2
conf_lower = alpha/2

z_alpha_upper = stats.norm.ppf(conf_upper)
z_alpha_lower = stats.norm.ppf(conf_lower)

red_theme_color = '#ba4a4f'

blue_theme_color = '#5c75d0'



mu_0 = 0

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



        z = np.linspace(-4, 4, num=160) * sigma_prime
        
        plt.figure(figsize=(10,6))
        # Plot the z-distribution here
        plt.plot(z, stats.norm.pdf(z, 0, sigma_prime), color = blue_theme_color, label='Null PDF:\n($\overline{X}_{2010-On}-\overline{X}_{Pre-2010}$)=0')
        
        plt.axvline(z_alpha_upper*sigma_prime, color='black', linestyle='-', label='$z_{a}$', linewidth=0.5)
        plt.axvline(z_alpha_lower*sigma_prime, color='black', linestyle='-', linewidth=0.5) # , label='$z_{a}$')
        shade_upper = np.linspace(z_alpha_upper*sigma_prime, np.max(z), 10)
        shade_lower = np.linspace(np.min(z), z_alpha_lower*sigma_prime, 10)
        
        plt.fill_between(shade_upper, stats.norm.pdf(shade_upper, 0, sigma_prime) ,  color='k', alpha=0.5, label='rejection region for\nalpha={}'.format(np.round(1-conf,2)))
        plt.fill_between(shade_lower, stats.norm.pdf(shade_lower, 0, sigma_prime) ,  color='k', alpha=0.5) #, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
        
        plt.axvline(z_score*sigma_prime, color=red_theme_color, linestyle='-', linewidth=3, label='z-test\n(for observed mean diff.)')
        plt.xlabel('($\overline{X}_{2010-On} - \overline{X}_{Pre-2010}$)')
        plt.ylabel('Probability Density Function')
        # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.ylim(bottom = 0)
        plt.legend(loc='upper right');
        
        plt.title('Puget Sound [DO] - All Seasons')
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/z_test_pre_post_2010_'+ key + '_' + depth + '_DO_present_.png', dpi=500)


# %%

for key in ['ps']:
    
    for depth in ['deep']:
        
        stat_df = depth_avgs_dict[key][depth_avgs_dict[key]['depth_bool'] == depth].copy()

        pre = stat_df[stat_df['period_label'] == 'pre']
        
        post = stat_df[stat_df['period_label'] == 'post']

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,6), sharex=True, sharey=True)
        
        ax1.hist(pre['DO_mg_L'], bins=10, color = blue_theme_color)
        #ax1.set_xlim((1e4,1.4e5))
        ax1.set_xlabel('DO [mg/L]')
        ax1.set_title('Pre-2010')
        
        ax2.hist(post['DO_mg_L'], bins=10, color = red_theme_color)
        #ax2.set_xlim((1e4,1.4e5))
        ax2.set_xlabel('DO [mg/L]')
        ax2.set_title('2010-On');
        
        plt.suptitle('Puget Sound [DO] - All Seasons')
        
        # ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        # ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        plt.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/z_test_pre_post_2010_ps_histos_.png', dpi=500)

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
        
        ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], color=red_theme_color, linewidth=2, label='least-squares linear regression')
        
        # ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , ':r', label='slope confidence limit')
        # ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , ':r')
        
        ax.plot(p_x, p_y_upper, ':', color=blue_theme_color, label='prediction interval')
        ax.plot(p_x, p_y_lower, ':', color=blue_theme_color)
        
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
        
        plt.title('Puget Sound [DO] - All Seasons')
                
        ax.set_ylim(0,10)
        
        ax.text(0.9, 0.9, '$r^2 = {}$'.format(np.round(r**2,2)), horizontalalignment='center', fontsize=14,
         verticalalignment='center', transform=ax.transAxes)

        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/reg_trend_'+ key + '_' + depth + '_post_2010_DO_present_.png', dpi=500)


# %%

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
        
        ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-', color=red_theme_color, label='least-squares linear regression')
        
        # ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , ':r', label='slope confidence limit')
        # ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , ':r')
        
        ax.plot(p_x, p_y_upper, ':', color=blue_theme_color, label='prediction interval')
        ax.plot(p_x, p_y_lower, ':', color=blue_theme_color)
        
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
        
        plt.title('Puget Sound [DO] - All Seasons')
                
        ax.set_ylim(0,10)
        
        ax.text(0.9, 0.9, '$r^2 = {}$'.format(np.round(r**2,2)), horizontalalignment='center', fontsize=14,
         verticalalignment='center', transform=ax.transAxes)

        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/reg_trend_'+ key + '_' + depth + '_DO_present_.png', dpi=500)

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

reject_null_dict = {}

Z_dict = {}

for key in odf_dict.keys():
    
    reject_null_dict[key] = {}
    
    Z_dict[key] = {}
    
    stat_df = depth_avgs_dict[key][depth_avgs_dict[key]['depth_bool'] == 'deep'].copy()
    
    reject_null, p_value, Z = mann_kendall(stat_df['DO_mg_L'].values, alpha)

    reject_null_dict[key]['all_year'] = reject_null
    
    Z_dict[key]['all_year'] = Z
    
            
    for season in ['winter','spring','summer','fall']:
        
            
            stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == 'deep') & (depth_avgs_dict[key]['season'] == season)].copy()

            reject_null, p_value, Z = mann_kendall(stat_df['DO_mg_L'].values, alpha)
            
            reject_null_dict[key][season] = reject_null
            
            Z_dict[key][season] = Z
            
# %%

for key in ['ps']:
    
    for depth in ['deep']:
        
        for season in ['fall']:
    
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
            
            ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-', color=red_theme_color, label='Least Squares Linear Regression Model')
            
            # ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            # ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            
            ax.plot(p_x, p_y_upper, ':', color=blue_theme_color, label='Upper Y prediction interval (95%)')
            ax.plot(p_x, p_y_lower, ':', color=blue_theme_color, label='Lower Y prediction interval (95%)')
            
            #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
            
            ax.set_xlabel('CT [deg C]')
            ax.set_ylabel('DO [mg/L]')
            
            r_2 = r**2
            
            # ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        
            
            plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            # ax.set_ylim(bottom = 0)
            
            ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r_2,2)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes,fontsize=14)
            
            plt.title('Puget Sound [DO] - Fall')

        
            fig.tight_layout()
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/temp_trend_'+ key + '_'  + season + '_' + depth +'_DO_.png', dpi=500)
            
# %%


for key in ['ps']:
    
    for depth in ['deep']:
        
        for season in ['fall']:
    
            stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == depth)].copy() # & (depth_avgs_dict[key]['season'] == season)].copy()
            
            stat_df = stat_df.groupby(['year','season']).mean().reset_index()
            
            x_temp =  stat_df[stat_df['season'] == 'spring']['NO3 (uM)'].to_numpy()
            y_temp = stat_df[stat_df['season'] == 'fall']['DO_mg_L'].to_numpy()
            
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
            
            ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-', color=red_theme_color, label='Least Squares Linear Regression Model')
            
            # ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            # ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            
            ax.plot(p_x, p_y_upper, ':', color=blue_theme_color, label='Upper Y prediction interval (95%)')
            ax.plot(p_x, p_y_lower, ':', color=blue_theme_color, label='Lower Y prediction interval (95%)')
            
            #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
            
            ax.set_xlabel('Spring NO3 [uM]')
            ax.set_ylabel('Fall DO [mg/L]')
            
            r_2 = r**2
            
            # ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        
            
            plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            # ax.set_ylim(bottom = 0)
            
            ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r_2,2)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes,fontsize=14)
            
            plt.title('Puget Sound [DO] - Fall')

        
            fig.tight_layout()
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/temp_trend_'+ key + '_'  + season + '_' + depth +'_NO3_.png', dpi=500)

            # quantiles = np.linspace(0,1,100)
            
            # # This is our empirical cdf of the Slide Canyon data, which also includes values down to 0 and up to 1.
            # NO3_spring_ordered = stats.mstats.mquantiles(x, quantiles)
            
            # # This is our empirical cdf of the Blue Canyon data, which also includes values down to 0 and up to 1.
            # DO_fall_ordered = stats.mstats.mquantiles(y, quantiles)
            
            # # Create our interpolation function for looking up a quantile given a value of SWE at Slide Canyon
            # f_NO3_spring = interp1d(NO3_spring_ordered, quantiles)
            # # Create our interpolation function for looking up SWE at Blue Canyon given a quantile
            # g_DO_fall = interp1d(quantiles, DO_fall_ordered)
            
            # plt.figure(figsize=(10,10))

            # # We can also create these by picking arbitrary quantile values, then using the scipy.stats.mstats.mquantiles function
            # quantiles = np.linspace(0,1,100) # 100 quantile values linearly spaced between 0 and 1
            # plt.plot(NO3_spring_ordered, quantiles, 
            #          'b.', label='Spring NO3 [uM]', alpha=0.7)
            # plt.plot(DO_fall_ordered, quantiles, 
            #          'r.', label='Fall DO [uM]', alpha=0.7)


# %%



for key in ['ps']:
    
    for depth in ['deep']:
        
        for season in ['fall']:
    
            stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == depth)].copy() # & (depth_avgs_dict[key]['season'] == season)].copy()
        
            x_temp =  stat_df[stat_df['season'] == 'spring']['NO3 (uM)'].to_numpy()
            y_temp = stat_df[stat_df['season'] == 'fall']['DO_mg_L'].to_numpy()
            
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
            
            ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-', color=red_theme_color, label='Least Squares Linear Regression Model')
            
            # ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            # ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
            
            ax.plot(p_x, p_y_upper, ':', color=blue_theme_color, label='Upper Y prediction interval (95%)')
            ax.plot(p_x, p_y_lower, ':', color=blue_theme_color, label='Lower Y prediction interval (95%)')
            
            #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
            
            ax.set_xlabel('NO3 [uM]')
            ax.set_ylabel('DO [mg/L]')
            
            r_2 = r**2
            
            # ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
        
            
            plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            # ax.set_ylim(bottom = 0)
            
            ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r_2,2)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes,fontsize=14)
            
            plt.title('Puget Sound [DO] - Fall')

        
            fig.tight_layout()
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/temp_trend_'+ key + '_'  + season + '_' + depth +'_NO3_.png', dpi=500)
            
            
# %%

enso = pd.read_csv('/Users/dakotamascarenas/Desktop/anomaly.txt', delimiter=r"\s+", lineterminator='\n')

enso = pd.melt(enso, id_vars='YEAR', value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT','NOV', 'DEC'], var_name='month', value_name='anomaly')

enso['day'] = 28

enso = enso.assign(timestring= enso['YEAR'].astype(str) + '-' + enso['month'] + '-' + enso['day'].astype(str),
    datetime=(lambda x: pd.to_datetime(x['timestring'], format='%Y-%b-%d')))


# %%

for key in ['ps']:
    
    for depth in ['deep']:
            
            plot_df = depth_avgs_dict[key][depth_avgs_dict[key]['depth_bool'] == depth].copy()
            

            fig, ax = plt.subplots(figsize=(10,6))
                        
            ax.scatter(plot_df['datetime'], plot_df['DO_mg_L'], color = 'k')
            
            ax2 = ax.twinx()
            
            plot_df2 = enso[enso['YEAR'] >=1999].sort_values(by=['datetime'])
                        
            ax2.plot(plot_df2['datetime'], plot_df2['anomaly'], color=red_theme_color)
            
            ax2.set_ylabel('SOI', color=red_theme_color, rotation=270)
            
            ax2.tick_params(axis='y', labelcolor=red_theme_color)

            
            
            ax.set_xlabel('Date')
            
            ax.set_ylabel('DO [mg/L]')
            
            #ax.set_ylim(0,15)
            
            #ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4, label = 'hypoxia [DO <2 mg/L]') #fill hypoxia
            
            plt.grid(color='lightgray', alpha=0.5, linestyle='--')
            
            #ax1.legend(['DO [mg/L]', 'SOI Index'])
            
            plt.title('Puget Sound [DO] Time Series with Southern Oscillation Indicator')
            
            fig.tight_layout()
            
            #ax3.get_legend().remove()
            
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + key+ '_month_avg_DO_ENSO.png', dpi=500)
  
# %%


            
for key in ['ps']:
    
    for depth in ['deep']:
            
        stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == depth)].copy() # & (depth_avgs_dict[key]['season'] == season)].copy()
    
        x_temp =  enso[enso['YEAR'] >=1999]['anomaly'].to_numpy()
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
        
        ax.plot([x.min(), x.max()], [B0 + B1*x.min(), B0 + B1*x.max()], '-', color=red_theme_color, label='Least Squares Linear Regression Model')
        
        # ax.plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
        # ax.plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
        
        ax.plot(p_x, p_y_upper, ':', color=blue_theme_color, label='Upper Y prediction interval (95%)')
        ax.plot(p_x, p_y_lower, ':', color=blue_theme_color, label='Lower Y prediction interval (95%)')
        
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
        
        ax.set_xlabel('ENSO Indicator')
        ax.set_ylabel('DO [mg/L]')
        
        r_2 = r**2
        
        # ax.axhspan(0, 2, facecolor='gray', alpha=0.1, zorder = 4) #fill hypoxia
    
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        # ax.set_ylim(bottom = 0)
        
        ax.text(0.9, 0.9, 'r^2 = {}'.format(np.round(r_2,2)), horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes,fontsize=14)
        
        plt.title('Puget Sound [DO]')

        plt.legend.remove()
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/temp_trend_'+ key + '_' + depth +'_ENSO_.png', dpi=500)

# %%

n_pre_dict = dict()

n_post_dict = dict()

X_pre_dict = dict()

X_post_dict = dict()

s_pre_dict = dict()

s_post_dict = dict()

t_test_dict = dict()

p_value_dict = dict()


for key in odf_dict.keys():
    
    n_pre_dict[key] = dict()

    n_post_dict[key] = dict()

    X_pre_dict[key] = dict()

    X_post_dict[key] = dict()

    s_pre_dict[key] = dict()

    s_post_dict[key] = dict()

    t_test_dict[key] = dict()

    p_value_dict[key] = dict()
    
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
        
        t_score, p_value = stats.ttest_ind_from_stats(X_2, s_2, n_2, X_1, s_1, n_1, equal_var=False, alternative='two-sided')
        
        # sigma_prime = np.sqrt(s_1**2/n_1 + s_2**2/n_2)

        # z_score = (X_2 - X_1 - mu_0)/sigma_prime

        # p_value = 1 - stats.norm.cdf(z_score)
        
        
        n_pre_dict[key]['all_year'] = n_1

        n_post_dict[key]['all_year'] = n_2

        X_pre_dict[key]['all_year'] = X_1

        X_post_dict[key]['all_year'] = X_2

        s_pre_dict[key]['all_year'] = s_1

        s_post_dict[key]['all_year'] = s_2

        t_test_dict[key]['all_year'] = t_score
        
        p_value_dict[key]['all_year'] = p_value

        print(key + 'allseasons: z = ' + str(t_score) + ' , p = ' + str(p_value))


        # z = np.linspace(-4, 4, num=160) * sigma_prime
        
        # plt.figure(figsize=(10,7))
        # # Plot the z-distribution here
        # plt.plot(z, stats.norm.pdf(z, 0, sigma_prime), label='Null PDF: ($\overline{X}_2 - \overline{X}_1$) = 0')
        
        # plt.axvline(z_alpha_upper*sigma_prime, color='black', linestyle='-', label='$z_{a}$')
        # plt.axvline(z_alpha_lower*sigma_prime, color='black', linestyle='-') # , label='$z_{a}$')
        # shade_upper = np.linspace(z_alpha_upper*sigma_prime, np.max(z), 10)
        # shade_lower = np.linspace(np.min(z), z_alpha_lower*sigma_prime, 10)
        
        # plt.fill_between(shade_upper, stats.norm.pdf(shade_upper, 0, sigma_prime) ,  color='k', alpha=0.5, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
        # plt.fill_between(shade_lower, stats.norm.pdf(shade_lower, 0, sigma_prime) ,  color='k', alpha=0.5) #, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
        
        # plt.axvline(z_score*sigma_prime, color='red', linestyle='-', label='z-test')
        # plt.xlabel('($\overline{X}_2 - \overline{X}_1$) [cfs]')
        # plt.ylabel('PDF')
        # # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        # # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # plt.ylim(bottom = 0)
        # plt.legend(loc='upper right');
        
        # plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/z_test_pre_post_2010_'+ key + '_' + depth + '_DO.png')
        
        
        
        
        for season in ['winter', 'spring', 'summer','fall']:
            
            stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == depth) & (depth_avgs_dict[key]['season'] == season)].copy()
    
            n_pre_dict[key][season] = dict()
    
            n_post_dict[key][season] = dict()
    
            X_pre_dict[key][season] = dict()
    
            X_post_dict[key][season] = dict()
    
            s_pre_dict[key][season] = dict()
    
            s_post_dict[key][season] = dict()
    
            t_test_dict[key][season] = dict()
            
            p_value_dict[key][season]= dict()
    
            
            pre = stat_df[stat_df['period_label'] == 'pre']
            
            post = stat_df[stat_df['period_label'] == 'post']
            
            
            n_1 = pre['year'].count()
            n_2 = post['year'].count()
            
    
            X_1 = pre['DO_mg_L'].mean()
            X_2 = post['DO_mg_L'].mean()
            
            s_1 = pre['DO_mg_L'].std(ddof=1)
            s_2 = post['DO_mg_L'].std(ddof=1)
            
            t_score, p_value = stats.ttest_ind_from_stats(X_2, s_2, n_2, X_1, s_1, n_1, equal_var=False, alternative='two-sided')
            
            # sigma_prime = np.sqrt(s_1**2/n_1 + s_2**2/n_2)
    
            # z_score = (X_2 - X_1 - mu_0)/sigma_prime
    
            # p_value = 1 - stats.norm.cdf(z_score)
            
            
            n_pre_dict[key][season] = n_1
    
            n_post_dict[key][season] = n_2
    
            X_pre_dict[key][season] = X_1
    
            X_post_dict[key][season] = X_2
    
            s_pre_dict[key][season]= s_1
    
            s_post_dict[key][season] = s_2
    
            t_test_dict[key][season] = t_score
            
            p_value_dict[key][season] = p_value
    
            print(key + season + ': z = ' + str(t_score) + ' , p = ' + str(p_value))
            
            # z = np.linspace(-4, 4, num=160) * sigma_prime
            
            # plt.figure(figsize=(10,7))
            # # Plot the z-distribution here
            # plt.plot(z, stats.norm.pdf(z, 0, sigma_prime), label='Null PDF: ($\overline{X}_2 - \overline{X}_1$) = 0')
            
            # plt.axvline(z_alpha_upper*sigma_prime, color='black', linestyle='-', label='$z_{a}$')
            # plt.axvline(z_alpha_lower*sigma_prime, color='black', linestyle='-') # , label='$z_{a}$')
            # shade_upper = np.linspace(z_alpha_upper*sigma_prime, np.max(z), 10)
            # shade_lower = np.linspace(np.min(z), z_alpha_lower*sigma_prime, 10)
            
            # plt.fill_between(shade_upper, stats.norm.pdf(shade_upper, 0, sigma_prime) ,  color='k', alpha=0.5, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
            # plt.fill_between(shade_lower, stats.norm.pdf(shade_lower, 0, sigma_prime) ,  color='k', alpha=0.5) #, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
            
            # plt.axvline(z_score*sigma_prime, color='red', linestyle='-', label='z-test')
            # plt.xlabel('($\overline{X}_2 - \overline{X}_1$) [cfs]')
            # plt.ylabel('PDF')
            # #plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            # #plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            # plt.ylim(bottom = 0)
            # plt.legend(loc='upper right');
            
            # plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/z_test_pre_post_2010_'+ key + '_' + season + '_DO.png')
            
            
            # for depth in ['shallow', 'deep']:
                
            #     stat_df = depth_avgs_dict[key][(depth_avgs_dict[key]['depth_bool'] == depth) & (depth_avgs_dict[key]['season'] == season)].copy()
    
            #     pre = stat_df[stat_df['period_label'] == 'pre']
                
            #     post = stat_df[stat_df['period_label'] == 'post']
    
    
            #     n_1 = pre['year'].count()
            #     n_2 = post['year'].count()
                
    
            #     X_1 = pre['DO_mg_L'].mean()
            #     X_2 = post['DO_mg_L'].mean()
                
            #     s_1 = pre['DO_mg_L'].std(ddof=1)
            #     s_2 = post['DO_mg_L'].std(ddof=1)
                
            #     sigma_prime = np.sqrt(s_1**2/n_1 + s_2**2/n_2)
    
    
            #     z_score = (X_2 - X_1 - mu_0)/sigma_prime
    
            #     p_value = 1 - stats.norm.cdf(z_score)
                
                
            #     n_pre_dict[key][season][depth] = n_1
    
            #     n_post_dict[key][season][depth] = n_2
    
            #     X_pre_dict[key][season][depth] = X_1
    
            #     X_post_dict[key][season][depth] = X_2
    
            #     s_pre_dict[key][season][depth] = s_1
    
            #     s_post_dict[key][season][depth] = s_2
    
            #     z_test_dict[key][season][depth] = z_score
                
            #     p_value_dict[key][season][depth] = p_value
    
    
    
            #     z = np.linspace(-4, 4, num=160) * sigma_prime
                
            #     plt.figure(figsize=(10,7))
            #     # Plot the z-distribution here
            #     plt.plot(z, stats.norm.pdf(z, 0, sigma_prime), label='Null PDF: ($\overline{X}_2 - \overline{X}_1$) = 0')
                
            #     plt.axvline(z_alpha_upper*sigma_prime, color='black', linestyle='-', label='$z_{a}$')
            #     plt.axvline(z_alpha_lower*sigma_prime, color='black', linestyle='-') # , label='$z_{a}$')
            #     shade_upper = np.linspace(z_alpha_upper*sigma_prime, np.max(z), 10)
            #     shade_lower = np.linspace(np.min(z), z_alpha_lower*sigma_prime, 10)
                
            #     plt.fill_between(shade_upper, stats.norm.pdf(shade_upper, 0, sigma_prime) ,  color='k', alpha=0.5, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
            #     plt.fill_between(shade_lower, stats.norm.pdf(shade_lower, 0, sigma_prime) ,  color='k', alpha=0.5) #, label='rejection region\nfor alpha={}'.format(np.round(1-conf,2)))
                
            #     plt.axvline(z_score*sigma_prime, color='red', linestyle='-', label='z-test')
            #     plt.xlabel('($\overline{X}_2 - \overline{X}_1$) [cfs]')
            #     plt.ylabel('PDF')
            #     # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            #     # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            #     plt.ylim(bottom = 0)
            #     plt.legend(loc='upper right');
                
            #     plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
                
            #     plt.savefig('/Users/dakotamascarenas/Desktop/pltz/z_test_pre_post_2010_'+ key + '_' + season + '_' + depth + '_DO.png')