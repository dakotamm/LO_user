#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepared for publication: 2026/01/07

Author: Dakota Mascarenas

Plotting code for: "Century-Scale Changes in Dissolved Oxygen, Temperature, and Salinity in Puget Sound" (Mascarenas et al., in review; submitted 2026/01/09 to Estuaries & Coasts)

This script processes data for and plots Figure 6 in corresponding manuscript. Please reach out to the author at dakotamm@uw.edu for any questions.

"""

# import modules
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import figure_functions_PSNME as ffun

plt.rcParams['font.size'] = 13  # base font (ticks, axis labels, legends); default 10

### FIGURE 6

# load pickled data frame for all sites' surface and bottom water (depth-averaged) cast data from user-specified directory
df_directory = '/Users/dakotamascarenas/Desktop/Mascarenas_etal_2026_R1/' #SPECIFY LOCAL DIRECTORY
site_depth_avg_var_DF = pd.read_pickle(df_directory + 'site_depth_avg_var_DF.p')

# apply DO filtering to casts in the bottom 50th percentile of annual seasonal bottom DO values
filter_DO_DF = ffun.filter_DO(site_depth_avg_var_DF)

# calculate Theil-Sen slopes with 95% confidence (alpha=0.05) for temperature, salinity, and DO
alpha = 0.05
slope_DF = ffun.calc_slopes_var(alpha, site_depth_avg_var_DF, filter_DO_DF)

# plot and save to user-specified directory
plot_directory = '/Users/dakotamascarenas/Desktop/pltz/' #SPECIFY SAVE LOCATION
red =     "#e04256" 
blue =     "#4565e8" 
site_list = ['point_jefferson', 'lynch_cove_mid']
palette = {'point_jefferson':red, 'lynch_cove_mid':blue}
site_label_dict = {'point_jefferson':'Point Jefferson (PJ)', 'lynch_cove_mid':'Lynch Cove (LC)'}
var_list = ['CT', 'SA', 'DO_mg_L']
mosaic = [['CT', 'CT'], ['SA', 'SA'], [ 'DO_mg_L', 'DO_mg_L']]
fig, axd = plt.subplot_mosaic(mosaic, figsize=(9,6), layout='constrained', gridspec_kw=dict(wspace=0.1), sharex = True)
for var in var_list:
    ax = axd[var]
    if 'DO' in var:
        label_var = '[DO]'
        ymin = 0
        ymax = 7
        marker = 'o'
        unit = r'[mg/L]'
    elif 'CT' in var:
        label_var = 'Temperature'
        ymin = 8
        ymax = 14
        marker = 'o' #'D'
        unit = r'[$^{\circ}$C]'
    else:
        label_var = 'Salinity'
        ymin = 29
        ymax = 32
        marker = 'o'#'s'
        unit = r'[g/kg]'
    for site in site_list:
        if var == 'DO_mg_L':
            plot_df = filter_DO_DF[(filter_DO_DF['season'] == 'loDO') & (filter_DO_DF['surf_deep'] == 'deep') & (filter_DO_DF['site'] == site)]
        else:
            plot_df = site_depth_avg_var_DF[(site_depth_avg_var_DF['season'] == 'loDO') & (site_depth_avg_var_DF['surf_deep'] == 'deep') & (site_depth_avg_var_DF['site'] == site) & (site_depth_avg_var_DF['var'] == var)]
        sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, color = palette[site], marker=marker)
        if var == 'CT':
            ax.scatter(x=0, y =0, color = palette[site], marker='o', label = site_label_dict[site])
        stat_df = slope_DF[(slope_DF['season'] == 'loDO') & (slope_DF['surf_deep'] == 'deep') & (slope_DF['site'] == site) & (slope_DF['var'] == var)]
        x = plot_df['date_ordinal']
        y = plot_df['val']
        x_plot = plot_df['datetime']
        B0 = stat_df['B0'].iloc[0]
        B1 = stat_df['B1'].iloc[0]
        ax.plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], alpha =0.7, color = palette[site], linewidth = 2)      
        ax.axhline(np.mean([B0 + B1*x.min(), B0 + B1*x.max()]), color = palette[site], linestyle = '--', alpha = 0.5)
        def norm0_1(x):
            return (x - x.min())/ (x.max()-x.min())
        x_norm = 2*norm0_1(x)-1
        res = stats.theilslopes(y, x_norm, alpha=0.05)
        x_vals = np.array([x_plot.min(), x_plot.max()])
        y_upper = res[1] + res[2] * np.array([x_norm.min(), x_norm.max()])
        y_lower = res[1] + res[3] * np.array([x_norm.min(), x_norm.max()])
        ax.fill_between(x_vals, y_lower, y_upper, color=palette[site], alpha=0.2)
    if var == 'DO_mg_L':  
        ax.axhspan(0,2, color = 'lightgray', alpha = 0.5, zorder=-5, label='Hypoxia')
        ax.legend(loc='lower left')
    elif var == 'CT':
        ax.legend(ncol=2, loc='upper left')
    ax.set_ylim(ymin, ymax) 
    ax.set_ylabel(label_var + ' ' + unit)
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    ax.set_xlabel('')
handles_CT, labels_CT = axd['CT'].get_legend_handles_labels()
handles_DO_mg_L, labels_DO_mg_L = axd['DO_mg_L'].get_legend_handles_labels()
handles = handles_CT + handles_DO_mg_L
labels = labels_CT + labels_DO_mg_L
fig.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.01),
    ncol=3
    )
axd['CT'].get_legend().remove()
axd['DO_mg_L'].get_legend().remove()
plt.savefig(plot_directory + 'figure_06.png', bbox_inches='tight', dpi=500, transparent=True)    
