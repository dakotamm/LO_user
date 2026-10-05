#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepared for publication: 2026/01/07

Author: Dakota Mascarenas

Plotting code for: "Century-Scale Changes in Dissolved Oxygen, Temperature, and Salinity in Puget Sound" (Mascarenas et al., in review; submitted 2026/01/09 to Estuaries & Coasts)

This script processes data for and plots Figure 9 in corresponding manuscript. Please reach out to the author at dakotamm@uw.edu for any questions.

"""

# import modules
import matplotlib.pyplot as plt
import pandas as pd
import figure_functions_PSNME as ffun

### FIGURE 9

# load pickled data frame for all sites' surface and bottom water (depth-averaged) cast data from user-specified directory
df_directory = '/Users/dakotamascarenas/Desktop/Mascarenas_etal_2026_R1/' #SPECIFY LOCAL DIRECTORY
site_depth_avg_var_DF = pd.read_pickle(df_directory + 'site_depth_avg_var_DF.p')

# apply DO filtering to casts in the bottom 50th percentile of annual seasonal bottom DO values
filter_DO_DF = ffun.filter_DO(site_depth_avg_var_DF)

# calculate DO saturation for depth-averaged cast data
DO_sat_DF = ffun.calc_DO_sat(site_depth_avg_var_DF)

# calculate Theil-Sen slopes with 95% confidence (alpha=0.05) for all variables and calculated DO saturation values
alpha = 0.05
slope_DF = ffun.calc_slopes_var(alpha, site_depth_avg_var_DF, filter_DO_DF, DO_sat_DF)

# plot and save to user-specified directory
plot_directory = '/Users/dakotamascarenas/Desktop/pltz/' #SPECIFY SAVE LOCATION
linecolor = 'k'
palette = {'Surface': 'white', 'Bottom': 'gray'}
yellow =     "#efbf04"
marker = 'o'
var = 'DO_mg_L'
depth = 'Bottom'
site_list = ['point_jefferson', 'near_seattle_offshore', 'carr_inlet_mid', 'saratoga_passage_mid', 'lynch_cove_mid']
season_list = ['Winter (Dec-Mar)', 'Spring (Apr-Jul)', 'Low-DO (Aug-Nov)']
mosaic = [['Winter (Dec-Mar)', 'Spring (Apr-Jul)', 'Low-DO (Aug-Nov)']]
fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, figsize=(9,2.2), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))
for season in season_list:
    ax_name = season
    ax = axd[ax_name]
    for site in site_list:
        plot_df = slope_DF[(slope_DF['var'] == var) & (slope_DF['season_label'] == season) & (slope_DF['site'] == site) & (slope_DF['depth_label'] == depth)]
        plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
        plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
        plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
        ax.scatter(plot_df['site_num'], plot_df['slope_datetime_cent'], color=palette[depth], edgecolors='k', marker=marker, s=50, label='Observed Bottom Trend')
        ax.plot([plot_df['site_num'], plot_df['site_num']],[plot_df['slope_datetime_cent_95lo'], plot_df['slope_datetime_cent_95hi']], color=linecolor, alpha =1, zorder = -5, linewidth=1, label=plot_df['site_type'].iloc[0])
        plot_df_DO_sol = slope_DF[(slope_DF['season_label'] == season) & (slope_DF['var'] == 'DO_sat') & (slope_DF['site'] == site) & (slope_DF['depth_label'] == depth)]
        plot_df_DO_sol['slope_datetime_cent'] = plot_df_DO_sol['slope_datetime']*100
        plot_df_DO_sol['slope_datetime_cent_95hi'] = plot_df_DO_sol['slope_datetime_s_hi']*100
        plot_df_DO_sol['slope_datetime_cent_95lo'] = plot_df_DO_sol['slope_datetime_s_lo']*100
        ax.scatter(plot_df_DO_sol['site_num'], plot_df_DO_sol['slope_datetime_cent'], color=yellow, marker=marker, s=100, alpha= 1, label= 'Sol.-Based Trend', zorder =6)
        ax.plot([plot_df_DO_sol['site_num'], plot_df_DO_sol['site_num']],[plot_df_DO_sol['slope_datetime_cent_95lo'], plot_df_DO_sol['slope_datetime_cent_95hi']], color=yellow, alpha =1, zorder = 6, linewidth=2)
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3, zorder = -7)
    ax.axhline(0, color='gray', linestyle = '--', zorder = -7) 
    ax.set_ylabel(slope_DF[slope_DF['var'] == var]['var_label'].iloc[0] + '/century')
    ax.set_xticks([1,2,3,4,5],['PJ', 'NS', 'CI', 'SP', 'LC'])
    ax.set_title(season, fontweight='bold', fontsize=10)
    ax.set_ylim(-2.5,1)
    if season == 'Spring (Apr-Jul)':
        handles, labels = ax.get_legend_handles_labels()
        selected_handles = [handles[0], handles[2]]
        selected_labels = [labels[0], labels[2]]
        ax.legend(selected_handles, selected_labels, loc ='upper left',  fontsize=12)
        ax.set_ylabel('')
    else:
        ax.set_ylabel('')
fig.legend(
    selected_handles, selected_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.01),
    ncol=len(selected_handles)
    )
axd['Spring (Apr-Jul)'].get_legend().remove()
plt.savefig(plot_directory + 'figure_09.png', bbox_inches='tight', dpi=500, transparent=True)    

    


