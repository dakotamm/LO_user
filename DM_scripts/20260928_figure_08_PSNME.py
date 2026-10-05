#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepared for publication: 2026/01/07

Author: Dakota Mascarenas

Plotting code for: "Century-Scale Changes in Dissolved Oxygen, Temperature, and Salinity in Puget Sound" (Mascarenas et al., in review; submitted 2026/01/09 to Estuaries & Coasts)

This script processes data for and plots Figure 8 in corresponding manuscript. Please reach out to the author at dakotamm@uw.edu for any questions.

"""

# import modules
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import figure_functions_PSNME as ffun

plt.rcParams['font.size'] = 13  # base font (ticks, axis labels, legends); default 10

### FIGURE 8

# load pickled data frame for all sites' surface and bottom water (depth-averaged) cast data from user-specified directory
df_directory = '/Users/dakotamascarenas/Desktop/Mascarenas_etal_2026_R1/' #SPECIFY LOCAL DIRECTORY
site_depth_avg_var_DF = pd.read_pickle(df_directory + 'site_depth_avg_var_DF.p')

# apply DO filtering to casts in the bottom 50th percentile of annual seasonal bottom DO values
filter_DO_DF = ffun.filter_DO(site_depth_avg_var_DF)

# calculate Theil-Sen slopes with 95% confidence (alpha=0.05) for temperature, salinity, and DO
alpha = 0.05
slope_DF = ffun.calc_slopes_var(alpha, site_depth_avg_var_DF, filter_DO_DF)

# flag the series whose trend shape departs from a monotonic straight line, using the
# canonical trend-shape classification (run_first_mono_lin_test.py -> shape_DF.p; both
# the reversal/monotonicity and curvature/linearity tests are BH-FDR corrected with a
# high-B reversal certification). Three de-emphasized tiers are drawn (all with a grey
# marker outline + non-solid grey CI whisker), from strongest to weakest evidence:
#   'rev'         : BH-significant non-monotonic reversal (group=='rev')        -> dark grey,  dashed CI
#   'raw_nonmono' : reversal test raw-significant but NOT surviving BH          -> light grey, dotted CI
#   'raw_nonlin'  : curvature raw-significant (nonlin_uncond) but NOT BH        -> light grey, dash-dot CI
# The two raw tiers are the nonlinear/nonmonotonic trends identified in the raw
# (uncorrected) tests that did not survive BH correction, shown for sensitivity.
shape_DF = pd.read_pickle(df_directory + 'shape_DF.p')
alpha_shape = 0.05
def _shape_key(r):
    return (r['var'], r['site'], r['season_label'], r['surf_deep'])
# BH-significant non-monotonic reversal
bh_nonmono_set = {_shape_key(r) for _, r in shape_DF[shape_DF['group'] == 'rev'].iterrows()}
# raw non-monotonic that did NOT survive BH correction
raw_nonmono_set = {_shape_key(r) for _, r in
                   shape_DF[(shape_DF['mono_p'] < alpha_shape) & (~shape_DF['nonmono_BH'])].iterrows()}
# raw non-linear (unconditional curvature significant) that did NOT survive BH correction
raw_nonlin_set = {_shape_key(r) for _, r in
                  shape_DF[(shape_DF['nonlin_uncond']) & (~shape_DF['nonlin_BH'])].iterrows()}
grey = '#8a8a8a'       # BH-significant reversal (strongest flag)
grey_raw = '#bdbdbd'   # raw-only flag (did not survive BH)
# per-tier (edge color, CI color, CI linestyle); priority rev > raw_nonmono > raw_nonlin
tier_style = {
    'rev':         dict(edge=grey,     ci=grey,     ls=(0, (1.5, 1.2))),        # dashed
    'raw_nonmono': dict(edge=grey_raw, ci=grey_raw, ls=(0, (1, 1.4))),          # dotted
    'raw_nonlin':  dict(edge=grey_raw, ci=grey_raw, ls=(0, (4, 1.2, 1, 1.2))),  # dash-dot
}
def shape_tier(var, site, season_label, depth):
    sdepth = 'surf' if depth == 'Surface' else 'deep'
    k = (var, site, season_label, sdepth)
    if k in bh_nonmono_set:
        return 'rev'
    if k in raw_nonmono_set:
        return 'raw_nonmono'
    if k in raw_nonlin_set:
        return 'raw_nonlin'
    return None

# plot and save to user-specified directory
plot_directory = '/Users/dakotamascarenas/Desktop/pltz/' #SPECIFY SAVE LOCATION
linecolors = {'Main Basin':'k', 'Sub-Basins':'k'}
var_colors = {'CT': '#e04256', 'SA': '#9bd400', 'DO_mg_L': '#4565e8'}  # red/green/blue from the Figure 2 palette
var_names = {'CT': 'Temperature', 'SA': 'Salinity', 'DO_mg_L': '[DO]'}
jitter = {'Surface': -0.1, 'Bottom': 0.1}
markers = {'DO_mg_L': 'o', 'SA': 'o', 'CT': 'o'}
ymins = {'DO_mg_L': -2.5, 'CT': -1, 'SA': -4}
ymaxs = {'DO_mg_L': 3, 'CT': 5.5, 'SA': 2}
plot_labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
var_list = ['CT', 'SA', 'DO_mg_L']
site_list = ['point_jefferson', 'near_seattle_offshore', 'carr_inlet_mid', 'saratoga_passage_mid', 'lynch_cove_mid']
season_list = ['Winter (Dec-Mar)', 'Spring (Apr-Jul)', 'Low-DO (Aug-Nov)']
depth_list = ['Surface', 'Bottom']
palette = {'Surface': 'white', 'Bottom': 'gray'}
mosaic = [['CT Winter (Dec-Mar)', 'CT Spring (Apr-Jul)', 'CT Low-DO (Aug-Nov)'],
          ['SA Winter (Dec-Mar)', 'SA Spring (Apr-Jul)', 'SA Low-DO (Aug-Nov)'],
          ['DO_mg_L Winter (Dec-Mar)', 'DO_mg_L Spring (Apr-Jul)', 'DO_mg_L Low-DO (Aug-Nov)']]
fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, figsize=(10,6.75), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))
c=0
for var in var_list:
    for season in season_list:
        ax_name = var + ' ' + season
        ax = axd[ax_name]
        for site in site_list:
            for depth in depth_list:
                plot_df = slope_DF[(slope_DF['var'] == var) & (slope_DF['season_label'] == season) & (slope_DF['site'] == site) & (slope_DF['depth_label'] == depth)]
                plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
                plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
                plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
                tier = shape_tier(var, site, season, depth)         # None, 'rev', 'raw_nonmono', or 'raw_nonlin'
                if tier:
                    edge_c = tier_style[tier]['edge']               # grey outline when shape departs from a monotonic line
                    ci_c = tier_style[tier]['ci']
                    ci_ls = tier_style[tier]['ls']                  # ...and a non-solid CI whisker keyed to the tier
                else:
                    edge_c = 'k'
                    ci_c = linecolors[plot_df['site_type'].iloc[0]]
                    ci_ls = '-'
                ax.scatter(plot_df['site_num'] + jitter[depth], plot_df['slope_datetime_cent'], color=palette[depth], edgecolors=edge_c, marker=markers[var], s=50, label=depth, zorder=3) #, marker=markers[depth], edgecolors=edgecolors[site])
                ax.plot([plot_df['site_num'] + jitter[depth], plot_df['site_num'] + jitter[depth]],[plot_df['slope_datetime_cent_95lo'], plot_df['slope_datetime_cent_95hi']], color=ci_c, alpha =1, zorder = -5, linewidth=1, linestyle=ci_ls, label=plot_df['site_type'].iloc[0])
        ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3, zorder = -6)
        ax.axhline(0, color='gray', linestyle = '--', zorder = -5) 
        if season == 'Winter (Dec-Mar)':
            # units line is the (black) y-label; the bold, colored variable name sits just left of it
            ax.set_ylabel(slope_DF[slope_DF['var'] == var]['var_label'].iloc[0] + '/cent.')
            ax.annotate(var_names[var], xy=(0, 0.5), xycoords=ax.yaxis.label, xytext=(-1, 0), textcoords='offset points',
                        rotation=90, ha='right', va='center', fontweight='bold', color=var_colors[var])
        if var == 'CT':
            ax.set_title(season, fontweight='bold', fontsize=13)
        else:
            ax.set_xlabel('')
        ax.set_ylim(ymins[var], ymaxs[var])
        ax.set_xticks([1,2,3,4,5],['PJ', 'NS', 'CI', 'SP', 'LC'])
        if ax_name == 'DO_mg_L Winter (Dec-Mar)':
            handles, labels = ax.get_legend_handles_labels()
            selected_handles = [handles[0], handles[2]]
            selected_labels = [labels[0], labels[2]]
            ax.legend(selected_handles, selected_labels, loc='upper left')
        c+=1
rev_handle = mlines.Line2D([], [], color=grey, marker='o', markerfacecolor='white',
    markeredgecolor=grey, markersize=8, lw=1.4, linestyle=(0, (1.5, 1.2)),
    label='non-monotonic (BH-significant)')
raw_nonmono_handle = mlines.Line2D([], [], color=grey_raw, marker='o', markerfacecolor='white',
    markeredgecolor=grey_raw, markersize=8, lw=1.4, linestyle=(0, (1, 1.4)),
    label='non-monotonic (raw, not BH)')
raw_nonlin_handle = mlines.Line2D([], [], color=grey_raw, marker='o', markerfacecolor='white',
    markeredgecolor=grey_raw, markersize=8, lw=1.4, linestyle=(0, (4, 1.2, 1, 1.2)),
    label='non-linear (raw, not BH)')
shape_handles = [rev_handle, raw_nonmono_handle, raw_nonlin_handle]
leg = fig.legend(
    selected_handles + shape_handles,
    selected_labels + [h.get_label() for h in shape_handles],
    loc='upper center',
    bbox_to_anchor=(0.5, -0.01),
    ncol=3
    )
axd['DO_mg_L Winter (Dec-Mar)'].get_legend().remove()
plt.savefig(plot_directory + 'figure_08.png', bbox_inches='tight', dpi=500, transparent=True)    



