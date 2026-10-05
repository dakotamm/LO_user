#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finalized for publication: 2026/01/07

Author: Dakota Mascarenas

Plotting code for: "Century-Scale Changes in Dissolved Oxygen, Temperature, and Salinity in Puget Sound" (Mascarenas et al., in review; submitted 2026/01/09 to Estuaries & Coasts)

This script processes data for and plots Figure 2 in corresponding manuscript. Please reach out to the author at dakotamm@uw.edu for any questions.

"""

# import modules
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import figure_functions_PSNME as ffun

plt.rcParams['hatch.linewidth'] = 0.5  # thinner hatch strokes (default ~1.0)
plt.rcParams['font.size'] = 13  # base font (ticks, axis labels, legends); default 10

### FIGURE 2

# load pickled data frame for all Puget Sound cast data from user-specified directory
df_directory = '/Users/dakotamascarenas/Desktop/Mascarenas_etal_2026_R1/' #SPECIFY LOCAL DIRECTORY
ps_casts_DF = pd.read_pickle(df_directory + 'ps_casts_DF.p')

# load grid attribute arrays from LiveOcean (MacCready et al., 2021; see text for more details and citations)
with open(df_directory + 'zm_inverse.p', 'rb') as fp:
    zm_inverse = pickle.load(fp)
with open(df_directory + 'plon.p', 'rb') as fp:
    plon = pickle.load(fp)
with open(df_directory + 'plat.p', 'rb') as fp:
    plat = pickle.load(fp)

# plot and save to user-specified directory
plot_directory = '/Users/dakotamascarenas/Desktop/pltz/' #SPECIFY SAVE LOCATION
palette = [
    "#e04256",
    "#4565e8",
    "#efbf04",
    "#9bd400"
]

# ---- coverage-timeline (panel b) encoding: source color + method-overlap hatch ----
sources = sorted(ps_casts_DF['data_source'].unique())
color_map = dict(zip(sources, palette))

# each sampling_type -> the set of base methods it represents (nceiSalish = both)
method_sets = {
    'Bottle': frozenset({'Bottle'}),
    'Bottle/CTD+DO': frozenset({'Bottle', 'CTD'}),
    'CTD': frozenset({'CTD'}),
    'CTD+DO': frozenset({'CTD'}),
    'Sonde (unknown type)': frozenset({'Sonde'}),
}
base_hatch = {'Bottle': '\\\\\\', 'CTD': '///', 'Sonde': '...'}  # back/fwd slashes, stipple (dense)
_method_order = ['Bottle', 'CTD', 'Sonde']

def combo_hatch(methods):
    """Overlay base hatches for every method present (bottle+CTD -> crosshatch)."""
    return ''.join(base_hatch[m] for m in _method_order if m in methods)

def combo_label(methods):
    return ' & '.join(m for m in _method_order if m in methods)

def method_runs(sub):
    """Yield (year_start, year_end, method_set) for contiguous same-method runs."""
    sub = sub.sort_values('year')
    new_run = (sub['mset'] != sub['mset'].shift()) | (sub['year'].diff().gt(1))
    sub = sub.assign(run=new_run.cumsum())
    for _, rr in sub.groupby('run'):
        yield rr['year'].min(), rr['year'].max(), rr['mset'].iloc[0]

# variable rows: CT & SA share a row (identical coverage); abbreviate CT->Temp., SA->Sal.
var_groups = [('Temp./Sal.', ['CT', 'SA']), ('DO', ['DO_mg_L'])]
short_src = {'Collias (Col.)': 'Col.', 'King County (KC)': 'KC',
             'Salish Cruises/PRISM (SCDP/P)': 'SCDP/P', 'WA Dept. of Ecology (Eco.)': 'Eco.'}

# manual coverage addition (not in per-cast data): WA Ecology used unknown-type
# sondes ~1973-1989; applied to both variable rows (carried from original Fig 2).
manual_spans = [
    ('WA Dept. of Ecology (Eco.)', 'Temp./Sal.', 1973, 1989, 'Sonde'),
    ('WA Dept. of Ecology (Eco.)', 'DO', 1973, 1989, 'Sonde'),
]

def coverage_for_group(gvars, glabel):
    """Per (source, year) union of methods present for the given variable group,
    including any manual_spans for that row label."""
    d = ps_casts_DF[ps_casts_DF['var'].isin(gvars)].copy()
    d['m'] = d['sampling_type'].map(method_sets)
    agg = (d.groupby(['data_source', 'year'])['m']
             .agg(lambda s: frozenset().union(*s)).reset_index())
    cells = {(r['data_source'], r['year']): set(r['m']) for _, r in agg.iterrows()}
    for src, lbl, y0, y1, meth in manual_spans:
        if lbl == glabel:
            for yr in range(y0, y1 + 1):
                cells.setdefault((src, yr), set()).add(meth)
    return pd.DataFrame([(s, y, frozenset(m)) for (s, y), m in cells.items()],
                        columns=['data_source', 'year', 'mset'])

mosaic = [['map_source', 'map_source','type_series', 'type_series', 'type_series'],
          ['map_source', 'map_source','type_series', 'type_series', 'type_series'],
          ['map_source', 'map_source','depth_time_series', 'depth_time_series', 'depth_time_series'],
          ['map_source', 'map_source','depth_time_series', 'depth_time_series', 'depth_time_series'],
          ['map_source', 'map_source','count_time_series','count_time_series','count_time_series'],
          ['map_source', 'map_source','count_time_series','count_time_series','count_time_series']]
fig, axd = plt.subplot_mosaic(mosaic, figsize=(9,7.5), layout='constrained', gridspec_kw=dict(wspace=0.13, width_ratios=[1.5, 1.5, 1, 1, 1]))  # wider map column so the map spans panels b-d
ax = axd['map_source']
# water (light blue) / land (off-white); drawn as a mesh so it survives transparent=True
ax.pcolormesh(plon, plat, np.where(np.isnan(zm_inverse), 0, 1), cmap=ListedColormap(['#cfe2f3', '#f5f6f7']), vmin=0, vmax=1, zorder=-5)
plot_df = ps_casts_DF.groupby(['data_source', 'cid']).first().reset_index()
sns.scatterplot(data=plot_df, x='lon', y='lat', hue='data_source', ax = ax, palette=palette, alpha=0.5, legend=False)
# PRISM/Salish Cruises is a small, low-visibility source; re-draw it last so it
# sits on top. Match seaborn's default white marker edge.
_prism = 'Salish Cruises/PRISM (SCDP/P)'
_prism_df = plot_df[plot_df['data_source'] == _prism]
ax.scatter(_prism_df['lon'], _prism_df['lat'], color=color_map[_prism],
           alpha=0.5, edgecolors='white', linewidths=0.5, zorder=10)
ax.set_xlim(-123.2, -122.1) 
ax.set_ylim(47,48.5)
ffun.add_coast(ax, df_directory)
ffun.dar(ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([-123.0, -122.6, -122.2], ['-123.0','-122.6', '-122.2'])
ax = axd['type_series']
ngrp = len(var_groups)
group_gap = 1.0
bar_h = 0.8
ypos, yticks, yticklabels = {}, [], []
for s_idx, src in enumerate(sources):
    base = s_idx * (ngrp + group_gap)
    for g_idx, (glabel, _) in enumerate(var_groups):
        y = base + g_idx
        ypos[(src, glabel)] = y
        yticks.append(y)
        yticklabels.append(glabel)
    ax.axhspan(base - 0.5, base + ngrp - 0.5, color=color_map[src], alpha=0.08, zorder=-5)

combos_seen = set()
for glabel, gvars in var_groups:
    cov = coverage_for_group(gvars, glabel)
    for src, sub in cov.groupby('data_source'):
        for yr0, yr1, methods in method_runs(sub):
            combos_seen.add(methods)
            ax.barh(y=ypos[(src, glabel)], width=yr1 - yr0 + 1, left=yr0, height=bar_h,
                    color=color_map[src], hatch=combo_hatch(methods), edgecolor='black', linewidth=0.4)

ax.set_yticks(yticks, yticklabels, fontsize=10)
ax.set_ylim(-0.7, (len(sources) - 1) * (ngrp + group_gap) + ngrp - 0.3)
ax.invert_yaxis()
ax.set_xlim(1930, 2027)
ax.set_xticklabels([])
ax.grid(color='lightgray', linestyle='--', alpha=0.5, axis='x')
# rotated source-group labels at far left
for s_idx, src in enumerate(sources):
    base = s_idx * (ngrp + group_gap)
    # fixed offset in points (not axes fraction) so it clears the tick labels at any panel width
    ax.annotate(short_src[src], xy=(0, base + (ngrp - 1) / 2), xycoords=ax.get_yaxis_transform(),
                xytext=(-70, 0), textcoords='offset points',
                ha='right', va='center', fontweight='bold', fontsize=11, rotation=90)
ax = axd['depth_time_series']
plot_df = ps_casts_DF.groupby(['data_source','year', 'cid']).min().reset_index()
plot_df = plot_df.groupby(['data_source', 'year']).mean(numeric_only=True).reset_index()
sns.scatterplot(data=plot_df, x='year', y='z', hue='data_source', ax=ax, palette=palette)
ax.set_xlabel('')
ax.set_ylabel('Annual Avg. Cast Depth [m]')
ax.set_ylim(-250,0)
ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
ax.set_xlim(1930, 2027)
ax.set_xticklabels([])
ax = axd['count_time_series']
plot_df = (ps_casts_DF
                      .groupby(['data_source','year']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      )
sns.scatterplot(data=plot_df, x='year', y='cid', hue='data_source', ax=ax, palette=palette, legend=False)
ax.set_xlabel('')
ax.set_ylabel('Annual Cast Count')
ax.set_ylim(0,1300)
ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
ax.set_xlim(1930, 2027)
handles_count, labels_count = axd['depth_time_series'].get_legend_handles_labels()
# legend shows the three base methods (overlaps in panel b combine these hatches)
method_legend = [('Bottle', base_hatch['Bottle']),
                 ('CTD', base_hatch['CTD']),
                 ('Sonde (unknown type)', base_hatch['Sonde'])]
hatch_handles = [
    patches.Patch(facecolor='lightgray', edgecolor='black', hatch=h, label=lab)
    for lab, h in method_legend
]
# Data Source legend at its natural (2-column) width, centered under the panels;
# Sampling Methods stretched (mode='expand') to that same width, stacked below with a fixed gap
axd['depth_time_series'].get_legend().remove()
fig.canvas.draw()  # settle constrained layout so the panel extents are final
to_fig = fig.transFigure.inverted()
panel_bbs = [to_fig.transform_bbox(a.get_tightbbox()) for a in axd.values()]
x_mid = (min(bb.x0 for bb in panel_bbs) + max(bb.x1 for bb in panel_bbs)) / 2
y_top = min(bb.y0 for bb in panel_bbs) - 0.01
leg_gap = 0.015
leg0 = fig.legend(
    handles_count, labels_count,
    loc='upper center',
    bbox_to_anchor=(x_mid, y_top),
    ncol=2,
    title='Data Source'
    )
fig.canvas.draw()
leg0_bb = to_fig.transform_bbox(leg0.get_window_extent())
leg1 = fig.legend(
    hatch_handles, [lab for lab, _ in method_legend],
    loc='upper left',
    bbox_to_anchor=(leg0_bb.x0, leg0_bb.y0 - leg_gap, leg0_bb.width, 0),
    mode='expand',
    borderaxespad=0,  # no inset from the anchor box, so the frame edges line up with leg0
    ncol=len(hatch_handles),
    title='Sampling Methods'
)
plt.savefig(plot_directory + 'figure_02.png', bbox_inches='tight', dpi=500, transparent=True)
