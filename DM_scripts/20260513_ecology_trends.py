#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:35:03 2025

@author: dakotamascarenas
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

import D_functions as dfun

import pickle

import math

from scipy.interpolate import interp1d

import gsw

import matplotlib.path as mpth

import matplotlib.patches as patches

import cmocean

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None




Ldir = Lfun.Lstart(gridname='cas7')


fng = Ldir['grid'] / 'grid.nc'
dsg = xr.open_dataset(fng)
lon = dsg.lon_rho.values
lat = dsg.lat_rho.values
m = dsg.mask_rho.values
xp, yp = pfun.get_plon_plat(lon,lat)
depths = dsg.h.values
depths[m==0] = np.nan

lon_1D = lon[0,:]

lat_1D = lat[:,0]

# weird, to fix

mask_rho = np.transpose(dsg.mask_rho.values)
zm = -depths.copy()
zm[np.transpose(mask_rho) == 0] = np.nan
zm[np.transpose(mask_rho) != 0] = -1

zm_inverse = zm.copy()

zm_inverse[np.isnan(zm)] = -1

zm_inverse[zm==-1] = np.nan


X = lon[0,:] # grid cell X values
Y = lat[:,0] # grid cell Y values

plon, plat = pfun.get_plon_plat(lon,lat)


j1 = 570
j2 = 1170
i1 = 220
i2 = 652

# %%

year_list = np.arange(1998, 2027)

source_list = ['ecology_nc']

otype_list = ['ctd']
# observations
ii = 0
for year in year_list:
    for source in source_list:
        for otype in otype_list:
            odir = Ldir['LOo'] / 'obs' / source / otype
            try:
                if ii == 0:
                    odf = pd.read_pickle( odir / (str(year) + '.p'))
                    if 'ecology_nc' in source_list:
                        if source == 'ecology_nc' and otype == 'bottle': #keep an eye on this for calculating confidence intervals!!!
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
                    if 'ecology_nc' in source_list:
                        if source == 'ecology_nc' and otype == 'bottle':
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
    
    
    
# %%

var_list = ['DO_mg_L']


odf = (odf
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
                          # segment=(lambda x: key),
                          decade=(lambda x: pd.cut(x['year'],
                                                  bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                                                  labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
                              )
                      )
    
odf.loc[odf['month'].isin([1,2,3,12]), 'season'] = 'winter'
    
odf.loc[odf['month'].isin([4,5,6,7]), 'season'] = 'grow'
    
odf.loc[odf['month'].isin([8,9,10,11]), 'season'] = 'loDO'


for var in var_list:
    
    if var not in odf.columns:
        
        odf[var] = np.nan
            
    odf = pd.melt(odf, id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'source', 'otype', 'decade', 'name'],
                                          value_vars=var_list, var_name='var', value_name = 'val')
    

#odf = pd.concat(odf_dict.values(), ignore_index=True)


odf['source_type'] = odf['source'] + '_' + odf['otype']


odf = odf.dropna()

odf = odf.assign(
    ix=(lambda x: x['lon'].apply(lambda x: zfun.find_nearest_ind(lon_1D, x))),
    iy=(lambda x: x['lat'].apply(lambda x: zfun.find_nearest_ind(lat_1D, x)))
)


odf['h'] = odf.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)


odf['yearday'] = odf['datetime'].dt.dayofyear


odf = odf[odf['val'] >0]

# %%

mins_dict = dict()

mins_dict['cast_mins'] = odf.groupby(['cid', 'name','var']).min().dropna().reset_index()

mins_dict['cast_maxdepth'] = odf.loc[odf.groupby(['cid', 'name', 'var'])['z'].idxmin()].dropna()

mins_dict['cast_bottom20p'] = odf[odf['z'] < odf['h']*0.8].groupby(['cid','name','var']).mean(numeric_only=True).dropna().reset_index()

mins_dict['cast_bottom20p'] = mins_dict['cast_bottom20p'].assign(datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))


# %%
    
def linRegDM(plot_df):
    
    x = plot_df['date_ordinal']
    
    x_plot = plot_df['datetime']
    
    y = plot_df['val']
    
    result = stats.linregress(x, y)

    B1 = result.slope

    B0 = result.intercept

    plot_df['B1'] = B1

    plot_df['B0'] = B0

    sB1 = result.stderr

    n = len(x)

    plot_df['n'] = n 

    dof = n-2
    
    alpha = 0.05

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
                                        
    plot_df_concat = plot_df[['name','var', 'p', 'n', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo', 'B1', 'B0']].head(1) #slope_datetime_unc_cent, slope_datetime_s

    return plot_df_concat

# %%

min_trend_df = pd.DataFrame()

for min_type in ['cast_mins','cast_maxdepth', 'cast_bottom20p']:

    for site in mins_dict[min_type]['name'].unique():
        
        plot_df = mins_dict[min_type].copy()
        
        plot_df = plot_df[plot_df['name'] == site]
        
        plot_df_concat = linRegDM(plot_df)
        
        plot_df_concat['min_type'] = min_type
        
        min_trend_df = pd.concat([min_trend_df, plot_df_concat])

# %%

out_dir = Ldir['LOo'] / 'DM_outs' / '20260513_ecology_trends'
out_dir.mkdir(parents=True, exist_ok=True)

min_type_list = ['cast_mins', 'cast_maxdepth', 'cast_bottom20p']

all_sites = set()
for min_type in min_type_list:
    all_sites.update(mins_dict[min_type]['name'].unique())

for site in sorted(all_sites):

    # compute shared y-axis limits across all min_types for this site
    all_vals = pd.concat([
        mins_dict[mt][mins_dict[mt]['name'] == site]['val']
        for mt in min_type_list
        if site in mins_dict[mt]['name'].values
    ])
    y_min = all_vals.min() - 0.5 if not all_vals.empty else 0
    y_max = all_vals.max() + 0.5 if not all_vals.empty else 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(site, fontsize=12, fontweight='bold')

    for ax, min_type in zip(axes, min_type_list):

        ts_df = mins_dict[min_type][mins_dict[min_type]['name'] == site].copy().sort_values('datetime')

        trend_row = min_trend_df[
            (min_trend_df['min_type'] == min_type) &
            (min_trend_df['name'] == site)
        ]

        if ts_df.empty or trend_row.empty:
            ax.set_title(min_type)
            ax.set_visible(False)
            continue

        B0 = trend_row['B0'].values[0]
        B1 = trend_row['B1'].values[0]
        p_val = trend_row['p'].values[0]
        slope = trend_row['slope_datetime'].values[0]

        x_ord = ts_df['date_ordinal'].values
        trend_vals = B0 + B1 * x_ord

        sig = p_val < 0.05
        trend_color = 'firebrick' if sig else 'gray'
        trend_lw = 2.0 if sig else 1.0
        trend_ls = '-' if sig else '--'
        sig_label = '(p<0.05)' if sig else '(n.s.)'

        ax.scatter(ts_df['datetime'], ts_df['val'], s=10, alpha=0.5, color='steelblue', label='observed')
        ax.plot(ts_df['datetime'], trend_vals, color=trend_color, linewidth=trend_lw, linestyle=trend_ls,
                label=f'trend: {slope:.3f} mg/L/yr (p={p_val:.3f}) {sig_label}')

        ax.set_title(min_type, fontsize=9)
        ax.set_ylabel('DO (mg/L)')
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    fig.tight_layout()

    safe_site = site.replace('/', '_').replace(' ', '_')
    fig.savefig(out_dir / f'{safe_site}.png', dpi=120)
    plt.close(fig)

print(f'plots saved to {out_dir}')

min_trend_df.to_csv(out_dir / 'min_trend_df.csv', index=False)
print(f'min_trend_df saved to {out_dir / "min_trend_df.csv"}')

# %% site map

# collect one unique lon/lat per site (use cast_mins as reference)
site_locs = (odf[odf['var'] == 'DO_mg_L']
             .groupby('name')[['lon', 'lat']]
             .mean()
             .reset_index())

fig, ax = plt.subplots(figsize=(8, 10))

pfun.add_coast(ax)
pfun.dar(ax)

ax.scatter(site_locs['lon'], site_locs['lat'], s=18, color='steelblue',
           edgecolor='white', linewidth=0.4, zorder=5)

site_locs = site_locs.sort_values(['lat', 'lon']).reset_index(drop=True)

pad = 0.3
lon_min = site_locs['lon'].min() - pad
lon_max = site_locs['lon'].max() + pad
lat_min = site_locs['lat'].min() - pad
lat_max = site_locs['lat'].max() + pad

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Ecology CTD sites')

# place text first at each point (with small offset), then iteratively
# push overlapping labels apart in display coordinates so leader lines
# can connect them back to their points.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

texts = []
for _, row in site_locs.iterrows():
    t = ax.text(row['lon'], row['lat'], '  ' + row['name'],
                fontsize=6, color='0.15', ha='left', va='center', zorder=6)
    texts.append(t)

def get_bboxes():
    return [t.get_window_extent(renderer=renderer) for t in texts]

# iterative repulsion in pixel space
n_iter = 200
step = 1.5  # pixels per iteration
for _ in range(n_iter):
    bboxes = get_bboxes()
    moved = False
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if bboxes[i].overlaps(bboxes[j]):
                # nudge both labels vertically apart
                ci = (bboxes[i].y0 + bboxes[i].y1) / 2
                cj = (bboxes[j].y0 + bboxes[j].y1) / 2
                if ci == cj:
                    cj += 0.1
                direction = 1 if ci > cj else -1
                # convert pixel step to data coords
                inv = ax.transData.inverted()
                _, dy_data = inv.transform((0, step)) - inv.transform((0, 0))
                xi, yi = texts[i].get_position()
                xj, yj = texts[j].get_position()
                texts[i].set_position((xi, yi + direction * dy_data))
                texts[j].set_position((xj, yj - direction * dy_data))
                moved = True
        bboxes = get_bboxes()
    if not moved:
        break

# add leader lines from each point to its (possibly moved) label
for (_, row), t in zip(site_locs.iterrows(), texts):
    tx, ty = t.get_position()
    # only draw a leader if the label moved meaningfully
    if abs(ty - row['lat']) > 0.005 or abs(tx - row['lon']) > 0.005:
        ax.plot([row['lon'], tx], [row['lat'], ty],
                color='0.6', lw=0.4, zorder=4)

fig.savefig(out_dir / 'site_map.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('site map saved')
