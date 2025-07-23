"""
Test finding hypoxic depth and volume.

Test on mac in ipython:
run test_hyp_vol_DM -gtx cas6_v0_live -source dfo -otype ctd -year 2019 -test False

"""
import sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from lo_tools import Lfun, zfun, zrfun
from lo_tools import extract_argfun as exfun
import cast_functions as cfun
from lo_tools import plotting_functions as pfun
import tef_fun as tfun
import pickle

from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from scipy.spatial import KDTree

import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import itertools

Ldir = exfun.intro() # this handles the argument passing

year_str = str(Ldir['year'])

month_num = ['01','02','03','04','05','06','07','08','09','10','11','12']

month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

#month_num = ['01']

#month_str = ['Jan']

hyp_vol_dict_obs = {}

hyp_vol_dict_LO = {}

hyp_array_dict_LO = {}

wtd_avg_dict_LO = {}

hyp_thick_dict_LO = {}

hyp_thick_dict_obs = {}

# get segment info
vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
v_df = pd.read_pickle(vol_dir / 'volumes.p')
j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
seg_list = list(v_df.index)

info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + year_str + '.p')

dt = pd.Timestamp('2022-01-01 01:30:00')
fn_his = cfun.get_his_fn_from_dt(Ldir, dt)

#grid info
G, S, T = zrfun.get_basic_info(fn_his)
Lon = G['lon_rho'][0, :]
Lat = G['lat_rho'][:, 0]
z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
dz = np.diff(z_w_grid,axis=0)
dv = dz*G['DX']*G['DY']

# %%

for seg_name in seg_list:
    
    if 'G1' in seg_name:
            
        jjj = j_dict[seg_name]
        iii = i_dict[seg_name]
        
        info_df = pd.read_pickle(info_fn)
        
        ij_ = []
        ii_ = []
        

        for cid in info_df.index:
            
            if info_df.loc[cid,'time'] >= datetime(2019,6,1) and info_df.loc[cid,'time'] <= datetime(2019,8,31):
            
                lon = info_df.loc[cid, 'lon']
                lat = info_df.loc[cid, 'lat']
                
                ij = zfun.find_nearest_ind(Lat, lat)
                ii = zfun.find_nearest_ind(Lon, lon)
                
                if (ii in iii) and (ij in jjj):
                    
                    ij_.append(ij)
                    ii_.append(ii)
                                   
            ii_cast = ii_
            ij_cast = ij_
        
        
xx = np.arange(min(iii), max(iii)+1)
yy = np.arange(min(jjj), max(jjj)+1)
x, y = np.meshgrid(xx, yy)
        
a = np.full([len(yy),len(xx)], -99)
a[jjj-min(jjj),iii-min(iii)] = -1
a = np.ma.masked_array(a,a==-99)

b = a.copy()
for n in range(len(ij_cast)):
    b[ij_cast[n]-min(jjj), ii_cast[n]-min(iii)] = n
    
c = b.copy()
c = np.ma.masked_array(c,c==-1)
    
xy_water = np.array((x[~a.mask],y[~a.mask])).T
xy_land = np.array((x[a.mask],y[a.mask])).T

xy_casts = np.array((x[~c.mask],y[~c.mask])).T

tree = KDTree(xy_casts)

tree_query = tree.query(xy_water)[1]

d = a.copy()
d[~a.mask] = b[~c.mask][tree_query]

# %%


sect_df = tfun.get_sect_df('cas6')

min_lat = [48, 48.4]
max_lat = [49, 48.7]
min_lon = [-124, -123.4]
max_lon = [-122.25,-122.4]


cmap = cm.get_cmap('viridis', len(ii_cast))

hyp_val = 0.0

# %%


for (mon_num, mon_str) in zip(month_num,month_str):

    dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
    fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
    
    dv_sliced = dv[:,min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    dz_sliced = dz[:,min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    ds_his = xr.open_dataset(fn_his)
        
    oxygen_mg_L = (ds_his.oxygen.squeeze()*32/1000).to_numpy() #molar mass of O2 ###double check this
    
    # fill oxygen_mg_L all below
    
    #oxygen_mg_L_np = oxygen_mg_L.isel(ocean_time = 0).to_numpy().reshape(30,1302,663)
    
    #oxygen_mg_L_np = oxygen_mg_L.values.squeeze() #squeezes out singleton dims
    
    oxygen_mg_L = oxygen_mg_L[:,min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    dv_hyp = dv_sliced.copy()
    
    dv_hyp[oxygen_mg_L > hyp_val] = 0
    
    dv_hyp[np.isnan(oxygen_mg_L)] = np.nan
                    
    #hyp_array_LO = np.ma.masked_where(oxygen_mg_L > hyp_val, oxygen_mg_L).filled(fill_value = 0)
    
    #hyp_vol = np.ma.masked_where(oxygen_mg_L > hyp_val, dv_sliced).filled(fill_value = 0)
        
    #wtd_avg_conc = np.nanmean(dv_sliced*1000*oxygen_mg_L)/np.nanmean(dv_sliced*1000)
        
    hyp_vol_sum = np.nansum(dv_hyp)
    
    # find alt to dictionaries
    
    dz_hyp = dz_sliced.copy()
    
    dz_hyp[oxygen_mg_L > hyp_val] = 0
    
    dz_hyp[np.isnan(oxygen_mg_L)] = np.nan
    
    hyp_thick_sum = np.nansum(dz_hyp, axis=0) #units of m...how thick hypoxic is...depth agnostic...
    
    #hyp_thick_sum[np.isnan(oxygen_mg_L[0,:,:])] = np.nan
    
    
  #  hyp_array_dict_LO[dt] = hyp_array_LO
    
    hyp_vol_dict_LO[dt] = hyp_vol_sum
    
    hyp_thick_dict_LO[dt] = hyp_thick_sum
    
   # wtd_avg_dict_LO[dt] = wtd_avg_conc
    
    #this is very hacky below - but works for now
    
    # info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + year_str + '.p')
    
    in_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' /
        (Ldir['source'] + '_' + Ldir['otype'] + '_' + year_str))
    
    fn_list = list(sorted(in_dir.glob('*' + mon_str + '_2022.nc')))
    
    foo = xr.open_dataset(fn_list[0])
    for vn in foo.data_vars:
        print('%14s: %s' % (vn, str(foo[vn].shape)))
    foo.close()
    
    tt0 = time()
    x = []; y = []
    s0 = []; s1 = []
    t0 = []; t1 = []
    z_rho = []
    oxygen = []
    
    for fn in fn_list:
        ds = xr.open_dataset(fn)
        z_rho.append(ds.z_rho.values)
        oxygen.append(ds.oxygen.values*32/1000)  # mg/L
        s0.append(ds.salt[0].values)
        t0.append(ds.temp[0].values)
        s1.append(ds.salt[-1].values)
        t1.append(ds.temp[-1].values)
        ds.close()
        print(min(z_rho[0]))
        print(fn)
    print('Took %0.2f sec' % (time()-tt0))
    
    
    # for seg_name in seg_list:
        
    #     if 'G1' in seg_name:
                
    #         jjj = j_dict[seg_name]
    #         iii = i_dict[seg_name]
            
    #         info_df = pd.read_pickle(info_fn)
            
    #         ij_ = []
    #         ii_ = []
            
    
    #         for cid in info_df.index:
                
    #             if info_df.loc[cid,'time'] >= datetime(2019,6,1) and info_df.loc[cid,'time'] <= datetime(2019,8,31):
                
    #                 lon = info_df.loc[cid, 'lon']
    #                 lat = info_df.loc[cid, 'lat']
                    
    #                 ij = zfun.find_nearest_ind(Lat, lat)
    #                 ii = zfun.find_nearest_ind(Lon, lon)
                    
    #                 if (ii in iii) and (ij in jjj):
                        
    #                     ij_.append(ij)
    #                     ii_.append(ii)
                                       
    #             ii_cast = ii_
    #             ij_cast = ij_
            
            
    # xx = np.arange(min(iii), max(iii)+1)
    # yy = np.arange(min(jjj), max(jjj)+1)
    # x, y = np.meshgrid(xx, yy)
            
    # a = np.full([len(yy),len(xx)], -99)
    # a[jjj-min(jjj),iii-min(iii)] = -1
    # a = np.ma.masked_array(a,a==-99)
    
    # b = a.copy()
    # for n in range(len(ij_cast)):
    #     b[ij_cast[n]-min(jjj), ii_cast[n]-min(iii)] = n
        
    # c = b.copy()
    # c = np.ma.masked_array(c,c==-1)
        
    # xy_water = np.array((x[~a.mask],y[~a.mask])).T
    # xy_land = np.array((x[a.mask],y[a.mask])).T
    
    # xy_casts = np.array((x[~c.mask],y[~c.mask])).T
    
    # tree = KDTree(xy_casts)
    
    # tree_query = tree.query(xy_water)[1]
    
    # d = a.copy()
    # d[~a.mask] = b[~c.mask][tree_query]
    
    
    # sect_df = tfun.get_sect_df('cas6')
    
    # min_lat = [48, 48.4]
    # max_lat = [49, 48.7]
    # min_lon = [-124, -123.4]
    # max_lon = [-122.25,-122.4]
    
    
    
    
    
    
    
    cmap = cm.get_cmap('viridis', len(ii_cast))
    
    # plt.close('all')
    # pfun.start_plot(fs=14, figsize=(20,10))
    # fig0, axes0 = plt.subplots(nrows=1, ncols=2, squeeze=False)
    # axes0[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],d)
    # for m in range(len(ii_cast)):
    #     axes0[0,0].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
    # axes0[0,0].set_xlim([min_lon[0],max_lon[0]])
    # axes0[0,0].set_ylim([min_lat[0],max_lat[0]])
    # axes0[0,0].tick_params(labelrotation=45)
    # pfun.add_coast(axes0[0,0])
    # pfun.dar(axes0[0,0])
    # axes0[0,1].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],d)
    # for m in range(len(ii_cast)):
    #     axes0[0,1].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
    # axes0[0,1].set_xlim([min_lon[1],max_lon[1]])
    # axes0[0,1].set_ylim([min_lat[1],max_lat[1]])
    # axes0[0,1].tick_params(labelrotation=45)
    # pfun.add_coast(axes0[0,1])
    # pfun.dar(axes0[0,1])
    # plt.title(mon_str + ' Cast Areas')
    # fig0.tight_layout()
    # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+mon_str+'_cast_areas.png')
    
   #  pfun.start_plot(fs=14, figsize=(7,5))
   #  fig1, axes1 = plt.subplots(nrows=1, ncols=1, squeeze=False)
   #  for i in range(len(ii_cast)):
   #      axes1[0,0].plot(oxygen[i],z_rho[i],'o',c=cmap(i))
   #  axes1[0,0].set_xlabel('[DO][mg/L]')
   #  axes1[0,0].set_ylabel('Depth [m]')
   # # plt.title(mon_str + ' DO Profiles')
   #  fig1.tight_layout()
   #  plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+mon_str+'_DO_profiles_small.png')
    
    
    cast_no = list(map(str, np.arange(len(ii_cast))))
    
    df0 = pd.DataFrame(oxygen)
    df0['cast_no'] = cast_no
    df0 = pd.melt(df0, id_vars=['cast_no'], var_name='sigma_level', value_name='DO')
    
    df1 = pd.DataFrame(z_rho)
    df1['cast_no'] = cast_no
    df1 = pd.melt(df1, id_vars=['cast_no'], var_name='sigma_level', value_name='z_rho')
    
    df = df0.merge(df1,on=['cast_no','sigma_level'])
    
    df_hyp = df[df['DO'] < hyp_val]
    
    hyp_casts = df_hyp['cast_no'].unique()
    
    hyp_casts = [int(l) for l in hyp_casts]
    
    e = d.copy()
    
    e = [[ele if ele in hyp_casts else -99 for ele in line] for line in e]
    
    e = np.array(e)
    
    e =np.ma.masked_array(e,e==-99)
    
    
    # cmap_colors = cmap.copy()
    # newcolors = cmap_colors(np.linspace(0, 1, len(cast_no)))
    # white = np.array([256/256, 256/256, 256/256, 1])
    # for cast in range(len(cast_no)):
    #     if cast not in hyp_casts:
    #         newcolors[int(cast), :] = white
    # new_cmap = ListedColormap(newcolors)
    
    # pfun.start_plot(fs=14, figsize=(10,10))
    # fig2, axes2 = plt.subplots(nrows=1, ncols=1, squeeze=False)
    # axes2[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],d,cmap=new_cmap)
    # for m in range(len(ii_cast)):
    #     axes2[0,0].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
    # axes2[0,0].set_xlim([min_lon[0],max_lon[0]])
    # axes2[0,0].set_ylim([min_lat[0],max_lat[0]])
    # axes2[0,0].tick_params(labelrotation=45)
    # pfun.add_coast(axes2[0,0])
    # plt.title(mon_str + ' <'+str(hyp_val)+'mg/L Cast Areas')
    # fig2.tight_layout()
    # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+mon_str+'_hyp_cast_areas.png')
    
    
    max_z_hyp = []
    
    hyp_vol = 0
    
    hyp_thick_array = np.empty(np.shape(z_rho_grid))
    hyp_thick_array.fill(0)

    
    for m in range(len(hyp_casts)):
        df_temp = df_hyp[df_hyp['cast_no']==str(hyp_casts[m])]
        max_z_hyp = df_temp['z_rho'].max()
        idx_hyp = np.where(e == hyp_casts[m])
        hyp_array = np.empty(np.shape(z_rho_grid))
        hyp_array.fill(0)
        
        for mm in range(len(idx_hyp[0])):
            jjj_hyp = idx_hyp[0][mm] + min(jjj)
            iii_hyp = idx_hyp[1][mm] + min(iii)
            zzz_hyp= np.where(z_rho_grid[:,jjj_hyp,iii_hyp] <= max_z_hyp)
            if zzz_hyp:
                #hyp_array[zzz_hyp,jjj_hyp,iii_hyp] = hyp_casts[m]
                hyp_array[zzz_hyp,jjj_hyp,iii_hyp] = dv[zzz_hyp,jjj_hyp,iii_hyp]
                hyp_thick_array[zzz_hyp,jjj_hyp,iii_hyp] = dz[zzz_hyp,jjj_hyp,iii_hyp]
                        
        hyp_vol = hyp_vol + np.sum(hyp_array)
        
    hyp_thick = np.sum(hyp_thick_array, axis=0)
    
    hyp_thick = hyp_thick[min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    hyp_thick[np.isnan(oxygen_mg_L[0,:,:])] = np.nan
        
    hyp_vol_dict_obs[dt] = hyp_vol
    
    hyp_thick_dict_obs[dt] = hyp_thick
    
    # hyp_deep = np.empty((54,151))
    
    # hyp_deep.fill(np.nan)
    
    # hyp_deep = hyp_array_LO[0,:,:]
    
    # hyp_deep = np.ma.masked_array(hyp_deep,hyp_deep == 0)
    
# %%

for (mon_num, mon_str) in zip(month_num,month_str):

    dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
    
    # cmap_colors = cmap.copy()
    # newcolors = cmap_colors(np.linspace(0, 1, len(cast_no)))
    # white = np.array([256/256, 256/256, 256/256, 1])
    # for cast in range(len(cast_no)):
    #     if cast not in hyp_casts:
    #         newcolors[int(cast), :] = white
    # new_cmap = ListedColormap(newcolors)
        
    pfun.start_plot(fs=14, figsize=(10,10))
    fig3, axes3 = plt.subplots(nrows=2, ncols=1, squeeze=False)
    c0 = axes3[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],hyp_thick_dict_obs[dt],cmap='Blues', vmin = 1, vmax = 150)
    for m in range(len(ii_cast)):
        axes3[0,0].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
    axes3[0,0].set_xlim([min_lon[1],max_lon[1]])
    axes3[0,0].set_ylim([min_lat[1],max_lat[1]])
    #axes3[0,0].set_xlabel('Casts')
    axes3[0,0].tick_params(labelrotation=45)
    axes3[0,0].set_title('Volume-From-Casts ('+mon_str+')')
    pfun.add_coast(axes3[0,0])
    
    hyp_thick_dict_LO[dt][hyp_thick_dict_LO[dt] <= 4] = 0
    c1 = axes3[1,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],hyp_thick_dict_LO[dt],cmap = 'Blues', vmin = 1, vmax = 150)
    for m in range(len(ii_cast)):
        axes3[1,0].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
    axes3[1,0].set_xlim([min_lon[1],max_lon[1]])
    axes3[1,0].set_ylim([min_lat[1],max_lat[1]])
    # axes3[0,1].set_xlabel('LO Volumes')
    axes3[1,0].tick_params(labelrotation=45)
    axes3[1,0].set_title('LO Volumes ('+mon_str+')')
    pfun.add_coast(axes3[1,0])
    
    fig3.colorbar(c0,ax=axes3[0,0], label = 'Subthreshold Thickness [m]')
    fig3.colorbar(c1,ax=axes3[1,0], label = 'Subthreshold Thickness [m]')
    
    #plt.title(mon_str + ' <'+str(hyp_val)+'mg/L Bottom Areas')
    fig3.tight_layout()
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+mon_str+'_comp_hyp_thick_'+str(hyp_val)+'.png')
    
    
    
# %%    

month_num_q = ['01','04','07','10']

month_str_q = ['Jan','Apr','Jul','Oct']

pfun.start_plot(fs=14, figsize=(50,20))
fig4, axes4 = plt.subplots(nrows=3, ncols=4, squeeze=False)

r = 0

c = 0

for (mon_num, mon_str) in zip(month_num,month_str):
    
    dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
    
    hyp_thick_dif = hyp_thick_dict_obs[dt] - hyp_thick_dict_LO[dt]

    c2 = axes4[r,c].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],hyp_thick_dif,cmap='PuOr', vmin = -150, vmax = 150)
    for m in range(len(ii_cast)):
        axes4[r,c].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
    axes4[r,c].set_xlim([min_lon[1],max_lon[1]])
    axes4[r,c].set_ylim([min_lat[1],max_lat[1]])
    #axes3[0,0].set_xlabel('Casts')
    axes4[r,c].tick_params(labelrotation=45)
    axes4[r,c].set_title(mon_str+' VFC-LO')
    pfun.add_coast(axes4[r,c])
    fig4.colorbar(c2, ax=axes4[r,c], label = '<'+str(hyp_val)+' [mg/L] Thick Diff [m]')
    c+=1
    if c >= 4:
        c = 0
        r+=1
    
fig4.tight_layout()
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/hyp_thick_dif_2022_'+str(hyp_val)+'.png')

    
 
    
# %%

pfun.start_plot(fs=14, figsize=(14,7))
fig4, axes4 = plt.subplots(nrows=1, ncols=1, squeeze=False)
hyp_vol_obs_ordered = sorted(hyp_vol_dict_obs.items())
hyp_vol_LO_ordered = sorted(hyp_vol_dict_LO.items())
x_obs, y_obs = zip(*hyp_vol_obs_ordered)
x_LO, y_LO = zip(*hyp_vol_LO_ordered)

plt.grid()
plt.plot(x_obs,y_obs,label ='Casts')
plt.plot(x_LO,y_LO,label = 'LO Volumes')
axes4[0,0].fill_between(x_obs,y_obs,y_LO, color='gray', alpha = 0.2)
axes4[0,0].set_xlim(x_obs[0], x_obs[-1])
#axes4[0,0].set_ylim(0, 5e10)
axes4[0,0].tick_params(axis ='x', labelrotation=45)
plt.legend()
axes4[0,0].set_title('Total Subthreshold Volumes [m^3]')
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/hyp_vol_2022_w_dif_'+str(hyp_val)+'.png')

# # %%

# cmap_colors = cmap.copy()
# newcolors = cmap_colors(np.linspace(0, 1, len(cast_no)))
# purp = np.array([72/256, 16/256, 106/256, 0.6])
# for cast in range(len(cast_no)):
#         newcolors[int(cast), :] = purp
# new_cmap = ListedColormap(newcolors)


# pfun.start_plot(fs=14, figsize=(10, 10))
# fig5, axes5 = plt.subplots(nrows=1, ncols=1, squeeze=False)
# axes5[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],d,cmap=new_cmap)
# pfun.add_coast(axes5[0,0])
# axes5[0,0].set_xlim([min_lon[0],max_lon[0]])
# axes5[0,0].set_ylim([min_lat[0],max_lat[0]])
# #axes5[0,0].set_xlabel('Casts')
# axes5[0,0].tick_params(labelrotation=45)


# #plt.plot(x_obs, y_obs, label='Casts')
# #plt.plot(x_LO, y_LO, label='LO Volumes')
# #plt.legend()
# axes5[0, 0].set_title('Volume-Cast-Method Domain')
# plt.savefig('/Users/dakotamascarenas/Desktop/pltz/domain.png')

# # %%

# pfun.start_plot(fs=14, figsize=(15, 10))
# fig1, axes1 = plt.subplots(nrows=1, ncols=1, squeeze=False)
# for i in range(len(ii_cast)):
#     axes1[0, 0].plot(oxygen[i], z_rho[i], 'o', c=cmap(i))
# axes1[0, 0].set_xlabel('Oxygen Concentration [mg/L]')
# axes1[0, 0].set_ylabel('Depth [m]')
# #plt.title(mon_str + ' DO Profiles')
# fig1.tight_layout()
# plt.savefig('/Users/dakotamascarenas/Desktop/pltz/DO_profiles_example.png')

#%%

# doesn't work yet!!!!


# cast_no = list(map(str, np.arange(len(ii_cast))))

# df0 = pd.DataFrame(oxygen[9])
# df0['cast_no'] = cast_no
# df0 = pd.melt(df0, id_vars=['cast_no'], var_name='sigma_level', value_name='DO')

# df1 = pd.DataFrame(z_rho[9])
# df1['cast_no'] = cast_no
# df1 = pd.melt(df1, id_vars=['cast_no'], var_name='sigma_level', value_name='z_rho')

# df = df0.merge(df1,on=['cast_no','sigma_level'])

# df_hyp = df[df['DO'] < hyp_val]

# hyp_casts = df_hyp['cast_no'].unique()

# hyp_casts = [int(l) for l in hyp_casts]

# cmap_colors = cmap.copy()
# newcolors = cmap_colors(np.linspace(0, 1, len(cast_no)))
# white = np.array([256/256, 256/256, 256/256, 1])
# for cast in range(len(cast_no)):
#     if cast not in hyp_casts:
#         newcolors[int(cast), :] = white
# new_cmap = ListedColormap(newcolors)

# pfun.start_plot(fs=14, figsize=(20,10))
# fig3, axes3 = plt.subplots(nrows=1, ncols=2, squeeze=False)
# axes3[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],d,cmap=new_cmap)
# for m in range(len(ii_cast)):
#     axes3[0,0].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
# axes3[0,0].set_xlim([min_lon[0],max_lon[0]])
# axes3[0,0].set_ylim([min_lat[0],max_lat[0]])
# axes3[0,0].set_xlabel('Casts')
# axes3[0,0].tick_params(labelrotation=45)
# pfun.add_coast(axes3[0,0])

# axes3[0,1].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],hyp_deep,vmin=0.1, vmax = 7,cmap='summer')
# for m in range(len(ii_cast)):
#     axes3[0,1].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
# axes3[0,1].set_xlim([min_lon[0],max_lon[0]])
# axes3[0,1].set_ylim([min_lat[0],max_lat[0]])
# axes3[0,1].set_xlabel('LO Volumes')
# axes3[0,1].tick_params(labelrotation=45)
# pfun.add_coast(axes3[0,1])

# plt.title(mon_str + ' <'+str(hyp_val)+'mg/L Bottom Areas')
# fig3.tight_layout()
# plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+mon_str+'_comp_hyp_areas.png')

# %%

# import pinfo
# from importlib import reload
# reload(pinfo)

# min_lat = [48, 48.4]
# max_lat = [49, 48.7]
# min_lon = [-124, -123.4]
# max_lon = [-122.25,-122.4]



# min_lon_sect = Lon[min(iii)]
# max_lon_sect = Lon[max(iii)]
# max_lat_sect = Lat[max(jjj)]



# for (mon_num, mon_str) in zip(month_num,month_str):

#     dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
#     fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
    
#     in_dict = {}

#     in_dict['fn'] = fn_his
    
    
#     ds = xr.open_dataset(fn_his)
#     # PLOT CODE
#     vn = 'oxygen'#'phytoplankton'
#     # if vn == 'salt':
#     #     pinfo.cmap_dict[vn] = 'jet'
#     # GET DATA
#     G, S, T = zrfun.get_basic_info(fn_his)
#     # CREATE THE SECTION
#     # create track by hand
#     lon = G['lon_rho']
#     lat = G['lat_rho']
#     zdeep = -350 
#     x = np.linspace(min_lon_sect, max_lon_sect, 10000)
#     y = max_lat_sect * np.ones(x.shape)
    
#     v2, v3, dist, idist0 = pfun.get_section(ds, vn, x, y, in_dict)
    

#     sf = pinfo.fac_dict[vn] * v3['sectvarf']

    
#     # PLOTTING
#     # map with section line
    
    
    
#     plt.close('all')
    
#     fs = 14
#     pfun.start_plot(fs=fs, figsize=(20,9))
#     fig6 = plt.figure()
    
#     ax6 = fig6.add_subplot(1, 3, 1)
#     cs = pfun.add_map_field(ax6, ds, vn, pinfo.vlims_dict,
#             cmap=pinfo.cmap_dict[vn], fac=pinfo.fac_dict[vn], do_mask_edges=True)
#     # fig.colorbar(cs, ax=ax) # It is identical to that of the section
#     pfun.add_coast(ax6)
#     aaf = [min_lon[1], max_lon[1], min_lat[1], max_lat[1]] # focus domain
#     ax6.axis(aaf)
#     pfun.dar(ax6)
#     pfun.add_info(ax6, fn_his, loc='upper_right')
#     ax6.set_title('Surface %s %s' % (pinfo.tstr_dict[vn],pinfo.units_dict[vn]))
#     ax6.set_xlabel('Longitude')
#     ax6.set_ylabel('Latitude')
#     # add section track
#     ax6.plot(x, y, '-r', linewidth=2)
#     ax6.plot(x[idist0], y[idist0], 'or', markersize=5, markerfacecolor='w',
#         markeredgecolor='r', markeredgewidth=2)
#     #ax6.set_xticks([-125, -124, -123])
#     #ax6.set_yticks([47, 48, 49, 50])
#     # section
#     ax6 = fig6.add_subplot(1, 3, (2, 3))
#     ax6.plot(dist, v2['zbot'], '-k', linewidth=2)
#     ax6.plot(dist, v2['zeta'], '-b', linewidth=1)
#     ax6.set_xlim(dist.min(), dist.max())
#     ax6.set_ylim(zdeep, 5)
#     # plot section
#     svlims = pinfo.vlims_dict[vn]
#     cs = ax6.pcolormesh(v3['distf'], v3['zrf'], sf,
#                         vmin=svlims[0], vmax=svlims[1], cmap=pinfo.cmap_dict[vn])
#     fig6.colorbar(cs, ax=ax6)
#     ax6.set_xlabel('Distance (km)')
#     ax6.set_ylabel('Z (m)')
#     ax6.set_title('Section %s %s' % (pinfo.tstr_dict[vn],pinfo.units_dict[vn]))
#     fig6.tight_layout()
#     # FINISH
#     ds.close()
#     pfun.end_plot()
#     plt.savefig('/Users/dakotamascarenas/Desktop/pltz/sect_LO_'+mon_str+'.png')
    
    
    # plt.close('all')
    
    # fs = 14
    # pfun.start_plot(fs=fs, figsize=(20,9))
    # fig7 = plt.figure()
    
    # ax7 = fig7.add_subplot(1, 3, 1)
    # cs = pfun.add_map_field(ax7, ds, vn, pinfo.vlims_dict,
    #         cmap=pinfo.cmap_dict[vn], fac=pinfo.fac_dict[vn], do_mask_edges=True)
    # # fig.colorbar(cs, ax=ax) # It is identical to that of the section
    # pfun.add_coast(ax6)
    # aaf = [min_lon[1], max_lon[1], min_lat[1], max_lat[1]] # focus domain
    # ax7.axis(aaf)
    # pfun.dar(ax7)
    # pfun.add_info(ax7, fn_his, loc='upper_right')
    # ax7.set_title('Surface %s %s' % (pinfo.tstr_dict[vn],pinfo.units_dict[vn]))
    # ax7.set_xlabel('Longitude')
    # ax7.set_ylabel('Latitude')
    # # add section track
    # ax7.plot(x, y, '-r', linewidth=2)
    # ax7.plot(x[idist0], y[idist0], 'or', markersize=5, markerfacecolor='w',
    #     markeredgecolor='r', markeredgewidth=2)
    # #ax6.set_xticks([-125, -124, -123])
    # #ax6.set_yticks([47, 48, 49, 50])
    # # section
    # ax7 = fig7.add_subplot(1, 3, (2, 3))
    # ax7.plot(dist, v2['zbot'], '-k', linewidth=2)
    # ax7.plot(dist, v2['zeta'], '-b', linewidth=1)
    # ax7.set_xlim(dist.min(), dist.max())
    # ax7.set_ylim(zdeep, 5)
    # # plot section
    # svlims = pinfo.vlims_dict[vn]
    
    # for
    
    # sf_new = 
    
    # cs = ax7.pcolormesh(v3['distf'], v3['zrf'], sf,
    #                    vmin=svlims[0], vmax=svlims[1], cmap=pinfo.cmap_dict[vn])
    # fig7.colorbar(cs, ax=ax6)
    # ax7.set_xlabel('Distance (km)')
    # ax7.set_ylabel('Z (m)')
    # ax7.set_title('Section %s %s' % (pinfo.tstr_dict[vn],pinfo.units_dict[vn]))
    # fig7.tight_layout()
    # # FINISH
    # ds.close()
    # pfun.end_plot()
    # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/sect_LO_'+mon_str+'.png')



# %%

# find iii/jjj indices of weird cells

dt = pd.Timestamp('2022-Jan-01 01:30:00')

