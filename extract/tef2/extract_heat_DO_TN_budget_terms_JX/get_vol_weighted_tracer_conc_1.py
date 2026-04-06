# get volume-weighted tracer concentration
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from lo_tools import zfun, zrfun, Lfun
import scipy, pickle, sys
from datetime import datetime, timedelta
from time import time
import pandas as pd

#% load salish sea index
seg_name = 'seg_info_dict_cas7_c2_noriv.p'
seg_df = pd.read_pickle(seg_name)
ji_list = seg_df['sog6_m']['ji_list']
jj = [x[0] for x in ji_list]
ii = [x[1] for x in ji_list]

#
Ldir = Lfun.Lstart()
Ldir['roms_out'] = Ldir['roms_out2']
Ldir['gtagex'] = 'cas7_t0_x4b'

ds0 = '2014.01.01'
ds1 = '2014.01.31'
Ldir['ds0'] = ds0
in_dir = Ldir['roms_out'] / Ldir['gtagex']
G, S, T = zrfun.get_basic_info(in_dir / ('f' + Ldir['ds0']) / 'ocean_his_0002.nc')

fn0 = xr.open_dataset(in_dir / ('f' + Ldir['ds0']) / 'ocean_his_0002.nc')
dx = 1/fn0.pm.values
dy = 1/fn0.pn.values
area = dx * dy
NX, NY = dx.shape

dt0 = datetime.strptime(ds0, Lfun.ds_fmt)
dt1 = datetime.strptime(ds1, Lfun.ds_fmt)
dt00 = dt0

# initialization
t = []
TN_all_depth = [] # includes the whole water column
TN_shallow = [] # shallower than 20m, for example
TN_deep = [] # deeper than 20m
TN_surface = [] # surface layer
TN_bottom = [] # bottom layer
NO3_all_depth = []
NO3_shallow = []
NO3_deep = []
NO3_surface = [] # surface layer
NO3_bottom = [] # bottom layer
NH4_all_depth = []
NH4_shallow = []
NH4_deep = []
NH4_surface = []
NH4_bottom = []
temp_all_depth = []
temp_shallow = []
temp_deep = []
temp_surface = []
temp_bottom = []
salt_all_depth = []
salt_shallow = []
salt_deep = []
salt_surface = []
salt_bottom = []
oxy_all_depth = []
oxy_shallow = [] 
oxy_deep = []
oxy_surface = []
oxy_bottom = []

z0 = -20 # meter

while dt00 <= dt1:
    print(dt00)
    sys.stdout.flush()
    ds00 = dt00.strftime(Lfun.ds_fmt)
    fn_list = Lfun.get_fn_list('hourly', Ldir, ds00, ds00)
    
    for fn in fn_list[0:-1]: 
        ds_his = xr.open_dataset(fn)
        h    = ds_his.h.values
        zeta = ds_his.zeta.values.squeeze()
        zw   = zrfun.get_z(h, zeta, S, only_w=True)
        z_rho = zrfun.get_z(h, zeta, S, only_rho=True)
        dz  = np.diff(zw, axis=0)
        vol = dx*dy*dz
        
        z_rho0 = z_rho.copy()
        z_rho0 = z_rho0 - z_rho0[-1,:,:] # adjust zero meter to free surface
        
        #-------- TN --------
        Phyt  = ds_his.phytoplankton.values.squeeze()
        Zoop  = ds_his.zooplankton.values.squeeze()
        NO3   = ds_his.NO3.values.squeeze()
        NH4   = ds_his.NH4.values.squeeze()
        LdetN = ds_his.LdetritusN.values.squeeze()
        SdetN = ds_his.SdetritusN.values.squeeze() 
        TN = Phyt + Zoop + NO3 + NH4 + LdetN + SdetN             
        # volume weighted TN concentration
        TN_all_depth.append((np.nansum(TN[:,jj,ii] * vol[:,jj,ii])) / np.nansum(vol[:,jj,ii]))
        TN_surface.append((np.nansum(TN[-1,jj,ii] * vol[-1,jj,ii])) / np.nansum(vol[-1,jj,ii]))
        TN_bottom.append((np.nansum(TN[0,jj,ii] * vol[0,jj,ii])) / np.nansum(vol[0,jj,ii]))
        # volume weighted NO3 concentration
        NO3_all_depth.append((np.nansum(NO3[:,jj,ii] * vol[:,jj,ii])) / np.nansum(vol[:,jj,ii]))
        NO3_surface.append((np.nansum(NO3[-1,jj,ii] * vol[-1,jj,ii])) / np.nansum(vol[-1,jj,ii]))
        NO3_bottom.append((np.nansum(NO3[0,jj,ii] * vol[0,jj,ii])) / np.nansum(vol[0,jj,ii]))
        # volume weighted NH4 concentration
        NH4_all_depth.append((np.nansum(NH4[:,jj,ii] * vol[:,jj,ii])) / np.nansum(vol[:,jj,ii]))
        NH4_surface.append((np.nansum(NH4[-1,jj,ii] * vol[-1,jj,ii])) / np.nansum(vol[-1,jj,ii]))
        NH4_bottom.append((np.nansum(NH4[0,jj,ii] * vol[0,jj,ii])) / np.nansum(vol[0,jj,ii]))
        
        #-------- temperature --------
        temp = ds_his.temp.values.squeeze()        
        temp_all_depth.append((np.nansum(temp[:,jj,ii] * vol[:,jj,ii])) / (np.nansum(vol[:,jj,ii])))
        temp_surface.append((np.nansum(temp[-1,jj,ii] * vol[-1,jj,ii])) / (np.nansum(vol[-1,jj,ii])))
        temp_bottom.append((np.nansum(temp[0,jj,ii] * vol[0,jj,ii])) / (np.nansum(vol[0,jj,ii])))
        
        #-------- salt --------
        salt = ds_his.salt.values.squeeze()
        salt_all_depth.append((np.nansum(salt[:,jj,ii] * vol[:,jj,ii])) / (np.nansum(vol[:,jj,ii])))
        salt_surface.append((np.nansum(salt[-1,jj,ii] * vol[-1,jj,ii])) / (np.nansum(vol[-1,jj,ii])))
        salt_bottom.append((np.nansum(salt[0,jj,ii] * vol[0,jj,ii])) / (np.nansum(vol[0,jj,ii])))
        
        #--------- oxygen -----------
        oxy = ds_his.oxygen.values.squeeze()
        oxy_all_depth.append((np.nansum(oxy[:,jj,ii] * vol[:,jj,ii])) / (np.nansum(vol[:,jj,ii])))
        oxy_surface.append((np.nansum(oxy[-1,jj,ii] * vol[-1,jj,ii])) / (np.nansum(vol[-1,jj,ii])))
        oxy_bottom.append((np.nansum(oxy[0,jj,ii] * vol[0,jj,ii])) / (np.nansum(vol[0,jj,ii])))
       
        #------ shallow and deep: tracer conc * vol --------
        oxy_vol_shallow = np.zeros(len(jj))
        oxy_vol_deep    = np.zeros(len(jj))
        temp_vol_shallow = np.zeros(len(jj))
        temp_vol_deep    = np.zeros(len(jj))
        salt_vol_shallow = np.zeros(len(jj))
        salt_vol_deep    = np.zeros(len(jj))
        TN_vol_shallow = np.zeros(len(jj))
        TN_vol_deep    = np.zeros(len(jj))
        NO3_vol_shallow = np.zeros(len(jj))
        NO3_vol_deep    = np.zeros(len(jj))
        NH4_vol_shallow = np.zeros(len(jj))
        NH4_vol_deep    = np.zeros(len(jj))
        
        vol_shallow = np.zeros(len(jj))
        vol_deep    = np.zeros(len(jj))
        
        for k in range(len(jj)): # loop inside the domain, e.g. Salish Sea
            index_s = np.argwhere(z_rho0[:,jj[k],ii[k]]>z0) # shallow than z0, z_rho0 is negative value
            oxy_vol_shallow[k]  = np.sum(oxy[index_s,jj[k],ii[k]] * vol[index_s,jj[k],ii[k]])
            TN_vol_shallow[k]   = np.sum(TN[index_s,jj[k],ii[k]]  * vol[index_s,jj[k],ii[k]])
            NO3_vol_shallow[k]  = np.sum(NO3[index_s,jj[k],ii[k]] * vol[index_s,jj[k],ii[k]])
            NH4_vol_shallow[k]  = np.sum(NH4[index_s,jj[k],ii[k]] * vol[index_s,jj[k],ii[k]])
            temp_vol_shallow[k] = np.sum(temp[index_s,jj[k],ii[k]] * vol[index_s,jj[k],ii[k]])
            salt_vol_shallow[k] = np.sum(salt[index_s,jj[k],ii[k]] * vol[index_s,jj[k],ii[k]])
            vol_shallow[k] = np.sum(vol[index_s,jj[k],ii[k]])
            
            index_d = np.argwhere(z_rho0[:,jj[k],ii[k]]<=z0) # deeper than z0
            oxy_vol_deep[k]  = np.sum(oxy[index_d,jj[k],ii[k]] * vol[index_d,jj[k],ii[k]])
            TN_vol_deep[k]   = np.sum(TN[index_d,jj[k],ii[k]]  * vol[index_d,jj[k],ii[k]])
            NO3_vol_deep[k]  = np.sum(NO3[index_d,jj[k],ii[k]] * vol[index_d,jj[k],ii[k]])
            NH4_vol_deep[k]  = np.sum(NH4[index_d,jj[k],ii[k]] * vol[index_d,jj[k],ii[k]])
            temp_vol_deep[k] = np.sum(temp[index_d,jj[k],ii[k]] * vol[index_d,jj[k],ii[k]])
            salt_vol_deep[k] = np.sum(salt[index_d,jj[k],ii[k]] * vol[index_d,jj[k],ii[k]])
            vol_deep[k] = np.sum(vol[index_d,jj[k],ii[k]])
            
        oxy_vol_deep[vol_deep==0] = np.nan # no "deep layer" in shallow regions
        TN_vol_deep[vol_deep==0]  = np.nan
        NO3_vol_deep[vol_deep==0] = np.nan
        NH4_vol_deep[vol_deep==0] = np.nan
        temp_vol_deep[vol_deep==0] = np.nan
        salt_vol_deep[vol_deep==0] = np.nan
        vol_deep[vol_deep==0] = np.nan
        
        oxy_shallow.append(np.sum(oxy_vol_shallow)/np.sum(vol_shallow))
        oxy_deep.append(np.nansum(oxy_vol_deep)/np.nansum(vol_deep))
        TN_shallow.append(np.sum(TN_vol_shallow)/np.sum(vol_shallow))
        TN_deep.append(np.nansum(TN_vol_deep)/np.nansum(vol_deep))
        NO3_shallow.append(np.sum(NO3_vol_shallow)/np.sum(vol_shallow))
        NO3_deep.append(np.nansum(NO3_vol_deep)/np.nansum(vol_deep))
        NH4_shallow.append(np.sum(NH4_vol_shallow)/np.sum(vol_shallow))
        NH4_deep.append(np.nansum(NH4_vol_deep)/np.nansum(vol_deep))
        temp_shallow.append(np.sum(temp_vol_shallow)/np.sum(vol_shallow))
        temp_deep.append(np.nansum(temp_vol_deep)/np.nansum(vol_deep))
        salt_shallow.append(np.sum(salt_vol_shallow)/np.sum(vol_shallow))
        salt_deep.append(np.nansum(salt_vol_deep)/np.nansum(vol_deep))
        
        t.append(ds_his.ocean_time.values)
        
    dt00 = dt00 + timedelta(days=1)

# save
dict_tmp = {'t': t,
            'TN_all_depth':    TN_all_depth,
            'TN_shallow':      TN_shallow,
            'TN_deep':         TN_deep,
            'TN_surface':      TN_surface,
            'TN_bottom':       TN_bottom,
            'NO3_all_depth':   NO3_all_depth,
            'NO3_shallow':     NO3_shallow,
            'NO3_deep':        NO3_deep,
            'NO3_surface':     NO3_surface,
            'NO3_bottom':      NO3_bottom,
            'NH4_all_depth':   NH4_all_depth,
            'NH4_shallow':     NH4_shallow,
            'NH4_deep':        NH4_deep,
            'NH4_surface':     NH4_surface,
            'NH4_bottom':      NH4_bottom,
            'temp_all_depth':  temp_all_depth,
            'temp_shallow':    temp_shallow,
            'temp_deep':       temp_deep,
            'temp_surface':    temp_surface,
            'temp_bottom':     temp_bottom,
            'salt_all_depth':  salt_all_depth,
            'salt_shallow':    salt_shallow,
            'salt_deep':       salt_deep,
            'salt_surface':    salt_surface,
            'salt_bottom':     salt_bottom,
            'oxy_all_depth':   oxy_all_depth,
            'oxy_shallow':     oxy_shallow,
            'oxy_deep':        oxy_deep,
            'oxy_surface':     oxy_surface,
            'oxy_bottom':      oxy_bottom,
           }
pickle.dump(dict_tmp, open("vol_weighted_tracer_conc_"+ds0+'_'+Ldir['gtagex']+'.p','wb'))
