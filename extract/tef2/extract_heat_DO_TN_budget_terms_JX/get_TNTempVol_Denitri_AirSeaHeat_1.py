# get TN*vol, Temp*vol, sediment TN loss, air-sea heat flux
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from lo_tools import zfun, zrfun, Lfun
import scipy, pickle, sys
from datetime import datetime, timedelta
from time import time
import pandas as pd

tt0 = time()

#% load salish sea j,i
seg_name = 'seg_info_dict_cas7_c2_noriv.p'
seg_df   = pd.read_pickle(seg_name)
ji_list  = seg_df['sog6_m']['ji_list']
jj = [x[0] for x in ji_list]
ii = [x[1] for x in ji_list]

#
Ldir = Lfun.Lstart()
Ldir['roms_out'] = Ldir['roms_out2']
Ldir['gtagex'] = 'cas7_t0_x4b'

ds0 = '2014.01.01'
ds1 = '2014.01.31'
Ldir['ds0'] = ds0
in_dir  = Ldir['roms_out'] / Ldir['gtagex']
G, S, T = zrfun.get_basic_info(in_dir / ('f' + Ldir['ds0']) / 'ocean_his_0002.nc')

fn0 = xr.open_dataset(in_dir / ('f' + Ldir['ds0']) / 'ocean_his_0002.nc')
dx  = 1/fn0.pm.values
dy  = 1/fn0.pn.values
area = dx * dy
NX, NY = dx.shape

dt0 = datetime.strptime(ds0, Lfun.ds_fmt)
dt1 = datetime.strptime(ds1, Lfun.ds_fmt)
dt00 = dt0

# settling velocity
Ws_L = 80 #m/d
Ws_S = 8 # m/d 
rOxNH4 = 106/16
t = []
TN_vol   = []
temp_vol = []
denitri_flux_sum  = []
detritus_loss_sum = []
NH4_gain_flux_sum = []
shflux_sum = []
latent_sum = []
sensible_sum = []
lwrad_sum = []
swrad_sum = []

while dt00 <= dt1:
    print(dt00)
    sys.stdout.flush()
    ds00 = dt00.strftime(Lfun.ds_fmt)
    fn_list = Lfun.get_fn_list('hourly', Ldir, ds00, ds00)
    
    for fn in fn_list[0:-1]:
        ds_his = xr.open_dataset(fn)
        #print(fn)
        h    = ds_his.h.values
        zeta = ds_his.zeta.values.squeeze()
        zw   = zrfun.get_z(h, zeta, S, only_w=True)
        dz   = np.diff(zw, axis=0)
        vol  = dx*dy*dz
        #-------- TN * vol --------
        Phyt  = ds_his.phytoplankton.values.squeeze()
        Zoop  = ds_his.zooplankton.values.squeeze()
        NO3   = ds_his.NO3.values.squeeze()
        NH4   = ds_his.NH4.values.squeeze()
        LdetN = ds_his.LdetritusN.values.squeeze()
        SdetN = ds_his.SdetritusN.values.squeeze() 
        TN = Phyt + Zoop + NO3 + NH4 + LdetN + SdetN
        TN_vol.append(np.nansum(TN[:,jj,ii] * vol[:,jj,ii])) # TN*vol, mmol N
        
        #-------- temperature * vol --------
        temp = ds_his.temp.values.squeeze()
        temp_vol.append(np.nansum(temp[:,jj,ii] * vol[:,jj,ii])) # temp * vol
        
        #-------- sediment denitrification --------
        LdetN_bot = ds_his.LdetritusN.values[0,0,:,:]  # bottom LdetritusN
        SdetN_bot = ds_his.SdetritusN.values[0,0,:,:]  
        Oxy_bot   = ds_his.oxygen.values[0,0,:,:]
        NO3_bot   = ds_his.NO3.values[0,0,:,:]
        
        NO3loss = 1.2 * (1/24) /dz[0,:,:] # 1.2 mmol/m2/day * day * (1/m) --> mmol/m3
              
        # SdetritusN
        FC_S = (SdetN_bot * Ws_S)/24    # mmol/m2/hr
        cff1_S = FC_S / dz[0,:,:] * 1   # mmol/m2/hr --> mmol/m3
        denitri_S  = np.zeros([NX,NY])  # mmol/m3, loss of NO3 to N2
        NH4_gain_S = np.zeros([NX,NY])
        for i in range(NX): # loop the whole domain
            for j in range(NY):
                if cff1_S[i,j]*rOxNH4 > Oxy_bot[i,j]: # directly go to denitrification if DO will be fully consumed
                    denitri_S[i,j] = min(NO3_bot[i,j], cff1_S[i,j])                  
                else: # use O2: PN + O2 --> NH4
                    denitri_S[i,j]  = 0
                    NH4_gain_S[i,j] = cff1_S[i,j]  # mmol/m3, NH4 gain from decomposing SdetN
                    if cff1_S[i,j] > NO3loss[i,j]: # denitrification: PN + NO3 --> N2
                        denitri_S[i,j] = min(NO3_bot[i,j], NO3loss[i,j])                                                
        denitri_flux_S  = denitri_S  * area * dz[0,:,:] # mmol, but the actual unit is mmol/hr
        NH4_gain_flux_S = NH4_gain_S * area * dz[0,:,:]
        
        # LdetritusN
        Oxy_bottom2 = Oxy_bot - NH4_gain_S*106/16 # decrease of bottom O2 from SdetritusN decomposition
        NO3_bottom2 = NO3_bot - denitri_S # decrease of bottom NO3 from SdetritusN decomposition
        FC_L   = (LdetN_bot * Ws_L)/24 # mmol/m2/hr
        cff1_L = FC_L / dz[0,:,:] * 1  # mmol/m2/hr --> mmol/m3
        denitri_L  = np.zeros([NX,NY]) # mmol/m3, loss of NO3 to N2
        NH4_gain_L = np.zeros([NX,NY])
        for i in range(NX): # loop the whole domain
            for j in range(NY):
                if cff1_L[i,j]*rOxNH4 > Oxy_bottom2[i,j]: # directly go to denitrification if DO will be fully consumed
                    denitri_L[i,j] = min(NO3_bottom2[i,j], cff1_L[i,j])                  
                else: # use O2: PN + O2 --> NH4
                    denitri_L[i,j]  = 0
                    NH4_gain_L[i,j] = cff1_L[i,j] #mmol/m3, NH4 gain from decomposing LdetN
                    if cff1_L[i,j] > NO3loss[i,j]: # denitr: PN + NO3 --> N2
                        denitri_L[i,j] = min(NO3_bottom2[i,j], NO3loss[i,j])                                                
        denitri_flux_L  = denitri_L  * area * dz[0,:,:] # mmol, but the actual unit is mmol/hr
        NH4_gain_flux_L = NH4_gain_L * area * dz[0,:,:]
                   
        denitri_flux  = denitri_flux_L + denitri_flux_S
        NH4_gain_flux = NH4_gain_flux_L + NH4_gain_flux_S
             
        denitri_flux_sum.append(np.nansum(denitri_flux[jj,ii]))   # denitrification loss of NO3
        NH4_gain_flux_sum.append(np.nansum(NH4_gain_flux[jj,ii])) # remineralization gain of NH4
        detritus_loss_sum.append(np.nansum((FC_L[jj,ii]+FC_S[jj,ii])*area[jj,ii])) # mmol/hr, loss of PN
        
        #-------- air-sea heat flux --------
        shflux   = ds_his.shflux.values.squeeze()
        latent   = ds_his.latent.values.squeeze()
        sensible = ds_his.sensible.values.squeeze()
        lwrad    = ds_his.lwrad.values.squeeze()
        swrad    = ds_his.swrad.values.squeeze()
        
        shflux_sum.append(np.nansum(area[jj,ii] * shflux[jj,ii]))  # watts (=J/s)
        latent_sum.append(np.nansum(area[jj,ii] * latent[jj,ii]))
        sensible_sum.append(np.nansum(area[jj,ii] * sensible[jj,ii]))
        lwrad_sum.append(np.nansum(area[jj,ii] * lwrad[jj,ii]))
        swrad_sum.append(np.nansum(area[jj,ii] * swrad[jj,ii]))
        
        t.append(ds_his.ocean_time.values)
        
    dt00 = dt00 + timedelta(days=1)
    
dict_tmp = {'t': t, 
            'TN_vol':   TN_vol,
            'temp_vol': temp_vol,
            'denitri_flux_sum':  denitri_flux_sum,
            'NH4_gain_flux_sum': NH4_gain_flux_sum,
            'detritus_loss': detritus_loss_sum,
            'shflux_sum':    shflux_sum,
            'swrad_sum':     swrad_sum,
            'lwrad_sum':     lwrad_sum,
            'latent_sum':    latent_sum,
            'sensible_sum':  sensible_sum
            }
pickle.dump(dict_tmp, open("TNTempVol_Denitri_AirSeaHeat"+ds0+'_'+Ldir['gtagex']+'.p','wb'))

print('total time = %0.1f sec' % (time()-tt0))
