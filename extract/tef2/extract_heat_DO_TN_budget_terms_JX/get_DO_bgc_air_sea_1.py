# calculate O2 production and comsumption from bgc process, and air-sea oxygen exchange
# O2 production from new (NO3) and regenrated production(NH4)
# O2 consumption in water column (nitrification + remi) + sediment (denitrification)
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from lo_tools import zfun, zrfun, Lfun
from datetime import datetime, timedelta
import scipy, pickle, sys
import pandas as pd

#%------------------------------------------------
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
lonr = fn0.lon_rho.values
latr = fn0.lat_rho.values
area = dx * dy
NX, NY = dx.shape

# We increased AttSW from 0.05 m-1 to 0.15 m-1 inside Salish Sea
AttSW = np.zeros((NX, NY)) + 0.05
AttSW[(lonr > -123.89) & (latr < 50.29) & (latr > 47.02)] = 0.15
AttSW[(lonr > -125.31) & (lonr < -123.89) & (latr < 51.02) & (latr > 49.13)] = 0.15

dt0 = datetime.strptime(ds0, Lfun.ds_fmt)
dt1 = datetime.strptime(ds1, Lfun.ds_fmt)
dt00 = dt0

#% load salish sea index
seg_name = 'seg_info_dict_cas7_c2_noriv.p'
seg_df = pd.read_pickle(seg_name)
ji_list = seg_df['sog6_m']['ji_list']
jj = [x[0] for x in ji_list]
ii = [x[1] for x in ji_list]

inDomain = np.zeros([NX, NY])
inDomain[jj,ii] = 1 # inside Salish, index=1, outside, index =0, this step is to reduce the size of spatial variable file (hopefully)

#% air-sea flux related parameters
#rho0 = 1025.0 # ROMS/Modules/mod_scalars.F
#sec2day = 1.0/86400
#dt = 3600 # sec, hourly history file
#BioIter = 1
#dtdays = dt * sec2day / BioIter
dtdays = 3600 * (1.0/86400) / 1
#cff1 = rho0 * 550.0
#cff1 = 1025.0 * 550.0
#RW14_OXYGEN_SC = False
#if RW14_OXYGEN_SC:   # cff2: s2/m
#    cff2 = dtdays * 0.251 * 24 / 100  # 0.251: (cm/h)(m/s)^(-2), 
#else:
#    cff2 = dtdays * 0.31 * 24 / 100
cff2_air = dtdays * 0.31 * 24 / 100
## formulation for Schmidt number coefficient
#OCMIP_OXYGEN_SC = False
#RW14_OXYGEN_SC = False
#if OCMIP_OXYGEN_SC: # Keeling et al., 1998
#    A_O2 = 1638.0; B_O2 = 81.83; C_O2 = 1.483
#    D_O2 = 0.008004; E_O2 = 0.0
#elif RW14_OXYGEN_SC: # Wanninkhof, 2014
#    A_O2 = 1920.4; B_O2 = 135.6; C_O2 = 5.2122
#    D_O2 = 0.10939; E_O2 = 0.00093777
#else: # Wanninkhof, 1992  # we used this option
#    A_O2 = 1953.4; B_O2 = 128.0; C_O2 = 3.9918
#    D_O2 = 0.050091; E_O2 = 0.0
A_O2 = 1953.4;   B_O2 = 128.0; C_O2 = 3.9918
D_O2 = 0.050091; E_O2 = 0.0
# Calculate O2 saturation concentration using Garcia and Gordon
#  L and O (1992) formula, (EXP(AA) is in ml/l).
OA0 = 2.00907       # Oxygen saturation coefficients
OA1 = 3.22014;      OA2 = 4.05010;       OA3 = 4.94457
OA4 = -0.256847;    OA5 = 3.88767;       OB0 = -0.00624523
OB1 = -0.00737614;  OB2 = -0.0103410;    OB3 = -0.00817083
OC0 = -0.000000488682    

# light attenuation
#PARfrac = 0.43
#Cp = 3985 # Joules/kg/degC
#rho0 = 1025 # kg/m3
#AttSW = 0.05
#AttChl = 0.012
#Vp = 1.7;
#PhyIS = 0.07 # initial slope of P-I curve [1/(Watts m-2 day)]
#rOxNO3 = 138/16
#rOxNH4 = 106/16
#K_NO3 = 10 # [1/(millimole_N m-3)]
#K_NH4 = 10
#SDeRRN = 0.1 # Small detritus remineralization rate N-fraction [1/day]
#LDeRRN = 0.1
#NitriR = 0.05 # Nitrification rate: oxidation of NH4 to NO3 [1/day]

#Ws_L = 80 # m/d
#Ws_S = 8 # m/d

# initialization
# these variables were integrated over the domain
Oxy_sed_sum  = [] # the 1st way to calculate SOD
Oxy_sed_sum2 = [] # the 2nd way to calculate SOD
Oxy_pro_sum  = []
Oxy_nitri_sum = []
Oxy_remi_sum  = []
Oxy_vol_sum   = []  # DO * vol
Oxy_air_flux_sum = []
t = []

# these variables save the 3D information; I run this code for each month, so the variable dimension is slightly smaller than 24 (hourly history file) * 32 (days)
Oxy_pro_spatial   = np.zeros((24*32, NX, NY))
Oxy_nitri_spatial = np.zeros((24*32, NX, NY))
Oxy_remi_spatial  = np.zeros((24*32, NX, NY))
Oxy_sed_spatial   = np.zeros((24*32, NX, NY))
Oxy_sed_spatial2  = np.zeros((24*32, NX, NY))
Oxy_air_flux_spatial = np.zeros((24*32, NX, NY))
diff_O2_spatial   = np.zeros((24*32, NX, NY))

cnt = 0
#%%
while dt00 <= dt1:  # loop each day and every history file
    print(dt00)
    sys.stdout.flush()
    ds00 = dt00.strftime(Lfun.ds_fmt)
    fn_list = Lfun.get_fn_list('hourly', Ldir, ds00, ds00)
    #%%
    for fn in fn_list[0:-1]: 
        ds    = xr.open_dataset(fn)
        swrad = ds.swrad.values.squeeze()   # w/m2
        chl   = ds.chlorophyll.values.squeeze()
        zeta  = ds.zeta.values.squeeze()
        h     = ds.h.values
        zw = zrfun.get_z(h, zeta, S, only_w=True)
        dz = np.diff(zw, axis=0)
        salt = ds.salt.values.squeeze()
        NH4  = ds.NH4.values.squeeze()
        NO3  = ds.NO3.values.squeeze()
        phy  = ds.phytoplankton.values.squeeze()
        SDeN = ds.SdetritusN.values.squeeze()
        LDeN = ds.LdetritusN.values.squeeze()
        Oxy  = ds.oxygen.values.squeeze()

        # PARsur = PARfrac * swrad #* rho0 * Cp # surface PAR, watts/m2
        Att = np.zeros(salt.shape)
        # ExpAtt = np.zeros(Att.shape)
        nk,ni,nj = Att.shape
        PAR = np.zeros([nk+1, ni, nj])
        PAR[-1,:,:] = 0.43 * swrad   # PAR[-1,:,:] = PARsur
#
        Oxy_pro   = np.zeros(Att.shape) # O2 production
        Oxy_nitri = np.zeros(Att.shape) # O2 consumption by nitrification in water column
        Oxy_remi  = np.zeros(Att.shape) # O2 consumption by remineralization in water column
        Oxy_sed   = np.zeros([ni,nj])   # O2 consumption by SOD
        
        z_w = zrfun.get_z(h, zeta, S, only_rho=False, only_w=True) # the same as zw...
        vol = np.diff(z_w,axis=0) * area # grid cell volume
        
        #Oxy_vol = Oxy * vol * stat # only account for Salish Sea
        #Oxy_vol_sum.append(np.nansum(Oxy_vol))
        Oxy_vol_sum.append(np.nansum(Oxy[:,jj,ii] * vol[:,jj,ii]))
        
        #if np.nanmin(PARsur) > 0: # doing photosysnthesis
        if np.nanmin(0.43*swrad) > 0: # doing photosysnthesis
            for k in np.arange(29,-1,-1): # surface layer to bottom layer
                # O2 production from photosynthesis, mmol O2/m3/hr ?
                # dz = z_w[k+1,:,:] - z_w[k,:,:]
                # Att[k,:,:] = (AttSW + AttChl*chl[k,:,:] - 0.0065*(salt[k,:,:]-32))*dz
                Att[k,:,:] = (AttSW + 0.012*chl[k,:,:] - 0.0065*(salt[k,:,:]-32))* (z_w[k+1,:,:] - z_w[k,:,:])                       
                # ExpAtt[k,:,:] = np.exp(-Att[k,:,:])
                # Itop = PAR[k+1,:,:]
                # PAR[k,:,:] = Itop * (1-ExpAtt[k,:,:])/Att[k,:,:]
                PAR[k,:,:] = PAR[k+1,:,:] * (1-np.exp(-Att[k,:,:]))/Att[k,:,:] # average at cell center
                fac1 = PAR[k,:,:] * 0.07  # fac1 = PAR[k,:,:] * PhyIS
                Epp  = 1.7/np.sqrt(1.7*1.7 + fac1*fac1) # Epp = Vp/np.sqrt(Vp*Vp + fac1*fac1)
                t_PPmax = Epp * fac1
                cff1 = NH4[k,:,:] * 10; cff1[cff1<0] = 0 #cff1 = NH4[k,:,:] * K_NH4;
                cff2 = NO3[k,:,:] * 10; cff2[cff2<0] = 0 #cff2 = NO3[k,:,:] * K_NO3;
                inhNH4 = 1.0/(1.0+cff1)

                # fac1 = dtdays * t_PPmax
                # cff4 = fac1 * K_NO3 * inhNH4/(1.0 + cff2 + 2.0*np.sqrt(cff2)) * phy[k,:,:]
                cff4 = dtdays * t_PPmax * 10 * inhNH4/(1.0 + cff2 + 2.0*np.sqrt(cff2)) * phy[k,:,:]
                # cff5 = fac1 * K_NH4 / (1.0 + cff1 + 2.0 * np.sqrt(cff1)) * phy[k,:,:]
                cff5 = dtdays * t_PPmax * 10 / (1.0 + cff1 + 2.0 * np.sqrt(cff1)) * phy[k,:,:]
                # N_Flux_NewProd = NO3[k,:,:] * cff4
                # N_Flux_RegProd = NH4[k,:,:] * cff5
                # Oxy_pro[k,:,:] = N_Flux_NewProd * rOxNO3 + N_Flux_RegProd * rOxNH4
                Oxy_pro[k,:,:]   = NO3[k,:,:] * cff4 * 138/16 + NH4[k,:,:] * cff5 * 106/16

                # O2 comsumption by nitrification in water column, mmol O2/m3/hr ?
                fac2 = Oxy[k,:,:];    fac2[fac2<0] = 0
                fac3 = fac2/(3+fac2); fac3[fac3<0] = 0
                # fac1 = dtdays * NitriR * fac3
                # cff3 = fac1
                # N_Flux_Nitrifi = NH4[k,:,:] * cff3
                # Oxy_nitri[k,:,:] = 2.0 * N_Flux_Nitrifi
                Oxy_nitri[k,:,:] = 2.0 * NH4[k,:,:] * dtdays * 0.05 * fac3
              
                # PAR[k,:,:] = Itop * ExpAtt[k,:,:]  # !!! light attenuation at the bottom of grid cell !!!!
                PAR[k,:,:] = PAR[k+1,:,:] * np.exp(-Att[k,:,:])  # !!! light attenuation at the bottom of grid cell !!!!
            
        else: # PARsur = 0, nitrification occurs at the maximum rate (NitriR)
            # cff3 = dtdays * NitriR
            # cff3 = dtdays * 0.05
            for k in np.arange(29,-1,-1):
                # N_Flux_Nitrifi = NH4[k,:,:] * cff3
                # Oxy_nitri[k,:,:] = 2.0 * N_Flux_Nitrifi
                Oxy_nitri[k,:,:] = 2.0 * NH4[k,:,:] * dtdays * 0.05
            
        #O2 consumption by remineralization in water column, mmol O2/m3/hr ??????????????
        for k in np.arange(0,30):
            # fac1 = 0; fac2 = 1
            # cff1 = dtdays * SDeRRN * fac2
            # cff3 = dtdays * LDeRRN * fac2
            # N_Flux_RemineS = SDeN[k,:,:]*cff1
            # N_Flux_RemineL = LDeN[k,:,:]*cff3
            # Oxy_remi[k,:,:] = (N_Flux_RemineS + N_Flux_RemineL)*rOxNH4
            Oxy_remi[k,:,:] = (SDeN[k,:,:]*dtdays*0.1*1 + LDeN[k,:,:]*dtdays*0.1*1) * 106/16

        # mmol O2/m3/hr to mmol O2/hr
        Oxy_pro   = Oxy_pro * vol
        Oxy_nitri = Oxy_nitri * vol #
        Oxy_remi  = Oxy_remi * vol #
        Oxy_pro_sum.append(np.nansum(Oxy_pro[:,jj,ii]))  # only save Salish Sea
        Oxy_nitri_sum.append(np.nansum(Oxy_nitri[:,jj,ii]))
        Oxy_remi_sum.append(np.nansum(Oxy_remi[:,jj,ii]))
        
        Oxy_pro_spatial[cnt,:,:]   = np.nansum(Oxy_pro * inDomain, axis=0) # vertical sum
        Oxy_nitri_spatial[cnt,:,:] = np.nansum(Oxy_nitri * inDomain, axis=0)
        Oxy_remi_spatial[cnt,:,:]  = np.nansum(Oxy_remi * inDomain, axis=0)
        
        #---------- sediment SOD, 1st way ----------
        # refer to Parker's code https://github.com/parkermac/LPM/blob/main/extract/moor/benthic_flux.py
        # LDeN_bot = LDeN[0,:,:]
        # SDeN_bot = SDeN[0,:,:]
        # F_Det = LDeN_bot * Ws_L + SDeN_bot * Ws_S # mmol N/m2/d
        # F_Det = LDeN_bot * 80 + SDeN_bot * 8 # mmol N/m2/d
        F_Det = LDeN[0,:,:] * 80 + SDeN[0,:,:] * 8 # mmol N/m2/d
        F_NO3 = F_Det.copy()
        F_NO3[F_Det > 1.2] = 1.2 # 1.2 mmol N/m2/d
        F_NH4 = F_Det - F_NO3
        F_NH4[F_NH4<0] = 0
        
        # Oxy_sed = F_NH4 * rOxNH4 * area, mmol O2/d
        # Oxy_sed_sum.append(np.nansum(Oxy_sed))
        # Oxy_sed_spatial[cnt,:,:] = Oxy_sed
        Oxy_sed_sum.append(np.nansum(F_NH4[jj,ii] * 106/16 * area[jj,ii]))
        Oxy_sed_spatial[cnt,:,:] = F_NH4 * inDomain * 106/16 * area

        #---------- sediment SOD: 2nd way, follow fennel.h ----------
        # LdetritusN decomposition in sediment
        # FC_L = (LdetN_bot * Ws_L)/24   # mmol/m2/hr
        # cff1_L = FC_L / dz[0,:,:] * 1  # mmol/m2/hr --> mmol/m3
        cff1_L = (LDeN[0,:,:] * 80)/24 / dz[0,:,:] * 1  # mmol/m3
        NH4_gain_L = np.zeros([NX,NY])        
        for i in range(NX): # loop the whole domain
            for j in range(NY):
                if cff1_L[i,j]*106/16 <= Oxy[0,:,:][i,j]:  # use O2: PN + O2 --> NH4; Oxy[0,:,:] is bottom oxygen                                        
                    NH4_gain_L[i,j] = cff1_L[i,j] #mmol/m3, NH4 gain from decomposing LdetN
        NH4_gain_flux_L = NH4_gain_L * area * dz[0,:,:] # mmol but the actual unit is mmol/hr
        # SdetritusN decomposition in sediment
        # FC_S = (SdetN_bot * Ws_S)/24 # mmol/m2/hr
        # cff1_S = FC_S / dz[0,:,:] * 1  # mmol/m2/hr --> mmol/m3 
        cff1_S = (SDeN[0,:,:] * 8)/24 / dz[0,:,:] * 1  # mmol/m3
        NH4_gain_S = np.zeros([NX,NY])
        for i in range(NX): # loop the whole domain
            for j in range(NY):
                if cff1_S[i,j]*106/16 <= Oxy[0,:,:][i,j]:  # use O2: PN + O2 --> NH4;
                    NH4_gain_S[i,j] = cff1_S[i,j] #mmol/m3, NH4 gain from decomposing LdetN
        NH4_gain_flux_S = NH4_gain_S * area * dz[0,:,:] # mmol but the actual unit is mmol/hr
        
        NH4_gain_flux = NH4_gain_flux_L + NH4_gain_flux_S
        Oxy_sed_sum2.append(np.nansum(NH4_gain_flux[jj,ii] * 106/16))    # mmol O2/hr  
        Oxy_sed_spatial2[cnt,:,:] = NH4_gain_flux * inDomain * 106/16
                
        #---------- air-sea flux ----------
        Uwind = ds.Uwind.values.squeeze()
        Vwind = ds.Vwind.values.squeeze()
        temp_surf = ds.temp.values[0,-1,:,:] # surface temp
        salt_surf = ds.salt.values[0,-1,:,:] # surface salt
        Oxy_surf  = ds.oxygen.values[0,-1,:,:] # surface O2
        # Compute O2 transfer velocity: u10squared (u10 in m/s)
        u10squ = Uwind * Uwind + Vwind * Vwind  # ifdef BULK_FLUXES
        
        SchmidtN_Ox = A_O2 - temp_surf*(B_O2 - temp_surf*(C_O2 - temp_surf*(D_O2 - temp_surf*E_O2)))

        cff3 = cff2_air * u10squ * np.sqrt(660.0/SchmidtN_Ox)  # m
        TS = np.log((298.15-temp_surf)/(273.15+temp_surf))
        AA = OA0 + TS*(OA1+TS*(OA2+TS*(OA3+TS*(OA4+TS*OA5)))) + salt_surf*(OB0+TS*(OB1+TS*(OB2+TS*OB3))) + OC0*salt_surf*salt_surf
        # Convert from ml/l to mmol/m3
        # l2mol = 1000./22.3916   # liter to mol
        # O2satu = l2mol * np.exp(AA) # mmol/m3
        O2satu = 1000./22.3916 * np.exp(AA) # mmol/m3
        
        # O2 gas exchange
        #O2_Flux = cff3 * (O2satu-Oxy_surf)  # mmol O2/m2/hr ?
        #O2_Flux1 = O2_Flux * area # mmol O2/hr
        #Oxy_air_flux_sum.append(np.nansum(O2_Flux1)) # 
        Oxy_air_flux_sum.append(np.nansum(cff3[jj,ii] * (O2satu[jj,ii]-Oxy_surf[jj,ii]) * area[jj,ii]))
        Oxy_air_flux_spatial[cnt,:,:] = cff3 * (O2satu-Oxy_surf) * inDomain # save spatial air-sea flux
        diff_O2_spatial[cnt,:,:] = (O2satu-Oxy_surf) * inDomain
        
        cnt += 1
        t.append(ds.ocean_time.values)
        ds.close()       
    dt00 = dt00 + timedelta(days=1)
        
# save netcdf

from netCDF4 import Dataset
nc   = Dataset('O2_bgc_'+ds0+'.nc','w')
time = nc.createDimension('time', len(t))
eta_rho = nc.createDimension('eta_rho', NX)
xi_rho  = nc.createDimension('xi_rho', NY)
s_rho   = nc.createDimension('s_rho', 30)

times       = nc.createVariable('time','f8',('time',))
times.units = 'seconds*1e9 since 1970-01-01 00:00:00'
Oxy_pro_sum_tmp       = nc.createVariable('Oxy_pro_sum','f4',('time',),compression='zlib',complevel=9)
Oxy_pro_sum_tmp.units = 'mmol O2/hr'
Oxy_nitri_sum_tmp       = nc.createVariable('Oxy_nitri_sum','f4',('time',),compression='zlib',complevel=9)
Oxy_nitri_sum_tmp.units = 'mmol O2/hr'
Oxy_remi_sum_tmp       = nc.createVariable('Oxy_remi_sum','f4',('time',),compression='zlib',complevel=9)
Oxy_remi_sum_tmp.units = 'mmol O2/hr'
Oxy_sed_sum_tmp        = nc.createVariable('Oxy_sed_sum','f4',('time',),compression='zlib',complevel=9)
Oxy_sed_sum_tmp.units  = 'mmol O2/day'
Oxy_sed_sum_tmp2       = nc.createVariable('Oxy_sed_sum2','f4',('time',),compression='zlib',complevel=9)
Oxy_sed_sum_tmp2.units = 'mmol O2/hr'
Oxy_vol_sum_tmp        = nc.createVariable('Oxy_vol_sum','f4', ('time',),compression='zlib',complevel=9)
Oxy_vol_sum_tmp.units  = 'mmol O2'
Oxy_air_flux_sum_tmp       = nc.createVariable('Oxy_air_flux_sum','f4', ('time',),compression='zlib',complevel=9)
Oxy_air_flux_sum_tmp.units = 'mmol O2/hr'
#s
Oxy_pro_spatial_tmp       = nc.createVariable('Oxy_pro_spatial','f4',('time','eta_rho','xi_rho'),compression='zlib',complevel=9)
Oxy_pro_spatial_tmp.units = 'mmol O2/hr'
Oxy_nitri_spatial_tmp       = nc.createVariable('Oxy_nitri_spatial','f4',('time','eta_rho','xi_rho'),compression='zlib',complevel=9)
Oxy_nitri_spatial_tmp.units = 'mmol O2/hr'
Oxy_remi_spatial_tmp       = nc.createVariable('Oxy_remi_spatial','f4',('time','eta_rho','xi_rho'),compression='zlib',complevel=9)
Oxy_remi_spatial_tmp.units = 'mmol O2/hr'
Oxy_sed_spatial_tmp       = nc.createVariable('Oxy_sed_spatial','f4',('time','eta_rho','xi_rho'),compression='zlib',complevel=9)
Oxy_sed_spatial_tmp.units = 'mmol O2/day'
Oxy_sed_spatial_tmp2       = nc.createVariable('Oxy_sed_spatial2','f4',('time','eta_rho','xi_rho'),compression='zlib',complevel=9)
Oxy_sed_spatial_tmp2.units = 'mmol O2/hr'
Oxy_air_flux_spatial_tmp       = nc.createVariable('Oxy_air_flux_spatial','f4',('time','eta_rho','xi_rho'),compression='zlib',complevel=9)
Oxy_air_flux_spatial_tmp.units = 'mmol O2/m2/hr'
diff_O2_spatial_tmp       = nc.createVariable('diff_O2_spatial','f4',('time','eta_rho','xi_rho'),compression='zlib',complevel=9)
diff_O2_spatial_tmp.units = 'mmol O2/m3'

#
times[:] = t
Oxy_pro_sum_tmp[:]   = Oxy_pro_sum
Oxy_nitri_sum_tmp[:] = Oxy_nitri_sum
Oxy_remi_sum_tmp[:]  = Oxy_remi_sum
Oxy_sed_sum_tmp[:]   = Oxy_sed_sum
Oxy_sed_sum_tmp2[:]  = Oxy_sed_sum2
Oxy_vol_sum_tmp[:]   = Oxy_vol_sum
Oxy_air_flux_sum_tmp[:] = Oxy_air_flux_sum
Oxy_pro_spatial_tmp[:]  = Oxy_pro_spatial[0:len(t),:,:]
Oxy_nitri_spatial_tmp[:] = Oxy_nitri_spatial[0:len(t),:,:]
Oxy_remi_spatial_tmp[:]  = Oxy_remi_spatial[0:len(t),:,:]
Oxy_sed_spatial_tmp[:]   = Oxy_sed_spatial[0:len(t),:,:]
Oxy_sed_spatial_tmp2[:]  = Oxy_sed_spatial2[0:len(t),:,:]
Oxy_air_flux_spatial_tmp[:] = Oxy_air_flux_spatial[0:len(t),:,:]
diff_O2_spatial_tmp[:]      = diff_O2_spatial[0:len(t),:,:]

nc.close()
