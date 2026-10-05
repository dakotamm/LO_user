"""
Module of functions to create job definitions for a box extraction.
"""

def get_box(job, Lon, Lat):
    vn_list = 'h,f,pm,pn,mask_rho,salt,temp,zeta,u,v,ubar,vbar' # default list
    # specific jobs
    if job == 'sequim0':
        aa = [-123.15120787, -122.89090010, 48.07302111, 48.19978336]
    elif job == 'taiping_hc':
        aa = [-122.66394, -122.61417, 47.93171, 47.94398]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,zeta,oxygen,phytoplankton,NO3'
    elif job == 'PS':
        # 3 MB per save (26 GB/year for hourly)
        aa = [-123.5, -122.05, 47, 49]
    elif job == 'garrison':
        aa = [-129.9, -122.05, 42.1, 51.9]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,oxygen'
    elif job == 'full':
        aa = [Lon[0], Lon[-1], Lat[0], Lat[-1]]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,oxygen'
    elif job == 'liu_wind':
        aa = [-123.5, -122.05, 47, 48.5]
        vn_list = 'h,mask_rho,Uwind,Vwind'
    elif job == 'liu_ps':
        aa = [-123.3, -122.2, 47, 49]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,Uwind,Vwind,shflux'
    elif job == 'surface0':
        aa = [Lon[0], Lon[-1], Lat[0], Lat[-1]]
        # For reasons I do not understand, this gets zeta even when it is not on the
        # list.  I will put it here to be explicit.
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,zeta'
    elif job == 'surface1':
        # For Samantha 2021.12.06
        aa = [Lon[0], Lon[-1], Lat[0], Lat[-1]]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,sustr,svstr,zeta'
    elif job == 'ubc0':
        aa = [-125.016452048434, -124.494612925929, 48.312, 48.7515055163539]
        # old version
        # vn_list = ('h,f,pm,pn,mask_rho,salt,temp,zeta,NO3,phytoplankton,'
        #         + 'zooplankton,detritus,Ldetritus,oxygen,TIC,alkalinity')
        # new version
        vn_list = ('h,f,pm,pn,mask_rho,salt,temp,zeta,NO3,NH4,phytoplankton,'
                + 'zooplankton,SdetritusN,LdetritusN,oxygen,TIC,alkalinity')
    elif job == 'cox':
        aa = [-123.204529, -122.728532, 48.393771, 48.726895]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,phytoplankton'
    elif job == 'jerry0':
        aa = [-122.52, -122.40, 47.40, 47.85]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,w,zeta,u,v,ubar,vbar,Uwind,Vwind'
    elif job == 'pisces0':
        aa = [-127, -124, 46, 48]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,zeta'
    elif job == 'desanto':
        aa = [-125.028, -124.8993, 45.2581, 45.3481]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,zeta,u,v'
    elif job == 'gheibi':
        aa = [-123.15, -122.84, 48.68, 48.95]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,zeta,u,v'
    elif job == 'byrd':
        aa = [Lon[0], Lon[-1], Lat[0], Lat[-1]]
        vn_list = 'h,mask_rho,Uwind,Vwind,u,v'
    elif job == 'barbanell':
        aa = [-125.5, -122.1, 47, 50.3]
        vn_list = 'h,mask_rho,temp'
    elif job == 'bass':
        aa = [-126.5, -124, 48.4, 49]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,oxygen'
    elif job == 'bass2':
        aa = [-125.5, -122.1, 47, 49]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,oxygen'
    elif job == 'harcourt':
        aa = [-125.6, -124.2, 46.6, 47.2]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,oxygen,zeta,u,v,w,Uwind,Vwind'
    elif job == 'gilliland':
        aa = [-126.05, -124.0077, 43.9892, 45.9748]
        vn_list = 'h,pm,pn,mask_rho,salt,temp'
    elif job == 'koepke':
        aa = [-123.52373780165013, -122.45300800245987, 48.09933878244702, 48.99110096840738]
        vn_list = 'h,pm,pn,mask_rho,zeta,u,v'
    elif job == 'afischer': # Alexis Fischer
        aa = [-127, -124, 42.2, 49]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,oxygen,NO3'
    elif job == 'valentine': # Kendall Valentine, for Morgan
        aa = [-(122 + 43/60 + 13.91/3600), -(122 + 17/60 + 7.93/3600),
            (47 + 13/60 + 23.69/3600), (48 +1/60 + 17.79/3600)]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,u,v,bustr,bvstr,zeta'
    elif job == 'valentine2': # Kendall Valentine, for Morgan, revised boundary 2024.04.30
        aa = [-(122 + 43/60 + 13.91/3600), -(122 + 11/60 + 0/3600),
            (47 + 13/60 + 23.69/3600), (48 + 26/60 + 0/3600)]
        vn_list = 'h,pm,pn,mask_rho,salt,temp,u,v,bustr,bvstr,zeta'
    elif job == 'kudela0':
        aa = [Lon[0], Lon[-1], Lat[0], Lat[-1]]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,NO3,phytoplankton'
    elif job == 'geyer0':
        aa = [-123, -122.2, 47.6, 48.4]
        vn_list = 'h,f,pm,pn,mask_rho,mask_u,mask_v,salt,temp,u,v,w,zeta,AKs,AKv,sustr,svstr,bustr,bvstr'
    elif job == 'barbosa0':
        aa = [-127.345470, -125.498403, 50.335856, 51.095854]
        vn_list = 'h,f,pm,pn,mask_rho,zeta,salt,temp,NO3,u,v'
    elif job == 'sienna':
        aa = [-123.9, -122.1, 47, 49]
        vn_list = 'h,f,pm,pn,mask_rho,zeta,salt,temp,mask_u,mask_v,u,v'
    elif job == 'sienna2':
        aa = [-123.9, -122.1, 47, 49]
        vn_list = 'h,f,pm,pn,mask_rho,salt'
    elif job == 'kastner0':
        aa = [-122.706, -122.457, 48.347, 48.53]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,Pair,Uwind,Vwind,shflux,ssflux,latent,sensible,lwrad,Tair,evaporation,rain,EminusP,swrad,sustr,svstr'
    elif job == 'weber':
        aa = [-125.5, -123.5, Lat[0], 49]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,oxygen,NO3,NH4,phytoplankton'
    elif job == 'iringan':
        aa = [-123.35, -122.428, 48.293, 48.827]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,phytoplankton,Uwind,Vwind'
    elif job == 'swot':
        aa = [Lon[0], Lon[-1], Lat[0], Lat[-1]]
        vn_list = 'h,f,pm,pn,mask_rho,zeta,u,v'
    elif job == 'meghana':
        aa = [Lon[0], Lon[-1], Lat[0], Lat[-1]]
        vn_list = 'h,mask_rho,f,pm,pn,Uwind,Vwind,u,v,salt,temp'
    elif job == 'townsend':
        aa = [-125, -124, 42.3, 46.3]
        vn_list = 'h,mask_rho,pm,pn,u,v,salt,temp'
    elif job == 'townsend2':
        aa = [-125, -124, Lat[0], 46.3]
        vn_list = 'h,mask_rho,pm,pn,temp,NO3'
    elif job == 'smolt0':
        aa = [-129, -123.5, 46, 51.5]
        vn_list = 'h,mask_rho,pm,pn,u,v,salt,temp,phytoplankton'
    elif job == 'surface2':
        aa = [Lon[0], Lon[-1], Lat[0], Lat[-1]]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,zeta,oxygen,phytoplankton,NO3'
    elif job == 'lundquist':
        aa = [Lon[0], Lon[-1], Lat[0], Lat[-1]]
        vn_list = 'h,f,pm,pn,mask_rho,temp,zeta'
    elif job == 'henderson': # Cassandra Henderson - Willapa
        aa = [Lon[0], Lon[-1], Lat[0], Lat[-1]]
        vn_list = 'h,f,pm,pn,mask_rho,zeta,u,v,mask_u,mask_v,wetdry_mask_rho,wetdry_mask_u,wetdry_mask_v'
    elif job == 'SSC':
        aa = [-122.67, -122.27, 47.57, 48.0]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,zeta,u,v,ubar,vbar,oxygen,NO3,phytoplankton,Uwind,Vwind'
    elif job == 'pc0': #DM added 2025/11/25
        aa = [-122.737995, -122.658319, 48.210705, 48.250686]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,zeta,u,v,ubar,vbar,oxygen,NO3,phytoplankton,Uwind,Vwind'

    elif job == 'pc_cove': #DM added 2026/08/18 -- lateral circulation in Penn Cove
        # Bounds pulled in to the cove water itself, not to the pc polygon: the
        # polygon runs ~1.2 km west of the last wet cove cell, so tracking it
        # buys columns of pure land. Flood-filling from inside the cove (sealed
        # at pc_lp) gives cove water over i 38-67, j 210-228; this keeps one
        # cell of margin W/N/S and captures all 317 cove cells.
        # The EAST edge is rho column 68, one past the pc_lp u-face. This is
        # the choice that only makes sense WITHOUT -uv_to_rho: ncks takes
        # xi_u = ilon0..ilon1-1, so stopping at rho 67 puts xi_u at 37..66 and
        # the pc_lp u-faces (xi_u = 67) are NOT IN THE BOX AT ALL. Going to 68
        # gives xi_u = 37..67, so the box carries the mouth section's own face
        # velocities -- exact on the native grid, and directly comparable to the
        # tef2 pc_lp extraction. It also makes u_rho two-sided at the mouth if
        # you average to rho points in post-processing.
        # v needs no such help: xi_v spans the FULL rho range, so v already
        # exists at every rho column including the mouth, and the only
        # one-sided rho rows are the all-land y margins.
        # Cost: 13 Saratoga cells enter the box, so mask_rho is no longer
        # exactly the cove. Slice them off with xi_rho <= 30 (one slice, not a
        # flood fill) when you want the cove alone.
        # Do NOT reuse the older pc0 job for mouth work: its east edge
        # (-122.6583) falls INSIDE the pc_lp section and cannot see the mouth.
        # 32 x 21 cells, 330 wet (317 cove + 13 Saratoga), ~11.4 GB 2024-2025.
        aa = [-122.735882, -122.652000, 48.213109, 48.249196]
        vn_list = 'h,f,pm,pn,mask_rho,salt,temp,zeta,u,v,w,ubar,vbar'

    elif job == 'pc_cove_o2': #DM added 2026/09/17 -- oxygen budget in Penn Cove
        # SAME FOOTPRINT as pc_cove, tracer-only. Read the pc_cove comment above
        # for why these bounds are what they are -- do not re-derive them, and do
        # not "tidy" the east edge in: rho column 68 is deliberate.
        #
        # WHY THIS EXISTS the pc_cove box carries no oxygen, so the cove's
        # oxygen inventory cannot be split vertically and no bottom-layer budget
        # can be formed. The whole-cove budget is net autotrophic in every
        # season (Winter +63, Spring +90, Low-DO +126 g/s internal source), so
        # depth-integrated numbers cannot explain hypoxia at all -- it is a
        # vertical redistribution. That needs the 3D oxygen field.
        #
        # It DOES NOT overwrite the velocity box. extract_box.py names output
        # <job>_<ds0>_<ds1>.nc and only cleans its own temp_<bname> directory,
        # so this writes pc_cove_o2_*.nc alongside pc_cove_*.nc and the 11.4 GB
        # file is untouched. Keep the job name distinct for that reason alone.
        #
        # salt and temp are here for density, so the pycnocline can be located
        # per column and per time rather than splitting at a fixed depth (a
        # fixed 10 m split moves the deep O2 supply by ~40% between seasons, so
        # the interface has to be real). No velocities: u/v/w are already in
        # pc_cove on the same grid and can be read from there.
        #
        # Use -lt hourly0 (ocean_his, instantaneous), matching pc_cove and
        # extract_segments_SV.py: the budget pairs an instantaneous state
        # difference with an avg-file flux across the same interval.
        #
        # hourly0, NOT hourly. Plain -lt hourly defaults to his_num=2, which
        # starts at ocean_his_0025.nc on the day BEFORE ds0 -- f2023.12.31 for a
        # 2024.01.01 start, which does not exist for this run, and the job dies
        # with FileNotFoundError before extracting anything. hourly0 reads
        # ocean_his_0001.nc of ds0 instead. Those are the SAME INSTANT
        # (f2023.12.31/0025 == f2024.01.01/0001 == 2024-01-01 00:00), so the
        # time axis is unchanged at 17545 points and still lines up cell for
        # cell with pc_cove and the segments file. -lt hourly -his_num 1 is
        # equivalent if you prefer.
        #
        # 32 x 21 cells, 30 levels. Three 3D rho tracers at ~1.4 GB each over
        # 2024-2025, so roughly 4-5 GB -- about 40% of the velocity box.
        aa = [-122.735882, -122.652000, 48.213109, 48.249196]
        vn_list = 'h,f,pm,pn,mask_rho,zeta,salt,temp,oxygen'

    return aa, vn_list
