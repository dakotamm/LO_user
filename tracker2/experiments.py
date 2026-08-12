"""
This is where you set the run initial condition using get_ic()
based on an experiment name passed by the calling code.

Thre are also some utility functions useful for making different
common release patterns.

"""

import numpy as np
    
def get_ic(TR):
    # routines to set particle initial locations, all numpy arrays
    
    # NOTE: "pcs" refers to fractional depth, and goes linearly from -1 to 0
    # between the local bottom and free surface.  It is how we keep track of
    # vertical position, only converting to z-position when needed.
    
    exp_name = TR['exp_name']
    gridname = TR['gridname']
    fn00 = TR['fn00']
        
    if exp_name == 'jdf0': # Mid-Juan de Fuca
        lonvec = np.linspace(-123.85, -123.6, 20)
        latvec = np.linspace(48.2, 48.4, 20)
        pcs_vec = np.array([-1])
        plon00, plat00, pcs00 = ic_from_meshgrid(lonvec, latvec, pcs_vec)
        
    elif exp_name == 'jdf1': # Mid-Juan de Fuca, just a line across the channel
        N = 10
        plon00 = -123.7 * np.ones(N)
        plat00 = np.linspace(48.15, 48.3, N)
        pcs00 = -1 * np.ones(N)

    elif exp_name == 'wgh0': # Designed for the new nested wgh2 grid.
        """
        north end of Long Beach at 46.497401, -124.064370
        and the middle of Twin Harbors beach at 46.809169, -124.10805
        """
        # Long Beach
        lon0 = -124.064370; lat0 = 46.497401 # center of the circle 
        radius_km = 2 # radius of the circle km
        N = 300 # number of particles
        # make random scattering of points in a circle
        plon00_1, plat00_1 = ic_random_in_circle(lon0, lat0, radius_km, N)
        # Twin Harbors
        lon0 = -124.10805; lat0 = 46.809169 # center of the circle 
        radius_km = 2 # radius of the circle km
        N = 300 # number of particles
        # make random scattering of points in a circle
        plon00_2, plat00_2 = ic_random_in_circle(lon0, lat0, radius_km, N)
        # Combine
        plon00 = np.concatenate((plon00_1, plon00_2))
        plat00 = np.concatenate((plat00_1, plat00_2))
        pcs00 = np.zeros(plon00.shape)
        
    elif exp_name == 'sect_AImid':
        lons = [-122.7, -122.6]
        lats = [48.075, 48.075]
        plon00, plat00, pcs00 = ic_sect(fn00, lons, lats, NPmax=10000)
        
    if exp_name == 'ai0': # Mid-Admiralty Inlet
        lonvec = np.array([-122.6])
        latvec = np.array([48])
        pcs_vec = np.linspace(-1,-0.9,num=1000)
        plon00, plat00, pcs00 = ic_from_list(lonvec, latvec, pcs_vec)
        
    if exp_name == 'aiN': # North Admiralty Inlet
        lonvec = np.array([-122.67])
        latvec = np.array([48.125])
        pcs_vec = np.linspace(-1,-0.9,num=300)
        plon00, plat00, pcs00 = ic_from_list(lonvec, latvec, pcs_vec)
        
    elif exp_name == 'vmix': # three vertical profiles to test mixing
        # use with the new flag: -no_advection True, so a full command would be
        # python tracker.py -exp vmix -3d True -clb True -no_advection True
        lonvec = np.array([-125.35, -124.0, -122.581])
        latvec = np.array([47.847, 48.3, 48.244])
        # These are: (Slope off JdF, Middle of JdF, Whidbey Basin)
        pcs_vec = np.linspace(-1,0,num=4000)
        plon00, plat00, pcs00 = ic_from_list(lonvec, latvec, pcs_vec)
        
    elif exp_name == 'dmMerhab':
        nyp = 7
        x0 = -126; x1 = -125; y0 = 48; y1 = 49
        clat_1 = np.cos(np.pi*(np.mean([y0, y1]))/180)
        xyRatio = clat_1 * (x1 - x0) / (y1 - y0)
        lonvec = np.linspace(x0, x1, (nyp * xyRatio).astype(int))
        latvec = np.linspace(y0, y1, nyp)
        lonmat_1, latmat_1 = np.meshgrid(lonvec, latvec)
        #
        x0 = -125.2; x1 = -124.2; y0 = 44; y1 = 45
        clat_2 = np.cos(np.pi*(np.mean([y0, y1]))/180)
        xyRatio = clat_2 * (x1 - x0) / (y1 - y0)
        lonvec = np.linspace(x0, x1, (nyp * xyRatio).astype(int))
        latvec = np.linspace(y0, y1, nyp)
        lonmat_2, latmat_2 = np.meshgrid(lonvec, latvec)
        lonmat = np.concatenate((lonmat_1.flatten(), lonmat_2.flatten()))
        latmat = np.concatenate((latmat_1.flatten(), latmat_2.flatten()))
        #
        plon00 = lonmat.flatten(); plat00 = latmat.flatten()
        pcs00 = np.zeros(plon00.shape)

    elif exp_name == 'dmMerhab2':
        # like dmMerhab but with many more particles
        nyp = 20
        x0 = -126; x1 = -125; y0 = 48; y1 = 49
        clat_1 = np.cos(np.pi*(np.mean([y0, y1]))/180)
        xyRatio = clat_1 * (x1 - x0) / (y1 - y0)
        lonvec = np.linspace(x0, x1, (nyp * xyRatio).astype(int))
        latvec = np.linspace(y0, y1, nyp)
        lonmat_1, latmat_1 = np.meshgrid(lonvec, latvec)
        #
        x0 = -125.2; x1 = -124.2; y0 = 44; y1 = 45
        clat_2 = np.cos(np.pi*(np.mean([y0, y1]))/180)
        xyRatio = clat_2 * (x1 - x0) / (y1 - y0)
        lonvec = np.linspace(x0, x1, (nyp * xyRatio).astype(int))
        latvec = np.linspace(y0, y1, nyp)
        lonmat_2, latmat_2 = np.meshgrid(lonvec, latvec)
        lonmat = np.concatenate((lonmat_1.flatten(), lonmat_2.flatten()))
        latmat = np.concatenate((latmat_1.flatten(), latmat_2.flatten()))
        #
        plon00 = lonmat.flatten(); plat00 = latmat.flatten()
        pcs00 = np.zeros(plon00.shape)
    elif exp_name == 'full': # the whole domain of cas6, with some edges trimmed
        # used by drifters0
        lonvec = np.linspace(-129, -122, 60)
        latvec = np.linspace(43, 51, 120)
        pcs_vec = np.array([0])
        plon00, plat00, pcs00 = ic_from_meshgrid(lonvec, latvec, pcs_vec)
        
    elif exp_name == 'PS': # nominally Puget Sound
        # used by drifters0
        lonvec = np.linspace(-123.6, -122, 60)
        latvec = np.linspace(47, 49, 120)
        pcs_vec = np.array([0])
        plon00, plat00, pcs00 = ic_from_meshgrid(lonvec, latvec, pcs_vec)

    elif exp_name == 'PS3deep': # nominally Puget Sound, over deeper depths
        lonvec = np.linspace(-123.6, -122, 60)
        latvec = np.linspace(47, 49, 120)
        pcs_vec = np.linspace(-1,-0.5,10)
        plon00, plat00, pcs00 = ic_from_meshgrid(lonvec, latvec, pcs_vec)

    elif exp_name == 'PS3shallow': # nominally Puget Sound, over shallower depths
        lonvec = np.linspace(-123.6, -122, 60)
        latvec = np.linspace(47, 49, 120)
        pcs_vec = np.linspace(-.5,0,10)
        plon00, plat00, pcs00 = ic_from_meshgrid(lonvec, latvec, pcs_vec)

    elif exp_name == 'depthrange': # the whole domain of cas7, with some edges trimmed
        # and then trim it to be in a specific depth range.
        # For James Murray June 2025
        lonvec = np.linspace(-129, -122, 500)
        latvec = np.linspace(43, 51, 1000)
        from lo_tools import zfun, zrfun
        G = zrfun.get_basic_info(fn00, only_G=True)
        lon, lat = np.meshgrid(lonvec, latvec)
        h = G['h']
        h[G['mask_rho']==0] = 0
        hh = zfun.interp2(lon,lat,G['lon_rho'],G['lat_rho'],h)
        hmin = 10; hmax = 50
        mask = (hh>=hmin) & (hh<=hmax) # keep only depth range hmin to hmax
        plon00 = lon[mask]
        plat00 = lat[mask]
        pcs00 = -np.ones(len(plon00))

    elif exp_name == 'willapa25':
        # Release from a number of locations in Willapa Bay, for Jim Thomson and
        # Christie Hegermiller, June 2025.
        sta_dict = {
            'OS': (-124.15, 46.721021), # Offshore
            'NC': (-124.089593, 46.733819), # North Channel
            'MC': (-124.083920,46.715140), # Mid Channel
            'MB': (-124.0036308196363, 46.69039774784152), # Mid Bay
            'EB': (-123.9551377630581, 46.694), # East Bay
            'SB': (-123.993093733228, 46.65901928767435), # South Bay
        }
        ii = 0
        radius_km = 0.2 # radius of the circle km
        N = 200 # number of particles
        for sta in sta_dict.keys():
            lon0, lat0 = sta_dict[sta]
            # make random scattering of points in a circle
            if ii == 0:
                plon00, plat00 = ic_random_in_circle(lon0, lat0, radius_km, N)
            else:
                plon00a, plat00a = ic_random_in_circle(lon0, lat0, radius_km, N)
                # Combine
                plon00 = np.concatenate((plon00, plon00a))
                plat00 = np.concatenate((plat00, plat00a))
            ii += 1
        pcs00 = np.zeros(plon00.shape)

    elif exp_name == 'haro0':
        # Looking at the difference between laminar and turbulent in Haro Strait
        plon00, plat00 = ic_random_in_circle(-123.231, 48.5383, 0.2, 1000)
        pcs00 = -0.5 * np.ones(plon00.shape)

    elif exp_name == 'pcret':
        # Penn Cove retention (DM 2026.08.05).
        #
        # Fill the three Penn Cove tef2 segments of the wb1_pc1 collection with
        # particles, all released at once so the three cohorts see identical
        # forcing and can be compared directly:
        #   pc_cp_m  inner cove   (landward of pc_cp)
        #   pc_cp_p  mid cove     (pc_cp to pc_lj)
        #   pc_lp_m  outer cove   (pc_lj to pc_lp)
        #
        # The cohort a particle belongs to is NOT recorded here. It is
        # recovered in the analysis from each particle's INITIAL position,
        # which lands exactly on a rho cell centre -- that is robust to the
        # tracker trimming particles on land, which would otherwise shift
        # every index and silently mislabel the cohorts.
        #
        # Vertical: DZ = 3 m gives about 5 levels in the 13 m inner cove and 8
        # in the 22 m outer cove, so the deep water is resolved separately from
        # the surface. That matters here -- the question is whether landward
        # BOTTOM water stagnates, and a surface-only release cannot see it.
        # pc_lp_p (upper Saratoga) is included as a CONTROL cohort: it is the
        # water the cove exchanges with, so it says whether any retention we
        # find is special to the cove or just what this part of the basin does.
        # It is capped, because at 2602 cells and ~30 m deep it would otherwise
        # contribute ~26000 particles and swamp the three cove cohorts.
        plon00, plat00, pcs00 = ic_from_tef2_segs(
            fn00, gridname, ctag='pc1', riv='trapsN00',
            seg_list=['pc_cp_m', 'pc_cp_p', 'pc_lp_m', 'pc_lp_p'], DZ=3,
            n_max_per_seg=1500)

    elif exp_name == 'pcbot':
        # Penn Cove INNER-COVE BOTTOM water (DM 2026.08.11).
        #
        # One cohort, not four: pc_cp_m only -- the tef2 segment landward of
        # the pc_cp (Coupeville) section, 165 wet cells, h = 7.0 to 21.2 m,
        # median 13 m. This is the water whose oxygen is drawn down, so it is
        # the water whose residence time the experiment is about.
        #
        # Only the BOTTOM of the column is seeded. Levels are set by HEIGHT
        # ABOVE THE BED, not by sigma, because the segment spans a factor of
        # three in depth: a fixed pcs would put "bottom" particles 1 m off the
        # bed in the shallows and 5 m off it in the deep, and the cohort would
        # not be one water mass. Height above bed also matches how the DO
        # drawdown is actually distributed -- it hugs the sediment.
        #
        # Paired with 20260811_pc_matched_weeks.py, which picks the two
        # tide-matched release weeks and prints the two tracker commands. Run
        # them as two SEPARATE commands with the start_hour each one gives.
        # Do NOT use -nsd/-dbs to make the pair in one command: -dbs steps in
        # whole days, and that is exactly how the earlier pcret pair ended up
        # ~9 h out of tidal phase, which made its early retention curves
        # incomparable.
        #
        # The same release is also used for the spring-neap pair picked by
        # 20260811_pc_springneap_weeks.py (sub_tags neap / spring), which
        # holds the season fixed and varies the tide instead. Nothing about
        # the release changes between the two experiments -- only the window.
        plon00, plat00, pcs00 = ic_from_tef2_segs_bottom(
            fn00, gridname, ctag='pc1', riv='trapsN00',
            seg_list=['pc_cp_m'], hab_max=5.0, dz=0.5, frac_max=0.5)

    return plon00, plat00, pcs00


def ic_from_tef2_segs_bottom(fn00, gridname, ctag, riv, seg_list,
                             hab_max=5.0, dz=0.5, frac_max=0.5, NPmax=20000):
    """Seed the BOTTOM layer only of a list of tef2 segments.

    Bottom-following version of ic_from_tef2_segs(). In every wet cell of every
    named segment, particles go at heights above the bed of dz/2, 3dz/2, ...
    up to hab_max, at the cell centre. pcs = -1 + hab/h.

    Two rules keep the cohort a single water mass:

      hab_max   the absolute thickness of the layer being called "bottom
                water", in metres off the bed. Fixed in metres rather than as
                a fraction of h, so a 7 m cell and a 21 m cell contribute
                water from the same height above the sediment.

      frac_max  no particle may start above frac_max of the way up the column.
                Without it, hab_max = 5 m in a 7 m cell would seed to 71 % of
                the depth -- surface water, wearing a bottom-water label -- and
                the shallow cells would quietly dominate the "bottom" cohort's
                early loss. With frac_max = 0.5 the shallowest cells simply
                get fewer levels.

    Particles sit at cell centres rather than jittered within the cell, as in
    ic_from_tef2_segs(). The centres come from the segment's ji_list and are
    therefore guaranteed wet, so the tracker's land-trimming does not eat the
    shoreline cells preferentially, and each particle's starting height above
    bed is recoverable exactly from its initial pcs.
    """
    import pickle
    import sys

    from lo_tools import Lfun, zrfun

    Ldir = Lfun.Lstart(gridname=gridname)
    tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
    seg_fn = tef2_dir / ('seg_info_dict_' + gridname + '_' + ctag + '_' + riv + '.p')
    if not seg_fn.is_file():
        # Cells do not depend on the river set (see ic_from_tef2_segs), so any
        # tag on this machine gives an identical release.
        alt = sorted(tef2_dir.glob('seg_info_dict_' + gridname + '_' + ctag + '_*.p'))
        if len(alt) == 0:
            raise FileNotFoundError(
                'no seg_info_dict for %s_%s in %s -- run create_seg_info_dict.py'
                % (gridname, ctag, tef2_dir))
        print('  %s not found, using %s (cells are river-independent)'
              % (seg_fn.name, alt[0].name))
        seg_fn = alt[0]
    seg_info = pickle.load(open(seg_fn, 'rb'))

    G = zrfun.get_basic_info(fn00, only_G=True)
    h = G['h']
    xp = G['lon_rho']
    yp = G['lat_rho']

    hab_vec = np.arange(dz / 2, hab_max, dz)   # heights above bed to try
    plon00 = np.array([]); plat00 = np.array([]); pcs00 = np.array([])
    for seg_name in seg_list:
        ji = np.array(seg_info[seg_name]['ji_list'])
        x_s = np.array([]); y_s = np.array([]); p_s = np.array([])
        h_used = []
        for j, i in ji:
            hh = h[j, i]
            if hh <= 0:
                continue
            hab = hab_vec[hab_vec <= frac_max * hh]
            if len(hab) == 0:
                continue
            x_s = np.append(x_s, xp[j, i] * np.ones(len(hab)))
            y_s = np.append(y_s, yp[j, i] * np.ones(len(hab)))
            p_s = np.append(p_s, -1 + hab / hh)
            h_used.append(hh)
        plon00 = np.append(plon00, x_s)
        plat00 = np.append(plat00, y_s)
        pcs00 = np.append(pcs00, p_s)
        print('  %-9s %4d cells (h %.1f-%.1f m) -> %5d particles, '
              '%.1f-%.1f m above the bed'
              % (seg_name, len(h_used), np.min(h_used), np.max(h_used),
                 len(x_s), hab_vec[0], hab_vec[-1]))

    NP = len(plon00)
    nstep = max(1, int(NP / NPmax))
    if nstep > 1:
        plon00 = plon00[::nstep]; plat00 = plat00[::nstep]; pcs00 = pcs00[::nstep]
    print('  total %d particles (subsample step %d)' % (len(plon00), nstep))
    sys.stdout.flush()
    return plon00, plat00, pcs00


def ic_from_tef2_segs(fn00, gridname, ctag, riv, seg_list, DZ,
                      n_max_per_seg=None, NPmax=20000):
    """Seed particles through the volume of a list of tef2 segments.

    This is the tef2 equivalent of ic_from_TEFsegs() above, which cannot be
    used here: that one reads LO_output/tef/volumes_<gridname>/j_dict.p, the
    old tef1 format. The tef2 workflow instead stores a seg_info_dict whose
    entries carry 'ji_list', a list of (j,i) rho indices per segment.

    Particles are placed at the centre of every wet cell of every named
    segment, stacked vertically with about DZ metres between levels, so the
    release is roughly volume-filling rather than area-filling. Deep cells
    therefore get more particles than shallow ones, which is what you want if
    the cohorts are to be compared per unit volume.

    Vertical placement is at sigma cell centres: n = round(h/DZ) levels at
    fractional depths -(k+0.5)/n. That spans the whole water column with a
    half-interval margin top and bottom. The obvious alternative, stepping DZ
    metres up from the bed, leaves up to DZ metres at the BOTTOM unsampled
    (h = 26 m with DZ = 3 puts the deepest particle at -24 m), which is exactly
    the water you care about when asking whether deep landward water stagnates.
    """
    import pickle
    import sys

    from lo_tools import Lfun, zrfun

    Ldir = Lfun.Lstart(gridname=gridname)
    tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
    seg_fn = tef2_dir / ('seg_info_dict_' + gridname + '_' + ctag + '_' + riv + '.p')
    if not seg_fn.is_file():
        # Fall back to whatever river tag is present. A segment's ji_list does
        # not depend on the river set -- create_seg_info_dict.py floods using
        # only sect_df and the land mask, and attaches riv_list afterwards --
        # so any tag gives identical cells and therefore an identical release.
        # Machines end up with different tags (the mac was built with riv00,
        # apogee with trapsN00) and that should not stop a run.
        alt = sorted(tef2_dir.glob('seg_info_dict_' + gridname + '_' + ctag + '_*.p'))
        if len(alt) == 0:
            raise FileNotFoundError(
                'no seg_info_dict for %s_%s in %s -- run create_seg_info_dict.py'
                % (gridname, ctag, tef2_dir))
        print('  %s not found, using %s (cells are river-independent)'
              % (seg_fn.name, alt[0].name))
        seg_fn = alt[0]
    seg_info = pickle.load(open(seg_fn, 'rb'))

    G = zrfun.get_basic_info(fn00, only_G=True)
    h = G['h']
    xp = G['lon_rho']
    yp = G['lat_rho']

    plon00 = np.array([]); plat00 = np.array([]); pcs00 = np.array([])
    for seg_name in seg_list:
        ji = np.array(seg_info[seg_name]['ji_list'])
        jj = ji[:, 0]; ii = ji[:, 1]
        x_s = np.array([]); y_s = np.array([]); p_s = np.array([])
        for j, i in zip(jj, ii):
            hh = h[j, i]
            if hh <= 0:
                continue
            n = max(1, int(round(hh / DZ)))          # levels in this column
            svec = -(np.arange(n) + 0.5) / n         # sigma cell centres
            x_s = np.append(x_s, xp[j, i] * np.ones(len(svec)))
            y_s = np.append(y_s, yp[j, i] * np.ones(len(svec)))
            p_s = np.append(p_s, svec)
        n_full = len(x_s)

        # Cap per segment, not globally. A global cap would preserve the
        # relative sizes and so let one big segment dominate; each cohort is
        # analysed on its own and normalised by its own count, so thinning one
        # segment costs coverage density but not comparability. Stride
        # subsampling keeps the release spread over the whole segment rather
        # than clustering it wherever the cell list happened to start.
        if n_max_per_seg is not None and n_full > n_max_per_seg:
            step = int(np.ceil(n_full / n_max_per_seg))
            x_s = x_s[::step]; y_s = y_s[::step]; p_s = p_s[::step]

        plon00 = np.append(plon00, x_s)
        plat00 = np.append(plat00, y_s)
        pcs00 = np.append(pcs00, p_s)
        print('  %-9s %5d cells -> %6d particles%s'
              % (seg_name, len(jj), len(x_s),
                 '  (capped from %d)' % n_full if len(x_s) < n_full else ''))

    NP = len(plon00)
    nstep = max(1, int(NP / NPmax))
    if nstep > 1:
        plon00 = plon00[::nstep]; plat00 = plat00[::nstep]; pcs00 = pcs00[::nstep]
    print('  total %d particles (subsample step %d)' % (len(plon00), nstep))
    sys.stdout.flush()
    return plon00, plat00, pcs00
    
def ic_from_meshgrid(lonvec, latvec, pcs_vec):
    # First create three vectors of initial locations (as done in some cases above).
    # plat00 and plon00 should be the same length, and the length of pcs00 is
    # as many vertical positions you have at each lat, lon
    # (expressed as fraction of depth -1 < pcs < 0).
    # Then we create full output vectors (each has one value per point).
    # This code takes each lat, lon location and then assigns it to NSP points
    # corresponding to the vector of pcs values.
    lonmat, latmat = np.meshgrid(lonvec, latvec)
    plon_vec = lonmat.flatten()
    plat_vec = latmat.flatten()
    if len(plon_vec) != len(plat_vec):
        print('WARNING: Problem with length of initial lat, lon vectors')
    NSP = len(pcs_vec)
    NXYP = len(plon_vec)
    plon_arr = plon_vec.reshape(NXYP,1) * np.ones((NXYP,NSP))
    plat_arr = plat_vec.reshape(NXYP,1) * np.ones((NXYP,NSP))
    pcs_arr = np.ones((NXYP,NSP)) * pcs_vec.reshape(1,NSP)
    plon00 = plon_arr.flatten()
    plat00 = plat_arr.flatten()
    pcs00 = pcs_arr.flatten()
    return plon00, plat00, pcs00
    
def ic_from_list(lonvec, latvec, pcs_vec):
    # Like ic_from_meshgrid() but treats the lon, lat lists like lists of mooring locations.
    plon_vec = lonvec
    plat_vec = latvec
    if len(plon_vec) != len(plat_vec):
        print('WARNING: Problem with length of initial lat, lon lists')
    NSP = len(pcs_vec)
    NXYP = len(plon_vec)
    plon_arr = plon_vec.reshape(NXYP,1) * np.ones((NXYP,NSP))
    plat_arr = plat_vec.reshape(NXYP,1) * np.ones((NXYP,NSP))
    pcs_arr = np.ones((NXYP,NSP)) * pcs_vec.reshape(1,NSP)
    plon00 = plon_arr.flatten()
    plat00 = plat_arr.flatten()
    pcs00 = pcs_arr.flatten()
    return plon00, plat00, pcs00
    
def ic_random_in_circle(lon0, lat0, radius_km, npoints):
    # Makes lon and lat of npoints scattered randomly in a circle.
    # I think the np.sqrt() used in calculating the radius makes these
    # evenly distributed over the whole circle.
    earth_r = 6371 # average earth radius [km]
    # radius of the circle km
    circle_r = radius_km
    # center of the circle (x, y)
    circle_x = lon0
    circle_y = lat0
    N = npoints # number of particles
    # random angle
    alpha = 2 * np.pi * np.random.rand(N)
    # random radius
    r = (circle_r/earth_r) * (180/np.pi) * np.sqrt(np.random.rand(N))
    # calculating coordinates
    plon00 = r * np.cos(alpha) / np.cos(circle_y*np.pi/180) + circle_x
    plat00 = r * np.sin(alpha) + circle_y
    # we leave it to the user to make pcs00
    return plon00, plat00
    
def ic_from_TEFsegs(fn00, gridname, seg_list, DZ, NPmax=10000):
    import pickle
    import sys
    # select the indir
    from lo_tools import Lfun, zrfun
    Ldir = Lfun.Lstart()
    indir = Ldir['LOo'] / 'tef' / ('volumes_' + gridname)
    # load data
    j_dict = pickle.load(open(indir / 'j_dict.p', 'rb'))
    i_dict = pickle.load(open(indir / 'i_dict.p', 'rb'))
    G = zrfun.get_basic_info(fn00, only_G=True)
    h = G['h']
    xp = G['lon_rho']
    yp = G['lat_rho']
    plon_vec = np.array([])
    plat_vec = np.array([])
    hh_vec = np.array([])
    for seg_name in seg_list:
        jjj = j_dict[seg_name]
        iii = i_dict[seg_name]
        # untested 2021.10.05
        hh_vec = np.append(hh_vec, h[jjj,iii])
        plon_vec = np.append(plon_vec, xp[jjj,iii])
        plat_vec = np.append(plat_vec, yp[jjj,iii])
        # ji_seg = ji_dict[seg_name]
        # for ji in ji_seg:
        #     plon_vec = np.append(plon_vec, xp[ji])
        #     plat_vec = np.append(plat_vec, yp[ji])
        #     hh_vec = np.append(hh_vec, h[ji])
    plon00 = np.array([]); plat00 = np.array([]); pcs00 = np.array([])
    for ii in range(len(plon_vec)):
        x = plon_vec[ii]
        y = plat_vec[ii]
        hdz = DZ*np.floor(hh_vec[ii]/DZ) # depth to closest DZ m (above the bottom)
        if hdz >= DZ:
            zvec = np.arange(-hdz,DZ,DZ) # a vector that goes from -hdz to 0 in steps of DZ m
            svec = zvec/hh_vec[ii]
            ns = len(svec)
            if ns > 0:
                plon00 = np.append(plon00, x*np.ones(ns))
                plat00 = np.append(plat00, y*np.ones(ns))
                pcs00 = np.append(pcs00, svec)
    # subsample the I.C. vectors to around max length around NPmax
    NP = len(plon00)
    print(len(plon00))
    nstep = max(1,int(NP/NPmax))
    plon00 = plon00[::nstep]
    plat00 = plat00[::nstep]
    pcs00 = pcs00[::nstep]
    print(len(plon00))
    sys.stdout.flush()
    return plon00, plat00, pcs00
    
def ic_sect(fn00, lons, lats, NPmax=10000):
    """
    This distributes NPmax particles evenly on a section defined by endpoints
    (lon0, lat0) - (lon1, lat1).
    
    For simplicity we force the section to be NS or EW, we put particles
    only on rho points.
    """
    from lo_tools import Lfun, zfun, zrfun

    Ldir = Lfun.Lstart()

    G = zrfun.get_basic_info(fn00, only_G=True)
    h = G['h']
    m = G['mask_rho']
    xr = G['lon_rho']
    yr = G['lat_rho']
    X = xr[0,:]
    Y = yr[:,0]

    lon0 = lons[0]; lon1 = lons[1]
    lat0 = lats[0]; lat1 = lats[1]

    ix0 = zfun.find_nearest_ind(X, lon0)
    ix1 = zfun.find_nearest_ind(X, lon1)
    iy0 = zfun.find_nearest_ind(Y, lat0)
    iy1 = zfun.find_nearest_ind(Y, lat1)

    # adjust indices to make it perfectly zonal or meridional
    dix = np.abs(ix1 - ix0)
    diy = np.abs(iy1 - iy0)
    if dix > diy: # EW section
        iy1 = iy0
    elif diy > dix: # NS section
        ix1 = ix0
    
    hvec = h[iy0:iy1+1, ix0:ix1+1].squeeze()
    mvec = m[iy0:iy1+1, ix0:ix1+1].squeeze()
    xvec = xr[iy0:iy1+1, ix0:ix1+1].squeeze()
    yvec = yr[iy0:iy1+1, ix0:ix1+1].squeeze()

    # add up total depth of water
    hnet = 0
    for ii in range(len(hvec)):
        if mvec[ii] == 1:
            hnet += hvec[ii]
    p_per_meter = NPmax/hnet
        
    # initialize result arrays
    plon00 = np.array([]); plat00 = np.array([]); pcs00 = np.array([])
    for ii in range(len(hvec)):
        if mvec[ii] == 1:
            this_h = hvec[ii]
            this_np = int(np.floor(p_per_meter * this_h))
            plon00 = np.concatenate((plon00,xvec[ii]*np.ones(this_np)))
            plat00 = np.concatenate((plat00,yvec[ii]*np.ones(this_np)))
            pcs00 = np.concatenate((pcs00,np.linspace(-1,0,this_np)))
    return plon00, plat00, pcs00
    