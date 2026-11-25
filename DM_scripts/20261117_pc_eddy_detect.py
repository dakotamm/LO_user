"""
Penn Cove inner-cove eddy, from the model extraction only (pc_cove box, hourly, native C-grid): WHEN is it there,
WHERE is it (horizontally and vertically), and HOW STRONG is it. One answer per hour and per depth bin.

STEP 1  read u, v, z_w from the box in time chunks; average u (u faces) and v (v faces) over EDGES bins of depth
        below the moving surface (thickness-weighted overlap of each s-level with the bin, z_w at a face = mean of
        the two rho cells). A rho cell has a bin only if its water column reaches the bin bottom at EVERY hour
        (H_min >= bin bottom), so a bin never wets and dries; a face has the bin only if both its rho cells do,
        otherwise it is a wall (u = 0).
STEP 2  Godin filter every face series (the eddy turns over in ~1.5-2 d; the tidal vorticity would make it blink on
        and off every tide). Vorticity zeta = dv/dx - du/dy and Okubo-Weiss OW = sn^2 + ss^2 - zeta^2 at the psi
        points with all four rho cells wet (no wall shear); u, v at the rho points (face means, walls 0).
STEP 3  CENTRE each hour and bin, no threshold: the inner-cove psi point (west of pc_cp) with the largest disk-mean
        vorticity  zeta_R = sum(zeta dA) / sum(dA) over the wet psi points within R0 (a circulation per area: the
        eddy is the strongest rotating patch about RO across, not the single noisiest psi point).
STEP 4  SIZE and STRENGTH from the azimuthal velocity around that centre: u_th(r) = SENSE * mean of the tangential
        velocity of the wet rho cells in rings of width DR out to RRING[-1]. r_max = ring of the largest u_th (the
        speed-based eddy radius, Chelton et al. 2011, Nencioli et al. 2010), U_max = u_th(r_max) [the strength],
        circ = 2 pi r_max U_max, zeta_c / f = zeta_R / f at the centre, T_turn = 2 pi r_max / U_max.
        PRESENT (geometry only): OW < 0 at the centre (rotation beats strain) AND u_th > 0 in all four quadrants of
        the r_max ring (closed circulation). Strength is kept for every hour, present or not.
STEP 5  EVENTS per bin: runs of present hours, kept if they last >= MIN_TURN turnovers (median T_turn of the run);
        'present_ev' = hour in a kept event. Save.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       LO_output/section_lines/pc_cp.p
OUTPUT LO_output/DM_outs/20261117_pc_eddy_detect/
           eddy_hourly.nc   (time, z): present, present_ev, lon_c, lat_c, x_c, y_c, r_max, U_max, circ, zeta_c_f,
                            ow_c, T_turn, n_quad; (time, z, r): u_th; (z, eta_psi, xi_psi): mean zeta
           events.csv       one line per kept event per bin

run 20261117_pc_eddy_detect.py           (then 20261117_pc_eddy_detect_plot.py)
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'     # the box file
EDGES = np.arange(0, 21, 2.0)             # depth bins below the moving surface [m]
SENSE = 1                                 # +1 counterclockwise eddy
R0 = 500.0                                # disk radius for the centre [m] (~2.6 dx)
DR = 190.0                                # ring width [m] (~1 dx)
RRING = np.arange(1, 7) * DR              # ring outer edges [m]: 190 ... 1140
MIN_TURN = 1.0                            # an event lasts at least this many turnovers
XI_COVE = 30                              # cove water: rho columns <= 30 (13 Saratoga cells east of it)
CHUNK = 720                               # hours read at a time
M_PER_DEG = 111.32e3
LAT0 = 48.23                              # local x, y frame: x from the pc_lp mouth, y from LAT0

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_eddy_detect'
Lfun.make_dir(out_dir)
LON_CP = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_cp.p').x.mean()    # N-S line
LON_LP = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()

# ------------------------------------------------------------------ STEP 1: face velocities in depth bins
ds = xr.open_dataset(box_fn)
ot = pd.DatetimeIndex(ds.ocean_time.values)
lon, lat = ds['lon_rho'].values, ds['lat_rho'].values
pm, pn, f_cor = ds['pm'].values, ds['pn'].values, ds['f'].values
mask = ds['mask_rho'].values == 1
mask[:, XI_COVE + 1:] = False
NT, (NR, NC), NB = len(ot), mask.shape, len(EDGES) - 1
Ub = np.zeros((NT, NB, NR, NC - 1), np.float32)     # bin-mean u on u faces
Vb = np.zeros((NT, NB, NR - 1, NC), np.float32)     # bin-mean v on v faces
Hmin = np.full((NR, NC), np.inf)                    # min water column depth over the record


def bin_mean(q, zw):
    """(t, NB, ...) thickness-weighted mean of q (t, k, ...) over each EDGES bin, zw (t, k+1, ...) at the same points;
    0 where the bin holds no water"""
    dw = zw[:, -1:] - zw                             # depth below the surface of each interface
    d_top, d_bot = dw[:, 1:], dw[:, :-1]
    out = np.zeros((q.shape[0], NB) + q.shape[2:], np.float32)
    for b in range(NB):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None)
        s = ov.sum(axis=1)
        out[:, b] = np.where(s > 0, (q * ov).sum(axis=1) / np.where(s > 0, s, 1), 0)
    return out


for i0 in range(0, NT, CHUNK):
    sl = slice(i0, min(i0 + CHUNK, NT))
    zw = ds['z_w'][sl].values
    u = np.nan_to_num(ds['u'][sl].values)
    v = np.nan_to_num(ds['v'][sl].values)
    Hmin = np.minimum(Hmin, (zw[:, -1] - zw[:, 0]).min(axis=0))
    Ub[sl] = bin_mean(u, 0.5 * (zw[..., :-1] + zw[..., 1:]))
    Vb[sl] = bin_mean(v, 0.5 * (zw[..., :-1, :] + zw[..., 1:, :]))
    print('  read %s to %s' % (ot[sl.start].date(), ot[sl.stop - 1].date()), flush=True)
ds.close()

MB = mask[None] & (Hmin[None] >= EDGES[1:, None, None])         # (NB, NR, NC) rho cells that have the bin
Ub *= (MB[:, :, :-1] & MB[:, :, 1:])[None]                      # faces next to a cell without the bin are walls
Vb *= (MB[:, :-1, :] & MB[:, 1:, :])[None]
for b in range(NB):
    print('bin %4.0f-%2.0f m: %3d rho cells' % (EDGES[b], EDGES[b + 1], MB[b].sum()))

# ------------------------------------------------------------------ STEP 2: Godin filter, vorticity, OW, rho velocities
NPAD = 35                                                       # half the Godin filter: NaN at each end
ok_t = slice(NPAD, NT - NPAD)
tt = ot[ok_t]
c4 = lambda a: 0.25 * (a[..., :-1, :-1] + a[..., :-1, 1:] + a[..., 1:, :-1] + a[..., 1:, 1:])   # rho -> psi
lon_p, lat_p, pm_p, pn_p, f_p = c4(lon), c4(lat), c4(pm), c4(pn), c4(f_cor)
area_p = 1 / (pm_p * pn_p)
to_x = lambda lo: (lo - LON_LP) * M_PER_DEG * np.cos(np.deg2rad(LAT0))
to_y = lambda la: (la - LAT0) * M_PER_DEG
xp, yp, xr_, yr_ = to_x(lon_p), to_y(lat_p), to_x(lon), to_y(lat)
DX = float(np.median(1 / pm[mask]))


def fields(b):
    """Godin-filtered bin b -> zeta, OW (t, NR-1, NC-1) at psi (NaN where not all four rho cells have the bin) and
    u, v (t, NR, NC) at rho (NaN at the box edge)"""
    Ug = zfun.lowpass(Ub[:, b].astype(float), f='godin')[ok_t]
    Vg = zfun.lowpass(Vb[:, b].astype(float), f='godin')[ok_t]
    dvdx = (Vg[:, :, 1:] - Vg[:, :, :-1]) * pm_p
    dudy = (Ug[:, 1:, :] - Ug[:, :-1, :]) * pn_p
    zeta, ss = dvdx - dudy, dvdx + dudy                         # vorticity, shear strain
    sn = np.full((len(Ug), NR, NC), np.nan)                     # normal strain at rho
    sn[:, 1:-1, 1:-1] = ((Ug[:, 1:-1, 1:] - Ug[:, 1:-1, :-1]) * pm[1:-1, 1:-1]
                         - (Vg[:, 1:, 1:-1] - Vg[:, :-1, 1:-1]) * pn[1:-1, 1:-1])
    ow = c4(sn) ** 2 + ss ** 2 - zeta ** 2
    wet4 = c4(MB[b].astype(float)) == 1
    ur, vr = np.full((len(Ug), NR, NC), np.nan), np.full((len(Ug), NR, NC), np.nan)
    ur[:, :, 1:-1] = 0.5 * (Ug[:, :, :-1] + Ug[:, :, 1:])
    vr[:, 1:-1, :] = 0.5 * (Vg[:, :-1, :] + Vg[:, 1:, :])
    return np.where(wet4, zeta, np.nan), np.where(wet4, ow, np.nan), ur, vr, wet4

# ------------------------------------------------------------------ STEP 3 + 4: centre, azimuthal profile, presence
MIN_DISK = 0.5                                                  # a centre's disk must be at least half wet
NRING = len(RRING)


def detect(zeta, ow, ur, vr, wet4, b):
    """one bin: hourly centre (largest disk-mean zeta), u_th(r) about it, r_max, U_max, quadrant closure"""
    # disk weights: search psi point s -> wet psi points within R0
    srch = np.flatnonzero((wet4 & (lon_p < LON_CP)).ravel())
    wetp = wet4.ravel()
    d = np.hypot(xp.ravel()[srch, None] - xp.ravel()[None], yp.ravel()[srch, None] - yp.ravel()[None])
    Wd = np.where((d <= R0) & wetp[None], area_p.ravel()[None], 0.0)          # (n_srch, n_psi)
    full = (d <= R0).astype(float) @ area_p.ravel()                          # disk area, land included
    keep = Wd.sum(1) >= MIN_DISK * full
    srch, Wd = srch[keep], Wd[keep]
    zR = np.nan_to_num(zeta.reshape(len(zeta), -1)) @ Wd.T / Wd.sum(1)        # (t, n_srch) disk-mean zeta
    ic = np.argmax(SENSE * zR, axis=1)                                        # centre each hour
    kc = srch[ic]                                                             # flat psi index of the centre
    out = {'x_c': xp.ravel()[kc], 'y_c': yp.ravel()[kc], 'lon_c': lon_p.ravel()[kc], 'lat_c': lat_p.ravel()[kc],
           'zeta_c_f': zR[np.arange(len(zR)), ic] / f_p.ravel()[kc], 'ow_c': ow.reshape(len(ow), -1)[np.arange(len(ow)), kc]}
    # ring and quadrant weights about each candidate centre (tangential unit vector folded in), applied hour by hour
    cell = np.flatnonzero((MB[b] & np.isfinite(ur[0]) & np.isfinite(vr[0])).ravel())
    U, V = ur.reshape(len(ur), -1)[:, cell], vr.reshape(len(vr), -1)[:, cell]
    uth = np.full((len(ur), NRING), np.nan)
    uq = np.full((len(ur), NRING, 4), np.nan)
    for s in np.unique(ic):
        h = np.flatnonzero(ic == s)
        dx_, dy_ = xr_.ravel()[cell] - xp.ravel()[srch[s]], yr_.ravel()[cell] - yp.ravel()[srch[s]]
        r = np.hypot(dx_, dy_)
        ring = np.searchsorted(RRING, r)                                     # 0 .. NRING-1, NRING = outside
        quad = np.clip(((np.arctan2(dy_, dx_) + np.pi) // (np.pi / 2)).astype(int), 0, 3)
        tx, ty = SENSE * -dy_ / r, SENSE * dx_ / r
        for j in range(NRING):
            m = ring == j
            if m.any():
                uth[h, j] = (U[h][:, m] * tx[m] + V[h][:, m] * ty[m]).mean(axis=1)
            for q in range(4):
                mq = m & (quad == q)
                if mq.any():
                    uq[h, j, q] = (U[h][:, mq] * tx[mq] + V[h][:, mq] * ty[mq]).mean(axis=1)
    jm = np.nanargmax(uth, axis=1)                                            # ring of the fastest swirl
    th = np.arange(len(uth))
    out['u_th'] = uth
    out['r_max'] = RRING[jm] - 0.5 * DR                                       # ring mid-radius
    out['U_max'] = uth[th, jm]
    out['n_quad'] = (uq[th, jm] > 0).sum(axis=1)                              # empty quadrant (NaN) counts as open
    out['circ'] = 2 * np.pi * out['r_max'] * out['U_max']
    out['T_turn'] = np.where(out['U_max'] > 0, 2 * np.pi * out['r_max'] / np.where(out['U_max'] > 0, out['U_max'], 1) / 3600, np.nan)
    out['present'] = (out['ow_c'] < 0) & (out['n_quad'] == 4)
    out['edge'] = jm == NRING - 1                                             # fastest swirl at the last ring
    return out

# ------------------------------------------------------------------ STEP 5: events, save
def runs(x):
    """[(start, stop), ...] of True runs in x (stop exclusive)"""
    k = np.flatnonzero(np.diff(np.r_[0, x.astype(int), 0]))
    return list(zip(k[::2], k[1::2]))


VN = ['x_c', 'y_c', 'lon_c', 'lat_c', 'zeta_c_f', 'ow_c', 'r_max', 'U_max', 'n_quad', 'circ', 'T_turn', 'present', 'edge']
R = {vn: np.full((len(tt), NB), np.nan) for vn in VN + ['present_ev']}
UTH = np.full((len(tt), NB, NRING), np.nan)
ZM = np.full((NB, NR - 1, NC - 1), np.nan)
EV = []
zc = 0.5 * (EDGES[1:] + EDGES[:-1])
for b in range(NB):
    zeta, ow, ur, vr, wet4 = fields(b)
    if (wet4 & (lon_p < LON_CP)).sum() == 0:
        print('bin %2.0f-%2.0f m: no inner-cove psi points' % (EDGES[b], EDGES[b + 1]))
        continue
    ZM[b] = np.nanmean(zeta, axis=0)
    o = detect(zeta, ow, ur, vr, wet4, b)
    for vn in VN:
        R[vn][:, b] = o[vn]
    UTH[:, b] = o['u_th']
    pe = np.zeros(len(tt), bool)
    for a, e in runs(o['present']):
        Tt = np.nanmedian(o['T_turn'][a:e])
        if e - a >= MIN_TURN * Tt:
            pe[a:e] = True
            EV.append({'z_top': EDGES[b], 'z_bot': EDGES[b + 1], 'start': tt[a], 'end': tt[e - 1], 'dur_h': e - a,
                       'T_turn_h': Tt, 'n_turns': (e - a) / Tt, 'lon_c': o['lon_c'][a:e].mean(), 'lat_c': o['lat_c'][a:e].mean(),
                       'r_max': np.median(o['r_max'][a:e]), 'U_max': np.median(o['U_max'][a:e]),
                       'zeta_c_f': np.median(o['zeta_c_f'][a:e])})
    R['present_ev'][:, b] = pe
    mo = tt.month
    P = o['present'] & pe
    print('bin %2.0f-%2.0f m: present %4.1f%% (raw %4.1f%%; Nov-Dec %4.1f, Aug-Oct %4.1f) | %3d events | when present: '
          'centre (%.4f, %.4f) r_max %3.0f m  U_max %.1f cm/s  zeta_c/f %.2f  T_turn %.0f h | edge-r_max %.0f%%'
          % (EDGES[b], EDGES[b + 1], 100 * pe.mean(), 100 * o['present'].mean(), 100 * pe[mo.isin([11, 12])].mean(),
             100 * pe[mo.isin([8, 9, 10])].mean(), sum(1 for x in EV if x['z_top'] == EDGES[b]),
             np.median(o['lon_c'][P]), np.median(o['lat_c'][P]), np.median(o['r_max'][P]), 100 * np.median(o['U_max'][P]),
             np.median(o['zeta_c_f'][P]), np.median(o['T_turn'][P]), 100 * o['edge'][P].mean()), flush=True)
    del zeta, ow, ur, vr

dout = xr.Dataset({vn: (('time', 'z'), R[vn]) for vn in R}, coords={'time': tt, 'z': zc, 'r': RRING - 0.5 * DR})
dout['u_th'] = (('time', 'z', 'r'), UTH)
dout['zeta_mean'] = (('z', 'eta_psi', 'xi_psi'), ZM)
dout['lon_psi'], dout['lat_psi'] = (('eta_psi', 'xi_psi'), lon_p), (('eta_psi', 'xi_psi'), lat_p)
dout['f_psi'] = (('eta_psi', 'xi_psi'), f_p)
dout['lon_rho'], dout['lat_rho'] = (('eta_rho', 'xi_rho'), lon), (('eta_rho', 'xi_rho'), lat)
dout['mask_bin'] = (('z', 'eta_rho', 'xi_rho'), MB.astype(int))
dout.attrs = {'gtagex': GTAGEX, 'edges_m': EDGES, 'R0_m': R0, 'DR_m': DR, 'sense': SENSE, 'min_turn': MIN_TURN,
              'lon_pc_cp': LON_CP, 'units': 'x_c, y_c, r_max m; U_max, u_th m/s; circ m2/s; T_turn h'}
dout.to_netcdf(out_dir / 'eddy_hourly.nc')
pd.DataFrame(EV).to_csv(out_dir / 'events.csv', index=False, float_format='%.5g')
print('saved %s, %s' % (out_dir / 'eddy_hourly.nc', out_dir / 'events.csv'))
