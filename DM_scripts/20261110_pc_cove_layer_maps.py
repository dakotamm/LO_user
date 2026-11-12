"""
Hourly layer velocities at every Penn Cove water cell, for flood / ebb / residual maps
(20261110_pc_cove_layer_maps_plot.py), of wb1_t0_xn11abbur00. Step 3 of the whole-cove
plan; the column analyses are 20261110_pc_column_metrics.py and _u0u1u2.py.

VELOCITY AT RHO POINTS from the native faces, masked faces = 0 (no normal flow at a
wall), as 20261008_tidal_ellipses.py: u_rho = mean of the faces west and east, v_rho = of
the faces south and north. u east, v north (wb1 is plaid, no rotation).

LAYERS DEFINED BY THE FLOW, not by a fixed thickness: each rho column xi has a divide
depth D(xi) BELOW THE MOVING SURFACE = the median two-layer divide of the u-face columns
either side (20261110_pc_column_metrics_plot.py, column_metrics_alongcove.csv, all hours;
rho column 1 takes u-face column 1). Upper = 0..D, lower = D..bed; a cell straddling D is
shared by overlap. Where the water is shallower than D the lower layer is empty (H = 0).

SAVED HOURLY per cell and layer (upper, lower):
    Ux = int u dz, Vy = int v dz over the layer  [m2/s]  (transport per unit width)
    H  = layer thickness [m]
so a composite is sum_t Ux / sum_t H (transport-weighted, as the section scripts), the
depth mean is (Ux_up + Ux_lo) / (H_up + H_lo), and the Godin filter can be taken per cell.

CENTRE-LINE CURTAIN (head to mouth, the cells of 20261103_pc_alongchannel_salt.py: per rho
column the cell midway between the shores, or the mean of the two either side): the
along-cove component u_al = u ax + v ay, (ax, ay) = (-0.9789, -0.2043), + toward the head,
composited per s-level over flood / ebb / all hours as sum_t u_al dz / sum_t dz, with the
phase-mean cell interfaces. Flood / ebb = sign of the hourly mouth Q (pc_lp, box xi_u 30).

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (hourly ocean_his snapshots)
       LO_output/DM_outs/20261110_pc_column_metrics/column_metrics_alongcove.csv
OUTPUT LO_output/DM_outs/20261110_pc_cove_layer_maps/
    layers_hourly.nc   Ux, Vy, H (time, layer, cell) + cell lon/lat/eta/xi/h, D
    curtain.p          centre-line composites
    curtain_hourly.nc  the hourly centre-line u_al and z_w (time, level, column), for condition
                       composites (-cycle in the plot script)

run 20261110_pc_cove_layer_maps.py
"""
import pickle
from time import time as clock

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                 # pc_lp section-positive (east) is out of the cove
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head
MOUTH = dict(XI_U=30, ETA=np.arange(19, 7, -1))
LAYERS = ['upper', 'lower']
BLK = 744                                 # hours per read (31 days)

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
div_fn = Ldir['LOo'] / 'DM_outs' / '20261110_pc_column_metrics' / 'column_metrics_alongcove.csv'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_cove_layer_maps'
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ cells, divide, centre line
ds = xr.open_dataset(in_fn)
time = ds.ocean_time.to_numpy()
NT = len(time)
mask = ds.mask_rho.values == 1
cove = mask.copy()
cove[:, 31:] = False                      # the cove: rho columns <= 30
JJ, II = np.nonzero(cove)                 # (cell,) indices
NCELL = len(JJ)
dv = pd.read_csv(div_fn).set_index('col').div_med_all
D_xi = {i: (dv[1] if i == 1 else 0.5 * (dv[i - 1] + dv[i])) for i in range(1, 31)}
D = np.array([D_xi[i] for i in II])       # (cell,) divide depth below the surface [m]

XC, JC = [], []                           # centre line, head to mouth (20261103_pc_alongchannel_salt.py)
for i in range(1, 31):
    jw = np.flatnonzero(cove[:, i])
    mid = 0.5 * (jw.min() + jw.max())
    XC.append(i)
    JC.append(np.unique([int(np.floor(mid)), int(np.ceil(mid))]))
print('%d cove cells; divide %.1f m (head) to %.1f m (mouth); centre line %d columns' % (NCELL, D_xi[1], D_xi[30], len(XC)))


def overlap(ztop, zbot, a, b):
    """thickness of each cell [ztop, zbot] (depths, + down) inside the depth interval [a, b]"""
    return np.clip(np.minimum(zbot, b) - np.maximum(ztop, a), 0, None)


# ------------------------------------------------------------------ main: read the box a block of hours at a time
OUT = {k: np.full((NT, 2, NCELL), np.nan, np.float32) for k in ['Ux', 'Vy', 'H']}
NZ = ds.s_rho.size
CURH = dict(ual=np.full((NT, NZ, len(XC)), np.nan, np.float32), zw=np.full((NT, NZ + 1, len(XC)), np.nan, np.float32))
CUR = {ph: dict(num=np.zeros((NZ, len(XC))), den=np.zeros((NZ, len(XC))), zw=np.zeros((NZ + 1, len(XC))), n=0)
       for ph in ['flood', 'ebb', 'all']}
Qm_all = np.full(NT, np.nan)
dy_m = (1 / ds.pn.values[MOUTH['ETA']][:, [30, 31]]).mean(axis=-1)
tt0 = clock()
for t0 in range(0, NT, BLK):
    tt = slice(t0, min(t0 + BLK, NT))
    U = np.nan_to_num(ds.u.isel(ocean_time=tt).to_numpy().astype(float))     # (t, k, eta_u, xi_u)
    V = np.nan_to_num(ds.v.isel(ocean_time=tt).to_numpy().astype(float))     # (t, k, eta_v, xi_v)
    ZW = ds.z_w.isel(ocean_time=tt).to_numpy()                                # (t, w, eta, xi)
    nt = U.shape[0]

    # mouth transport: the flood / ebb clock
    dzm = np.diff(ZW[:, :, MOUTH['ETA']][:, :, :, [30, 31]], axis=1).mean(axis=-1)
    Qm = SIGN * (U[:, :, MOUTH['ETA'], 30] * dzm * dy_m[None, None, :]).sum(axis=(1, 2))
    Qm_all[tt] = Qm

    # velocity at rho points, faces at walls = 0
    ur = np.zeros(U.shape[:2] + ZW.shape[2:])
    vr = np.zeros(U.shape[:2] + ZW.shape[2:])
    ur[..., 1:31] = 0.5 * (U[..., 0:30] + U[..., 1:31])     # rho i: faces xi_u i-1 (west) and i (east)
    vr[:, :, 1:20, :] = 0.5 * (V[:, :, 0:19, :] + V[:, :, 1:20, :])

    # layers, per cell
    zw = ZW[:, :, JJ, II]                                    # (t, w, cell)
    dw = zw[:, -1:, :] - zw                                  # depth of each w level below the surface
    dtop, dbot = dw[:, 1:, :], dw[:, :-1, :]                 # (t, k, cell), k = 0 at the bed
    uc, vc = ur[:, :, JJ, II], vr[:, :, JJ, II]
    for n, (a, b) in enumerate([(0, D), (D, np.inf)]):
        ov = overlap(dtop, dbot, a, b)
        OUT['Ux'][tt, n] = (uc * ov).sum(axis=1)
        OUT['Vy'][tt, n] = (vc * ov).sum(axis=1)
        OUT['H'][tt, n] = ov.sum(axis=1)

    # centre-line curtain composites
    ual = np.stack([(ur[:, :, j, i] * AXIS[0] + vr[:, :, j, i] * AXIS[1]).mean(axis=-1) for i, j in zip(XC, JC)], axis=-1)   # (t, k, n)
    zc = np.stack([ZW[:, :, j, i].mean(axis=-1) for i, j in zip(XC, JC)], axis=-1)                                          # (t, w, n)
    dzc = np.diff(zc, axis=1)
    CURH['ual'][tt], CURH['zw'][tt] = ual, zc
    for ph, m in [('flood', Qm > 0), ('ebb', Qm < 0), ('all', np.ones(nt, bool))]:
        CUR[ph]['num'] += (ual[m] * dzc[m]).sum(axis=0)
        CUR[ph]['den'] += dzc[m].sum(axis=0)
        CUR[ph]['zw'] += zc[m].sum(axis=0)
        CUR[ph]['n'] += m.sum()
    print('  hours %d-%d of %d, %.0f s' % (tt.start, tt.stop, NT, clock() - tt0))
lon, lat, h = ds.lon_rho.values, ds.lat_rho.values, ds.h.values
pm, pn = ds.pm.values, ds.pn.values
ds.close()

# ------------------------------------------------------------------ save
dsout = xr.Dataset({k: (('time', 'layer', 'cell'), OUT[k], {'units': 'm' if k == 'H' else 'm2/s'}) for k in OUT},
                   coords={'time': time, 'layer': LAYERS, 'cell': np.arange(NCELL)},
                   attrs={'source': str(in_fn),
                          'note': 'Ux, Vy = layer-integrated u (east), v (north) at rho points; see the docstring of 20261110_pc_cove_layer_maps.py'})
dsout['Qmouth'] = (('time',), Qm_all, {'units': 'm3/s', 'long_name': 'hourly transport through pc_lp, + into the cove'})
for k, a in [('eta', JJ), ('xi', II), ('lon', lon[JJ, II]), ('lat', lat[JJ, II]), ('h', h[JJ, II]), ('D', D),
             ('dx', 1 / pm[JJ, II]), ('dy', 1 / pn[JJ, II])]:
    dsout[k] = (('cell',), a)
dsout.to_netcdf(out_dir / 'layers_hourly.nc')
C = {ph: dict(u=c['num'] / c['den'], zw=c['zw'] / c['n'], n=c['n']) for ph, c in CUR.items()}
dxc = np.array([np.mean(1 / pm[j, i]) for i, j in zip(XC, JC)])
with open(out_dir / 'curtain.p', 'wb') as f:
    pickle.dump(dict(C=C, xi=np.array(XC), jc=JC, dx=dxc, D=np.array([D_xi[i] for i in XC]),
                     h=np.array([h[j, i].mean() for i, j in zip(XC, JC)])), f)
xr.Dataset({'ual': (('time', 's_rho', 'col'), CURH['ual'], {'units': 'm/s', 'long_name': 'along-cove u, + toward the head'}),
            'zw': (('time', 's_w', 'col'), CURH['zw'], {'units': 'm'})},
           coords={'time': time, 'col': np.array(XC)},
           attrs={'note': 'centre-line cells of 20261110_pc_cove_layer_maps.py, hourly; for condition composites'}
           ).to_netcdf(out_dir / 'curtain_hourly.nc')
print('saved to ' + str(out_dir))

# checks: layers fill the water column; flood / ebb hour counts
Htot = OUT['H'].sum(axis=1)
print('layers fill the column: mean H_up + H_lo %.2f m vs mean h %.2f m (the difference is mean zeta)' % (np.nanmean(Htot), h[JJ, II].mean()))
print('flood %d h, ebb %d h; curtain u_al mean (all) %.4f m/s' % (C['flood']['n'], C['ebb']['n'], np.nanmean(C['all']['u'])))
