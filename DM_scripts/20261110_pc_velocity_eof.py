"""
EOFs (empirical orthogonal functions) of Penn Cove velocity, wb1_t0_xn11abbur00 2024-2025,
done the way of the UW data-analysis Lab 8-1 (SVD of a space x time anomaly matrix):
    X (space, time) = anomalies
    U, S, V = svd(X, full_matrices=False)
    U[:, i] = EOF i (spatial pattern),  V[i, :] = PC i (its time series)
    var_exp = S**2 / sum(S**2)
    X ~= U[:, :k] * S[:k] @ V[:k, :]     (reconstruction from the first k modes)

TWO FAMILIES OF FIELD
  section  the velocity NORMAL to pc_cp, pc_lj, pc_lp (box xi_u 15, 23, 30), + into the cove,
           at every element (s-level k, face p) of the section; one SVD per section. The
           elements are the space points, as in MacCready 2011 Sec. 2
           (20261110_pc_column_u0u1u2.py).
  cove     u (east) and v (north) of BOTH flow-defined layers (upper, lower) at every cove
           rho cell, stacked into one column [u_up; v_up; u_lo; v_lo] (a vector EOF, real
           SVD). The layer velocity Ux / H is taken from layers_hourly.nc of
           20261110_pc_cove_layer_maps.py. The lower layer empties at low water in 38 shallow
           cells; those hours are bridged by linear interpolation in time, as that script's
           ratio map.

DEPARTURES FROM THE LAB, and why
  1. Remove the time mean only; do NOT standardize. The lab divided each pixel by its std so
     that wet mountain pixels would not swamp the rest. Velocity has one unit everywhere, and
     the energetic parts of the flow should lead the modes. The time mean (the residual
     circulation) is saved as mode 0.
  2. Weight each row by sqrt(w) before the SVD, so that S**2 is area- (section) or volume-
     (cove) integrated variance. w = time-mean element area dA0 (section) or dx dy Hbar (cove,
     Hbar = time-mean layer thickness). Without it the thin surface sigma layers and the
     small cells would count as much as the big ones. The weights are divided back out of
     every EOF that is saved.
  3. Band (-band): hourly = everything, so EOF 1 is the tide. subtidal = Godin low-pass, the
     variability of the residual circulation. The 35 h NaN ends are dropped.
  4. Scaling for plotting: EOF_i = U[:, i] * S[i] / sqrt(nt) / sqrt(w)  [m/s per 1 std of PC]
     and PC_i = V[i, :] * sqrt(nt)  [unit variance]. The product is unchanged, so this is
     still exactly U S V. The raw U, S, V are saved too.
  5. Sign (the lab: flipping both EOF and PC changes nothing): each mode is flipped so that
     PC > 0 means UPPER-LAYER water moving toward the head. Section: the dA0-weighted mean of
     the EOF above the column's median two-layer divide (column_metrics_alongcove.csv of
     20261110_pc_column_metrics) is positive. (The top s-level alone flipped modes with
     little surface signal, e.g. a slab mode at pc_cp.) Cove: the area-weighted along-cove component of the
     upper-layer EOF is positive.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (section u, z_w)
       LO_output/DM_outs/20261110_pc_cove_layer_maps/layers_hourly.nc (cove layers, Qmouth)
OUTPUT LO_output/DM_outs/20261110_pc_velocity_eof/eof_<band>.p
       a dict with, per field (pc_cp, pc_lj, pc_lp, cove): eof (space, NM) physical, pc (NM, t),
       var_exp (all modes), S, U (space, NM) and V (NM, t) raw, w, mean, geometry, time.
       Plotted by 20261110_pc_velocity_eof_plot.py.

run 20261110_pc_velocity_eof.py -band hourly
run 20261110_pc_velocity_eof.py -band subtidal
"""
import argparse
import pickle
from time import time as clock

import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import svd

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-band', default='hourly', choices=['hourly', 'subtidal'])
BAND = parser.parse_args().band

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                 # section-positive (east) is out of the cove
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
NM = 10                                   # modes saved
BLK = 744                                 # hours per read (31 days)

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
div_fn = Ldir['LOo'] / 'DM_outs' / '20261110_pc_column_metrics' / 'column_metrics_alongcove.csv'
lay_fn = Ldir['LOo'] / 'DM_outs' / '20261110_pc_cove_layer_maps' / 'layers_hourly.nc'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_velocity_eof'
out_dir.mkdir(parents=True, exist_ok=True)


def bridge(a):
    """linear interpolation in time (axis 0) across NaN hours, so the Godin filter does not spread NaN"""
    sh = a.shape
    return pd.DataFrame(a.reshape(sh[0], -1)).interpolate(limit_direction='both').to_numpy().reshape(sh)


def eof(data, w, sgn_fn):
    """data (time, space) hourly field, w (space,) weights, sgn_fn(eof_phys) -> the number whose
    sign must be + (upper layer toward the head). Returns the dict saved per field."""
    # band, then anomalies: X (space, time) as the lab, the time mean removed per row
    if BAND == 'subtidal':
        data = zfun.lowpass(data, f='godin')
    ok = np.isfinite(data).all(axis=1)                        # drops the Godin NaN ends
    mean = data[ok].mean(axis=0)
    X = (data[ok] - mean).T * np.sqrt(w)[:, None]             # (space, time), weighted
    nt = X.shape[1]

    # SVD: U = EOFs, S = singular values, V = PCs (the lab's names; scipy returns V as the rows)
    U, S, V = svd(X, full_matrices=False)
    var_exp = S**2 / np.sum(S**2)

    # check: all modes give X back, and how much the first NM modes keep
    err = np.abs(np.dot(U * S, V) - X).max() / np.abs(X).max()
    Xk = np.dot(U[:, :NM] * S[:NM], V[:NM, :])
    print('    reconstruction error (all modes) %.1e; first %d modes keep %.1f%% of the variance'
          % (err, NM, 100 * (1 - ((X - Xk)**2).sum() / (X**2).sum())))

    # keep NM modes; sign so that PC > 0 = upper layer toward the head
    U, V = U[:, :NM].copy(), V[:NM, :].copy()
    for i in range(NM):
        if sgn_fn(U[:, i] / np.sqrt(w)) < 0:
            U[:, i], V[i, :] = -U[:, i], -V[i, :]

    # physical scaling: EOF in m/s per 1 std of PC, PC unit variance; EOF @ PC = U S V exactly
    eof_phys = U * S[:NM] / np.sqrt(nt) / np.sqrt(w)[:, None]
    pc = V * np.sqrt(nt)
    print('    ' + ', '.join('EOF%d %.1f%%' % (i + 1, 100 * var_exp[i]) for i in range(5)))
    return dict(eof=eof_phys, pc=pc, var_exp=var_exp, S=S, U=U, V=V, w=w, mean=mean, ok=ok)


# ------------------------------------------------------------------ section fields: u normal to the section at every element (k, p)
ds = xr.open_dataset(in_fn)
time = ds.ocean_time.to_numpy()
NT = len(time)
mask = ds.mask_rho.values == 1
pn = ds.pn.values
SEC = {}
for SN, XI_U in SECT_XI_U.items():
    ETA = np.flatnonzero(mask[:, XI_U] & mask[:, XI_U + 1])[::-1]        # north to south
    SEC[SN] = dict(XI_U=XI_U, ETA=ETA, dy=(1 / pn[ETA][:, [XI_U, XI_U + 1]]).mean(axis=-1),
                   u=np.empty((NT, 30, len(ETA)), np.float32), dz=np.empty((NT, 30, len(ETA)), np.float32))
tt0 = clock()
for t0 in range(0, NT, BLK):
    tt = slice(t0, min(t0 + BLK, NT))
    U = ds.u.isel(ocean_time=tt).to_numpy()
    ZW = ds.z_w.isel(ocean_time=tt).to_numpy()
    for C in SEC.values():
        ETA, XI_U = C['ETA'], C['XI_U']
        C['u'][tt] = SIGN * U[:, :, ETA, XI_U]
        C['dz'][tt] = np.diff(ZW[:, :, ETA][:, :, :, [XI_U, XI_U + 1]], axis=1).mean(axis=-1)
    print('  hours %d-%d of %d, %.0f s' % (tt.start, tt.stop, NT, clock() - tt0))
ds.close()

OUT = dict(band=BAND, time=time, NM=NM)
DIV = pd.read_csv(div_fn).set_index('col').div_med_all                  # median two-layer divide [m below the surface]
for SN, C in SEC.items():
    print('%s (%d faces x 30 levels), %s' % (SN, len(C['ETA']), BAND))
    nk, npf = C['u'].shape[1:]
    dzm = C['dz'].astype(float).mean(axis=0)                             # (k, p) time-mean thickness, k = 0 at the bed
    dA0 = dzm * C['dy'][None, :]                                         # time-mean element area
    z0 = np.cumsum(dzm, axis=0) - 0.5 * dzm - dzm.sum(axis=0)            # cell-centre z, 0 at the mean surface
    top = z0 > -DIV[C['XI_U']]                                           # the upper layer: above the median divide
    sgn = lambda e, top=top.ravel(), a=dA0.ravel(): (e[top] * a[top]).sum()
    R = eof(C['u'].reshape(NT, -1).astype(float), dA0.ravel(), sgn)
    R.update(shape=(nk, npf), dy=C['dy'], z0=z0, dA0=dA0, D=DIV[C['XI_U']], ETA=C['ETA'], XI_U=C['XI_U'])
    OUT[SN] = R
del SEC


# ------------------------------------------------------------------ cove field: [u_up; v_up; u_lo; v_lo] at every cove rho cell
d = xr.open_dataset(lay_fn)
assert (d.time.values == time).all()
Ux, Vy, H = d.Ux.values.astype(float), d.Vy.values.astype(float), d.H.values.astype(float)   # (t, layer, cell)
NCELL = H.shape[-1]
area = (d.dx.values * d.dy.values)                                       # (cell,)
print('cove (%d cells x 2 layers x u, v), %s; lower layer empty in %.2f%% of cell-hours'
      % (NCELL, BAND, 100 * (H[:, 1] == 0).mean()))
with np.errstate(invalid='ignore', divide='ignore'):
    u, v = np.where(H > 0, Ux / H, np.nan), np.where(H > 0, Vy / H, np.nan)
u, v = bridge(u), bridge(v)                                              # empty lower-layer hours, as the ratio map
data = np.concatenate([u[:, 0], v[:, 0], u[:, 1], v[:, 1]], axis=1)      # (t, 4 * cell)
Hbar = H.mean(axis=0)                                                    # (layer, cell), empty hours count as 0
w = np.concatenate([area * Hbar[0], area * Hbar[0], area * Hbar[1], area * Hbar[1]])
sgn = lambda e: ((e[:NCELL] * AXIS[0] + e[NCELL:2 * NCELL] * AXIS[1]) * area).sum()   # upper-layer along-cove, + toward the head
R = eof(data, w, sgn)
R.update(ncell=NCELL, order=['u_upper', 'v_upper', 'u_lower', 'v_lower'], area=area, Hbar=Hbar,
         eta=d.eta.values, xi=d.xi.values, lon=d.lon.values, lat=d.lat.values, h=d.h.values, D=d.D.values)
OUT['cove'] = R
OUT['Qmouth'] = d.Qmouth.values
d.close()

# ------------------------------------------------------------------ save
out_fn = out_dir / ('eof_%s.p' % BAND)
with open(out_fn, 'wb') as f:
    pickle.dump(OUT, f)
print('saved ' + str(out_fn))
