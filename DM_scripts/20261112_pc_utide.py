"""
Penn Cove tidal circulation by harmonic analysis (utide), built step by step.

STEP 1: load what utide needs from the pc_cove box extraction of wb1_t0_xn11abbur00
(2024-2025 hourly snapshots, native C-grid, no -uv_to_rho):
    time    ocean_time (UTC)
    lat     lat_rho (utide uses it for nodal corrections / inference)
    mask    mask_rho from the run output, not grid.nc
    zeta    sea surface height (rho)
    ubar, vbar   depth-averaged velocity (u- and v-faces)
    u, v    3D velocity (u- and v-faces), kept lazy: ~1.4 GB each

run 20261112_pc_utide.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')        # one BLAS thread: utide is ~3x faster this way
import multiprocessing as mp
import pickle
from time import time as clock

import numpy as np
import utide
import xarray as xr

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_' + DS0 + '_' + DS1 + '.nc')

ds = xr.open_dataset(box_fn)

T = ds.ocean_time.values
lon = ds.lon_rho.values
lat = ds.lat_rho.values
mask = ds.mask_rho.values == 1

zeta = ds.zeta.values
ubar = ds.ubar.values
vbar = ds.vbar.values

u = ds.u
v = ds.v

print('%d hours, %s to %s' % (len(T), str(T[0])[:16], str(T[-1])[:16]))
print('rho %s, %d wet' % (str(mask.shape), mask.sum()))
print('zeta %s  ubar %s  vbar %s' % (zeta.shape, ubar.shape, vbar.shape))
print('u %s  v %s (lazy)' % (u.shape, v.shape))

"""
STEP 2: depth-averaged velocity onto rho points, cove cells only.
    C-grid -> rho: average the two faces on either side of each rho cell. Masked
    (land) faces are set to 0 first, the no-normal-flow wall value, so a cell
    against the shore gets the average of the wall and the interior face.
    The box edge ring has only one face and stays NaN (it is all land in the cove).
    wb1 is a plaid lon/lat grid (no angle), so u is east, v is north.
    Cove only: xi_rho <= 30. Column 31 is in the box only to carry the pc_lp
    u-faces (xi_u 30); its wet cells are Saratoga Passage.
"""
NT, NR, NC = len(T), mask.shape[0], mask.shape[1]

def to_rho(uf, vf):
    """(NT, NR, NC-1) u-faces, (NT, NR-1, NC) v-faces -> (NT, NR, NC) rho."""
    uf = np.nan_to_num(uf, nan=0.0)
    vf = np.nan_to_num(vf, nan=0.0)
    ur = np.full((uf.shape[0], NR, NC), np.nan, dtype=np.float32)
    vr = np.full((vf.shape[0], NR, NC), np.nan, dtype=np.float32)
    ur[:, :, 1:-1] = 0.5 * (uf[:, :, :-1] + uf[:, :, 1:])
    vr[:, 1:-1, :] = 0.5 * (vf[:, :-1, :] + vf[:, 1:, :])
    return ur, vr

ubar_r, vbar_r = to_rho(ubar, vbar)

cove = mask.copy()
cove[:, 31:] = False
jj, ii = np.nonzero(cove)

# (time, cell) arrays: one column per cove cell, ready for utide
Z = zeta[:, jj, ii]
U = ubar_r[:, jj, ii]
V = vbar_r[:, jj, ii]
LATc = lat[jj, ii]

print('cove cells %d (Saratoga dropped %d)' % (cove.sum(), mask.sum() - cove.sum()))
print('Z %s  U %s  V %s' % (Z.shape, U.shape, V.shape))
print('NaN: Z %d  U %d  V %d' % (np.isnan(Z).sum(), np.isnan(U).sum(), np.isnan(V).sum()))

"""
STEP 3: utide fit per cove cell.
    zeta:        scalar fit -> amplitude A [m], Greenwich phase g [deg]
    (ubar, vbar): 2-D fit -> ellipse: Lsmaj, Lsmin [m s-1] (Lsmin > 0 = CCW),
                 inclination theta [deg CCW from east], phase g [deg]
    Constituents: the 8 in the parent's tide forcing (tide00: M2 S2 N2 K2 K1 O1
    P1 Q1), the shallow-water overtides M4 MS4 M6, and S1. S1 is not a forced tide
    here; it is the daily (sea-breeze) wind band, and without it K1/P1 absorb it.
    2 years resolves S1 from K1 and P1 (Rayleigh 1/730 cpd), so nothing is inferred.
    Settings: OLS, nodal corrections at the record centre, no trend, linear 95% CI.
    Run in parallel over cells (fork: workers inherit Z/U/V, nothing is pickled).
"""
CONS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'MS4', 'M6', 'S1']
KW = dict(method='ols', conf_int='linear', constit=CONS, nodal=True, trend=False, verbose=False)
Z_OUT = ['A', 'g', 'A_ci', 'g_ci']
UV_OUT = ['Lsmaj', 'Lsmin', 'theta', 'g', 'Lsmaj_ci', 'Lsmin_ci', 'theta_ci', 'g_ci']
NPROC = 8

def reorder(c, names):
    """utide sorts constituents by percent energy (differs cell to cell); put them back in CONS order."""
    ii = [list(c.name).index(n) for n in CONS]
    return np.array([np.asarray(c[vn])[ii] for vn in names])

def fit_z(k):
    return reorder(utide.solve(T, Z[:, k], lat=LATc[k], **KW), Z_OUT)

def fit_uv(k):
    return reorder(utide.solve(T, U[:, k], V[:, k], lat=LATc[k], **KW), UV_OUT)

NK = len(jj)
tt = clock()
with mp.get_context('fork').Pool(NPROC) as pool:
    fz = np.array(pool.map(fit_z, range(NK)))      # (cell, Z_OUT, con)
    fuv = np.array(pool.map(fit_uv, range(NK)))    # (cell, UV_OUT, con)
print('%d zeta + %d ubar fits in %.0f s' % (NK, NK, clock() - tt))

FIT = {'cons': CONS, 'jj': jj, 'ii': ii, 'lon': lon[jj, ii], 'lat': LATc,
       'zeta': {vn: fz[:, n, :] for n, vn in enumerate(Z_OUT)},
       'bar': {vn: fuv[:, n, :] for n, vn in enumerate(UV_OUT)},
       'kw': {k: v for k, v in KW.items() if k != 'constit'}}

out_dir = Ldir['LOo'] / 'DM_outs' / '20261112_pc_utide'
Lfun.make_dir(out_dir)
pickle.dump(FIT, open(out_dir / 'fits.p', 'wb'))
print('saved ' + str(out_dir / 'fits.p'))

# cove-median summary, cm and cm/s
print('%4s %8s %8s %8s %8s' % ('con', 'zeta A', 'Lsmaj', 'Lsmin', 'CCW%'))
for n, con in enumerate(CONS):
    print('%4s %8.2f %8.2f %8.2f %8.0f' % (con, 100 * np.median(FIT['zeta']['A'][:, n]),
        100 * np.median(FIT['bar']['Lsmaj'][:, n]), 100 * np.median(FIT['bar']['Lsmin'][:, n]),
        100 * (FIT['bar']['Lsmin'][:, n] > 0).mean()))
