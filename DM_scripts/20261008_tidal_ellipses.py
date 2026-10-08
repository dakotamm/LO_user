"""
TIDAL ELLIPSES (and cotidal zeta) from a box extraction, by harmonic analysis.

WHY THIS EXISTS
LO output carries no tidal constituents. wb1 is nested (forcing ocn = ocnN), so
the tide enters only through the open boundaries as hourly zeta/ubar/vbar from
the parent cas7 run, which is itself forced by tide00 with 8 TPXO
constituents (M2 S2 N2 K2 K1 O1 P1 Q1). Constituents therefore have to be
recovered from hourly model velocity by harmonic analysis. This script does that
with utide (Greenwich phase, nodal corrections at the record's central time),
one cell at a time, parallelized over cells.

INPUT: a box extraction made WITHOUT -uv_to_rho (native C-grid)
  -job pc_cove   Penn Cove, 2024-2025 hourly, already on the mac. Has ubar/vbar
                 AND 3D u/v, so the vertical structure of the ellipses is fit too.
  -job sp_head_tide  head of Saratoga Passage (sp_head.p), made on apogee
                 twice, with -surf True and with -bot True (commands in
                 extract/box/job_definitions.py). ubar/vbar plus top- and
                 bottom-layer u/v. The files are sp_head_tide_surf_*.nc and
                 sp_head_tide_bot_*.nc; the script finds and uses both.

RUN
  mac:    python 20261008_tidal_ellipses.py -gtx wb1_t0_xn11abbur00 -job pc_cove -0 2024.01.01 -1 2025.12.31 -Nproc 8
  apogee: python 20261008_tidal_ellipses.py -gtx wb1_t0_xn11abbur00 -job sp_head_tide -0 2024.01.01 -1 2025.12.31 -Nproc 20 > tidal_ellipses_sp_head.log &
utide costs ~0.17 s per cell for 2 years of hourly data: pc_cove (330 wet
cells x [zeta, ubar, 30 layers]) is ~10k fits; sp_head_tide is 3363 cells x
[zeta, ubar, surf, bot] = ~13k.

C-GRID -> RHO
u faces are averaged to rho points (and v likewise) with masked faces set to 0,
the no-normal-flow wall value, so a cell against the shore gets the linear
interpolation between the wall and the interior face. The outermost ring of the
box has only one face and is left NaN. mask_rho comes from the extraction, never
grid.nc (they disagree at 16 cells around Penn Cove, [[wb1-grid-vs-run-mask]]).
wb1 is a plaid lon/lat grid (no `angle`), so u is east and v is north with no
rotation.

OUTPUT: LO_output/extract/<gtx>/tidal_ellipses/<job>_<ds0>_<ds1>.nc
dims (con, eta_rho, xi_rho), plus s_rho for the 3D fit. For each velocity
field P in {bar, surf, bot, 3d}:
  P_Lsmaj, P_Lsmin  semi-major/minor axis [m s-1]; Lsmin > 0 = counterclockwise
  P_theta           inclination of the major axis [deg CCW from east, 0-180]
  P_g               Greenwich phase of max current along +theta [deg]
  and *_ci          95% confidence half-widths (utide 'linear')
  P_umean, P_vmean   EULERIAN residual: time mean of the rho-point velocity
                    over the record [m s-1]. Plain mean, not utide's mean term
                    (identical to within noise over a 2-year record). This is
                    NOT the Lagrangian residual -- Stokes drift is not in it.
zeta_A [m], zeta_g [deg] (+ _ci) for the cotidal maps. z0_rho (s_rho, eta,
xi) is the depth of each layer at zeta = 0, for plotting the 3D fit vs depth.

-mean_only True adds/overwrites only the *_umean/*_vmean fields in an EXISTING
output file -- no fits, a few minutes of reading. For files made before the
means were added.
"""

import sys
import argparse
import multiprocessing as mp
from time import time

import numpy as np
import xarray as xr
import utide

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-job', default='pc_cove', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-Nproc', default=8, type=int)
parser.add_argument('-do_3d', default=True, type=Lfun.boolean_string) # fit every layer of 3D u/v if present
parser.add_argument('-test', default=False, type=Lfun.boolean_string) # one month, every 4th cell
parser.add_argument('-mean_only', default=False, type=Lfun.boolean_string) # add Eulerian means to an existing output file
args = parser.parse_args()

# the 8 constituents in the parent's tide forcing (tide00) + shallow-water overtides
CONS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'MS4', 'M6']
KW = dict(method='ols', conf_int='linear', constit=CONS, nodal=True, trend=False, verbose=False)
UV_OUT = ['Lsmaj', 'Lsmin', 'theta', 'g', 'Lsmaj_ci', 'Lsmin_ci', 'theta_ci', 'g_ci']
Z_OUT = ['A', 'g', 'A_ci', 'g_ci']
MIN_GOOD = 0.9 # fraction of finite hours required to fit a cell

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'box'
out_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'tidal_ellipses'
Lfun.make_dir(out_dir)

dstr = args.ds0 + '_' + args.ds1
# '' = full 3D box (pc_cove); 'surf' / 'bot' = extract_box -surf / -bot runs
found = {k: box_dir / (args.job + ('_' + k if k else '') + '_' + dstr + '.nc') for k in ['', 'surf', 'bot']}
found = {k: fn for k, fn in found.items() if fn.is_file()}
if len(found) == 0:
    print('*** No box file for %s %s in %s' % (args.job, dstr, box_dir))
    sys.exit()
box_fn = list(found.values())[0] # zeta, ubar/vbar and the grid come from here
out_fn = out_dir / (args.job + '_' + dstr + ('_test' if args.test else '') + '.nc')
print('Reading ' + str(box_fn))

def open_box(fn):
    d = xr.open_dataset(fn)
    return d.isel(ocean_time=slice(0, 24 * 31 + 1)) if args.test else d

ds = open_box(box_fn)
T = ds.ocean_time.values
mask = ds.mask_rho.values == 1
lat = ds.lat_rho.values
NR, NC = mask.shape
wet = np.flatnonzero(mask)
if args.test:
    wet = wet[::4]
print('%d hours, %d x %d rho, %d wet cells to fit' % (len(T), NR, NC, len(wet)))

def to_rho(u, v):
    """(NT, NR, NC-1) u and (NT, NR-1, NC) v on faces -> (NT, NR, NC) at rho."""
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    ur = np.full((u.shape[0], NR, NC), np.nan, dtype=np.float32)
    vr = np.full((v.shape[0], NR, NC), np.nan, dtype=np.float32)
    ur[:, :, 1:-1] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    vr[:, 1:-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    return ur, vr

# globals read by the workers (inherited through fork, not pickled)
A = None
B = None
LAT = lat.ravel()[wet]

def fit_uv(k):
    a, b = A[:, k], B[:, k]
    if np.isfinite(a).mean() < MIN_GOOD:
        return np.full((len(UV_OUT), len(CONS)), np.nan)
    c = utide.solve(T, a, b, lat=LAT[k], **KW)
    ii = [list(c.name).index(n) for n in CONS]
    return np.array([np.asarray(c[vn])[ii] for vn in UV_OUT])

def fit_z(k):
    a = A[:, k]
    if np.isfinite(a).mean() < MIN_GOOD:
        return np.full((len(Z_OUT), len(CONS)), np.nan)
    c = utide.solve(T, a, lat=LAT[k], **KW)
    ii = [list(c.name).index(n) for n in CONS]
    return np.array([np.asarray(c[vn])[ii] for vn in Z_OUT])

def run(fun, a, b=None):
    """Fit every wet column of a (and b); returns (nout, ncon, NR, NC)."""
    global A, B
    A = a.reshape(a.shape[0], -1)[:, wet]
    B = None if b is None else b.reshape(b.shape[0], -1)[:, wet]
    tt0 = time()
    with mp.get_context('fork').Pool(args.Nproc) as pool:
        res = pool.map(fun, range(len(wet)), chunksize=max(1, len(wet) // (20 * args.Nproc)))
    res = np.stack(res, axis=-1) # (nout, ncon, nwet)
    full = np.full(res.shape[:2] + (NR * NC,), np.nan)
    full[:, :, wet] = res
    print('  %d fits in %0.1f s' % (len(wet), time() - tt0))
    sys.stdout.flush()
    return full.reshape(res.shape[:2] + (NR, NC))

def rho_mean(ur, vr):
    """Eulerian residual: time mean at each wet rho point."""
    return np.where(mask, np.nanmean(ur, axis=0), np.nan), np.where(mask, np.nanmean(vr, axis=0), np.nan)

dims2 = ('con', 'eta_rho', 'xi_rho')
if args.mean_only:
    print('Adding means to ' + str(out_fn))
    out = xr.load_dataset(out_fn)
else:
    out = xr.Dataset(coords={'con': CONS, 'lon_rho': (('eta_rho', 'xi_rho'), ds.lon_rho.values), 'lat_rho': (('eta_rho', 'xi_rho'), lat)})
    out['h'] = (('eta_rho', 'xi_rho'), ds.h.values)
    out['mask_rho'] = (('eta_rho', 'xi_rho'), ds.mask_rho.values)
    print('zeta')
    res = run(fit_z, ds.zeta.values)
    for i, vn in enumerate(Z_OUT):
        out['zeta_' + vn] = (dims2, res[i])

pairs = [('bar', ds, 'ubar', 'vbar')]
for k in ['surf', 'bot']: # -surf / -bot boxes: u/v are one layer only
    if k in found:
        print('Reading ' + str(found[k]))
        dk = open_box(found[k])
        assert np.array_equal(dk.ocean_time.values, T) and np.array_equal(dk.mask_rho.values, ds.mask_rho.values)
        pairs.append((k, dk, 'u', 'v'))
for P, dp, vu, vv in pairs:
    print(P)
    ur, vr = to_rho(dp[vu].values, dp[vv].values)
    um, vm = rho_mean(ur, vr)
    out[P + '_umean'] = (('eta_rho', 'xi_rho'), um)
    out[P + '_vmean'] = (('eta_rho', 'xi_rho'), vm)
    if not args.mean_only:
        res = run(fit_uv, ur, vr)
        for i, vn in enumerate(UV_OUT):
            out[P + '_' + vn] = (dims2, res[i])
    del ur, vr

if args.do_3d and ('u' in ds) and (ds.u.ndim == 4):
    NZ = ds.sizes['s_rho']
    res3 = np.full((len(UV_OUT), len(CONS), NZ, NR, NC), np.nan)
    mean3 = np.full((2, NZ, NR, NC), np.nan)
    for iz in range(NZ):
        print('3d layer %d of %d' % (iz + 1, NZ))
        ur, vr = to_rho(ds.u[:, iz].values, ds.v[:, iz].values)
        mean3[:, iz] = rho_mean(ur, vr)
        if not args.mean_only:
            res3[:, :, iz] = run(fit_uv, ur, vr)
    out['3d_umean'] = (('s_rho', 'eta_rho', 'xi_rho'), mean3[0])
    out['3d_vmean'] = (('s_rho', 'eta_rho', 'xi_rho'), mean3[1])
    dims3 = ('con', 's_rho', 'eta_rho', 'xi_rho')
    if not args.mean_only:
        for i, vn in enumerate(UV_OUT):
            out['3d_' + vn] = (dims3, res3[i])
    # layer depths at zeta = 0 (Vtransform 2): z = h * (hc*s + Cs*h) / (hc + h)
    h = ds.h.values[None]
    s = ds.s_rho.values[:, None, None]
    Cs = ds.Cs_r.values[:, None, None]
    hc = float(ds.hc)
    out['z0_rho'] = (('s_rho', 'eta_rho', 'xi_rho'), h * (hc * s + Cs * h) / (hc + h))
    out = out.assign_coords(s_rho=ds.s_rho.values)

for vn in out.data_vars:
    if ('Lsm' in vn) or vn.endswith('mean'):
        out[vn].attrs['units'] = 'm s-1'
    elif vn.endswith(('theta', 'g', 'theta_ci', 'g_ci')):
        out[vn].attrs['units'] = 'deg'
if not args.mean_only:
    out.attrs = {'source': ', '.join(str(fn) for fn in found.values()), 'times': '%s to %s, %d hourly' % (str(T[0])[:16], str(T[-1])[:16], len(T)),
        'method': 'utide.solve ' + str({k: v for k, v in KW.items() if k != 'constit'}),
        'sign': 'Lsmin > 0 counterclockwise; theta deg CCW from east; g Greenwich phase deg'}
out.to_netcdf(out_fn, encoding={vn: {'dtype': 'float32'} for vn in out.data_vars})
print('Saved ' + str(out_fn))
