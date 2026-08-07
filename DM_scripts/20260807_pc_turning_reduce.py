"""
Reduce the Penn Cove velocity field to along-cove transport profiles, on apogee.

THE QUESTION: at what longitude along the cove does the flow stop being inflow
and become outflow -- i.e. where does the inflowing limb turn south and head
back out -- and how does that point move?

WHY A NET TRANSPORT PROFILE CANNOT ANSWER IT. The depth- and width-integrated
transport through every cross-cove section is ~0 at subtidal timescales (the
cove neither fills nor drains for long), so Q(x) has no useful zero crossing.
The turning point only exists once the section is SPLIT, and there are two
ways to split it that make physical sense, both kept here:

  LATERAL   north half vs south half. Water enters Penn Cove along the north
            shore and leaves along the south shore (verified at the mouth in
            20260806_pc_mouth_salinity_tides.py: pc_lp faces 0-5 negative,
            6-11 positive). The north-limb transport Q_N(x) starts strongly
            INTO the cove at the mouth and weakens westward as water crosses
            to the south side; how far west it gets before it has decayed is
            the turning point. (In a 15-day test it decays to a pinch off
            Coupeville and then GROWS again -- the inner cove runs a second
            circulation of the same sense -- so the plotting script measures
            the decay rather than looking for a zero crossing.)
  VERTICAL  bottom third vs top third. If the cove ran as a classical
            estuarine exchange instead, the inflow would be deep and the
            outflow shallow, and Q_bot(x) would carry the same zero crossing.

Both are computed at every longitude and every time step, so the answer is not
assumed -- the recirculating transport per column, E_lat vs E_vert, says which
decomposition the cove actually runs.

WHY THIS READS HOURLY HISTORY FILES AND FILTERS AFTERWARDS, rather than
reading lowpassed.nc. lowpassed.nc holds the Godin-filtered VELOCITY and the
Godin-filtered ZETA, and the transport this script needs is the filtered
PRODUCT:  <u dz> != <u><dz>.  The difference is the tidal Stokes transport,
the correlation between the tidal velocity and the tidal thickness, and in
Penn Cove it is not small -- a ~2 m range on a ~16 m column is over 10% of the
water depth, and the tidal velocity is 3x the residual. Reading lowpassed.nc
in a 15-day test left a spurious 44 m3/s of net inflow at the mouth (~11% of
the north-limb transport) and a continuity check of only r = +0.19. Computing
u*dz*dy hourly and Godin-filtering the transport gets both right.

So the default is -src his over the whole run: ~17,500 files, but only the
cove box is read from each -- 0.08 s per file on the mac -- so of order an
hour. Run it under nohup. It holds ~380 MB of transports in memory and writes
~105 MB (16 MB of daily subtidal fields plus 90 MB of hourly QU2/QV2).
-src lowpass is kept for a quick look and is flagged as stokes_missing.

WHAT IS SAVED. Per-face transports on the cove's own grid, NOT a fitted
turning point: the splitting latitude, the zero crossings, the decay scales
are all choices, and choices should be revisable at home without a second pass
over the run.

  subtidal, daily (Godin-filtered hourly transports, sampled at 12:00)
    QU2[t, j, iu]      depth-integrated eastward transport, m3/s, on every u
                       face of the cove (iu = face between rho i and i+1)
    QU2_top/_bot       the same over the top and bottom third of the local
                       water column (fractional depth, not sigma levels)
    QV2, QV2_top/_bot  northward transport on every v face -- this is the flow
                       doing the turning, and by continuity it is the along-
                       cove derivative of QU2
    QUkx[t, k, iu]     row-summed eastward transport per sigma level: the full
                       vertical structure of each cross-cove section
    QVky[t, k, jv]     the same for the cross-cove flow
  hourly
    QU2_h, QV2_h       the unfiltered depth-integrated transports, so the
                       turning point can also be tracked through the ebb-flood
                       cycle. -hourly_all keeps the bands and the per-level
                       arrays hourly too (4x the file size).

Transports, not velocities, because they add: a column sum of QU2 IS the
section transport, with no area weighting to get wrong. Divide by the stored
face geometry for a velocity.

GEOMETRY. The cove is a flood fill west of the pc_lp section face column (u
index 67), seeded from pc.p but NOT bounded by it: pc.p's east edge cuts
through rho column 67 diagonally and would clip the mouth. The fill gives 317
water cells, i 38-67, j 210-228, every column contiguous, and its only opening
is the mouth. dz comes from the actual free surface at every hour.

run 20260807_pc_turning_reduce.py
run 20260807_pc_turning_reduce.py -gtx wb1_r0_xn11b -0 2017.09.04 -1 2017.09.18
run 20260807_pc_turning_reduce.py -src lowpass          # quick, Stokes-biased
"""
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndi
from scipy.ndimage import convolve1d
from datetime import datetime
from pathlib import Path
from matplotlib.path import Path as MplPath

from lo_tools import Lfun, zfun, zrfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-src', '--source', default='his', type=str,
                    choices=['his', 'lowpass'])
parser.add_argument('-hstride', default=1, type=int,
                    help='keep every Nth hour when -src his; 1 (all 24) is '
                         'required for an honest Godin filter')
parser.add_argument('-hourly_all', action='store_true',
                    help='keep the bands and per-level arrays hourly as well')
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gtx = args.gtagex
DATE0, DATE1, SOURCE = args.ds0, args.ds1, args.source
HIS_HOURS = range(1, 25, args.hstride)

MOUTH_IU = 67          # u-face column of pc_lp; the cove is everything west
FRAC = 1.0 / 3.0       # top / bottom band thickness as a fraction of depth
ROMS_OUT_KEYS = ['roms_out2', 'roms_out1', 'roms_out']

out_dir = Ldir['LOo'] / 'DM_outs' / '20260807_pc_turning'
Lfun.make_dir(out_dir)
out_fn = out_dir / ('turning_%s_%s_%s_%s.p' % (SOURCE, gtx, DATE0, DATE1))


# ------------------------------------------------------------------ files ---
def find_files():
    """{'YYYY.MM.DD': [paths]} over the run; the first roms_out key wins."""
    days = {}
    for key in ROMS_OUT_KEYS:
        base = Ldir.get(key)
        if base is None or str(base).endswith('BLANK'):
            continue
        base = Path(base) / gtx
        if not base.is_dir():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith('f'):
                continue
            ds_ = run_dir.name[1:]
            if not (DATE0 <= ds_ <= DATE1) or ds_ in days:
                continue
            if SOURCE == 'lowpass':
                fns = [run_dir / 'lowpassed.nc']
            else:
                fns = [run_dir / ('ocean_his_%04d.nc' % i) for i in HIS_HOURS]
            fns = [f for f in fns if f.is_file()]
            if fns:
                days[ds_] = fns
    return dict(sorted(days.items()))


FN = find_files()
if not FN:
    raise SystemExit('no %s files for %s between %s and %s under\n  %s'
                     % (SOURCE, gtx, DATE0, DATE1,
                        '\n  '.join(str(Ldir.get(k)) for k in ROMS_OUT_KEYS)))
FLIST = [f for v in FN.values() for f in v]
nfile = len(FLIST)
print('%s: %d days, %d %s files, %s to %s'
      % (gtx, len(FN), nfile, SOURCE, min(FN), max(FN)))
if SOURCE == 'lowpass':
    print('** -src lowpass: transports are <u><dz>, so the tidal Stokes term\n'
          '   is missing. Fine for a quick look, not for a budget. **')

# --------------------------------------------------------------- geometry ---
dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
LON, LAT = dsg.lon_rho.values, dsg.lat_rho.values
MASK = dsg.mask_rho.values == 1
H = dsg.h.values
PM, PN = dsg.pm.values, dsg.pn.values
dsg.close()

# The cove: water connected to Penn Cove, west of the mouth face. pc.p only
# seeds the fill -- it is not the boundary, because its east edge cuts through
# rho column 67 diagonally and would clip the mouth.
w = MASK.copy()
w[:, MOUTH_IU + 1:] = False
lab, _ = ndi.label(w)
pc_poly = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc.p')
seed = MplPath(np.column_stack([pc_poly.x.values, pc_poly.y.values])
               ).contains_points(np.column_stack([LON.ravel(), LAT.ravel()])
                                 ).reshape(LON.shape) & (lab > 0)
COVE = lab == np.bincount(lab[seed]).argmax()
if COVE.sum() > 4 * seed.sum():
    raise SystemExit('the fill escaped the cove (%d cells from %d seeds) -- '
                     'check MOUTH_IU' % (COVE.sum(), seed.sum()))
jj, ii = np.where(COVE)
J0, J1 = jj.min() - 1, jj.max() + 2          # rho row slice (python, exclusive)
I0, I1 = ii.min() - 1, ii.max() + 2          # rho col slice; I1-2 = mouth face
print('cove: %d cells, i %d-%d, j %d-%d, lon %.4f to %.4f'
      % (COVE.sum(), ii.min(), ii.max(), jj.min(), jj.max(),
         LON[0, ii.min()], LON[0, ii.max()]))

sl_r = (slice(J0, J1), slice(I0, I1))        # rho box
nj, ni = J1 - J0, I1 - I0
nu, njv = ni - 1, nj - 1

lon_b, lat_b = LON[sl_r], LAT[sl_r]
h_b, mask_b, cove_b = H[sl_r], MASK[sl_r], COVE[sl_r]
pm_b, pn_b = PM[sl_r], PN[sl_r]

# u faces sit between rho i and i+1, v faces between rho j and j+1. A face
# belongs to the cove if EITHER neighbour is a cove cell -- that is what keeps
# the mouth column, whose east neighbour is Saratoga Passage.
UM = (mask_b[:, :-1] & mask_b[:, 1:]) & (cove_b[:, :-1] | cove_b[:, 1:])
VM = (mask_b[:-1, :] & mask_b[1:, :]) & (cove_b[:-1, :] | cove_b[1:, :])
DYU = 2.0 / (pn_b[:, :-1] + pn_b[:, 1:])     # ROMS on_u, m
DXV = 2.0 / (pm_b[:-1, :] + pm_b[1:, :])     # ROMS om_v, m
lon_u = 0.5 * (lon_b[:, :-1] + lon_b[:, 1:])
lat_u = 0.5 * (lat_b[:, :-1] + lat_b[:, 1:])
lon_v = 0.5 * (lon_b[:-1, :] + lon_b[1:, :])
lat_v = 0.5 * (lat_b[:-1, :] + lat_b[1:, :])
iu_glob = np.arange(I0, I1 - 1)              # global u index of each column
jv_glob = np.arange(J0, J1 - 1)
AREA = (1 / pm_b) * (1 / pn_b)
nface_col = UM.sum(axis=0)
print('box %d x %d rho, %d u-face columns (%d..%d); mouth column iu=%d has %d '
      'faces' % (nj, ni, nu, iu_glob[0], iu_glob[-1], MOUTH_IU,
                 nface_col[list(iu_glob).index(MOUTH_IU)]))

# vertical coordinate from the output itself (lowpassed.nc keeps it too)
S = None
for fns in FN.values():
    for src in [fns[0], fns[0].parent / 'ocean_his_0002.nc']:
        try:
            S = zrfun.get_basic_info(src, only_S=True)
            print('vertical coordinate from %s (N = %d)' % (src.name, S['N']))
            break
        except Exception:
            continue
    if S is not None:
        break
if S is None:
    raise SystemExit('could not read the s-coordinate from any output file')
N = S['N']

# ------------------------------------------------------------- containers ---
SHAPE = dict(QU2=(nj, nu), QU2_top=(nj, nu), QU2_bot=(nj, nu),
             QV2=(njv, ni), QV2_top=(njv, ni), QV2_bot=(njv, ni),
             QUkx=(N, nu), QVky=(N, njv))
A = {k: np.full((nfile,) + s, np.nan, dtype=np.float32)
     for k, s in SHAPE.items()}
tarr = np.full(nfile, np.datetime64('NaT'), dtype='datetime64[ns]')
zeta_m = np.full(nfile, np.nan)
vol = np.full(nfile, np.nan)
print('holding %.0f MB of hourly transports in memory'
      % (sum(a.nbytes for a in A.values()) / 1e6))


# --------------------------------------------------------------- the work ---
def band_dz(zw, f0, f1):
    """Thickness of each level inside the depth band [f0, f1] of the column.

    Fractions run from the free surface (0) to the bed (1), so the band edges
    follow the real column at every face and every hour instead of pinning the
    split to a sigma level, whose depth changes with h and zeta.
    """
    Hc = zw[-1] - zw[0]
    z_hi = zw[-1] - f0 * Hc
    z_lo = zw[-1] - f1 * Hc
    return np.clip(zw[1:], z_lo, z_hi) - np.clip(zw[:-1], z_lo, z_hi)


def read_one(fn):
    """(time, u box, v box, zeta box) with the time axis dropped."""
    ds = xr.open_dataset(fn)
    tname = 'ocean_time' if 'ocean_time' in ds.variables else 'time'
    t = pd.to_datetime(np.atleast_1d(ds[tname].values)[-1])

    def get(da, ndim):
        a = da.values
        return a[-1] if a.ndim == ndim + 1 else a

    u = get(ds['u'].isel(eta_u=slice(J0, J1), xi_u=slice(I0, I1 - 1)), 3)
    v = get(ds['v'].isel(eta_v=slice(J0, J1 - 1), xi_v=slice(I0, I1)), 3)
    zeta = get(ds['zeta'].isel(eta_rho=slice(J0, J1), xi_rho=slice(I0, I1)), 2)
    ds.close()
    return t, u, v, zeta


bad = []
print('\nreducing %d files...' % nfile)
for n, fn in enumerate(FLIST):
    try:
        t, u, v, zeta = read_one(fn)
    except Exception as ex:
        bad.append((fn.parent.name, fn.name, str(ex)))
        continue

    zw = zrfun.get_z(h_b, zeta, S, only_w=True)              # (N+1, nj, ni)
    zw_u = 0.5 * (zw[:, :, :-1] + zw[:, :, 1:])
    zw_v = 0.5 * (zw[:, :-1, :] + zw[:, 1:, :])
    uq = np.where(UM, u, np.nan) * DYU                       # m2/s per unit dz
    vq = np.where(VM, v, np.nan) * DXV

    QU3 = uq * np.diff(zw_u, axis=0)                         # m3/s per cell
    QV3 = vq * np.diff(zw_v, axis=0)
    A['QU2'][n] = np.nansum(QU3, axis=0)
    A['QV2'][n] = np.nansum(QV3, axis=0)
    A['QU2_top'][n] = np.nansum(uq * band_dz(zw_u, 0.0, FRAC), axis=0)
    A['QU2_bot'][n] = np.nansum(uq * band_dz(zw_u, 1 - FRAC, 1.0), axis=0)
    A['QV2_top'][n] = np.nansum(vq * band_dz(zw_v, 0.0, FRAC), axis=0)
    A['QV2_bot'][n] = np.nansum(vq * band_dz(zw_v, 1 - FRAC, 1.0), axis=0)
    A['QUkx'][n] = np.nansum(QU3, axis=1)
    A['QVky'][n] = np.nansum(QV3, axis=2)

    dz = np.diff(zw, axis=0)
    zeta_m[n] = float(np.nanmean(np.where(cove_b, zeta, np.nan)))
    vol[n] = float(np.nansum((dz * AREA)[:, cove_b]))
    tarr[n] = np.datetime64(t)

    if (n + 1) % 1000 == 0:
        print('  %d / %d files' % (n + 1, nfile))

ok = ~np.isnat(tarr)
if not ok.any():
    raise SystemExit('no usable files -- first failure: %s' % (bad[:1],))
order = np.argsort(tarr[ok])
idx = np.where(ok)[0][order]
A = {k: a[idx] for k, a in A.items()}
time = pd.DatetimeIndex(tarr[idx])
zeta_m, vol = zeta_m[idx], vol[idx]
nt = len(time)

# --------------------------------------------------------------- continuity ---
# The whole reduction rests on the transports being right, and there is one
# check that can fail: the net transport into the cove must equal the rate the
# cove volume changes. This is the number to look at before trusting anything
# downstream.
im = list(iu_glob).index(MOUTH_IU)
Q_mouth = np.nansum(A['QU2'], axis=1)[:, im]                 # + = out of cove
sec = (time - time[0]).total_seconds().values
r_cont, res_frac = np.nan, np.nan
if nt > 3:
    dVdt = np.gradient(vol, sec)
    g = np.isfinite(dVdt) & np.isfinite(Q_mouth)
    r_cont = float(np.corrcoef(-Q_mouth[g], dVdt[g])[0, 1])
    res_frac = float(np.std(dVdt[g] + Q_mouth[g]) / np.std(dVdt[g]))

# --------------------------------------------------------------- filtering ---
GODIN = zfun.godin_shape()
NPAD = len(GODIN) // 2


def godin_t(a):
    """Godin lowpass along the time axis only, NaN at the ends."""
    out = convolve1d(np.nan_to_num(a), GODIN, axis=0, mode='nearest')
    out[:NPAD] = np.nan
    out[-NPAD:] = np.nan
    return out.astype(np.float32)


hourly = {}
if SOURCE == 'his' and args.hstride == 1 and nt > 2 * NPAD:
    print('\nGodin-filtering %d hourly steps and sampling daily...' % nt)
    keep = ['QU2', 'QV2'] + (list(SHAPE) if args.hourly_all else [])
    hourly = {k + '_h': A[k] for k in dict.fromkeys(keep)}
    hourly['time_h'] = time
    hourly['zeta_h'] = zeta_m
    hourly['V_cove_h'] = vol
    sub = np.arange(NPAD, nt - NPAD)
    sub = sub[time[sub].hour == 12]                  # daily at 12:00, like LO
    A = {k: godin_t(a)[sub] for k, a in A.items()}
    zeta_lp = zfun.lowpass(zeta_m, f='godin')[sub]
    vol_lp = zfun.lowpass(vol, f='godin')[sub]
    time_lp = time[sub]
    stokes_missing = False
else:
    if SOURCE == 'his':
        print('\n** not filtering: hstride=%d or too short a record. The saved '
              '"subtidal" fields are raw. **' % args.hstride)
    time_lp, zeta_lp, vol_lp = time, zeta_m, vol
    stokes_missing = (SOURCE == 'lowpass')

T = pd.DataFrame(dict(zeta=zeta_lp, V_cove=vol_lp), index=time_lp)

pd.to_pickle(dict(
    time=time_lp, T=T, **A, **hourly,
    lon_u=lon_u, lat_u=lat_u, lon_v=lon_v, lat_v=lat_v,
    lon_rho=lon_b, lat_rho=lat_b, h=h_b, area=AREA,
    mask_rho=mask_b, cove=cove_b, UM=UM, VM=VM, DYU=DYU, DXV=DXV,
    iu_glob=iu_glob, jv_glob=jv_glob, J0=J0, J1=J1, I0=I0, I1=I1,
    mouth_iu=MOUTH_IU, frac_band=FRAC, N=N,
    info=dict(gtx=gtx, source=SOURCE, date0=DATE0, date1=DATE1,
              hstride=args.hstride, ncell_cove=int(COVE.sum()),
              stokes_missing=stokes_missing, nt_raw=nt,
              continuity_r=r_cont, continuity_residual=res_frac,
              made=datetime.now().isoformat(timespec='seconds'))), out_fn)

# ----------------------------------------------------------------- sanity ---
print('\n%d raw steps -> %d saved subtidal steps, %s to %s'
      % (nt, len(time_lp), time_lp[0], time_lp[-1]))
if bad:
    print('** %d files failed, e.g. %s' % (len(bad), bad[:3]))

print('\ncontinuity at the mouth: corr(-Q_mouth, dV/dt) = %+.3f, '
      'unexplained %.1f%% of dV/dt' % (r_cont, 100 * res_frac))
print('  r near +1 means the transports close; a low r on -src his means '
      'something is wrong,\n  a low r on -src lowpass is the missing Stokes '
      'term and is expected.')

qbar = np.nanmean(A['QU2'], axis=0)
print('\nmean along-cove structure (positive = eastward = OUT of the cove, '
      'm3/s):')
print('%5s %10s %6s %10s %10s %10s' % ('iu', 'lon', 'nface', 'net', 'north',
                                       'south'))
for c in range(nu):
    if nface_col[c] == 0:
        continue
    q = qbar[:, c]
    # split where the mean row transport changes sign: the row that maximises
    # the north-half inflow IS the sign change
    js = int(np.nanargmax([-np.nansum(q[j:]) for j in range(nj)]))
    print('%5d %10.4f %6d %10.1f %10.1f %10.1f'
          % (iu_glob[c], lon_u[0, c], nface_col[c], np.nansum(q),
             np.nansum(q[js:]), np.nansum(q[:js])))
print('  north negative = inflow along the north shore. The longitude where '
      'that column\n  reaches zero is the mean turning point; '
      '20260807_pc_turning_point.py does it properly.')

print('\nsaved %s  (%.1f MB)' % (out_fn, out_fn.stat().st_size / 1e6))
print('\nbring it home with:\n  scp apogee:%s \\\n      ~/%s'
      % (out_fn, out_fn.parent.relative_to(Path.home())))
