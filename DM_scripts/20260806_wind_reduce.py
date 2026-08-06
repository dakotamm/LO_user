"""
Reduce the wb1_t0_xn11abbur00 surface wind forcing to one pickle, on apogee.

Same idea as 20260806_river_reduce.py: the forcing is ~730 daily directories of
atm00 files, each Uwind.nc / Vwind.nc holding 25 hourly full-grid fields
(368 x 272), which is ~30 GB of wind that nobody wants to move. This walks the
run, reduces it, and writes one file of ~66 MB that
20260806_wind_characterize.py reads at home:

  scp apogee:LO_output/DM_outs/20260806_wind/wind_hourly_*.p \\
      ~/LO_output/DM_outs/20260806_wind/

The daily files overlap by an hour (hour 00 of a day is also hour 24 of the day
before), so only each file's own 24 hours are kept -- the same
one-entry-per-day rule the river reducer uses.

Four things are kept, chosen so that no second trip is needed:

  SERIES  hourly wind averaged over each region polygon (water cells only),
          which is the wind the model actually applied there, plus the scalar
          met fields (Pair, Tair, rain, swrad) over the same regions. Both the
          vector wind and the mean of the SPEED are kept, because they are not
          the same thing: a day of reversing sea breeze has a large mean speed
          and a near-zero mean vector, and that difference is exactly what a
          daily-mean file loses.
  CUBE    hourly u,v on every 16th grid point, whole domain. 55 MB, and it is
          what makes any spatial question answerable later -- principal axis
          maps, coherence with the domain mean, storm structure.
  FIELDS  monthly and whole-run mean fields at full time resolution: vector
          wind, speed, wind stress, and u*^3. These are accumulated hour by
          hour here, which is the only place it can be done, because stress is
          nonlinear in wind -- the stress of a monthly mean wind is not the
          monthly mean stress. Same reason the region-mean tau and ustar3 are
          computed cell by cell and averaged afterwards.
  PROV    per day: how many DISTINCT (u,v) pairs exist in the domain and in
          each region, plus what the forcing log says about which WRF nests
          were available. atm00 interpolates WRF to the ROMS grid by NEAREST
          NEIGHBOUR, so the number of distinct values IS the number of WRF
          cells covering that area -- a direct measurement of the real
          resolution of the wind, which is nothing like the 200 m grid it is
          written onto. Any claim about spatial structure of the wind in this
          run has to live inside that number.

u*^3 = (tau/rho_w)^1.5 is the usual stand-in for the rate of turbulent energy
input to the surface layer, i.e. the quantity that decides whether wind can
break the stratification.

Wind stress uses Large & Pond (1981) Cd. ROMS itself has BULK_FLUXES defined
and computes its own stress with COARE from these same 10 m winds, so this is a
proxy with the right form and magnitude, not the model's exact stress. It is
here for relative comparisons (which month, which storm), not for a budget.

Uwind/Vwind are already rotated to true East/North by the atm00 forcing code,
so no grid-angle rotation is needed.

run 20260806_wind_reduce.py
"""
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from pathlib import Path
from matplotlib.path import Path as MplPath

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
gtx = 'wb1_t0_xn11abbur00'
FRC = 'atm00'
DATE0, DATE1 = '2024.01.01', '2025.12.31'

STRIDE = 16          # cube subsampling, 368x272 -> 23x17
FSTRIDE = 4          # monthly-mean-field subsampling, -> 92x68
RHO_A, RHO_W = 1.22, 1025.0

# region polygons, in the order they get reported
POLYS = ['pc', 'skagit_delta', 'wb_north', 'wb']
# scalar met fields to carry along, cheap next to the wind
MET = ['Pair', 'Tair', 'rain', 'swrad']

out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_wind'
Lfun.make_dir(out_dir)
out_fn = out_dir / ('wind_hourly_%s_%s_%s.p' % (FRC, DATE0, DATE1))

frc_dir = Ldir['LOo'] / 'forcing' / Ldir['gridname']
dates = pd.date_range(DATE0.replace('.', '-'), DATE1.replace('.', '-'),
                      freq='D')
print('reducing %d days of %s forcing from %s' % (len(dates), FRC, frc_dir))


def cd_lp(spd):
    """Large & Pond (1981) neutral drag coefficient."""
    return np.where(spd < 10.0, 1.14e-3, (0.49 + 0.065 * spd) * 1e-3)


# ------------------------------------------------------------------ masks ---
dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = dsg.lon_rho.values, dsg.lat_rho.values
water = dsg.mask_rho.values == 1
dsg.close()
XY = np.column_stack([lon.ravel(), lat.ravel()])

R = {'domain': water}
for nm in POLYS:
    p = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (nm + '.p'))
    inp = MplPath(np.column_stack([p.x.values, p.y.values])).contains_points(
        XY).reshape(lon.shape)
    R[nm] = inp & water
regions = list(R)
for nm in regions:
    print('  region %-14s %6d water cells (%.1f km2)'
          % (nm, R[nm].sum(), R[nm].sum() * 0.2 * 0.2))

# ----------------------------------------------------------------- arrays ---
W, M, cu, cv, ct = {}, {}, [], [], []
diag, missing = {}, []
MFacc = {}                      # (year, month) -> dict of summed fields
NR, NC = lon.shape
FLD = ['u', 'v', 'spd', 'taux', 'tauy', 'tau', 'ustar3']


def new_acc():
    d = {k: np.zeros((NR, NC)) for k in FLD}
    d['spd2'] = np.zeros((NR, NC))    # for the standard deviation of speed
    d['n'] = 0
    return d


run_acc = new_acc()

for k, dt in enumerate(dates):
    day_dir = frc_dir / ('f' + dt.strftime('%Y.%m.%d')) / FRC
    fnu, fnv = day_dir / 'Uwind.nc', day_dir / 'Vwind.nc'
    if not (fnu.is_file() and fnv.is_file()):
        missing.append(dt)
        continue

    du, dv = xr.open_dataset(fnu), xr.open_dataset(fnv)
    t = pd.to_datetime(du.wind_time.values)
    keep = (t >= dt) & (t < dt + pd.Timedelta('1D'))   # this file's own day
    if keep.sum() == 0:
        missing.append(dt)
        du.close(); dv.close()
        continue
    tt = t[keep]
    u = du.Uwind.values[keep, :, :]
    v = dv.Vwind.values[keep, :, :]
    du.close(); dv.close()

    spd = np.hypot(u, v)
    cd = cd_lp(spd)
    taux, tauy = RHO_A * cd * spd * u, RHO_A * cd * spd * v
    tau = np.hypot(taux, tauy)
    ustar3 = (tau / RHO_W) ** 1.5      # wind-mixing energy proxy, m3 s-3

    # hourly region means
    row = {}
    for nm in regions:
        m = R[nm]
        row['u_' + nm] = u[:, m].mean(axis=1)
        row['v_' + nm] = v[:, m].mean(axis=1)
        row['spd_' + nm] = spd[:, m].mean(axis=1)   # mean of speed, not speed
        row['tau_' + nm] = tau[:, m].mean(axis=1)   # of the mean -- both are
        row['ustar3_' + nm] = ustar3[:, m].mean(axis=1)   # nonlinear
    W[dt] = pd.DataFrame(row, index=tt)

    # cube
    cu.append(u[:, ::STRIDE, ::STRIDE].astype(np.float32))
    cv.append(v[:, ::STRIDE, ::STRIDE].astype(np.float32))
    ct.append(tt)

    # accumulate mean fields
    key = (dt.year, dt.month)
    if key not in MFacc:
        MFacc[key] = new_acc()
    for acc in (MFacc[key], run_acc):
        acc['u'] += u.sum(axis=0)
        acc['v'] += v.sum(axis=0)
        acc['spd'] += spd.sum(axis=0)
        acc['spd2'] += (spd ** 2).sum(axis=0)
        acc['taux'] += taux.sum(axis=0)
        acc['tauy'] += tauy.sum(axis=0)
        acc['tau'] += tau.sum(axis=0)
        acc['ustar3'] += ustar3.sum(axis=0)
        acc['n'] += len(tt)

    # how many distinct wind values actually exist, at midday
    i = min(12, len(tt) - 1)
    d1 = dict(nt=len(tt))
    for nm in regions:
        m = R[nm]
        d1['nuniq_' + nm] = np.unique(np.column_stack([u[i][m], v[i][m]]),
                                      axis=0).shape[0]
    # what the forcing log says about the WRF nests: these lines appear only
    # when a nest could not be processed, so the count is failed hours
    log_fn = day_dir / 'Info' / 'screen_output.txt'
    if log_fn.is_file():
        txt = log_fn.read_text()
        d1['miss_d3'] = txt.count('_d3.')
        d1['miss_d4'] = txt.count('_d4.')
    diag[dt] = d1

    # scalar met fields over the same regions
    mrow = {}
    for vn in MET:
        fn = day_dir / (vn + '.nc')
        if not fn.is_file():
            continue
        ds = xr.open_dataset(fn)
        tn = [d for d in ds[vn].dims if 'time' in d][0]
        tm = pd.to_datetime(ds[tn].values)
        km = (tm >= dt) & (tm < dt + pd.Timedelta('1D'))
        a = ds[vn].values[km, :, :]
        ds.close()
        if a.shape[0] != len(tt):
            continue
        for nm in ['domain', 'pc']:
            mrow['%s_%s' % (vn, nm)] = a[:, R[nm]].mean(axis=1)
    if mrow:
        M[dt] = pd.DataFrame(mrow, index=tt)

    if (k + 1) % 100 == 0:
        print('  %d / %d' % (k + 1, len(dates)))

if not W:
    raise SystemExit('no %s forcing found under %s' % (FRC, frc_dir))

W = pd.concat(W.values()).sort_index()
M = pd.concat(M.values()).sort_index() if M else pd.DataFrame()
D = pd.DataFrame(diag).T.sort_index()

cube = dict(u=np.concatenate(cu, axis=0), v=np.concatenate(cv, axis=0),
            time=pd.DatetimeIndex(np.concatenate(ct)),
            lon=lon[::STRIDE, ::STRIDE], lat=lat[::STRIDE, ::STRIDE],
            water=water[::STRIDE, ::STRIDE], stride=STRIDE)


def finish(acc, sub):
    """Turn summed fields into means, subsampled, float32."""
    n = acc['n']
    o = {k: (acc[k] / n)[::sub, ::sub].astype(np.float32) for k in FLD}
    o['spd_sd'] = np.sqrt(np.maximum(acc['spd2'] / n - (acc['spd'] / n) ** 2,
                                     0))[::sub, ::sub].astype(np.float32)
    o['n'] = n
    return o


MF = {k: finish(a, FSTRIDE) for k, a in sorted(MFacc.items())}
RUN = finish(run_acc, 1)

pd.to_pickle(dict(W=W, M=M, diag=D, cube=cube, MF=MF, RUN=RUN,
                  lon=lon, lat=lat, water=water,
                  fstride=FSTRIDE, regions=regions,
                  ncell={nm: int(R[nm].sum()) for nm in regions},
                  missing=missing,
                  info=dict(gtx=gtx, frc=FRC, date0=DATE0, date1=DATE1,
                            rho_a=RHO_A, rho_w=RHO_W, cd='Large & Pond 1981',
                            made=datetime.now().isoformat(timespec='seconds'))),
             out_fn)

print('\n%d hours x %d regions, %s to %s'
      % (len(W), len(regions), W.index[0], W.index[-1]))
if missing:
    print('** %d model days had no usable %s forcing, e.g. %s'
          % (len(missing), FRC,
             ', '.join(d.strftime('%Y.%m.%d') for d in missing[:5])))
print('domain-mean wind speed %.2f m/s, max hourly %.2f m/s'
      % (W.spd_domain.mean(), W.spd_domain.max()))
print('distinct (u,v) pairs at midday: domain %d-%d, Penn Cove %d-%d'
      % (D.nuniq_domain.min(), D.nuniq_domain.max(),
         D.nuniq_pc.min(), D.nuniq_pc.max()))
print('cube %s float32, %d monthly mean fields'
      % (str(cube['u'].shape), len(MF)))

print('\nsaved %s  (%.1f MB)' % (out_fn, out_fn.stat().st_size / 1e6))
print('\nbring it home with:\n  scp apogee:%s \\\n      ~/%s'
      % (out_fn, out_fn.parent.relative_to(Path.home())))
