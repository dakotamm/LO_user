"""
Reduce the wb1_t0_xn11abbur00 oxygen field to hypoxic area and volume series,
on apogee.

Same shape as 20260806_wind_reduce.py: the model output is ~730 daily
directories that nobody wants to move, so this walks the run once, reduces it,
and writes one small pickle that 20260806_hypoxia_timeseries.py reads at home:

  scp apogee:LO_output/DM_outs/20260806_hypoxia/hypoxia_*.p \\
      ~/LO_output/DM_outs/20260806_hypoxia/

Penn Cove is the target, but every quantity is computed for four polygons and
the whole domain, so the cove can be read against the basin it sits in.

WHY AREA AND VOLUME ARE DIFFERENT QUESTIONS, and why two areas are kept:

  VOLUME    sum of dz*DX*DY over every cell below the threshold. This scales
            with the oxygen deficit, and it is the one quantity that cannot be
            recovered later from a map -- it needs the full 3D field at the
            time it was written.
  A_bot     area of cells whose BOTTOM cell is below the threshold. This is
            what a benthic organism experiences and what a bottom-water mooring
            or a grab survey would report.
  A_col     area of cells with ANY level below the threshold -- the footprint,
            i.e. the water-column-minimum criterion used by
            20260609_hypoxic_days_map.py. A_col >= A_bot always, and the two
            separate whenever the low-oxygen layer lifts off the bed.

  V / A_col is then the mean thickness of the hypoxic layer where it exists,
  which is how an area series and a volume series get reconciled.

Volume uses the actual free surface (zeta), not a flat one. Penn Cove averages
~16 m deep, so a 2 m tidal range is well over 10% of the water column and a
flat-surface volume would put that straight into the series.

THRESHOLDS: 2 mg/L is the conventional hypoxia line and 5 mg/L the low-DO line
already used in wb1_penncove_region.py; 0.5 and 3 are carried so the answer
does not rest on one arbitrary cut. Penn Cove is shallow and well flushed, so
2 mg/L may be crossed rarely or never there -- in which case the 5 mg/L series
is the one carrying the signal, and it is better to have it in hand than to
make a second pass over the run for it.

SOURCE: 'lowpass' reads lowpassed.nc, one Godin-filtered field per day, which
is the right series for a seasonal question -- the tide is removed rather than
aliased into it, and two years is ~730 files (a quarter of a second of work
each, so the whole cost is reading them). 'his' reads the hourly history files
instead, 24x the I/O, and answers the different question of how much hypoxic
volume is built and destroyed within a tidal cycle; -hstride 3 makes that
tractable over a long window. Start with lowpass.

Salinity comes along (region surface and bottom means) because stratification
is the first thing anyone asks when the volume series moves, and one more
variable out of an already-open file is free next to a second pass.

Also kept, since this is the only pass over the 3D field:
  MONTHLY   mean bottom DO, the fraction of the month each cell was hypoxic at
            the bed and anywhere in the column, and mean hypoxic layer
            thickness -- at FSTRIDE, enough to say WHERE the series lives.
  RUN/YEAR  the same at full resolution, plus the run minimum of bottom DO and
            of column-minimum DO at every cell.

run 20260806_hypoxia_reduce.py
run 20260806_hypoxia_reduce.py -gtx wb1_t1_xn11abbur00
run 20260806_hypoxia_reduce.py -src his -0 2024.08.01 -1 2024.09.30
"""
import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from pathlib import Path
from matplotlib.path import Path as MplPath

from lo_tools import Lfun, zrfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-src', '--source', default='lowpass', type=str,
                    choices=['lowpass', 'his'])
parser.add_argument('-hstride', default=1, type=int,
                    help='keep every Nth hour when -src his (24 files/day at 1)')
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gtx = args.gtagex
DATE0, DATE1, SOURCE = args.ds0, args.ds1, args.source

# ocean_his_0001 is 00:00 of the day and 0025 is 00:00 of the next, so 1..24
# keeps each day's own hours exactly once
HIS_HOURS = range(1, 25, args.hstride)

THRESH = [0.5, 2.0, 3.0, 5.0]      # mg/L, series thresholds
FIELD_THRESH = [2.0, 5.0]          # mg/L, thresholds carried as map fields
DO_MMOL_TO_MGL = 32.0 / 1000.0     # mmol m-3 (uM) -> mg L-1

POLYS = ['pc', 'skagit_delta', 'wb_north', 'wb']
FSTRIDE = 2                        # monthly-field subsample, 368x272 -> 184x136

# roms_out keys in priority order (roms_out2 = /dat2/dakotamm on apogee)
ROMS_OUT_KEYS = ['roms_out2', 'roms_out1', 'roms_out']

out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_hypoxia'
Lfun.make_dir(out_dir)
out_fn = out_dir / ('hypoxia_%s_%s_%s_%s.p' % (SOURCE, gtx, DATE0, DATE1))

TKEY = ['%g' % t for t in THRESH]
FKEY = ['%g' % t for t in FIELD_THRESH]


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
    raise SystemExit(
        'no %s files for %s between %s and %s under\n  %s\n'
        'if lowpassed.nc was never made for this run, use the history files:\n'
        '  python %s -gtx %s -src his -hstride 3'
        % (SOURCE, gtx, DATE0, DATE1,
           '\n  '.join(str(Ldir.get(k)) for k in ROMS_OUT_KEYS),
           Path(__file__).name, gtx))
nfile = sum(len(v) for v in FN.values())
print('%s: %d days, %d %s files, %s to %s'
      % (gtx, len(FN), nfile, SOURCE, min(FN), max(FN)))

# ------------------------------------------------------------------ masks ---
dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = dsg.lon_rho.values, dsg.lat_rho.values
water = dsg.mask_rho.values == 1
h = dsg.h.values
area = (1 / dsg.pm.values) * (1 / dsg.pn.values)      # m2, cell by cell
dsg.close()
XY = np.column_stack([lon.ravel(), lat.ravel()])

R = {'domain': water}
for nm in POLYS:
    p = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (nm + '.p'))
    inp = MplPath(np.column_stack([p.x.values, p.y.values])).contains_points(
        XY).reshape(lon.shape)
    R[nm] = inp & water
regions = list(R)
IDX = {nm: np.where(R[nm]) for nm in regions}         # index once, reuse daily
for nm in regions:
    m = R[nm]
    print('  region %-14s %6d cells, %8.1f km2, mean h %5.1f m, max h %5.1f m'
          % (nm, m.sum(), area[m].sum() / 1e6, h[m].mean(), h[m].max()))

# vertical coordinate: from the output itself if it kept the s-coordinate
# variables (lowpassed.nc does), else from a history file in the same directory
S = None
for fns in FN.values():
    for src in [fns[0], fns[0].parent / 'ocean_his_0002.nc']:
        try:
            S = zrfun.get_basic_info(src, only_S=True)
            print('vertical coordinate from %s (N = %d, Vtransform = %d)'
                  % (src.name, S['N'], int(S['Vtransform'])))
            break
        except Exception:
            continue
    if S is not None:
        break
if S is None:
    raise SystemExit('could not read the s-coordinate from any output file')
N = S['N']

# ------------------------------------------------------------- containers ---
FLD = ['bot_do'] + ['%s_%s' % (a, k) for k in FKEY
                    for a in ['fhyp_bot', 'fhyp_col', 'thick']]


def new_acc():
    d = {k: np.zeros(lon.shape) for k in FLD}
    d['n'] = 0
    return d


MFacc, YRacc = {}, {}
run_acc = new_acc()
bot_min = np.full(lon.shape, np.inf)      # run minimum of bottom DO
col_min = np.full(lon.shape, np.inf)      # run minimum over the whole column
rows = {nm: {} for nm in regions}
bad = []


def read_one(fn):
    """(time, DO in mg/L, zeta, salt or None) from one output file."""
    ds = xr.open_dataset(fn)
    if 'oxygen' not in ds.data_vars:
        ds.close()
        raise KeyError('no oxygen variable')
    tname = 'ocean_time' if 'ocean_time' in ds.variables else 'time'
    t = pd.to_datetime(np.atleast_1d(ds[tname].values)[-1])

    def get(vn, ndim):
        """The field without its time axis, whether or not one is present."""
        if vn not in ds.data_vars:
            return None
        a = ds[vn].values
        return a[-1] if a.ndim == ndim + 1 else a

    do = get('oxygen', 3) * DO_MMOL_TO_MGL
    zeta, salt = get('zeta', 2), get('salt', 3)
    ds.close()
    return t, do, zeta, salt


def wmean(x, w):
    """Area- or volume-weighted mean that ignores missing cells."""
    g = np.isfinite(x)
    return float(np.sum(x[g] * w[g]) / np.sum(w[g])) if g.any() else np.nan


# --------------------------------------------------------------- the pass ---
print('\nreducing...')
warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN columns over land
for k, (ds_, fns) in enumerate(FN.items()):
    for fn in fns:
        try:
            t, do, zeta, salt = read_one(fn)
        except Exception as ex:
            bad.append((ds_, fn.name, str(ex)))
            continue

        # geometry at this instant: dz from the real free surface
        dz = np.diff(zrfun.get_z(h, zeta, S, only_w=True), axis=0)
        dv = dz * area                                       # m3, (N, M, L)

        # land and fill values must not be able to count as hypoxic
        do = np.where(np.broadcast_to(water, do.shape) & (do > 0), do, np.nan)
        hyp = {kk: do < th for kk, th in zip(TKEY, THRESH)}   # nan -> False
        col_do = np.nanmin(do, axis=0)

        # ---------------------------------------------------- region series ---
        for nm in regions:
            i, j = IDX[nm]
            o = do[:, i, j]                                  # (N, ncell)
            d_ = dv[:, i, j]
            a_ = area[i, j]
            row = dict(V_tot=float(d_.sum()), A_tot=float(a_.sum()),
                       do_bot_mean=wmean(o[0], a_),
                       do_bot_min=float(np.nanmin(o[0])),
                       do_surf_mean=wmean(o[-1], a_),
                       do_vol_mean=wmean(o.ravel(), d_.ravel()),
                       do_min=float(np.nanmin(o)))
            for kk in TKEY:
                hm = hyp[kk][:, i, j]
                row['V_' + kk] = float(d_[hm].sum())
                row['A_bot_' + kk] = float(a_[hm[0]].sum())
                row['A_col_' + kk] = float(a_[hm.any(axis=0)].sum())
            if salt is not None:
                s_ = salt[:, i, j]
                row['salt_bot'] = wmean(s_[0], a_)
                row['salt_surf'] = wmean(s_[-1], a_)
                # positive = stratified, bottom saltier than surface
                row['dsalt'] = row['salt_bot'] - row['salt_surf']
            rows[nm][t] = row

        # ----------------------------------------------------- map fields ---
        mkey = (t.year, t.month)
        if mkey not in MFacc:
            MFacc[mkey] = new_acc()
        if t.year not in YRacc:
            YRacc[t.year] = new_acc()
        for acc in (MFacc[mkey], YRacc[t.year], run_acc):
            acc['bot_do'] += np.nan_to_num(do[0])
            for kf in FKEY:
                hm = hyp[kf]
                acc['fhyp_bot_' + kf] += hm[0]
                acc['fhyp_col_' + kf] += hm.any(axis=0)
                acc['thick_' + kf] += (dz * hm).sum(axis=0)
            acc['n'] += 1
        np.fmin(bot_min, np.where(water, do[0], np.nan), out=bot_min)
        np.fmin(col_min, np.where(water, col_do, np.nan), out=col_min)

    if (k + 1) % 50 == 0:
        print('  %d / %d days' % (k + 1, len(FN)))

if not rows['domain']:
    raise SystemExit('no usable files -- first failure: %s' % (bad[:1],))

T = {nm: pd.DataFrame.from_dict(rows[nm], orient='index').sort_index()
     for nm in regions}


def finish(acc, sub):
    """Summed fields -> per-time-step means (so fhyp_* are fractions)."""
    n = acc['n']
    o = {k: (acc[k] / n)[::sub, ::sub].astype(np.float32) for k in FLD}
    o['n'] = n
    return o


MF = {k: finish(a, FSTRIDE) for k, a in sorted(MFacc.items())}
YF = {k: finish(a, 1) for k, a in sorted(YRacc.items())}
RUN = finish(run_acc, 1)
RUN['bot_do_min'] = np.where(np.isfinite(bot_min), bot_min,
                             np.nan).astype(np.float32)
RUN['col_do_min'] = np.where(np.isfinite(col_min), col_min,
                             np.nan).astype(np.float32)

pd.to_pickle(dict(T=T, MF=MF, YF=YF, RUN=RUN,
                  lon=lon, lat=lat, water=water, h=h, area=area,
                  fstride=FSTRIDE, regions=regions,
                  thresh=THRESH, field_thresh=FIELD_THRESH,
                  ncell={nm: int(R[nm].sum()) for nm in regions},
                  bad=bad,
                  info=dict(gtx=gtx, source=SOURCE, date0=DATE0, date1=DATE1,
                            do_conv=DO_MMOL_TO_MGL, N=N,
                            made=datetime.now().isoformat(timespec='seconds'))),
             out_fn)

# ----------------------------------------------------------------- sanity ---
print('\n%d time steps, %s to %s' % (len(T['domain']), T['domain'].index[0],
                                     T['domain'].index[-1]))
if bad:
    print('** %d files failed, e.g. %s' % (len(bad), bad[:3]))

print('\n--- does anything actually go hypoxic, and where ---')
print('%-14s %8s %8s %9s   %s'
      % ('region', 'km3', 'min DO', 'mean bot',
         ' '.join('%6s' % ('<%g' % t) for t in THRESH)))
for nm in regions:
    d = T[nm]
    frac = ' '.join('%5.0f%%' % (100 * (d['V_' + k] > 0).mean()) for k in TKEY)
    print('%-14s %8.3f %8.2f %9.2f   %s'
          % (nm, d.V_tot.mean() / 1e9, d.do_min.min(), d.do_bot_mean.mean(),
             frac))
print('  the last columns are the % of time steps with ANY water below each')
print('  threshold in that region; the series worth plotting is the lowest')
print('  threshold that is neither 0% nor 100% of the time')

print()
for nm in ['pc', 'domain']:
    d = T[nm]
    for kk in TKEY:
        v = d['V_' + kk]
        if v.max() <= 0:
            continue
        print('%-8s <%-4s peak %9.2f x1e6 m3 (%4.1f%% of the region) on %s'
              % (nm, kk, v.max() / 1e6, 100 * v.max() / d.V_tot.mean(),
                 v.idxmax().strftime('%Y-%m-%d')))

print('\nsaved %s  (%.1f MB)' % (out_fn, out_fn.stat().st_size / 1e6))
print('\nbring it home with:\n  scp apogee:%s \\\n      ~/%s'
      % (out_fn, out_fn.parent.relative_to(Path.home())))
