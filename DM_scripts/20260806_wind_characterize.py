"""
Surface winds of wb1_t0_xn11abbur00, for the whole run.

Third of the forcing set, after the tides and the rivers, and the answer it
gives is the opposite of the river one. There, 29 of 32 sources were a
climatology that repeats exactly, so almost none of the run's variability came
from the rivers. The wind is WRF, hourly, a real atmospheric model: nothing
repeats, both years differ, and it is the only forcing that puts energy in at
the sea surface everywhere at once.

This runs at home off the pickle 20260806_wind_reduce.py makes on apogee, so
it is the forcing as the model ran it rather than a reconstruction.

There is already a wind extraction in the tef2 folder --
extract/tef2/reduce_wind_cove.py, which wrote wind_*_wb1_pc1.nc. That was
built for one question (does wind drive the Penn Cove gyre) and is DAILY MEAN
wind over the three cove segments, subsampled every 6th hour. It is used here
as an independent check, but it cannot characterise the wind on its own, for
three reasons that this script is built around:

  HOURLY    a daily mean of a reversing sea breeze is close to zero, and the
            daily mean of a storm is well below its peak. The diurnal cycle
            and the event peaks are both invisible in a daily mean.
  SPEED     |daily mean vector| is not mean speed, and wind stress goes as
            speed squared and mixing as speed cubed, so averaging first
            biases every energy-like quantity low. Both are carried here.
  ELSEWHERE Penn Cove is 12 km2. The reduce_wind_cove file says nothing about
            the Skagit delta, the basin, or the domain.

What limits every spatial claim below is resolution, so it is measured rather
than assumed. atm00 interpolates WRF to the ROMS grid by NEAREST NEIGHBOUR, so
the number of distinct (u,v) pairs inside a region is exactly the number of
WRF cells covering it. The reducer counts them. Over this run the domain gets
~1100 distinct values over 1550 km2, i.e. the 1.3 km d4 nest -- but Penn Cove
still resolves to only ~14 of them. A 200 m grid is being forced by a wind
field that has about fourteen numbers in the cove.

Wind stress is Large & Pond (1981), a proxy: ROMS has BULK_FLUXES defined and
computes its own stress with COARE from these same 10 m winds. Use it for
relative comparisons, not for a budget. u*^3 = (tau/rho_w)^1.5 is the usual
stand-in for the rate of turbulent energy input to the surface layer, which is
the quantity that matters for whether wind can break the stratification.

Directions are METEOROLOGICAL -- the direction the wind blows FROM.

run 20260806_wind_characterize.py
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.path import Path as MplPath
from scipy.signal import find_peaks

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
gtx = 'wb1_t0_xn11abbur00'
FRC = 'atm00'
DATE0, DATE1 = '2024.01.01', '2025.12.31'
TZ = 'America/Los_Angeles'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_wind'
Lfun.make_dir(out_dir)

CACHE = out_dir / ('wind_hourly_%s_%s_%s.p' % (FRC, DATE0, DATE1))

CB = dict(blue='#0072B2', orange='#D55E00', green='#009E73', red='#CC0000',
          purple='#7B3294', dgreen='#1B7837', yellow='#E69F00', pink='#CC79A7')
SEAS = dict(DJF=[12, 1, 2], MAM=[3, 4, 5], JJA=[6, 7, 8], SON=[9, 10, 11])
SEAS_C = dict(DJF=CB['blue'], MAM=CB['green'], JJA=CB['red'], SON=CB['orange'])
MN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# ------------------------------------------------------------------- load ---
if not CACHE.is_file():
    raise SystemExit(
        'no reduced wind file at\n  %s\n'
        'Run 20260806_wind_reduce.py on apogee and copy it here:\n'
        '  scp apogee:LO_output/DM_outs/20260806_wind/%s \\\n'
        '      ~/LO_output/DM_outs/20260806_wind/' % (CACHE, CACHE.name))

C = pd.read_pickle(CACHE)
W, MET, DG = C['W'], C['M'], C['diag']
cube, MF, RUN = C['cube'], C['MF'], C['RUN']
lon, lat, water = C['lon'], C['lat'], C['water']
fs = C['fstride']
regions = C['regions']

W.index = W.index.tz_localize('UTC').tz_convert(TZ)
if len(MET):
    MET.index = MET.index.tz_localize('UTC').tz_convert(TZ)
lt = W.index                                   # local time, for the diurnal
print('reduced wind forcing from %s' % CACHE.name)
print('  made %s from %s on the %s grid'
      % (C['info']['made'], C['info']['frc'], Ldir['gridname']))
print('%s: %d hours, %s to %s (%.1f days), %d regions'
      % (gtx, len(W), lt[0], lt[-1], len(W) / 24, len(regions)))
if C['missing']:
    print('  ** %d model days had no %s forcing file, e.g. %s'
          % (len(C['missing']), FRC,
             ', '.join(d.strftime('%Y.%m.%d') for d in C['missing'][:5])))

# ------------------------------------------------------------- provenance ---
# unique (u,v) count == number of WRF cells covering the region, because atm00
# interpolates by nearest neighbour
print('\n--- what resolution is this wind, really ---')
print('%-14s %8s %10s %14s' % ('region', 'km2', 'WRF cells', 'km per cell'))
for nm in regions:
    a = C['ncell'][nm] * 0.04
    n = DG['nuniq_' + nm]
    print('%-14s %8.1f %5d - %-4d %14.2f'
          % (nm, a, n.min(), n.max(), np.sqrt(a / n.median())))
if 'miss_d4' in DG:
    nd4 = (DG.miss_d4 == 0).sum()
    nd3 = (DG.miss_d3 == 0).sum()
    print('forcing log: d4 (1.3 km) complete on %d of %d days, '
          'd3 (4 km) complete on %d' % (nd4, len(DG), nd3))
    print('  the ROMS grid is 200 m, so the wind is smoother than the '
          'bathymetry by a factor of ~%.0f'
          % (np.sqrt(1552.5 / DG.nuniq_domain.median()) / 0.2))

# --------------------------------------------------------------- geometry ---
# axis of each region from its own water cells, to ask whether the wind is
# steered along the waterway or just blows across it
XY = np.column_stack([lon.ravel(), lat.ravel()])
COS = np.cos(np.deg2rad(lat.mean()))


def princ_axis(x, y):
    """Major-axis heading (deg clockwise from N, mod 180) and its variance
    fraction, for a cloud of points or a cloud of (u,v) vectors."""
    c = np.cov(np.column_stack([x, y]).T)
    ev, EV = np.linalg.eigh(c)
    e = EV[:, np.argmax(ev)]
    hdg = np.rad2deg(np.arctan2(e[0], e[1])) % 180
    return hdg, ev.max() / ev.sum()


GEOM = {}
for nm in regions:
    if nm == 'domain':
        m = water
    else:
        p = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (nm + '.p'))
        m = MplPath(np.column_stack([p.x.values, p.y.values])).contains_points(
            XY).reshape(lon.shape) & water
    GEOM[nm] = princ_axis(lon[m] * COS * 111e3, lat[m] * 111e3)

# ------------------------------------------------- per-region statistics ---
def met_dir(u, v):
    """Direction the wind blows FROM, deg clockwise from N."""
    return (270 - np.rad2deg(np.arctan2(v, u))) % 360


S = pd.DataFrame(index=regions)
for nm in regions:
    u, v, s = W['u_' + nm], W['v_' + nm], W['spd_' + nm]
    S.loc[nm, 'km2'] = C['ncell'][nm] * 0.04
    S.loc[nm, 'mean_spd'] = s.mean()
    S.loc[nm, 'median_spd'] = s.median()
    S.loc[nm, 'p95_spd'] = s.quantile(0.95)
    S.loc[nm, 'max_spd'] = s.max()
    S.loc[nm, 'max_when'] = s.idxmax().strftime('%Y-%m-%d %H')
    S.loc[nm, 'calm_pct'] = 100 * (s < 1).mean()
    S.loc[nm, 'gale_pct'] = 100 * (s > 13.9).mean()     # Beaufort 7
    S.loc[nm, 'vec_u'] = u.mean()
    S.loc[nm, 'vec_v'] = v.mean()
    S.loc[nm, 'vec_spd'] = np.hypot(u.mean(), v.mean())
    S.loc[nm, 'vec_dir_from'] = met_dir(u.mean(), v.mean())
    # steadiness: 1 = the wind never changes direction, 0 = no net direction
    S.loc[nm, 'steadiness'] = S.loc[nm, 'vec_spd'] / S.loc[nm, 'mean_spd']
    hdg, frac = princ_axis(u.values, v.values)
    S.loc[nm, 'wind_axis'] = hdg
    S.loc[nm, 'wind_axis_frac'] = frac
    S.loc[nm, 'geom_axis'] = GEOM[nm][0]
    S.loc[nm, 'axis_offset'] = min(abs(hdg - GEOM[nm][0]),
                                   180 - abs(hdg - GEOM[nm][0]))
    S.loc[nm, 'tau'] = W['tau_' + nm].mean()
    S.loc[nm, 'ustar3_e6'] = 1e6 * W['ustar3_' + nm].mean()

print('\n--- how hard and how steadily does it blow, by region ---')
print('%-14s %6s %6s %6s %7s %6s %6s %9s' % ('region', 'mean', 'p95', 'max',
                                             'calm%', 'gale%', 'steady',
                                             'from'))
for nm in regions:
    r = S.loc[nm]
    print('%-14s %6.2f %6.2f %6.2f %7.1f %6.2f %6.2f %6.0f deg'
          % (nm, r.mean_spd, r.p95_spd, r.max_spd, r.calm_pct, r.gale_pct,
             r.steadiness, r.vec_dir_from))
print('  (speeds m/s; calm < 1 m/s; gale > 13.9 m/s = Beaufort 7; steadiness')
print('   = |mean vector| / mean speed, so 1 means it never turns)')

print('\n--- is the wind steered along the waterway? ---')
print('%-14s %10s %10s %8s %10s' % ('region', 'wind axis', 'basin axis',
                                    'offset', 'along-axis'))
for nm in regions:
    r = S.loc[nm]
    print('%-14s %8.0f T %8.0f T %6.0f d %8.0f%%'
          % (nm, r.wind_axis, r.geom_axis, r.axis_offset,
             100 * r.wind_axis_frac))
print('  axes are bidirectional (mod 180). along-axis = % of wind variance on')
print('  the major axis; 50% would mean no preferred direction at all')

# ------------------------------------------------------- along-cove frame ---
# The Penn Cove axis, recomputed here from the polygon so this script stands
# alone, and checked against the one reduce_wind_cove.py stored in its file.
hdg_pc = np.deg2rad(GEOM['pc'][0])
ax_, ay_ = np.sin(hdg_pc), np.cos(hdg_pc)
if ax_ > 0:                      # point the axis WEST->EAST -> into the cove
    ax_, ay_ = -ax_, -ay_
W['along_pc'] = W.u_pc * ax_ + W.v_pc * ay_      # + = blowing INTO the cove
W['cross_pc'] = -W.u_pc * ay_ + W.v_pc * ax_     # + = toward the north shore
print('\nPenn Cove axis from the polygon: (%.3f, %.3f), mouth -> head'
      % (ax_, ay_))

# ------------------------------------------------------------- daily/mon ---
d = W.resample('1D').mean()
d['spd_domain_max'] = W.spd_domain.resample('1D').max()
d['spd_pc_max'] = W.spd_pc.resample('1D').max()
# The forcing is stamped UTC and this is Pacific, so the first few hours of the
# run fall in the previous calendar year and would otherwise show up as a third
# "year" of one day. Only years the run actually covers count.
yrs = [y for y in sorted(set(d.index.year)) if (d.index.year == y).sum() > 300]
print('  full years in the run: %s' % ', '.join(str(y) for y in yrs))

mo = W.groupby([W.index.year, W.index.month]).mean()
mo.index.names = ['year', 'month']

print('\n--- seasonal cycle of the domain wind ---')
print('%-6s %6s %6s %7s %7s %8s %9s' % ('month', 'speed', '|vec|', 'u', 'v',
                                        'from', 'u*3 x1e6'))
for m in range(1, 13):
    q = W[W.index.month == m]
    print('%-6s %6.2f %6.2f %7.2f %7.2f %8.0f %9.2f'
          % (MN[m - 1], q.spd_domain.mean(),
             np.hypot(q.u_domain.mean(), q.v_domain.mean()),
             q.u_domain.mean(), q.v_domain.mean(),
             met_dir(q.u_domain.mean(), q.v_domain.mean()),
             1e6 * q.ustar3_domain.mean()))

# harmonic fit, same form as the tidal seasonality script: annual + semiannual
def harm_fit(t_days, y):
    """Return R2 for annual-only and for annual+semiannual, and amplitudes."""
    w = 2 * np.pi / 365.25
    ones = np.ones_like(t_days)
    X1 = np.column_stack([ones, np.cos(w * t_days), np.sin(w * t_days)])
    X2 = np.column_stack([X1, np.cos(2 * w * t_days), np.sin(2 * w * t_days)])
    out = []
    for X in (X1, X2):
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ b
        out.append((1 - r.var() / y.var(), b))
    a1 = np.hypot(out[1][1][1], out[1][1][2])
    a2 = np.hypot(out[1][1][3], out[1][1][4])
    ph = (np.rad2deg(np.arctan2(out[1][1][2], out[1][1][1])) / 360 * 365.25) % 365.25
    return out[0][0], out[1][0], a1, a2, ph


td = (d.index - d.index[0]).days.values.astype(float)
print('\n--- how much of the wind is a seasonal cycle at all ---')
for lab, y in [('daily mean speed', d.spd_domain.values),
               ('daily along-cove wind', d.along_pc.values),
               ('daily u*^3 (mixing)', d.ustar3_domain.values * 1e6)]:
    r1, r2, a1, a2, ph = harm_fit(td, y)
    print('  %-22s annual %4.1f%% of variance, +semiannual %4.1f%%   '
          'annual amp %.2f (peak day %.0f), semiannual amp %.2f'
          % (lab, 100 * r1, 100 * r2, a1, ph, a2))
print('  the rest is weather: everything a smooth seasonal curve cannot say')

print('\n--- do the two years repeat, as the rivers did? ---')
# the river forcing was 29/32 sources with a max between-year difference of 0.
# Here the same test, on day of year.
if len(yrs) >= 2:
    doy = d.index.dayofyear
    a = d[d.index.year == yrs[0]].set_index(doy[d.index.year == yrs[0]])
    b = d[d.index.year == yrs[-1]].set_index(doy[d.index.year == yrs[-1]])
    com = a.index.intersection(b.index)
    rr = np.corrcoef(a.loc[com, 'spd_domain'], b.loc[com, 'spd_domain'])[0, 1]
    print('  daily mean speed, %d vs %d on day of year: r = %.3f, '
          'mean |difference| %.2f m/s' % (yrs[0], yrs[-1], rr,
          (a.loc[com, 'spd_domain'] - b.loc[com, 'spd_domain']).abs().mean()))
    for yr in yrs:
        q = W[W.index.year == yr]
        if len(q) < 8000:
            continue
        print('  %d  mean %.2f m/s, max %.1f m/s, %d hours over 10 m/s'
              % (yr, q.spd_domain.mean(), q.spd_domain.max(),
                 (q.spd_domain > 10).sum()))
    print('  no source of wind here repeats -- the contrast with the rivers, '
          'where 29 of 32 sources repeated exactly, is the point')

# ------------------------------------------------------------ sea breeze ---
print('\n--- diurnal cycle (local time): is there a sea breeze ---')
print('%-5s %9s %9s %9s %11s' % ('seas', 'range', 'peak hr', 'min hr',
                                 'cross-cove'))
DI = {}
for s, ms in SEAS.items():
    q = W[W.index.month.isin(ms)]
    h = q.groupby(q.index.hour).mean()
    DI[s] = h
    print('%-5s %9.2f %9d %9d %11.2f'
          % (s, h.spd_domain.max() - h.spd_domain.min(), h.spd_domain.idxmax(),
             h.spd_domain.idxmin(),
             h.cross_pc.max() - h.cross_pc.min()))
print('  range = peak-to-trough of the hour-of-day mean domain speed (m/s).')
print('  a sea breeze shows up as an afternoon peak that is large in summer')
print('  and absent in winter, and it is invisible in a daily mean.')

# --------------------------------------------------------------- events ---
print('\n--- wind events (peaks in domain-mean hourly speed) ---')
thr = W.spd_domain.quantile(0.95)
pk, _ = find_peaks(W.spd_domain.values, height=thr, distance=24)
ev = W.spd_domain.iloc[pk].sort_values(ascending=False)
E = pd.DataFrame(index=ev.index[:15])
for t0 in E.index:
    w = W.loc[t0 - pd.Timedelta('36h'):t0 + pd.Timedelta('36h')]
    E.loc[t0, 'peak_spd'] = W.spd_domain[t0]
    E.loc[t0, 'from_deg'] = met_dir(W.u_domain[t0], W.v_domain[t0])
    E.loc[t0, 'hours_over_p90'] = (w.spd_domain > W.spd_domain.quantile(0.9)).sum()
    E.loc[t0, 'pc_spd'] = W.spd_pc[t0]
    E.loc[t0, 'along_pc'] = W.along_pc[t0]
print('%-17s %8s %8s %9s %8s %9s' % ('local time', 'domain', 'from', 'hrs>p90',
                                     'PennCove', 'along-cove'))
for t0, r in E.iterrows():
    print('%-17s %8.2f %6.0f d %9d %8.2f %9.2f'
          % (t0.strftime('%Y-%m-%d %H'), r.peak_spd, r.from_deg,
             r.hours_over_p90, r.pc_spd, r.along_pc))
print('%d peaks above the 95th percentile (%.2f m/s) in %d hours'
      % (len(pk), thr, len(W)))
em = pd.Series(ev.index.month).value_counts().reindex(range(1, 13), fill_value=0)
print('  by month: ' + ', '.join('%s %d' % (MN[m - 1], n)
                                 for m, n in em.items() if n))

# ------------------------------------------------------- mixing energy ---
print('\n--- wind mixing energy (u*^3), and how event-dominated it is ---')
e = W.ustar3_domain
for p in [1, 5, 10]:
    top = e.sort_values(ascending=False).head(int(len(e) * p / 100))
    print('  the windiest %2d%% of hours deliver %4.1f%% of the total u*^3'
          % (p, 100 * top.sum() / e.sum()))


def center_of(x):
    """Day by which half the period's total has been delivered."""
    c = x.cumsum() / x.sum()
    return c.index[np.searchsorted(c.values, 0.5)]


for yr in yrs:
    q = e[e.index.year == yr]
    if len(q) < 8000:
        continue
    print('  %d  half the year\'s mixing energy is in by %s; '
          'Nov-Mar holds %.0f%% of it'
          % (yr, center_of(q).strftime('%b %d'),
             100 * q[q.index.month.isin([11, 12, 1, 2, 3])].sum() / q.sum()))
print('  compare the rivers, where half the volume arrives by late Feb -- '
      'freshwater and wind energy peak in the same season')

# --------------------------------------------------- spatial structure ---
print('\n--- spatial structure, within the resolution measured above ---')
cu, cv = cube['u'], cube['v']
cs = np.hypot(cu, cv)
cw = cube['water']
dom = W.spd_domain.values
if len(dom) == cs.shape[0]:
    flat = cs.reshape(cs.shape[0], -1)
    dm = dom - dom.mean()
    fm = flat - flat.mean(axis=0)
    coh = ((fm * dm[:, None]).sum(axis=0)
           / np.sqrt((fm ** 2).sum(axis=0) * (dm ** 2).sum())
           ).reshape(cs.shape[1:])
else:
    coh = np.full(cs.shape[1:], np.nan)
ratio = cs.mean(axis=0) / dom.mean()
axmap = np.full(cs.shape[1:], np.nan)
for i in range(cs.shape[1]):
    for j in range(cs.shape[2]):
        axmap[i, j] = princ_axis(cu[:, i, j], cv[:, i, j])[0]
print('  correlation of local hourly speed with the domain mean: '
      'median %.3f over water, minimum %.3f'
      % (np.nanmedian(coh[cw]), np.nanmin(coh[cw])))
print('  local mean speed / domain mean speed: %.2f to %.2f over water '
      '(sheltering and channelling)' % (ratio[cw].min(), ratio[cw].max()))
print('  local wind axis spans %.0f-%.0f deg T over water'
      % (np.nanmin(axmap[cw]), np.nanmax(axmap[cw])))
print('  Penn Cove mean speed is %.2f m/s = %.0f%% of the domain mean'
      % (S.loc['pc', 'mean_spd'],
         100 * S.loc['pc', 'mean_spd'] / S.loc['domain', 'mean_spd']))

# ------------------------------------------- check against the tef2 file ---
print('\n--- check against extract/tef2/wind_*_wb1_pc1.nc ---')
try:
    fn = (Ldir['LOo'] / 'extract' / gtx / 'tef2'
          / ('wind_%s_%s_wb1_pc1.nc' % (DATE0, DATE1)))
    dw = xr.open_dataset(fn)
    tw = pd.to_datetime(dw.day.values)
    old = pd.DataFrame(dict(U=dw.Uwind.values, V=dw.Vwind.values,
                            along=dw.w_along.values), index=tw)
    dw.close()
    # that file is UTC-stamped daily means of a 6-hourly subsample, from the
    # his files; this one is a daily mean of every hour, from the forcing.
    # They should agree closely but not exactly.
    new = W[['u_pc', 'v_pc', 'along_pc']].tz_convert('UTC').tz_localize(None)
    new = new.resample('1D').mean()
    j = old.join(new, how='inner').dropna()
    print('  %d overlapping days' % len(j))
    for a, b, lab in [('U', 'u_pc', 'u'), ('V', 'v_pc', 'v'),
                      ('along', 'along_pc', 'along-cove')]:
        r = np.corrcoef(j[a], j[b])[0, 1]
        print('    %-11s r = %.4f, bias %+.3f m/s, rms difference %.3f m/s'
              % (lab, r, (j[b] - j[a]).mean(),
                 np.sqrt(((j[b] - j[a]) ** 2).mean())))
    print('  the residual is the 6-hourly subsampling in that file plus the '
          'segment-vs-polygon difference in which cells are averaged')
    # and the reason a daily mean is not enough
    dm_ = np.hypot(new.u_pc, new.v_pc)
    hm = W.spd_pc.tz_convert('UTC').tz_localize(None).resample('1D').mean()
    print('  daily |mean vector| averages %.2f m/s but the daily mean SPEED '
          'is %.2f m/s -- a daily-mean file understates the wind by %.0f%%'
          % (dm_.mean(), hm.mean(), 100 * (1 - dm_.mean() / hm.mean())))
except Exception as ex:
    print('  skipped: %s' % ex)

# ------------------------------------------------------------------ save ---
d.round(4).to_csv(out_dir / 'daily_wind.csv', index_label='date')
mo.round(4).to_csv(out_dir / 'monthly_wind.csv')
S.round(4).to_csv(out_dir / 'region_summary.csv', index_label='region')
E.round(3).to_csv(out_dir / 'wind_events.csv', index_label='local_time')

# ------------------------------------------------------- figure 1: run ---
plt.close('all')
fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True,
                        layout='constrained',
                        gridspec_kw=dict(height_ratios=[2, 1.4, 1.4, 1.4]))

A = axs[0]
A.fill_between(W.index, 0, W.spd_domain, color='0.85', lw=0)
A.plot(W.index, W.spd_domain, color='0.6', lw=0.4, label='hourly')
A.plot(d.index, d.spd_domain, color='k', lw=1.3, label='daily mean')
A.plot(E.index[:8], E.peak_spd[:8], 'v', color=CB['red'], ms=7,
       label='8 largest events')
A.axhline(thr, color=CB['purple'], ls=':', lw=1,
          label='95th pctl (%.1f m/s)' % thr)
A.set_ylabel('domain-mean wind speed (m s$^{-1}$)')
A.set_title('%s -- surface wind as the model ran it (%s, WRF), %s to %s'
            % (gtx, FRC, DATE0, DATE1))
A.grid(color='lightgray', ls='--', alpha=0.5)
A.legend(fontsize=8, ncol=4, loc='upper right')

B = axs[1]
# stick plot: daily mean vector, so direction and its reversals are visible
B.quiver(d.index, np.zeros(len(d)), d.u_domain, d.v_domain,
         color=CB['blue'], width=0.0012, scale=90, headwidth=0, headlength=0,
         headaxislength=0)
B.axhline(0, color='k', lw=0.6)
B.set_ylim(-8, 8)
B.set_ylabel('daily mean\nwind vector (m s$^{-1}$)')
B.grid(color='lightgray', ls='--', alpha=0.5)
B.text(0.005, 0.9, 'up = toward the north; sticks point downwind',
       transform=B.transAxes, fontsize=8, color='0.3')

Cx = axs[2]
Cx.fill_between(d.index, 0, d.along_pc, where=d.along_pc >= 0,
                color=CB['orange'], alpha=0.6, label='into the cove')
Cx.fill_between(d.index, 0, d.along_pc, where=d.along_pc < 0,
                color=CB['blue'], alpha=0.6, label='out of the cove')
Cx.axhline(0, color='k', lw=0.6)
Cx.set_ylabel('Penn Cove\nalong-cove wind (m s$^{-1}$)')
Cx.grid(color='lightgray', ls='--', alpha=0.5)
Cx.legend(fontsize=8, ncol=2, loc='upper right')

D = axs[3]
D.plot(W.index, 1e6 * W.ustar3_domain, color='0.6', lw=0.4)
D.plot(d.index, 1e6 * d.ustar3_domain, color=CB['dgreen'], lw=1.3)
D.set_ylabel('wind mixing\n$u_*^3$ ($10^{-6}$ m$^3$ s$^{-3}$)')
D.set_xlabel('date (%s)' % TZ)
D.grid(color='lightgray', ls='--', alpha=0.5)
D.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
D.text(0.005, 0.85, 'cubic in speed, so this is where the storms show up',
       transform=D.transAxes, fontsize=8, color='0.3')

fn1 = out_dir / 'wind_overview.png'
fig.savefig(fn1, dpi=200, bbox_inches='tight')

# ------------------------------------------------ figure 2: seasonality ---
fig, axs = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')

A = axs[0, 0]
for yr, ls in zip(yrs, ['-', '--']):
    q = W[W.index.year == yr]
    if len(q) < 2000:
        continue
    m = q.spd_domain.groupby(q.index.month).mean()
    A.plot(m.index, m.values, ls, color=CB['blue'], lw=1.8, marker='o', ms=4,
           label='%d domain' % yr)
    m = q.spd_pc.groupby(q.index.month).mean()
    A.plot(m.index, m.values, ls, color=CB['orange'], lw=1.4, marker='s', ms=4,
           label='%d Penn Cove' % yr)
A.set_xticks(range(1, 13))
A.set_xticklabels(MN, fontsize=8)
A.set_ylabel('mean wind speed (m s$^{-1}$)')
A.set_title('monthly mean speed -- the two years do not repeat')
A.grid(color='lightgray', ls='--', alpha=0.5)
A.legend(fontsize=8, ncol=2)

B = axs[0, 1]
# monthly mean vector wind on a compass, so the winter/summer reversal is one
# picture rather than two curves
for m in range(1, 13):
    q = W[W.index.month == m]
    uu, vv = q.u_domain.mean(), q.v_domain.mean()
    c = [SEAS_C[s] for s, ms in SEAS.items() if m in ms][0]
    B.annotate('', xy=(uu, vv), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=c, lw=2))
    B.text(uu * 1.12, vv * 1.12, MN[m - 1], fontsize=8, color=c,
           ha='center', va='center')
lim = 1.35 * max(np.hypot(W.u_domain.groupby(W.index.month).mean(),
                          W.v_domain.groupby(W.index.month).mean()))
B.set_xlim(-lim, lim); B.set_ylim(-lim, lim)
B.axhline(0, color='0.7', lw=0.8); B.axvline(0, color='0.7', lw=0.8)
B.set_aspect('equal')
B.set_xlabel('eastward (m s$^{-1}$)')
B.set_ylabel('northward (m s$^{-1}$)')
B.set_title('monthly mean wind VECTOR, both years pooled\n'
            '(arrow points downwind)', fontsize=10)
B.grid(color='lightgray', ls='--', alpha=0.5)

Cx = axs[1, 0]
for s, h in DI.items():
    Cx.plot(h.index, h.spd_domain, color=SEAS_C[s], lw=1.8, marker='o', ms=3,
            label='%s domain' % s)
    Cx.plot(h.index, h.spd_pc, color=SEAS_C[s], lw=1.0, ls='--',
            label='%s Penn Cove' % s)
Cx.set_xticks(range(0, 24, 3))
Cx.set_xlabel('hour of day (%s)' % TZ)
Cx.set_ylabel('mean wind speed (m s$^{-1}$)')
Cx.set_title('diurnal cycle by season -- the sea breeze')
Cx.grid(color='lightgray', ls='--', alpha=0.5)
Cx.legend(fontsize=7, ncol=2)

D = axs[1, 1]
for yr in yrs:
    q = e[e.index.year == yr]
    if len(q) < 8000:
        continue
    c = 100 * q.cumsum() / q.sum()
    D.plot(q.index.dayofyear + q.index.hour / 24, c.values, lw=1.6,
           label='%d mixing energy' % yr)
    cv = center_of(q)
    D.plot(cv.dayofyear, 50, 'o', ms=7, mfc='none', mew=1.6,
           color=D.lines[-1].get_color())
D.axhline(50, color='0.6', lw=0.8)
D.plot([0, 366], [0, 100], color='0.7', ls=':', lw=1,
       label='if it were spread evenly')
D.set_xlabel('day of year')
D.set_ylabel('cumulative % of annual $u_*^3$')
D.set_title('when the wind energy arrives (circle = half the year\'s total)')
D.grid(color='lightgray', ls='--', alpha=0.5)
D.legend(fontsize=8, loc='upper left')

fn2 = out_dir / 'wind_seasonality.png'
fig.savefig(fn2, dpi=200, bbox_inches='tight')

# ----------------------------------------------------- figure 3: roses ---
RB = [0, 2, 4, 6, 8, 10, 100]
RC = ['#deebf7', '#9ecae1', '#4292c6', '#2171b5', '#08519c', '#08306b']
nd = 16
edges = np.deg2rad(np.arange(nd + 1) * 360 / nd - 180 / nd)

# count first, draw second, so every panel can share one radial scale -- with
# per-panel autoscaling a calm season looks exactly like a stormy one
RO = {}
for nm in ['domain', 'pc']:
    for s, ms in SEAS.items():
        q = W[W.index.month.isin(ms)]
        dr = np.deg2rad(met_dir(q['u_' + nm], q['v_' + nm]).values)
        sp = q['spd_' + nm].values
        ib = ((dr + np.pi / nd) % (2 * np.pi)) // (2 * np.pi / nd)
        RO[(nm, s)] = (np.array([[100 * ((ib == i)
                                         & (sp >= RB[k]) & (sp < RB[k + 1])
                                         ).sum() / len(sp)
                                 for i in range(nd)]
                                for k in range(len(RB) - 1)]),
                       q['spd_' + nm].mean())
rmax = max(v[0].sum(axis=0).max() for v in RO.values()) * 1.05

fig, axs = plt.subplots(2, 4, figsize=(16, 8.5), layout='constrained',
                        subplot_kw=dict(projection='polar'))
for r_, nm in enumerate(['domain', 'pc']):
    for c_, s in enumerate(SEAS):
        ax = axs[r_, c_]
        cnt, msp = RO[(nm, s)]
        bot = np.zeros(nd)
        for k in range(len(RB) - 1):
            ax.bar(edges[:-1] + np.pi / nd, cnt[k], width=2 * np.pi / nd,
                   bottom=bot, color=RC[k], edgecolor='w', lw=0.3,
                   label='%d-%d' % (RB[k], RB[k + 1]) if RB[k + 1] < 99
                   else '>%d' % RB[k])
            bot += cnt[k]
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, rmax)
        ax.set_title('%s  %s\nmean %.1f m/s' % (nm, s, msp), fontsize=10)
        ax.tick_params(labelsize=7)
        if r_ == 0 and c_ == 3:
            ax.legend(fontsize=7, title='m s$^{-1}$', title_fontsize=7,
                      loc='upper left', bbox_to_anchor=(1.05, 1.1))
fig.suptitle('%s -- wind roses by season (direction the wind blows FROM, '
             '%% of hours)' % gtx, fontsize=12)

fn3 = out_dir / 'wind_roses.png'
fig.savefig(fn3, dpi=180, bbox_inches='tight')

# ------------------------------------------------------ figure 4: maps ---
lo_f, la_f = lon[::fs, ::fs], lat[::fs, ::fs]
wa_f = water[::fs, ::fs]
asp = 1 / np.cos(np.deg2rad(lat.mean()))


def seas_field(ms, k):
    n = sum(MF[key]['n'] for key in MF if key[1] in ms)
    a = sum(MF[key][k].astype(float) * MF[key]['n'] for key in MF
            if key[1] in ms)
    return a / n


fig, axs = plt.subplots(1, 4, figsize=(17, 8), layout='constrained')

q = 8


def vecs(ax, u_, v_, ref):
    """Mean-vector arrows with their own key, because the annual mean vector
    is much smaller than the seasonal ones -- the seasons largely cancel, and
    a shared arrow scale would just hide the annual panel."""
    h = ax.quiver(lo_f[::q, ::q], la_f[::q, ::q],
                  np.where(wa_f, u_, np.nan)[::q, ::q],
                  np.where(wa_f, v_, np.nan)[::q, ::q],
                  color='w', scale=ref * 12, width=0.006)
    ax.quiverkey(h, 0.78, 0.03, ref, '%g m s$^{-1}$' % ref, labelpos='E',
                 fontproperties=dict(size=8))


A = axs[0]
sp = np.where(water, RUN['spd'], np.nan)
p = A.pcolormesh(lon, lat, sp, cmap='viridis', shading='nearest')
fig.colorbar(p, ax=A, orientation='horizontal', pad=0.03,
             label='mean speed (m s$^{-1}$)')
vecs(A, RUN['u'][::fs, ::fs], RUN['v'][::fs, ::fs], 0.5)
A.set_title('whole-run mean speed\nand mean vector wind', fontsize=10)

for ax, ms, lab in [(axs[1], SEAS['DJF'], 'DJF'), (axs[2], SEAS['JJA'], 'JJA')]:
    us, vs = seas_field(ms, 'u'), seas_field(ms, 'v')
    ss = np.where(wa_f, seas_field(ms, 'spd'), np.nan)
    p = ax.pcolormesh(lo_f, la_f, ss, cmap='viridis', shading='nearest',
                      vmin=np.nanmin(sp), vmax=np.nanmax(sp))
    fig.colorbar(p, ax=ax, orientation='horizontal', pad=0.03,
                 label='%s mean speed (m s$^{-1}$)' % lab)
    vecs(ax, us, vs, 2.0)
    ax.set_title('%s mean wind' % lab, fontsize=10)

D = axs[3]
cl, ca = cube['lon'], cube['lat']
p = D.pcolormesh(cl, ca, np.where(cw, coh, np.nan), cmap='magma',
                 shading='nearest', vmin=0.5, vmax=1)
fig.colorbar(p, ax=D, orientation='horizontal', pad=0.03,
             label='r of local hourly speed with domain mean')
D.set_title('coherence with the domain mean\n(every %dth grid point)'
            % cube['stride'], fontsize=10)

for ax in axs:
    ax.contour(lon, lat, water.astype(float), [0.5], colors='k',
               linewidths=0.5)
    for nm, col in [('pc', CB['red']), ('skagit_delta', CB['yellow'])]:
        p_ = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (nm + '.p'))
        ax.plot(p_.x, p_.y, color=col, lw=1.2)
    ax.set_aspect(asp)
    ax.tick_params(labelsize=7)
fig.suptitle('%s -- wind fields. The wind has ~%d distinct values over the '
             'whole domain, so nothing finer than ~%.1f km is real.'
             % (gtx, DG.nuniq_domain.median(),
                np.sqrt(1552.5 / DG.nuniq_domain.median())), fontsize=11)

fn4 = out_dir / 'wind_maps.png'
fig.savefig(fn4, dpi=180, bbox_inches='tight')

for f in [fn1, fn2, fn3, fn4, out_dir / 'daily_wind.csv',
          out_dir / 'monthly_wind.csv', out_dir / 'region_summary.csv',
          out_dir / 'wind_events.csv']:
    print('saved %s' % f)
