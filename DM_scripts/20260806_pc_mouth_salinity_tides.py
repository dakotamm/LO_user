"""
How does top/bottom salinity at the Penn Cove mouth vary with ebb/flood,
spring/neap, and season?

Uses the two points across the mouth from 20260806_wbnorth_tidal_movie.py --
north (lat ~48.2438) and south (lat ~48.2275) on pc_lp -- and the tidal-phase
framework from 20260806_tidal_phase_calendar.py.

NO NEW EXTRACTION IS NEEDED. extract_sections_avg.py already stored
salt(time, z, p) and q(time, z, p) hourly for two years at every face of
pc_lp, so the two points ARE faces of an existing file:

    extractions_avg_<ds0>_<ds1>/pc_lp.nc   salt, q per face   (--source faces)
    hourly_flux_<ds0>_<ds1>_<gctag>.nc     ssh, qnet          (tidal phase)
    strat_<ds0>_<ds1>_<gctag>.nc           section-mean s_top/s_bot
    DM_outs/20260806_tidal_phase/phase_daily.csv   spring/neap per day

--source strat falls back to the section-MEAN top/bottom salinity in
strat_*.nc, which is small enough to live on the mac. Same analysis, one
location instead of two, useful as a cross-check and for developing offline.

Why these two points and not a mid-channel one: the mean per-face transport
across pc_lp changes sign across the mouth (north faces one way, south faces
the other), so the mouth carries a LATERAL exchange as well as a vertical one.
A single point cannot see that; a north/south pair at matched depth can.

Four figures, each answering one question:
  1. ebb/flood   composite over the tidal cycle, split spring vs neap
  2. spring/neap daily response to the fortnightly envelope
  3. seasonal    monthly climatology + annual/semiannual harmonic fit
  4. variance    how much of the salinity signal each band actually carries,
                 with utide constituents for the record

Sign conventions are VERIFIED from the data, not assumed: the script checks
qnet against d(ssh)/dt and reports which sign is flood.

On apogee:
    python 20260806_pc_mouth_salinity_tides.py
On the mac (section mean only):
    python 20260806_pc_mouth_salinity_tides.py --source strat
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.signal import find_peaks
from scipy.stats import t as tdist

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('--sect', default='pc_lp')
p.add_argument('--ds0', default='2024.01.01')
p.add_argument('--ds1', default='2025.12.31')
p.add_argument('--source', default='faces', choices=['faces', 'strat'],
               help='faces = per-point from extractions_avg; strat = section mean')
p.add_argument('--lats', default='48.2438,48.2275',
               help='latitudes of the two mouth points (nearest face is used)')
p.add_argument('--faces', default='',
               help='override --lats with face indices, e.g. 2,11')
p.add_argument('--tz', default='America/Los_Angeles')
p.add_argument('--no-utide', dest='utide', action='store_false')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_pc_mouth_salinity_tides'
Lfun.make_dir(out_dir)

# her three 4-month seasons; December folds forward into the next year
SEASONS = {'Winter (Dec-Mar)': [12, 1, 2, 3],
           'Spring (Apr-Jul)': [4, 5, 6, 7],
           'Low-DO (Aug-Nov)': [8, 9, 10, 11]}
SEASON_COLOR = {'Winter (Dec-Mar)': '#4565e8', 'Spring (Apr-Jul)': '#009E73',
                'Low-DO (Aug-Nov)': '#e04256'}
LCOLOR = {'north': '#0072B2', 'south': '#D55E00', 'section mean': '#555555'}
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)


def savefig(fig, name):
    fn = out_dir / name
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    print('saved %s' % fn)


# ------------------------------------------------------------------- load ---
fn_flux = tef2 / ('hourly_flux_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag))
if not fn_flux.is_file():
    cand = sorted(tef2.glob('hourly_flux_*_%s.nc' % gctag))
    if not cand:
        raise SystemExit('no hourly_flux file in %s' % tef2)
    fn_flux = cand[0]
d = xr.open_dataset(fn_flux)
ssh = d.ssh.sel(sect=args.sect).to_pandas()
qnet = d.qnet.sel(sect=args.sect).to_pandas()
d.close()
print('tidal forcing from %s (section %s)' % (fn_flux.name, args.sect))

# everything is put on one local-time hourly index
idx = ssh.index.tz_localize('UTC').tz_convert(args.tz)
ssh.index = idx
qnet.index = idx

LOC = {}          # label -> dict(s_top, s_bot, q or None, lat, h)
if args.source == 'faces':
    in_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
    fn_sect = in_dir / (args.sect + '.nc')
    if not fn_sect.is_file():
        raise SystemExit('%s not found -- run with --source strat on the mac, '
                         'or check the extraction date range' % fn_sect)
    ds = xr.open_dataset(fn_sect)
    # extractions_avg carries h and dd per face but NOT lon/lat -- those are
    # computed from sect_df and only survive in structure_*.nc, so the face
    # latitudes come from there.
    if args.faces:
        pk = [int(v) for v in args.faces.split(',')]
        fn_str = sorted(tef2.glob('structure_*_%s.nc' % gctag))
        flat = (xr.open_dataset(fn_str[0])['%s_lat' % args.sect].values
                if fn_str else np.full(ds.sizes['p'], np.nan))
    else:
        fn_str = sorted(tef2.glob('structure_*_%s.nc' % gctag))
        if not fn_str:
            raise SystemExit(
                'need structure_*_%s.nc for the face latitudes (run '
                'reduce_extractions_structure.py), or pass --faces with face '
                'indices directly' % gctag)
        flat = xr.open_dataset(fn_str[0])['%s_lat' % args.sect].values
        want = [float(v) for v in args.lats.split(',')]
        pk = [int(np.argmin(np.abs(flat - w))) for w in want]
    order = np.argsort([-flat[k] for k in pk])       # north first
    pk = [pk[o] for o in order]
    names = ['north', 'south']
    print('%s has %d faces; using:' % (fn_sect.name, ds.sizes['p']))
    for nm, k in zip(names, pk):
        s = ds.salt.values[:, :, k]                  # (time, z)
        q = ds.q.values[:, :, k].sum(axis=1) if 'q' in ds.data_vars else None
        LOC[nm] = dict(s_top=pd.Series(s[:, -1], index=idx),
                       s_bot=pd.Series(s[:, 0], index=idx),
                       q=None if q is None else pd.Series(q, index=idx),
                       lat=float(flat[k]), h=float(ds.h.values[k]), face=k)
        print('  %-6s face p=%2d  lat %.4f  h %.1f m' % (nm, k, flat[k],
                                                         ds.h.values[k]))
    ds.close()
else:
    fn_st = sorted(tef2.glob('strat_*_%s.nc' % gctag))
    if not fn_st:
        raise SystemExit('no strat file in %s' % tef2)
    ds = xr.open_dataset(fn_st[0])
    LOC['section mean'] = dict(
        s_top=pd.Series(ds.s_top.sel(sect=args.sect).values, index=idx),
        s_bot=pd.Series(ds.s_bot.sel(sect=args.sect).values, index=idx),
        q=qnet, lat=np.nan, h=np.nan, face=-1)
    ds.close()
    print('section-mean top/bottom salinity from %s' % fn_st[0].name)

for nm, L in LOC.items():
    L['dstrat'] = L['s_bot'] - L['s_top']

print('%d hourly steps, %s to %s (%s)'
      % (len(idx), idx[0], idx[-1], args.tz))


# ------------------------------------------------- tidal phase and flood/ebb ---
def godin(s):
    """Godin lowpass of an hourly Series, NaN-padded at the ends."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pd.Series(zfun.lowpass(s.values.astype(float), f='godin'),
                         index=s.index)


# high waters: distance=8 h keeps both semidiurnal highs of a mixed tide but
# will not split one high into two on the flat top of a diurnal day
i_hw, _ = find_peaks(ssh.values, distance=8)
t_hw = ssh.index[i_hw]
print('%d high waters (%.2f per day)'
      % (len(t_hw), len(t_hw) / ((idx[-1] - idx[0]).total_seconds() / 86400)))

# signed hours from the nearest high water, the tidal-cycle coordinate
hw64 = t_hw.tz_convert('UTC').tz_localize(None).values.astype('datetime64[ns]')
all64 = idx.tz_convert('UTC').tz_localize(None).values.astype('datetime64[ns]')
k = np.searchsorted(hw64, all64)
k = np.clip(k, 1, len(hw64) - 1)
d_prev = (all64 - hw64[k - 1]) / np.timedelta64(1, 'h')
d_next = (all64 - hw64[k]) / np.timedelta64(1, 'h')
hrs_hw = np.where(np.abs(d_prev) <= np.abs(d_next), d_prev, d_next)

# Which sign of qnet is flood? Verify rather than assume: during a rising tide
# water must be entering the volume landward of the section.
dssh = np.gradient(ssh.values)
r_check = np.corrcoef(qnet.values, dssh)[0, 1]
FLOOD_SIGN = -1.0 if r_check < 0 else 1.0
print('corr(qnet, d(ssh)/dt) = %+.2f  ->  flood (into the cove) is qnet %s 0'
      % (r_check, '<' if FLOOD_SIGN < 0 else '>'))
is_flood = (FLOOD_SIGN * qnet.values) > 0

T = pd.DataFrame(index=idx)
T['ssh'] = ssh
T['qnet'] = qnet
T['hrs_hw'] = hrs_hw
T['flood'] = is_flood
T['month'] = idx.month
T['season'] = ''
for nm, months in SEASONS.items():
    T.loc[T.month.isin(months), 'season'] = nm
# December folds forward, so the season year is not the calendar year
T['season_year'] = np.where(T.month == 12, idx.year + 1, idx.year)

for nm, L in LOC.items():
    for vn in ['s_top', 's_bot', 'dstrat']:
        T['%s_%s' % (nm, vn)] = L[vn]
        T['%s_%s_sub' % (nm, vn)] = godin(L[vn])        # subtidal
        T['%s_%s_tid' % (nm, vn)] = (L[vn] - godin(L[vn]))   # tidal band
    if L['q'] is not None:
        T['%s_q' % nm] = L['q']

# ----------------------------------------------------------- daily / phase ---
# fixed 24 h from local midnight, same convention as the tidal calendar, so a
# DST day is never 23 h and never clips one of the day's extremes
midn = pd.date_range(idx[0].ceil('D'), idx[-1].floor('D'), freq='D', tz=args.tz)
pos = idx.searchsorted(midn)
ok = (pos + 24) <= len(idx)
midn, pos = midn[ok], pos[ok]


def daily(v, how='mean'):
    w = np.lib.stride_tricks.sliding_window_view(np.asarray(v, dtype=float), 24)[pos]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if how == 'mean':
            return pd.Series(np.nanmean(w, axis=1), index=midn)
        if how == 'range':
            return pd.Series(np.nanmax(w, axis=1) - np.nanmin(w, axis=1), index=midn)
        if how == 'std':
            return pd.Series(np.nanstd(w, axis=1), index=midn)


D = pd.DataFrame(index=midn)
D['tide_range'] = daily(ssh.values, 'range')
for nm in LOC:
    for vn in ['s_top', 's_bot', 'dstrat']:
        D['%s_%s' % (nm, vn)] = daily(T['%s_%s' % (nm, vn)].values, 'mean')
        # amplitude of the tidal band that day = how hard the tide works the point
        D['%s_%s_amp' % (nm, vn)] = daily(T['%s_%s_tid' % (nm, vn)].values, 'std')

fn_phase = (Ldir['LOo'] / 'DM_outs' / '20260806_tidal_phase' / 'phase_daily.csv')
if fn_phase.is_file():
    P = pd.read_csv(fn_phase, parse_dates=['date_local']).set_index('date_local')
    P.index = P.index.tz_localize(args.tz)
    D = D.join(P[['range_smooth_m', 'days_from_syzygy', 'synodic_phase',
                  'model_phase']])
    print('joined spring/neap labels from %s' % fn_phase.name)
else:
    # self-contained fallback: terciles of the 3-day smoothed daily range
    print('no phase_daily.csv -- labelling spring/neap from the daily range')
    rs = D.tide_range.rolling(3, center=True, min_periods=1).mean()
    q = rs.quantile([1 / 3, 2 / 3]).values
    D['range_smooth_m'] = rs
    D['model_phase'] = np.where(rs >= q[1], 'spring',
                                np.where(rs <= q[0], 'neap', 'transition'))
    D['days_from_syzygy'] = np.nan
    D['synodic_phase'] = np.nan
D['month'] = D.index.month
D['season'] = ''
for nm, months in SEASONS.items():
    D.loc[D.month.isin(months), 'season'] = nm

# map the daily spring/neap label back onto the hourly frame
T['model_phase'] = D.model_phase.reindex(
    idx.floor('D')).values if 'model_phase' in D else ''

names = list(LOC)


# ------------------------------------------------------------------ stats ---
def eff_corr(x, y):
    """Pearson r with a p-value corrected for serial correlation.

    Daily values here are strongly autocorrelated, so the nominal n would make
    almost anything significant. n_eff uses the lag-1 autocorrelation of both
    series (Bretherton et al. 1999).
    """
    g = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(g) < 10:
        return np.nan, np.nan, np.nan, 0
    r = g.x.corr(g.y)
    slope = np.polyfit(g.x, g.y, 1)[0]
    r1x = g.x.autocorr(1)
    r1y = g.y.autocorr(1)
    n_eff = len(g) * (1 - r1x * r1y) / (1 + r1x * r1y)
    n_eff = float(np.clip(n_eff, 3, len(g)))
    tstat = r * np.sqrt((n_eff - 2) / max(1e-12, 1 - r ** 2))
    pval = 2 * (1 - tdist.cdf(abs(tstat), n_eff - 2))
    return r, slope, pval, n_eff


def seasonal_fit(y):
    """Mean + annual + semiannual least squares. Returns fit, params, R2."""
    g = y.dropna()
    if len(g) < 60:
        return None
    t = (g.index - g.index[0]).total_seconds().values / 86400.0
    w = 2 * np.pi / 365.25
    X = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t),
                         np.cos(2 * w * t), np.sin(2 * w * t)])
    b, *_ = np.linalg.lstsq(X, g.values, rcond=None)
    fit = pd.Series(X @ b, index=g.index)
    ss = 1 - np.var(g.values - fit.values) / np.var(g.values)
    # variance explained by each harmonic on its own
    r2_ann = 1 - np.var(g.values - X[:, [0, 1, 2]] @ b[[0, 1, 2]]) / np.var(g.values)
    r2_semi = 1 - np.var(g.values - X[:, [0, 3, 4]] @ b[[0, 3, 4]]) / np.var(g.values)
    return dict(fit=fit, R2=ss, R2_annual=r2_ann, R2_semiannual=r2_semi,
                amp_annual=np.hypot(b[1], b[2]),
                amp_semiannual=np.hypot(b[3], b[4]), mean=b[0])


# =========================================================== 1. EBB / FLOOD ===
# Composite over the tidal cycle, on the TIDAL ANOMALY (hourly minus Godin),
# not on raw salinity. The tidal excursion here is a few tenths of a g/kg
# riding on a seasonal range of several g/kg, so a raw composite is a flat
# line inside a band that is just the seasonal cycle in disguise. Removing the
# subtidal background is what makes the ebb/flood shape visible at all.
BINS = np.arange(-6.5, 6.6, 1.0)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
T['hw_bin'] = pd.cut(T.hrs_hw, BINS, labels=CTR)

comp_rows = []
fig, axs = plt.subplots(len(names), 3, figsize=(15, 4.2 * len(names)),
                        squeeze=False, layout='constrained')
for i, nm in enumerate(names):
    for j, (vn, lab) in enumerate([('s_top', 'surface salinity'),
                                   ('s_bot', 'bottom salinity'),
                                   ('dstrat', 'bottom - surface')]):
        ax = axs[i][j]
        col = '%s_%s_tid' % (nm, vn)
        for ph, ls in [('spring', '-'), ('neap', '--')]:
            sel = T[T.model_phase == ph]
            if len(sel) == 0:
                continue
            g = sel.groupby('hw_bin', observed=False)[col]
            m, se = g.mean(), g.std() / np.sqrt(g.count())
            ax.plot(CTR, m.values, ls, color=LCOLOR.get(nm, 'k'), lw=2,
                    label=ph)
            ax.fill_between(CTR, m - 2 * se, m + 2 * se,
                            color=LCOLOR.get(nm, 'k'), alpha=0.20, lw=0)
            for c, mm, ss, nn in zip(CTR, m.values, g.std().values,
                                     g.count().values):
                comp_rows.append(dict(location=nm, var=vn, phase=ph,
                                      hrs_from_HW=c, mean_anomaly=mm, std=ss,
                                      n=nn))
        g = T.groupby('hw_bin', observed=False)[col].mean()
        ax.plot(CTR, g.values, ':', color='k', lw=1.4, label='all days')
        ax.axhline(0, color='0.5', lw=0.8)
        ax.axvline(0, color='0.5', lw=0.8)
        ax.grid(**GRID)
        ax.set_xlabel('hours from high water')
        pk = np.nanmax(g.values) - np.nanmin(g.values)
        ax.set_title('%s -- %s\ntidal excursion %.3f g kg$^{-1}$ peak to peak'
                     % (nm, lab, pk), color=LCOLOR.get(nm, 'k'), fontsize=10)
        if j == 0:
            ax.set_ylabel('tidal anomaly (g kg$^{-1}$)\nsubtidal removed')
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
        # shade the flood half, using the verified sign; after the ylim is set
        # by the data, so the shading spans exactly the axes
        fl = T.groupby('hw_bin', observed=False)['flood'].mean()
        y0, y1 = ax.get_ylim()
        ax.fill_between(CTR, y0, y1, where=(fl.values > 0.5),
                        color='#4565e8', alpha=0.08, lw=0, zorder=0)
        ax.set_ylim(y0, y1)
fig.suptitle('%s -- %s mouth: salinity through the tidal cycle\n'
             'blue shading = flood (into the cove); band = +/- 2 standard errors'
             % (args.gtx, args.sect), fontsize=12)
savefig(fig, 'fig1_tidal_cycle.png')
plt.close(fig)
pd.DataFrame(comp_rows).to_csv(out_dir / 'tidal_composite.csv', index=False,
                               float_format='%.4f')

# flood-vs-ebb means, the single-number version of figure 1
fe_rows = []
for nm in names:
    for vn in ['s_top', 's_bot', 'dstrat']:
        raw, tid = '%s_%s' % (nm, vn), '%s_%s_tid' % (nm, vn)
        fe_rows.append(dict(
            location=nm, var=vn,
            flood_mean=T.loc[T.flood, raw].mean(),
            ebb_mean=T.loc[~T.flood, raw].mean(),
            # the anomaly version is the clean one: it cannot be biased by a
            # seasonal cycle happening to be sampled unevenly by flood vs ebb
            flood_minus_ebb=(T.loc[T.flood, tid].mean()
                             - T.loc[~T.flood, tid].mean())))
FE = pd.DataFrame(fe_rows)
FE.to_csv(out_dir / 'flood_ebb_means.csv', index=False, float_format='%.4f')
print('\nflood minus ebb, tidal anomaly (g/kg):')
print(FE.pivot(index='location', columns='var',
               values='flood_minus_ebb').round(3).to_string())

# ========================================================== 2. SPRING/NEAP ===
# The fortnightly response has to be measured on SEASONAL ANOMALIES. Daily
# stratification here swings over ~20 g/kg across a year while the spring/neap
# effect is a few tenths, so a raw scatter against tidal range is almost all
# season and returns r ~ 0 no matter what the tide is doing. Subtracting a
# centred 30-day rolling mean removes the seasonal and synoptic background
# while leaving the 14.8 d cycle intact -- 30 days is about two spring-neap
# cycles, so the rolling mean cannot absorb the signal being tested.
ROLL = 30


def anom(s):
    return s - s.rolling(ROLL, center=True, min_periods=ROLL // 2).mean()


D['tide_range_anom'] = anom(D.tide_range)
for nm in names:
    for vn in ['s_top', 's_bot', 'dstrat']:
        D['%s_%s_anom' % (nm, vn)] = anom(D['%s_%s' % (nm, vn)])
        D['%s_%s_amp_anom' % (nm, vn)] = anom(D['%s_%s_amp' % (nm, vn)])

sn_rows = []
fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')

A = axs[0][0]
for nm in names:
    x, y = D.tide_range_anom, D['%s_dstrat_anom' % nm]
    r, slope, pv, ne = eff_corr(x, y)
    A.plot(x, y, '.', ms=3, alpha=0.4, color=LCOLOR.get(nm, 'k'),
           label='%s: r=%+.2f, %.2f per m (p=%.4f, n_eff=%.0f)'
                 % (nm, r, slope, pv, ne))
    g = pd.DataFrame({'x': x, 'y': y}).dropna()
    xx = np.linspace(g.x.min(), g.x.max(), 10)
    A.plot(xx, np.polyval(np.polyfit(g.x, g.y, 1), xx), '-',
           color=LCOLOR.get(nm, 'k'), lw=2)
    sn_rows.append(dict(location=nm, var='dstrat', vs='tide range',
                        detrended='30 d anomaly', r=r, slope=slope, p=pv,
                        n_eff=ne))
    # the raw, confounded version, kept so the difference is on the record
    r0, s0, p0, n0 = eff_corr(D.tide_range, D['%s_dstrat' % nm])
    sn_rows.append(dict(location=nm, var='dstrat', vs='tide range',
                        detrended='none (seasonal cycle left in)', r=r0,
                        slope=s0, p=p0, n_eff=n0))
A.axhline(0, color='0.5', lw=0.8)
A.axvline(0, color='0.5', lw=0.8)
A.set_xlabel('daily tidal range anomaly (m)')
A.set_ylabel('stratification anomaly (g kg$^{-1}$)')
A.set_title('stratification vs the fortnightly envelope\n'
            '(both as 30-day anomalies)')
A.grid(**GRID)
A.legend(fontsize=8)

B = axs[0][1]
for nm in names:
    x, y = D.tide_range_anom, D['%s_dstrat_amp_anom' % nm]
    r, slope, pv, ne = eff_corr(x, y)
    B.plot(x, y, '.', ms=3, alpha=0.4, color=LCOLOR.get(nm, 'k'),
           label='%s: r=%+.2f (p=%.4f)' % (nm, r, pv))
    g = pd.DataFrame({'x': x, 'y': y}).dropna()
    xx = np.linspace(g.x.min(), g.x.max(), 10)
    B.plot(xx, np.polyval(np.polyfit(g.x, g.y, 1), xx), '-',
           color=LCOLOR.get(nm, 'k'), lw=2)
    sn_rows.append(dict(location=nm, var='dstrat tidal amplitude',
                        vs='tide range', detrended='30 d anomaly', r=r,
                        slope=slope, p=pv, n_eff=ne))
B.axhline(0, color='0.5', lw=0.8)
B.axvline(0, color='0.5', lw=0.8)
B.set_xlabel('daily tidal range anomaly (m)')
B.set_ylabel('anomaly of the within-day tidal sd (g kg$^{-1}$)')
B.set_title('how hard the tide works the water column')
B.grid(**GRID)
B.legend(fontsize=8)

C = axs[1][0]
if D.days_from_syzygy.notna().any():
    sb = np.arange(-7.5, 7.6, 1.0)
    sc = 0.5 * (sb[:-1] + sb[1:])
    cut = pd.cut(D.days_from_syzygy, sb, labels=sc)
    for nm in names:
        g = D.groupby(cut, observed=False)['%s_dstrat_anom' % nm]
        se = g.std() / np.sqrt(g.count())
        C.errorbar(sc, g.mean().values, yerr=2 * se.values, fmt='o-',
                   color=LCOLOR.get(nm, 'k'), ms=4, lw=1.6, capsize=2, label=nm)
    C.axvline(0, color='0.5', lw=0.8)
    C.axhline(0, color='0.5', lw=0.8)
    C.set_xlabel('days from syzygy (0 = new/full moon = spring)')
    C.set_ylabel('stratification anomaly (g kg$^{-1}$)')
    C.set_title('composite over the 14.8 d synodic cycle\n'
                'error bars = +/- 2 standard errors')
    C.grid(**GRID)
    C.legend(fontsize=8)
else:
    C.text(0.5, 0.5, 'no phase_daily.csv -- run the tidal phase calendar',
           ha='center', transform=C.transAxes)

E = axs[1][1]
E2 = E.twinx()
E2.plot(D.index, D.tide_range, '-', color='0.7', lw=0.8, zorder=0)
E2.set_ylabel('daily tidal range (m)', color='0.5')
E2.tick_params(axis='y', colors='0.5')
for nm in names:
    E.plot(D.index, D['%s_dstrat' % nm], '-', lw=1.2,
           color=LCOLOR.get(nm, 'k'), label=nm, zorder=3)
E.set_zorder(E2.get_zorder() + 1)
E.patch.set_visible(False)
E.set_ylabel('bottom - surface salinity (g kg$^{-1}$)')
E.set_title('daily stratification over the record (grey = tidal range)')
E.grid(**GRID)
E.legend(fontsize=8, loc='upper left')
E.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

fig.suptitle('%s -- %s mouth: spring/neap response' % (args.gtx, args.sect),
             fontsize=12)
savefig(fig, 'fig2_spring_neap.png')
plt.close(fig)

# ============================================================= 3. SEASONAL ===
fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
mon = np.arange(1, 13)

A = axs[0][0]
for nm in names:
    for vn, ls in [('s_top', '-'), ('s_bot', '--')]:
        g = D.groupby('month')['%s_%s' % (nm, vn)]
        A.plot(mon, g.mean().reindex(mon).values, ls, lw=2,
               color=LCOLOR.get(nm, 'k'), label='%s %s' % (nm, vn))
        A.fill_between(mon, g.quantile(0.25).reindex(mon).values,
                       g.quantile(0.75).reindex(mon).values,
                       color=LCOLOR.get(nm, 'k'), alpha=0.10, lw=0)
A.set_xticks(mon)
A.set_xlabel('month')
A.set_ylabel('salinity (g kg$^{-1}$)')
A.set_title('monthly climatology, surface (solid) and bottom (dashed)\n'
            'band = interquartile range of daily means')
A.grid(**GRID)
A.legend(fontsize=8, ncol=2)

B = axs[0][1]
for nm in names:
    g = D.groupby('month')['%s_dstrat' % nm]
    B.plot(mon, g.mean().reindex(mon).values, 'o-', lw=2, ms=4,
           color=LCOLOR.get(nm, 'k'), label=nm)
    B.fill_between(mon, g.quantile(0.25).reindex(mon).values,
                   g.quantile(0.75).reindex(mon).values,
                   color=LCOLOR.get(nm, 'k'), alpha=0.10, lw=0)
B.set_xticks(mon)
B.set_xlabel('month')
B.set_ylabel('bottom - surface salinity (g kg$^{-1}$)')
B.set_title('seasonal cycle of stratification')
B.grid(**GRID)
B.legend(fontsize=8)

C = axs[1][0]
for nm in names:
    g = D.groupby('month')['%s_dstrat_amp' % nm]
    C.plot(mon, g.mean().reindex(mon).values, 'o-', lw=2, ms=4,
           color=LCOLOR.get(nm, 'k'), label=nm)
C.set_xticks(mon)
C.set_xlabel('month')
C.set_ylabel('within-day sd of the tidal band (g kg$^{-1}$)')
C.set_title('seasonal modulation of the TIDAL salinity signal')
C.grid(**GRID)
C.legend(fontsize=8)

E = axs[1][1]
sfit_rows = []
for nm in names:
    y = D['%s_dstrat' % nm]
    E.plot(D.index, y, '-', lw=0.8, alpha=0.35, color=LCOLOR.get(nm, 'k'))
    f = seasonal_fit(y)
    if f is None:
        continue
    E.plot(f['fit'].index, f['fit'].values, '-', lw=2.2,
           color=LCOLOR.get(nm, 'k'),
           label='%s: R2=%.2f (annual %.2f, semiannual %.2f)'
                 % (nm, f['R2'], f['R2_annual'], f['R2_semiannual']))
    sfit_rows.append(dict(location=nm, var='dstrat', mean=f['mean'],
                          amp_annual=f['amp_annual'],
                          amp_semiannual=f['amp_semiannual'], R2=f['R2'],
                          R2_annual=f['R2_annual'],
                          R2_semiannual=f['R2_semiannual']))
    for vn in ['s_top', 's_bot']:
        f2 = seasonal_fit(D['%s_%s' % (nm, vn)])
        if f2 is not None:
            sfit_rows.append(dict(location=nm, var=vn, mean=f2['mean'],
                                  amp_annual=f2['amp_annual'],
                                  amp_semiannual=f2['amp_semiannual'],
                                  R2=f2['R2'], R2_annual=f2['R2_annual'],
                                  R2_semiannual=f2['R2_semiannual']))
E.set_ylabel('bottom - surface salinity (g kg$^{-1}$)')
E.set_title('annual + semiannual harmonic fit')
E.grid(**GRID)
E.legend(fontsize=8)
E.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

fig.suptitle('%s -- %s mouth: seasonal cycle.  Two years = two samples per '
             'calendar month:\nenough for a cycle that is large and '
             'phase-locked, NOT enough to call any single month anomalous.'
             % (args.gtx, args.sect), fontsize=12)
savefig(fig, 'fig3_seasonal.png')
plt.close(fig)
pd.DataFrame(sfit_rows).to_csv(out_dir / 'seasonal_harmonic_fit.csv',
                               index=False, float_format='%.4f')

# monthly + seasonal climatology tables
clim = D.groupby('month')[[c for c in D.columns
                           if any(c.startswith(n) for n in names)]].mean()
clim.to_csv(out_dir / 'monthly_climatology.csv', float_format='%.4f')
seas = D.groupby('season')[[c for c in D.columns
                            if any(c.startswith(n) for n in names)]].mean()
seas.to_csv(out_dir / 'seasonal_climatology.csv', float_format='%.4f')
print('\nseasonal means (bottom - surface salinity, g/kg):')
print(seas[[('%s_dstrat' % n) for n in names]].round(3).to_string())

# ============================================================= 4. VARIANCE ===
# how much of the hourly signal lives in each band
var_rows = []
for nm in names:
    for vn in ['s_top', 's_bot', 'dstrat']:
        raw = T['%s_%s' % (nm, vn)]
        sub = T['%s_%s_sub' % (nm, vn)]
        tid = T['%s_%s_tid' % (nm, vn)]
        ok = raw.notna() & sub.notna()
        v_tot = float(np.var(raw[ok]))
        v_tid = float(np.var(tid[ok]))
        v_sub = float(np.var(sub[ok]))
        f = seasonal_fit(D['%s_%s' % (nm, vn)])
        v_seas = (f['R2'] * float(np.var(D['%s_%s' % (nm, vn)].dropna()))
                  if f else np.nan)
        var_rows.append(dict(location=nm, var=vn, sd_total=np.sqrt(v_tot),
                             frac_tidal=v_tid / v_tot, frac_subtidal=v_sub / v_tot,
                             frac_seasonal_of_total=v_seas / v_tot))
V = pd.DataFrame(var_rows)
V.to_csv(out_dir / 'variance_partition.csv', index=False, float_format='%.4f')
print('\nvariance partition (fraction of hourly variance):')
print(V.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), layout='constrained')
A = axs[0]
lbl = ['%s\n%s' % (r.location, r['var']) for _, r in V.iterrows()]
x = np.arange(len(V))
A.bar(x, V.frac_tidal, color='#4565e8', label='tidal band (hourly - Godin)')
A.bar(x, V.frac_subtidal, bottom=V.frac_tidal, color='#e04256',
      label='subtidal (Godin)')
A.plot(x, V.frac_seasonal_of_total, 'k^', ms=7,
       label='of which annual+semiannual')
A.set_xticks(x)
A.set_xticklabels(lbl, fontsize=8)
A.set_ylabel('fraction of hourly variance')
A.set_title('where the salinity variance lives')
A.grid(**GRID)
A.legend(fontsize=8)

B = axs[1]
if args.utide:
    try:
        import utide
        con_rows = []
        tnum = idx.tz_convert('UTC').tz_localize(None).to_pydatetime()
        for nm in names:
            for vn in ['s_top', 's_bot']:
                y = T['%s_%s' % (nm, vn)].values
                sol = utide.solve(pd.to_datetime(tnum), y, lat=48.23,
                                  method='ols', conf_int='none', nodal=True,
                                  trend=False, verbose=False)
                for cn, amp in zip(sol.name, sol.A):
                    con_rows.append(dict(location=nm, var=vn, constituent=cn,
                                         amplitude=amp))
        sol_ssh = utide.solve(pd.to_datetime(tnum), ssh.values, lat=48.23,
                              method='ols', conf_int='none', nodal=True,
                              trend=False, verbose=False)
        for cn, amp in zip(sol_ssh.name, sol_ssh.A):
            con_rows.append(dict(location='ssh', var='ssh', constituent=cn,
                                 amplitude=amp))
        CN = pd.DataFrame(con_rows)
        CN.to_csv(out_dir / 'utide_constituents.csv', index=False,
                  float_format='%.5f')
        MAIN = ['M2', 'S2', 'N2', 'K1', 'O1', 'P1']
        sub = CN[CN.constituent.isin(MAIN)]
        wide = sub.pivot_table(index='constituent',
                               columns=['location', 'var'], values='amplitude')
        wide = wide.reindex(MAIN)
        cols = [c for c in wide.columns if c[0] != 'ssh']
        w = 0.8 / max(1, len(cols))
        xx = np.arange(len(MAIN))
        for kk, c in enumerate(cols):
            B.bar(xx + kk * w, wide[c].values, w,
                  color=LCOLOR.get(c[0], 'k'),
                  alpha=1.0 if c[1] == 's_top' else 0.5,
                  label='%s %s' % (c[0], c[1]))
        B.set_xticks(xx + 0.4 - w / 2)
        B.set_xticklabels(MAIN)
        B.set_ylabel('salinity amplitude (g kg$^{-1}$)')
        B.set_title('tidal constituents of SALINITY (utide)')
        B.grid(**GRID)
        B.legend(fontsize=7, ncol=2)
        print('\nssh constituent amplitudes (m):')
        print(CN[(CN.location == 'ssh') & CN.constituent.isin(MAIN)]
              [['constituent', 'amplitude']].round(3).to_string(index=False))
    except Exception as e:
        B.text(0.5, 0.5, 'utide failed:\n%s' % e, ha='center', va='center',
               transform=B.transAxes, fontsize=8)
        print('utide failed: %s' % e)
else:
    B.axis('off')

fig.suptitle('%s -- %s mouth: variance and constituents'
             % (args.gtx, args.sect), fontsize=12)
savefig(fig, 'fig4_variance.png')
plt.close(fig)

D.to_csv(out_dir / 'daily_series.csv', float_format='%.4f')
pd.DataFrame(sn_rows).to_csv(out_dir / 'spring_neap_stats.csv', index=False,
                             float_format='%.4f')
print('\nsaved daily_series.csv, spring_neap_stats.csv and the tables above to')
print('  %s' % out_dir)
