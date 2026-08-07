"""
Along-channel salinity and velocity gradients in Penn Cove, and what the wind
does to them.

The three cross-cove sections are a transect, not three unrelated places:

    pc_lp   lon -122.6534   the MOUTH      x = 0.00 km
    pc_lj   lon -122.6723   mid-cove       x = 1.51 km
    pc_cp   lon -122.6939   inner / head   x = 3.12 km

so the along-channel coordinate x runs mouth -> head and every quantity below
is a property of that axis rather than of one section. x is positive INTO the
cove, which is the same sign convention the wind already uses (w_along > 0
blows from the mouth toward the head), so a positive wind and a positive
velocity mean the same thing and the two can be read against each other
without a sign table.

WHAT IS COMPUTED

  ds/dx     the along-channel salinity gradient, from a least-squares fit of
            section-mean salinity against x across all three sections, done
            separately for the top 3 m, the bottom 3 m and the full depth.
            Pairwise gradients (mouth->mid, mid->head) are carried too,
            because a wind that pushes the front up the cove steepens one half
            while flattening the other, and the single fitted slope averages
            exactly that away.

            DO NOT read the sign as a textbook estuary would. Penn Cove has no
            river worth the name -- Coupeville and Penn Cove STP together are
            ~0.009 m3/s -- so its fresh water arrives from OUTSIDE, at the
            mouth, as Skagit-influenced Saratoga Passage water. The measured
            mean gradients are therefore
                surface  ds/dx = +0.07 g/kg/km   (the HEAD is saltier)
                bottom   ds/dx = -0.08 g/kg/km   (the head is fresher)
            which is not a horizontal salinity front at all but a
            STRATIFICATION gradient: the cove mixes vertically toward its
            closed end (top-to-bottom 4.14 g/kg at the mouth, 3.68 at the
            head). Surface and bottom gradients having opposite signs is the
            single most important fact about this transect, and it is why they
            are never combined into one number.

  u         section-normal velocity, positive INTO the cove, from q/(dd*DZ),
            layer-averaged the same way. u_top - u_bot is the exchange, and it
            is NEGATIVE in the mean (-0.0055 m/s), i.e. out at the surface and
            in at depth -- the estuarine sense, driven from the mouth rather
            than by a local river. The wind can reverse it.

  du/dx     the along-channel gradient of that velocity: at the surface, where
            it is order 10^-6 s^-1, and depth-mean, where it is three orders
            smaller because on subtidal timescales the cove neither fills nor
            drains.

THREE THINGS THIS ANALYSIS IS BUILT AROUND

  ANOMALIES  the seasonal cycle in Whidbey salinity is an order of magnitude
             larger than anything the wind does in three days, so every
             composite is built on 30-day centred rolling anomalies of the
             Godin-lowpassed series, never on raw values. Composited raw, a
             real wind response reads as a flat line inside a band that is
             just the season in disguise.

  STRESS,    events are picked on the along-cove wind STRESS, not the wind
  NOT SPEED  speed. Momentum into the water goes as speed squared, and the
             sign is the whole point -- a 10 m/s wind blowing across the cove
             does almost nothing to an along-channel gradient.

  TWO        the wind events are not one population, and compositing them
  FAMILIES   together cancels the response. Split by the sign of the along-cove
             stress they come out cleanly separated in direction, and the two
             families are not each other's mirror image:

               DOWN-COVE  n=36, from 257 +/- 24 deg -- westerlies. The cove
                          axis is 255 deg true, so these blow straight down
                          it: along 4.10 m/s against cross 1.35.
               UP-COVE    n=22, from 142 +/- 16 deg -- south-easterlies, the
                          region's prevailing direction. These hit the axis
                          obliquely and are mostly CROSS-cove: along 1.98 m/s
                          against cross 4.58.

             So "up-cove event" is a weaker along-axis forcing than
             "down-cove event" at the same wind speed (5.70 vs 5.68 m/s),
             and the difference in response amplitude between the families is
             partly just that. The along and cross components are reported per
             event in wind_events_alongcove.csv.

The continuous test is a lagged correlation of the along-cove stress against
each response, on DAILY means, with an autocorrelation-corrected n_eff (lag-1,
Bretherton et al. 1999) -- subtidal values are massively autocorrelated and
nominal n makes anything at all significant. Daily rather than hourly because
at hourly sampling of an already-lowpassed series the lag-1 estimator is not
usable; see the comment at the correlation block.

Both the wind and the response are Godin-lowpassed (71 hours), so lags shorter
than about a day are not resolvable. A composite peak at -0.1 d means
simultaneous, not the ocean leading the wind.

Note u is the section-NORMAL component only; the tef2 extraction never carries
the along-section component, so |u| is a lower bound on the true speed.

Runs on the mac from the local extractions_avg plus the reduced wind pickle.

run 20260807_pc_alongchannel_wind.py
run 20260807_pc_alongchannel_wind.py --hlay 4 --minsep 5
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.path import Path as MplPath
from scipy.signal import find_peaks
from scipy import stats

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('--ds0', default='2024.01.01')
p.add_argument('--ds1', default='2025.12.31')
p.add_argument('--sects', default='pc_lp,pc_lj,pc_cp', help='mouth first')
p.add_argument('--hlay', type=float, default=3.0,
               help='thickness (m) of the surface and bottom layers')
p.add_argument('--anom_days', type=float, default=30.0,
               help='window for the rolling mean that defines the anomaly')
p.add_argument('--minsep', type=float, default=4.0,
               help='minimum days between wind events')
p.add_argument('--pct', type=float, default=90.0,
               help='percentile of |along-cove stress| that defines an event')
p.add_argument('--lead', type=float, default=3.0, help='days before the peak')
p.add_argument('--lag', type=float, default=5.0, help='days after the peak')
p.add_argument('--tz', default='America/Los_Angeles')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
in_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
           / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260807_pc_alongchannel_wind'
Lfun.make_dir(out_dir)

SECTS = [s.strip() for s in args.sects.split(',')]
SLAB = {'pc_lp': 'mouth', 'pc_lj': 'mid-cove', 'pc_cp': 'head'}
CB = dict(blue='#0072B2', orange='#D55E00', green='#009E73', red='#CC0000',
          purple='#7B3294', yellow='#E69F00', pink='#CC79A7', grey='#7f7f7f')
SC = {'pc_lp': CB['blue'], 'pc_lj': CB['green'], 'pc_cp': CB['orange']}
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
RHO0 = 1025.0
O2_MMOL_TO_MGL = 32.0 / 1000.0


def met_dir(u, v):
    """Direction the wind blows FROM, deg clockwise from N."""
    return (270 - np.rad2deg(np.arctan2(v, u))) % 360


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def anom(s):
    """Deviation from a centred rolling mean -- the seasonal cycle removed."""
    w = int(round(args.anom_days * 24))
    return s - s.rolling(w, center=True, min_periods=int(w * 0.6)).mean()


def neff_r(x, y):
    """corr with an autocorrelation-corrected n, and its two-sided p."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    n = len(x)
    if n < 20:
        return np.nan, np.nan, 0
    r = np.corrcoef(x, y)[0, 1]
    r1 = [np.corrcoef(v[:-1], v[1:])[0, 1] for v in (x, y)]
    f = (1 - r1[0] * r1[1]) / (1 + r1[0] * r1[1])
    ne = max(3.0, n * min(1.0, max(f, 1e-6)))
    t = r * np.sqrt((ne - 2) / max(1e-12, 1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), ne - 2), int(ne)


# ------------------------------------------------------------- geometry ---
# The channel axis is defined by the sections themselves -- the line joining
# the mouth section's centroid to the head section's -- because that is the
# axis the gradients are actually differenced along. It is then checked
# against the principal axis of the Penn Cove polygon, which is what the wind
# reduction used; if the two disagreed, every projection below would be
# measuring a different direction from the wind it is being compared with.
dstr = xr.open_dataset(tef2 / ('structure_%s_%s_%s.nc' % (args.ds0, args.ds1,
                                                          gctag)))
CEN = {sn: (float(np.mean(dstr['%s_lon' % sn].values)),
            float(np.mean(dstr['%s_lat' % sn].values))) for sn in SECTS}
lat0 = np.mean([c[1] for c in CEN.values()])
COS = np.cos(np.deg2rad(lat0))


def xy_km(lo, la):
    return lo * COS * 111.32, la * 111.32


x0, y0 = xy_km(*CEN[SECTS[0]])
xN, yN = xy_km(*CEN[SECTS[-1]])
axl = np.hypot(xN - x0, yN - y0)
ax_, ay_ = (xN - x0) / axl, (yN - y0) / axl      # unit vector, mouth -> head
X = {}
for sn in SECTS:
    xs, ys = xy_km(*CEN[sn])
    X[sn] = (xs - x0) * ax_ + (ys - y0) * ay_
xv = np.array([X[sn] for sn in SECTS])

# The polygon axis, for the cross-check. Taken over the grid's WATER cells
# inside pc.p, exactly as 20260806_wind_characterize.py did it -- the axis of
# the polygon's vertices is not the same thing and is not what the wind was
# projected onto.
C = pd.read_pickle(wind_fn)
pcp = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc.p')
glon, glat, water = C['lon'], C['lat'], C['water']
m = MplPath(np.column_stack([pcp.x.values.astype(float),
                             pcp.y.values.astype(float)])).contains_points(
    np.column_stack([glon.ravel(), glat.ravel()])).reshape(glon.shape) & water
c = np.cov(np.column_stack([glon[m] * COS * 111.32, glat[m] * 111.32]).T)
ev, EV = np.linalg.eigh(c)
e = EV[:, np.argmax(ev)]
if e[0] > 0:
    e = -e                                        # point it mouth -> head
off = np.rad2deg(np.arccos(np.clip(abs(e[0] * ax_ + e[1] * ay_), -1, 1)))

print('--- along-channel axis ---')
print('sections define (%.4f, %.4f), i.e. %.0f deg true, mouth -> head'
      % (ax_, ay_, np.rad2deg(np.arctan2(ax_, ay_)) % 360))
print('pc polygon principal axis (%.4f, %.4f); the two differ by %.1f deg'
      % (e[0], e[1], off))
for sn in SECTS:
    print('  %-6s (%-8s) x = %5.2f km' % (sn, SLAB[sn], X[sn]))

# ------------------------------------------------------ flood sign check ---
dflux = xr.open_dataset(tef2 / ('hourly_flux_%s_%s_%s.nc'
                                % (args.ds0, args.ds1, gctag)))
SGN = {}
print('\n--- which sign of q runs into the cove (verified, not assumed) ---')
for sn in SECTS:
    q = dflux.qnet.sel(sect=sn).values
    z = dflux.ssh.sel(sect=sn).values
    r = np.corrcoef(q, np.gradient(z))[0, 1]
    SGN[sn] = -1.0 if r < 0 else 1.0
    print('  %-6s corr(qnet, d(ssh)/dt) = %+.2f -> inflow is q %s 0'
          % (sn, r, '<' if r < 0 else '>'))
dflux.close()

# ------------------------------------------------- per-section reduction ---
# Layer means use PARTIAL cells: a sigma cell straddling the 3 m level
# contributes the fraction of itself that lies inside the layer. Without that
# the surface layer would be whatever the top few sigma cells happen to add
# up to, which changes with the tide and with h across the section, and the
# resulting "gradient" would partly be a gradient in layer thickness.
print('\n--- reducing sections (top/bottom %.1f m layers) ---' % args.hlay)
R = {}
for sn in SECTS:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    tt = pd.DatetimeIndex(ds.time.values)
    dd = ds.dd.values[None, None, :]                       # (1,1,p) face width
    DZ = ds.DZ.values                                      # (t,z,p)
    salt = ds.salt.values
    oxy = ds.oxygen.values * O2_MMOL_TO_MGL                # mmol m-3 -> mg/L
    q = ds.q.values * SGN[sn]                              # + = into the cove

    cum_hi = np.cumsum(DZ, axis=1)                         # above the bed
    cum_lo = cum_hi - DZ
    H = cum_hi[:, -1:, :]
    d_hi = H - cum_hi                                      # below the surface
    w_bot = np.clip(args.hlay - cum_lo, 0, DZ) / DZ
    w_top = np.clip(args.hlay - d_hi, 0, DZ) / DZ
    w_all = np.ones_like(DZ)

    z_c = 0.5 * (cum_hi + cum_lo)                          # height above bed
    d = {}
    for nm, w in [('top', w_top), ('bot', w_bot), ('bar', w_all)]:
        A = dd * DZ * w                                    # face area, m2
        As = A.sum(axis=(1, 2))
        d['s_' + nm] = (salt * A).sum(axis=(1, 2)) / As
        d['o_' + nm] = (oxy * A).sum(axis=(1, 2)) / As
        d['u_' + nm] = (q * w).sum(axis=(1, 2)) / As
        d['Q_' + nm] = (q * w).sum(axis=(1, 2))
        d['z_' + nm] = (z_c * A).sum(axis=(1, 2)) / As     # layer centroid
    d['A'] = (dd * DZ).sum(axis=(1, 2))
    R[sn] = pd.DataFrame(d, index=tt)
    print('  %-6s %d faces, area %.3f km2, mean H %.1f m, layer centroids '
          '%.1f / %.1f m above bed'
          % (sn, ds.sizes['p'], R[sn].A.mean() / 1e6,
             float(H.mean()), R[sn].z_top.mean(), R[sn].z_bot.mean()))
    ds.close()

TT = R[SECTS[0]].index
for sn in SECTS:
    assert R[sn].index.equals(TT), 'section time axes differ'

# ------------------------------------------------------- subtidal series ---
# Everything from here on is the Godin lowpass. The wind response being asked
# about is a multi-day one; leaving the tide in would swamp it with a signal
# an order of magnitude larger that has nothing to do with the question.
S = pd.DataFrame(index=TT)
for sn in SECTS:
    for c_ in ['s_top', 's_bot', 's_bar', 'u_top', 'u_bot', 'u_bar',
               'o_top', 'o_bot', 'o_bar', 'Q_top', 'Q_bot']:
        S['%s_%s' % (sn, c_)] = godin(R[sn][c_].values)
    S['%s_dstrat' % sn] = S['%s_s_bot' % sn] - S['%s_s_top' % sn]
    dz = (R[sn].z_top - R[sn].z_bot).mean()
    S['%s_shear' % sn] = (S['%s_u_top' % sn] - S['%s_u_bot' % sn]) / dz

# gradients: least-squares slope of section means against x, per hour
den = ((xv - xv.mean()) ** 2).sum()
for lay in ['top', 'bot', 'bar']:
    M = np.column_stack([S['%s_s_%s' % (sn, lay)].values for sn in SECTS])
    S['dsdx_' + lay] = ((M - M.mean(axis=1, keepdims=True))
                        * (xv - xv.mean())).sum(axis=1) / den
    U = np.column_stack([S['%s_u_%s' % (sn, lay)].values for sn in SECTS])
    S['dudx_' + lay] = ((U - U.mean(axis=1, keepdims=True))
                        * (xv - xv.mean())).sum(axis=1) / den / 1e3   # per s
    O = np.column_stack([S['%s_o_%s' % (sn, lay)].values for sn in SECTS])
    S['dodx_' + lay] = ((O - O.mean(axis=1, keepdims=True))
                        * (xv - xv.mean())).sum(axis=1) / den

# pairwise, so a front moving along the cove is not averaged away
for a, b, nm in [(0, 1, 'outer'), (1, 2, 'inner')]:
    for lay in ['top', 'bot', 'bar']:
        S['dsdx_%s_%s' % (lay, nm)] = (
            (S['%s_s_%s' % (SECTS[b], lay)] - S['%s_s_%s' % (SECTS[a], lay)])
            / (xv[b] - xv[a]))
S['shear_cove'] = S[['%s_shear' % sn for sn in SECTS]].mean(axis=1)
S['exch_cove'] = S[['%s_u_top' % sn for sn in SECTS]].mean(axis=1) - \
    S[['%s_u_bot' % sn for sn in SECTS]].mean(axis=1)
S['dstrat_cove'] = S[['%s_dstrat' % sn for sn in SECTS]].mean(axis=1)
S['do_cove'] = S[['%s_o_bot' % sn for sn in SECTS]].mean(axis=1)

# ------------------------------------------------------------------ wind ---
W = C['W']
wa = W.u_pc.values * ax_ + W.v_pc.values * ay_        # + = blowing INTO cove
wc = -W.u_pc.values * ay_ + W.v_pc.values * ax_       # + = toward north shore
spd = W.spd_pc.values
# signed along-cove stress: the stress magnitude (quadratic in speed) times
# the fraction of the wind that lies on the cove axis, keeping the sign.
# Momentum input scales with speed squared and the sign is the whole point.
tau = W.tau_pc.values * wa / np.maximum(spd, 1e-6)
Wd = pd.DataFrame({'w_along': wa, 'w_cross': wc, 'spd': spd, 'tau_along': tau},
                  index=W.index)
# the section time axis is on the half hour, the wind on the hour
Wd = (Wd.reindex(Wd.index.union(TT)).interpolate('time').reindex(TT))
for c_ in ['w_along', 'w_cross', 'spd', 'tau_along']:
    S[c_] = godin(Wd[c_].values)

print('\n--- along-cove wind at Penn Cove, subtidal ---')
print('  mean %+.2f m/s, sd %.2f, range %+.2f to %+.2f (+ = into the cove)'
      % (S.w_along.mean(), S.w_along.std(), S.w_along.min(), S.w_along.max()))
print('  blows into the cove %.0f%% of hours; |along| / speed = %.2f'
      % (100 * (S.w_along > 0).mean(),
         np.nanmean(np.abs(S.w_along)) / np.nanmean(S.spd)))

# --------------------------------------------------------------- anomalies ---
A = pd.DataFrame({c_: anom(S[c_]) for c_ in S.columns}, index=TT)

# ------------------------------------------------------------- the events ---
# Peaks in the anomaly of the along-cove stress, split by sign. Two families,
# not one: see the docstring. The threshold is on |anomaly| so the same bar is
# applied to both directions.
thr = A.tau_along.abs().quantile(args.pct / 100)
dist = int(args.minsep * 24)
# An event is kept only if its whole -lead..+lag window lies inside the range
# where every series is finite. Without this an event near an end contributes
# to part of its own composite and drops out of the rest, which puts a step in
# the mean at whatever lag it vanishes -- a feature of the record's edge, not
# of the wind.
fin = A.notna().all(axis=1)
v0, v1 = TT[fin][0], TT[fin][-1]
EV, ndrop = {}, 0
for nm, sgn in [('up-cove', 1.0), ('down-cove', -1.0)]:
    v = sgn * A.tau_along.values
    v = np.where(np.isfinite(v), v, -np.inf)
    pk, _ = find_peaks(v, height=thr, distance=dist)
    t = TT[pk]
    keep = ((t - pd.Timedelta(days=args.lead) >= v0)
            & (t + pd.Timedelta(days=args.lag) <= v1))
    ndrop += int((~keep).sum())
    EV[nm] = t[keep]

print('\n--- wind events: peaks in the along-cove stress anomaly ---')
print('  threshold |tau_along anomaly| > %.4f Pa (the %.0fth percentile), '
      'min %.0f d apart' % (thr, args.pct, args.minsep))
print('  all series finite %s to %s; %d peaks dropped for an incomplete '
      '-%.0f/+%.0f d window'
      % (v0.date(), v1.date(), ndrop, args.lead, args.lag))
ERows = []
for nm in EV:
    print('  %-9s %d events' % (nm, len(EV[nm])))
    for t in EV[nm]:
        w = S.loc[t - pd.Timedelta('36h'):t + pd.Timedelta('36h')]
        ERows.append(dict(
            family=nm, time=t, tau_along=S.tau_along[t],
            w_along=S.w_along[t], w_cross=S.w_cross[t], spd=S.spd[t],
            peak_spd_36h=w.spd.max(),
            from_deg=met_dir(S.w_along[t] * ax_ - S.w_cross[t] * ay_,
                             S.w_along[t] * ay_ + S.w_cross[t] * ax_),
            month=t.month))
E = pd.DataFrame(ERows).sort_values(['family', 'time'])
E.to_csv(out_dir / 'wind_events_alongcove.csv', index=False,
         float_format='%.4f')
SEAS = dict(DJF=[12, 1, 2], MAM=[3, 4, 5], JJA=[6, 7, 8], SON=[9, 10, 11])
for nm in EV:
    g = E[E.family == nm]
    print('  %-9s by season: %s' % (nm, '  '.join(
        '%s %d' % (k, int(g.month.isin(v).sum())) for k, v in SEAS.items())))
print('  along vs cross composition of each family (mean |component|):')
for nm in EV:
    g = E[E.family == nm]
    print('    %-9s from %3.0f +/- %2.0f deg   along %.2f   cross %.2f   '
          'speed %.2f m/s'
          % (nm, g.from_deg.mean(), g.from_deg.std(), g.w_along.abs().mean(),
             g.w_cross.abs().mean(), g.spd.mean()))
print('  a family whose cross component beats its along component is not a')
print('  clean along-channel forcing, whatever its speed.')
print('  the two families are NOT drawn from the same time of year, so their')
print('  composites are compared at different background stratification. The')
print('  30-day anomaly removes the seasonal mean but not the seasonal')
print('  sensitivity, so a larger response in one family is partly a')
print('  statement about when it happens to blow.')

# ------------------------------------------------------------- composites ---
RESP = [('tau_along', 'along-cove wind stress', 'Pa', 1.0),
        ('w_along', 'along-cove wind', 'm s$^{-1}$', 1.0),
        ('dsdx_top', 'ds/dx surface', 'g kg$^{-1}$ km$^{-1}$', 1.0),
        ('dsdx_bot', 'ds/dx bottom', 'g kg$^{-1}$ km$^{-1}$', 1.0),
        ('dsdx_bar', 'ds/dx depth-mean', 'g kg$^{-1}$ km$^{-1}$', 1.0),
        ('dsdx_bar_outer', 'ds/dx outer half', 'g kg$^{-1}$ km$^{-1}$', 1.0),
        ('dsdx_bar_inner', 'ds/dx inner half', 'g kg$^{-1}$ km$^{-1}$', 1.0),
        ('exch_cove', 'exchange  u_top - u_bot', 'm s$^{-1}$', 1.0),
        ('shear_cove', 'shear  d(u)/dz', '10$^{-3}$ s$^{-1}$', 1e3),
        ('dudx_top', 'du/dx surface', '10$^{-6}$ s$^{-1}$', 1e6),
        ('dudx_bar', 'du/dx depth-mean', '10$^{-6}$ s$^{-1}$', 1e6),
        ('dstrat_cove', 'stratification  s_bot - s_top', 'g kg$^{-1}$', 1.0),
        ('do_cove', 'bottom DO, cove mean', 'mg L$^{-1}$', 1.0),
        ('dodx_bot', 'dDO/dx bottom', 'mg L$^{-1}$ km$^{-1}$', 1.0)]
for sn in SECTS:
    RESP += [('%s_u_top' % sn, 'u surface, %s' % SLAB[sn], 'm s$^{-1}$', 1.0),
             ('%s_u_bot' % sn, 'u bottom, %s' % SLAB[sn], 'm s$^{-1}$', 1.0),
             ('%s_s_top' % sn, 's surface, %s' % SLAB[sn], 'g kg$^{-1}$', 1.0),
             ('%s_o_bot' % sn, 'bottom DO, %s' % SLAB[sn], 'mg L$^{-1}$', 1.0)]

lags = np.arange(-int(args.lead * 24), int(args.lag * 24) + 1)
lagd = lags / 24.0
BASE = (lagd >= -args.lead) & (lagd <= -args.lead + 1)   # the lead-in day
COMP = {}
crows = []
for nm in EV:
    D = {}
    for vn, lab, un, sc in RESP:
        v = A[vn].values
        M = np.full((len(EV[nm]), len(lags)), np.nan)
        for i, t in enumerate(EV[nm]):
            k = TT.get_loc(t)
            j = k + lags
            ok = (j >= 0) & (j < len(TT))
            M[i, ok] = v[j[ok]]
        M = M - np.nanmean(M[:, BASE], axis=1, keepdims=True)
        n = np.sum(np.isfinite(M), axis=0)
        mu = np.nanmean(M, axis=0)
        se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
        D[vn] = dict(mean=mu * sc, se=se * sc, n=n)
        k = np.nanargmax(np.abs(mu))
        crows.append(dict(family=nm, var=vn, label=lab, unit=un,
                          n_events=len(EV[nm]),
                          peak_dev=mu[k] * sc, peak_lag_d=lagd[k],
                          peak_se=se[k] * sc,
                          sig=abs(mu[k]) > 2 * se[k],
                          dev_at_0=mu[lagd == 0][0] * sc,
                          dev_at_1d=mu[np.argmin(abs(lagd - 1))] * sc,
                          dev_at_2d=mu[np.argmin(abs(lagd - 2))] * sc))
    COMP[nm] = D
CO = pd.DataFrame(crows)
CO.to_csv(out_dir / 'event_composites.csv', index=False, float_format='%.5f')

print('\n--- composite response, deviation from the lead-in day ---')
print('  both the wind and the response are Godin-lowpassed, a 71-hour '
      'filter,\n  so lags smaller than about a day are not resolvable and a '
      'peak at\n  -0.1 d should be read as simultaneous, not as the ocean '
      'leading.')
print('%-10s %-28s %10s %8s %6s' % ('family', 'response', 'peak dev',
                                    'lag (d)', '2SE?'))
for nm in EV:
    for vn, lab, un, sc in RESP[:14]:
        r = CO[(CO.family == nm) & (CO['var'] == vn)].iloc[0]
        print('%-10s %-28s %+10.4f %8.2f %6s'
              % (nm, lab, r.peak_dev, r.peak_lag_d, 'yes' if r.sig else '--'))

# ------------------------------------------------------ lagged correlation ---
# The continuous version of the same question, so the answer does not rest on
# which hours got labelled "events". Positive lag = the response follows the
# wind.
#
# Run on DAILY means, not on the hourly series. The n_eff correction needs a
# lag-1 autocorrelation, and at hourly sampling of an already-lowpassed series
# r1 is ~0.999, where n_eff = n(1-r1a.r1b)/(1+r1a.r1b) is the small difference
# of two numbers near 1 and is wildly unstable -- it returned n_eff ~ 13-30
# out of 17544 hours, i.e. a decorrelation time of over a month, which the
# series plainly does not have. Daily sampling puts r1 in a range where the
# estimator behaves, and it is the resolution the question is asked at anyway.
AD = A.resample('1D').mean()
LAGS = np.arange(-3, 8)
lrows = []
tv = AD.tau_along.values
for vn, lab, un, sc in RESP:
    if vn in ('tau_along', 'w_along'):
        continue
    v = AD[vn].values
    for L in LAGS:
        a_ = tv[:len(tv) - L] if L >= 0 else tv[-L:]
        b_ = v[L:] if L >= 0 else v[:len(v) + L]
        r, pp, ne = neff_r(a_, b_)
        lrows.append(dict(var=vn, label=lab, lag_d=float(L), r=r, p=pp,
                          n_eff=ne))
LC = pd.DataFrame(lrows)
LC.to_csv(out_dir / 'lagged_correlations.csv', index=False,
          float_format='%.4f')

print('\n--- lagged correlation with the along-cove stress anomaly ---')
print('%-28s %7s %8s %10s %8s' % ('response', 'max |r|', 'lag (d)', 'p',
                                  'n_eff'))
best = []
for vn, lab, un, sc in RESP:
    if vn in ('tau_along', 'w_along'):
        continue
    g = LC[LC['var'] == vn].dropna(subset=['r'])
    if not len(g):
        continue
    b = g.iloc[g.r.abs().values.argmax()]
    best.append(b)
    print('%-28s %+7.2f %8.2f %10.2g %8d'
          % (lab, b.r, b.lag_d, b.p, b.n_eff))
print('  computed on %d daily means; p uses an autocorrelation-corrected'
      % len(AD))
print('  n_eff, which here runs about %d-%d of those days'
      % (LC.n_eff.replace(0, np.nan).min(), LC.n_eff.max()))

# =========================================================== figure 1: series
sub = S.dropna(subset=['tau_along'])
fig, axs = plt.subplots(7, 1, figsize=(15, 18), sharex=True,
                        layout='constrained')

ax = axs[0]
ax.fill_between(sub.index, 0, sub.w_along, where=sub.w_along >= 0,
                color=CB['blue'], alpha=0.6, lw=0, label='into the cove')
ax.fill_between(sub.index, 0, sub.w_along, where=sub.w_along < 0,
                color=CB['orange'], alpha=0.6, lw=0, label='out of the cove')
ax.set_ylabel('along-cove wind\n(m s$^{-1}$), subtidal')
ax.legend(fontsize=8, ncol=2, loc='upper left')

ax = axs[1]
for lay, c_, lab in [('top', CB['blue'], 'surface'),
                     ('bot', CB['orange'], 'bottom'),
                     ('bar', 'k', 'depth-mean')]:
    ax.plot(sub.index, sub['dsdx_' + lay], lw=1.4, color=c_, label=lab)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('ds/dx\n(g kg$^{-1}$ km$^{-1}$)')
ax.legend(fontsize=8, ncol=3, loc='upper left')

ax = axs[2]
for a_, b_, c_, lab in [('outer', None, CB['purple'], 'mouth -> mid'),
                        ('inner', None, CB['green'], 'mid -> head')]:
    ax.plot(sub.index, sub['dsdx_bar_' + a_], lw=1.4, color=c_, label=lab)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('ds/dx by half\n(g kg$^{-1}$ km$^{-1}$)')
ax.legend(fontsize=8, ncol=2, loc='upper left')

# Surface and bottom get their OWN panels and their own y-scales. Sharing an
# axis was hiding the bottom flow: its subtidal range is about a fifth of the
# surface's, so it collapsed onto the zero line and the deep inflow -- the
# half of the exchange that actually ventilates the cove -- was unreadable.
# The rms of each is printed in the panel label so the scales stay comparable
# by eye rather than by accident.
ax = axs[3]
for sn in SECTS:
    ax.plot(sub.index, sub['%s_u_top' % sn], lw=1.2, color=SC[sn],
            label='%s (rms %.3f)' % (SLAB[sn],
                                     np.sqrt(np.nanmean(
                                         sub['%s_u_top' % sn] ** 2))))
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('u SURFACE, + into cove\n(m s$^{-1}$)')
ax.legend(fontsize=7, ncol=3, loc='upper left')

ax = axs[4]
for sn in SECTS:
    ax.plot(sub.index, sub['%s_u_bot' % sn], lw=1.2, color=SC[sn],
            label='%s (rms %.3f)' % (SLAB[sn],
                                     np.sqrt(np.nanmean(
                                         sub['%s_u_bot' % sn] ** 2))))
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('u BOTTOM, + into cove\n(m s$^{-1}$)')
ax.legend(fontsize=7, ncol=3, loc='upper left')

ax = axs[5]
for sn in SECTS:
    ax.plot(sub.index, sub['%s_o_bot' % sn], lw=1.3, color=SC[sn],
            label=SLAB[sn])
ax.axhline(2.0, color=CB['red'], lw=1.0, ls='--', alpha=0.8)
ax.axhline(5.0, color=CB['yellow'], lw=1.0, ls='--', alpha=0.8)
ax.text(sub.index[5], 2.05, 'hypoxic 2 mg/L', fontsize=7, color=CB['red'])
ax.text(sub.index[5], 5.05, '5 mg/L', fontsize=7, color=CB['yellow'])
ax.set_ylabel('bottom DO\n(mg L$^{-1}$)')
ax.legend(fontsize=7, ncol=3, loc='upper left')

ax = axs[6]
ax.plot(sub.index, 1e3 * sub.shear_cove, lw=1.4, color='k', label='d(u)/dz')
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('shear\n(10$^{-3}$ s$^{-1}$)')
ax2 = ax.twinx()
ax2.plot(sub.index, sub.dstrat_cove, lw=1.2, color=CB['pink'],
         label='s_bot - s_top')
ax2.set_ylabel('stratification (g kg$^{-1}$)', color=CB['pink'])
ax2.tick_params(axis='y', colors=CB['pink'])
ax.legend(fontsize=8, loc='upper left')

for ax in axs:
    ax.grid(**GRID)
    for nm, c_ in [('up-cove', CB['blue']), ('down-cove', CB['orange'])]:
        for t in EV[nm]:
            ax.axvline(t, color=c_, lw=0.8, alpha=0.35, zorder=0)
axs[-1].xaxis.set_major_locator(MonthLocator())
axs[-1].xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
axs[-1].set_xlim(sub.index[0], sub.index[-1])
fig.suptitle('%s -- Penn Cove along-channel structure, Godin-lowpassed.  '
             'x is positive mouth -> head, so ds/dx < 0 means a fresher head '
             'and > 0 a saltier one.\nnote the SURFACE gradient sits above '
             'zero and the bottom below it: the cove destratifies toward its '
             'closed end rather than freshening.\n'
             'vertical lines: along-cove wind events '
             '(blue = into the cove, orange = out)' % args.gtx, fontsize=11)
fn = out_dir / 'fig1_alongchannel_series.png'
fig.savefig(fn, dpi=170, bbox_inches='tight')
plt.close(fig)
print('\nsaved %s' % fn)

# ====================================================== figure 2: composites
SHOW = ['tau_along', 'dsdx_bar', 'dsdx_top', 'dsdx_bot', 'dsdx_bar_outer',
        'dsdx_bar_inner',
        'pc_lp_u_top', 'pc_lp_u_bot', 'exch_cove', 'shear_cove',
        'do_cove', 'dodx_bot']
LOOK = {vn: (lab, un, sc) for vn, lab, un, sc in RESP}
fig, axs = plt.subplots(6, 2, figsize=(13, 17), sharex=True,
                        layout='constrained')
FC = {'down-cove': CB['orange'], 'up-cove': CB['blue']}
for i, vn in enumerate(SHOW):
    ax = axs[i % 6][i // 6]
    lab, un, sc = LOOK[vn]
    for nm in ['down-cove', 'up-cove']:
        d = COMP[nm][vn]
        ax.plot(lagd, d['mean'], lw=2, color=FC[nm],
                label='%s (n=%d)' % (nm, len(EV[nm])))
        ax.fill_between(lagd, d['mean'] - 2 * d['se'], d['mean'] + 2 * d['se'],
                        color=FC[nm], alpha=0.16, lw=0)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.grid(**GRID)
    ax.set_title('%s  (%s)' % (lab, un), fontsize=10)
    y0_, y1_ = ax.get_ylim()
    ax.fill_between([lagd[BASE][0], lagd[BASE][-1]], y0_, y1_, color='0.6',
                    alpha=0.12, lw=0, zorder=0)
    ax.set_ylim(y0_, y1_)
    if i == 0:
        ax.legend(fontsize=8)
for ax in axs[-1]:
    ax.set_xlabel('days from the peak of the wind event')
fig.suptitle('Penn Cove: composite response to along-cove wind events\n'
             '30-day rolling anomalies of the subtidal series, referenced to '
             'the shaded lead-in day; band = +/- 2 SE across events',
             fontsize=12)
fn = out_dir / 'fig2_event_composites.png'
fig.savefig(fn, dpi=170, bbox_inches='tight')
plt.close(fig)
print('saved %s' % fn)

# ============================================== figure 3: lagged correlation
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')
ax = axs[0]
for vn, c_ in [('dsdx_bar', 'k'), ('dsdx_top', CB['blue']),
               ('dsdx_bot', CB['orange']), ('dsdx_bar_outer', CB['purple']),
               ('dsdx_bar_inner', CB['green'])]:
    g = LC[LC['var'] == vn]
    ax.plot(g.lag_d, g.r, '-', lw=2, color=c_, marker='o', ms=4, mfc='none',
            label=LOOK[vn][0])
    m = (g.p < 0.05).values
    ax.plot(g.lag_d[m], g.r[m], 'o', ms=7, color=c_)
ax.set_title('salinity gradient vs along-cove stress', fontsize=11)
ax = axs[1]
# exch_cove is omitted here on purpose: it is shear_cove times a constant
# depth, so plotting both draws one line exactly under the other. It is kept
# in the composites and in the CSVs, where the m/s units are easier to read.
for vn, c_ in [('shear_cove', CB['purple']),
               ('dstrat_cove', CB['pink']), ('dudx_top', CB['green']),
               ('dudx_bar', CB['yellow']),
               ('pc_lp_u_top', CB['blue']), ('pc_lp_u_bot', CB['red']),
               ('do_cove', CB['grey'])]:
    g = LC[LC['var'] == vn]
    ax.plot(g.lag_d, g.r, '-', lw=2, color=c_, marker='o', ms=4, mfc='none',
            label=LOOK[vn][0])
    m = (g.p < 0.05).values
    ax.plot(g.lag_d[m], g.r[m], 'o', ms=7, color=c_)
ax.set_title('velocity, stratification and bottom DO vs along-cove stress',
             fontsize=11)
for ax in axs:
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.grid(**GRID)
    ax.set_xlabel('lag (days); positive = the response follows the wind')
    ax.set_ylabel('correlation with along-cove stress anomaly')
    ax.legend(fontsize=8)
fig.suptitle('Lagged correlation, 30-day anomalies of the subtidal series.  '
             'daily means; large filled markers: p < 0.05 with an '
             'autocorrelation-corrected n_eff', fontsize=12)
fn = out_dir / 'fig3_lagged_correlation.png'
fig.savefig(fn, dpi=170, bbox_inches='tight')
plt.close(fig)
print('saved %s' % fn)

# ------------------------------------------------------------------ tables ---
MEANS = pd.DataFrame(index=SECTS)
for sn in SECTS:
    MEANS.loc[sn, 'x_km'] = X[sn]
    for c_ in ['s_top', 's_bot', 'u_top', 'u_bot', 'dstrat', 'shear']:
        MEANS.loc[sn, c_] = S['%s_%s' % (sn, c_)].mean()
MEANS.to_csv(out_dir / 'section_means.csv', float_format='%.5f')
S.to_csv(out_dir / 'alongchannel_subtidal.csv', float_format='%.5f')

print('\n--- time-mean state of the transect ---')
print(MEANS.round(4).to_string())
print('\nmean ds/dx  surface %+.4f, bottom %+.4f, depth-mean %+.4f '
      'g/kg per km' % (S.dsdx_top.mean(), S.dsdx_bot.mean(),
                       S.dsdx_bar.mean()))
print('mean exchange u_top - u_bot = %+.4f m/s (negative = estuarine: out at '
      'the surface, in at depth)' % S.exch_cove.mean())
for fn_ in ['wind_events_alongcove.csv', 'event_composites.csv',
            'lagged_correlations.csv', 'section_means.csv',
            'alongchannel_subtidal.csv']:
    print('saved %s' % (out_dir / fn_))
