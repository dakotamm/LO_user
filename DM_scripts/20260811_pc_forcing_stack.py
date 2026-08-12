"""
Penn Cove bottom DO against its forcing, 2025, wb1_t0_xn11abbur00.

Four panels on one shared time axis, in the order the causal argument runs --
what happened, then the two things that could have caused it, then the thing
that drives those:

  a  BOTTOM DO at cp_mid (mid-Coupeville line, inner cove, h = 20.6 m) and at
     M5 (Saratoga Passage, outside, h = 37.4 m). The pair separates a drawdown
     inside the cove from a low-oxygen source arriving from the passage: only
     cp_mid falling while M5 holds is a cove signal.

  b  DENSITY STRATIFICATION at pc_lp (mouth) and pc_cp (Coupeville):
     bottom-minus-top sigma0, so positive is stably stratified. The thing that
     suppresses vertical exchange, and the first place to look for the drawdown
     in panel a. Salinity difference alone would miss the thermal term, which
     opposes the haline one once the surface warms in summer.

  c  QPRISM at the same two sections. The tidal exchange that would ventilate
     the cove. High stratification with low Qprism is the combination that
     should let bottom water go hypoxic; read the two panels together.

  d  ALONG-COVE WIND VELOCITY, positive blowing INTO the cove (mouth -> head).
     The subtidal forcing that drives the exchange in b and can break down the
     stratification in c.

SIGN AND AXIS. The along-cove axis is the line joining the pc_lp centroid to
the pc_cp centroid, taken from structure_*.nc -- the same construction as
20260807_pc_alongchannel_wind.py, so "into the cove" means the same thing in
both. w_along is the region-mean wind vector projected onto that axis.

WHY VELOCITY AND NOT STRESS. Stress would be the more physical forcing, since
momentum input scales with speed squared, but the stored stress cannot be
projected cleanly: tau_pc in the wind pickle is the mean over cells of the
per-cell stress MAGNITUDE, so multiplying it by the direction of the vector
mean mixes two different averages and biases the result high wherever wind
direction varies across the cove. Velocity is a linear projection of the vector
mean and has no such problem. It also avoids leaning on the Large & Pond Cd in
20260806_wind_reduce.py, which is a proxy for the COARE stress ROMS actually
used. A properly projected stress means re-running the reduction with a
per-cell projection; until then this panel is velocity.

The cross-cove component is not plotted here, which sidesteps the known
mislabelling of cross-cove sign in reduce_wind_cove.py / wind_characterize.py.

FILTERING. The four sources arrive at different cadences and are put on one
daily axis:
  moorings   extracted with -lt lowpass, already Godin, one sample/day at 12:00
  qprism     from bulk_calc_avg.py, already Godin then subsampled daily
  sigma0     built from hourly s/t in strat_*.nc -- Godin filtered here
  wind       hourly -- Godin filtered here
Filtering is done on the FULL 2024-2025 record and only then windowed to 2025,
so the January edge is not blanked by the filter's own half-width. Nothing in
this figure is raw: every panel is subtidal, and a brief tidal excursion (a dip
below a DO threshold, a single strong sea breeze) cannot appear in any of them.

run 20260811_pc_forcing_stack.py
run 20260811_pc_forcing_stack.py -year 2024
run 20260811_pc_forcing_stack.py -gtx wb1_t1_xn11abbur00
"""
import argparse
import sys
import warnings

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from cmcrameri import cm as cmc

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-job', default='pc4', type=str)
# the tef2/wind products span the full run; the moor files were extracted over
# the days that actually have a lowpassed.nc, hence the different range
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-m0', '--mds0', default='2024.01.02', type=str)
p.add_argument('-m1', '--mds1', default='2025.12.30', type=str)
p.add_argument('-year', default='2025', type=str)
p.add_argument('--tz', default='America/Los_Angeles')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
moor_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'moor' / args.job
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
           / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_forcing_stack'
Lfun.make_dir(out_dir)

O2_MMOL_TO_MGL = 32.0 / 1000.0
CB = dict(blue='#0072B2', red='#CC0000', grey='#7f7f7f')
# Station colors are the map's: cmcrameri lajolla at 0.05 / 0.50 / 0.97, so
# inner / mouth / outside mean the same color here as in
# 20260811_pc4_points_map.py. The outside color is a pale cream that is nearly
# invisible as a line, so that one series gets a thin dark stroke -- see PE below.
C_CP = mcolors.to_hex(cmc.lajolla(0.05))    # inner cove / pc_cp
C_LP = mcolors.to_hex(cmc.lajolla(0.50))    # mouth / pc_lp
C_M5 = mcolors.to_hex(cmc.lajolla(0.97))    # Saratoga Passage, outside
PE = [pe.Stroke(linewidth=3.0, foreground='#7a6a2a'), pe.Normal()]
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
MOUTH, HEAD = 'pc_lp', 'pc_cp'


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def need(fn):
    if not fn.is_file():
        print('*** missing: %s' % fn)
        sys.exit(1)
    return fn


# ---------------------------------------------------------------------------
# a. bottom DO from the moorings (already daily Godin)
# ---------------------------------------------------------------------------
DO = {}
for sn in ['cp_mid', 'M5']:
    fn = need(moor_dir / ('%s_%s_%s.nc' % (sn, args.mds0, args.mds1)))
    ds = xr.open_dataset(fn)
    DO[sn] = pd.Series(ds.oxygen.values[:, 0] * O2_MMOL_TO_MGL,  # 0 = bottom
                       index=pd.to_datetime(ds.ocean_time.values), name=sn)
    print('%-7s h = %5.1f m, %d daily samples' % (sn, float(ds.h), len(DO[sn])))
    ds.close()

# the daily axis everything else is put onto
TT = DO['cp_mid'].index

# ---------------------------------------------------------------------------
# b. Qprism (already Godin + daily-subsampled by bulk_calc_avg.py)
# ---------------------------------------------------------------------------
QP = {}
for sn in [MOUTH, HEAD]:
    fn = need(tef2 / ('bulk_avg_%s_%s' % (args.ds0, args.ds1)) / (sn + '.nc'))
    ds = xr.open_dataset(fn)
    QP[sn] = pd.Series(ds.qprism.values,
                       index=pd.to_datetime(ds.time.values), name=sn)
    ds.close()

# ---------------------------------------------------------------------------
# section geometry: centroid for the wind axis, width-weighted mean depth for
# the bottom pressure used in the density calculation below
# ---------------------------------------------------------------------------
dstr = xr.open_dataset(need(tef2 / ('structure_%s_%s_%s.nc'
                                    % (args.ds0, args.ds1, args.gctag))))
CEN = {sn: (float(np.mean(dstr['%s_lon' % sn].values)),
            float(np.mean(dstr['%s_lat' % sn].values))) for sn in [MOUTH, HEAD]}
HBAR = {}
for sn in [MOUTH, HEAD]:
    hh, dd = dstr['%s_h' % sn].values, dstr['%s_dd' % sn].values
    HBAR[sn] = float(np.sum(hh * dd) / np.sum(dd))   # width-weighted, as strat is
dstr.close()

# ---------------------------------------------------------------------------
# c/d. stratification and speed, both from the hourly section extractions
#
# extractions_avg/*.nc is hourly despite the name (17544 records at :30) and
# holds the full (time, z, p) fields, so both quantities are built per FACE and
# then width-weighted across the section -- the same choice 20260811_pc_pycnocline.py
# makes, and for the same reason: faces of a section have different depths, so a
# sigma level is not a fixed z across the section and averaging profiles first
# would smear the structure before it is measured.
#
# N2  Depth-averaged buoyancy frequency squared, the column's mean stratification:
#         N2_avg = (g/rho0) * (sigma_bot - sigma_top) / H        [s-2]
#     which is the exact depth-average of the local N2 = -(g/rho0) dsigma/dz,
#     since integrating dsigma/dz over the column telescopes to the endpoints.
#     NOTE what that means: a depth-averaged N2 carries no information about
#     interior structure beyond what the endpoints and H already say. What it
#     adds over a bare top-minus-bottom density difference is the 1/H
#     normalisation, which matters here because pc_lp is deeper than pc_cp, so
#     the same density difference is weaker stratification at the mouth. If you
#     want a measure that does respond to interior structure, the one to use is
#     N2_max (pycnocline sharpness) from 20260811_pc_pycnocline.py, not this.
#     sigma0 comes from the standard gsw chain: p_from_z -> SA_from_SP ->
#     CT_from_pt -> sigma0. Positive N2 is stable.
#
# SPD Area-weighted mean of |u| over the section. Cell velocity is q/(DZ*dd), so
#         sum(|u| * DZ*dd) / sum(DZ*dd)  =  sum(|q|) / sum(DZ*dd)
#     and the velocity never has to be formed. Taking |.| PER CELL before summing
#     keeps inflow and outflow layers from cancelling, so this is the strength of
#     the exchange rather than the net through-section flow.
#
# Both are Godin filtered afterwards, so each is the subtidal envelope of an
# hourly series -- the speed does not go to zero at slack water the way an
# instantaneous section speed would.
# ---------------------------------------------------------------------------
G, RHO0 = 9.81, 1025.0
DS, SPD = {}, {}
for sn in [MOUTH, HEAD]:
    fn = need(tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
              / (sn + '.nc'))
    ds = xr.open_dataset(fn)
    tt = pd.to_datetime(ds.time.values)
    h, dd = ds.h.values, ds.dd.values                  # (p,)
    DZ = ds.DZ.values                                  # (time, z, p)
    zeta = ds.zeta.values
    slon, slat = CEN[sn]

    z_w = -h[None, None, :] + np.cumsum(DZ, axis=1)    # top of each layer
    z_rho = z_w - 0.5 * DZ
    SA = gsw.SA_from_SP(ds.salt.values.astype(float),
                        gsw.p_from_z(z_rho, slat), slon, slat)
    CT = gsw.CT_from_pt(SA, ds.temp.values.astype(float))
    sig = gsw.sigma0(SA, CT)                           # z index 0 = bed

    H = h[None, :] + zeta                              # column depth (time, p)
    n2 = (G / RHO0) * (sig[:, 0, :] - sig[:, -1, :]) / H
    DS[sn] = pd.Series(godin((n2 * dd[None, :]).sum(axis=1) / dd.sum()),
                       index=tt, name=sn)

    area = ds.DZ * ds.dd
    ubar = (np.abs(ds.q).sum(dim=['z', 'p']) / area.sum(dim=['z', 'p'])).values
    SPD[sn] = pd.Series(godin(ubar), index=tt, name=sn)
    print('%s: width-weighted mean depth %.1f m' % (sn, HBAR[sn]))
    ds.close()

# ---------------------------------------------------------------------------
# Skagit discharge, from the river reduction of the run's own forcing
# (trapsN00), so this is the freshwater the model actually saw and not a gauge
# record standing in for it. The Skagit is one of the three USGS-gauged sources
# in this forcing -- most of the other 29 are day-of-year climatology that
# repeats exactly year to year, so year-specific structure would be meaningless
# from them. Already daily; no filtering needed.
# ---------------------------------------------------------------------------
riv_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_river_hydrographs'
          / ('river_daily_trapsN00_%s_%s.p' % (args.ds0, args.ds1)))
RIV = pd.read_pickle(need(riv_fn))['Q']['skagit']

# ---------------------------------------------------------------------------
# along-cove wind velocity
# ---------------------------------------------------------------------------
COS = np.cos(np.deg2rad(np.mean([c[1] for c in CEN.values()])))
x0, y0 = CEN[MOUTH][0] * COS * 111.32, CEN[MOUTH][1] * 111.32
xN, yN = CEN[HEAD][0] * COS * 111.32, CEN[HEAD][1] * 111.32
axl = np.hypot(xN - x0, yN - y0)
ax_, ay_ = (xN - x0) / axl, (yN - y0) / axl        # unit vector, mouth -> head
print('\nalong-cove axis (%.4f, %.4f), %.0f deg true, mouth -> head, %.2f km'
      % (ax_, ay_, np.rad2deg(np.arctan2(ax_, ay_)) % 360, axl))

C = pd.read_pickle(need(wind_fn))
W = C['W']
# Along-cove wind VELOCITY, not stress. This is a linear projection of the
# region-mean wind vector onto the cove axis, so it inherits no averaging-order
# problem. The stress version had one: tau_pc is the mean of the per-cell
# stress MAGNITUDE, so scaling it by the direction of the vector mean mixes two
# different averages and biases the result high wherever wind direction varies
# across the cove. Velocity also avoids leaning on a Large & Pond Cd that is a
# proxy for what ROMS actually computed with COARE.
WA = pd.Series(godin(W.u_pc.values * ax_ + W.v_pc.values * ay_),
               index=W.index, name='w_along')       # + = blowing INTO the cove

# ---------------------------------------------------------------------------
# common daily axis, then window to the year
# ---------------------------------------------------------------------------
def on_daily(s):
    """Sample a series onto TT; the hourly ones are on the hour, TT on the
    half hour, so interpolate rather than nearest-neighbour."""
    return s.reindex(s.index.union(TT)).interpolate('time').reindex(TT)


S = pd.DataFrame(index=TT)
S['do_cp'] = DO['cp_mid']
S['do_M5'] = DO['M5']
S['qp_lp'] = on_daily(QP[MOUTH])
S['qp_cp'] = on_daily(QP[HEAD])
S['n2_lp'] = on_daily(DS[MOUTH])
S['n2_cp'] = on_daily(DS[HEAD])
S['sp_lp'] = on_daily(SPD[MOUTH])
S['sp_cp'] = on_daily(SPD[HEAD])
S['wa'] = on_daily(WA)
S['riv'] = on_daily(RIV)

if args.year.lower() != 'all':
    yr = int(args.year)
    S = S[S.index.year == yr]
    span_lbl = str(yr)
    if len(S) == 0:
        print('*** no samples in %d' % yr)
        sys.exit(1)
else:
    span_lbl = '%s to %s' % (S.index[0].date(), S.index[-1].date())
S.to_csv(out_dir / ('pc_forcing_stack_%s_%s.csv' % (args.gtagex, args.year)))

print('\n--- %s, %d days ---' % (span_lbl, len(S)))
for c_, lbl, u in [('do_cp', 'bottom DO cp_mid', 'mg/L'),
                   ('do_M5', 'bottom DO M5', 'mg/L'),
                   ('qp_lp', 'Qprism pc_lp', 'm3/s'),
                   ('qp_cp', 'Qprism pc_cp', 'm3/s'),
                   ('n2_lp', 'N2 pc_lp', 's-2'),
                   ('n2_cp', 'N2 pc_cp', 's-2'),
                   ('sp_lp', 'mean |u| pc_lp', 'm/s'),
                   ('sp_cp', 'mean |u| pc_cp', 'm/s'),
                   ('wa', 'w_along', 'm/s'),
                   ('riv', 'Skagit Q', 'm3/s')]:
    print('  %-18s min %8.4f  mean %8.4f  max %8.4f  %s'
          % (lbl, S[c_].min(), S[c_].mean(), S[c_].max(), u))
print('  wind blows into the cove %.0f%% of days' % (100 * (S.wa > 0).mean()))
print('\ncorrelation with bottom DO at cp_mid (daily, no lag):')
for c_ in ['do_M5', 'n2_lp', 'n2_cp', 'qp_lp', 'qp_cp', 'sp_lp', 'sp_cp',
           'wa', 'riv']:
    print('  %-6s r = %+.2f' % (c_, S.do_cp.corr(S[c_])))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
FS = 15                                   # axis label / tick font size
t = S.index


def p_do(ax):
    """Bottom DO at the inner-cove and outside moorings, with hypoxia lines."""
    for thr, col in [(2.0, '#7f2704'), (3.0, '#d94801'), (5.0, '#fdae6b')]:
        ax.axhline(thr, color=col, lw=0.8, ls='--', zorder=1)
    # Explicit light tint rather than a low alpha: with transparent=True there is
    # no white page behind the axes for alpha to blend into, so an alpha-shaded
    # band saves as a solid slab of color.
    ax.axhspan(0, 2.0, color='#f2ddd3', lw=0, zorder=0)
    ax.plot(t, S.do_cp, color=C_CP, lw=1.8, zorder=3)
    ax.plot(t, S.do_M5, color=C_M5, lw=1.8, zorder=3, path_effects=PE)
    ax.set_ylabel('bottom DO\n[mg L$^{-1}$]', fontsize=FS)
    ax.set_ylim(0, max(10.5, float(S[['do_cp', 'do_M5']].max().max()) + 0.5))


def p_riv(ax):
    ax.fill_between(t, 0, S.riv, color='#bcd5e8', lw=0)
    ax.plot(t, S.riv, color='#2b6ca3', lw=1.6)
    ax.set_ylabel('Skagit $Q_r$\n[m$^3$ s$^{-1}$]', fontsize=FS)
    ax.set_ylim(0, float(S.riv.max()) * 1.05)


def p_wind(ax):
    # this panel keeps its legend: the sign convention is not recoverable from
    # the colors the way the station identities are recoverable from the map
    ax.fill_between(t, 0, S.wa, where=S.wa >= 0, color='#e08b8b',
                    lw=0, interpolate=True, label='into the cove')
    ax.fill_between(t, 0, S.wa, where=S.wa < 0, color='#8fb8db',
                    lw=0, interpolate=True, label='out of the cove')
    ax.plot(t, S.wa, color='k', lw=0.7)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('along-cove wind\n[m s$^{-1}$]', fontsize=FS)
    ax.legend(loc='lower left', frameon=False, fontsize=FS - 2, ncol=2)


def p_n2(ax):
    ax.plot(t, S.n2_lp, color=C_LP, lw=1.8)
    ax.plot(t, S.n2_cp, color=C_CP, lw=1.8)
    ax.set_ylabel('$N^2$ depth-avg\n[s$^{-2}$]', fontsize=FS)


def p_spd(ax):
    ax.plot(t, S.sp_lp, color=C_LP, lw=1.8)
    ax.plot(t, S.sp_cp, color=C_CP, lw=1.8)
    ax.set_ylabel('mean $|u|$\n[m s$^{-1}$]', fontsize=FS)


def p_qp(ax):
    ax.plot(t, S.qp_lp, color=C_LP, lw=1.8)
    ax.plot(t, S.qp_cp, color=C_CP, lw=1.8)
    ax.set_ylabel('$Q_{prism}$\n[m$^3$ s$^{-1}$]', fontsize=FS)


def build(panels, h_per_panel, tag):
    fig, axes = plt.subplots(len(panels), 1, sharex=True,
                             figsize=(11.5, h_per_panel * len(panels)))
    axes = np.atleast_1d(axes)
    for fn_p, ax in zip(panels, axes):
        fn_p(ax)
        ax.grid(**GRID)
        ax.tick_params(labelsize=FS - 2)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].set_xlim(t[0], t[-1])
    fig.tight_layout()
    fn_out = out_dir / ('pc_%s_%s_%s.png' % (tag, args.gtagex, args.year))
    fig.savefig(fn_out, dpi=500, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print('wrote ' + str(fn_out))


print()
# forcing stack: what happened, the two external drivers, then the cove's own
# response in stratification and exchange speed
build([p_do, p_riv, p_wind, p_n2, p_spd], 1.96, 'forcing_stack')
# tidal stack: the same DO series against the tidal exchange and the wind
build([p_do, p_qp, p_wind], 2.6, 'tidal_stack')
