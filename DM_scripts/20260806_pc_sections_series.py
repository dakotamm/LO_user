"""
Six points across Penn Cove -- north and south at each of the three
cross-cove sections -- on one time axis, under the forcing that drives them.

This is the multi-section version of 20260806_pc_mouth_series.py. That script
showed two points at the mouth and put surface and bottom in separate panels.
Here surface and bottom share an axis, which is the only way to see the
stratification and the exchange as one thing, and the cove is followed from
mouth to head:

    pc_lp   lon -122.653   the MOUTH        12 faces, h to 26.8 m
    pc_lj   lon -122.672   mid-cove          8 faces, h to 25.4 m
    pc_cp   lon -122.694   inner / head      8 faces, h to 21.5 m

WHICH SIX POINTS, AND WHY
The mouth pair was fixed earlier as pc_lp p=2 / p=11 on the grounds that they
are depth-matched (19.8 and 20.0 m) and sit on opposite sides of the lateral
exchange. Rather than eyeball the other two sections, that reasoning is
written down as a rule and applied to all three:

    take one face from each side of the sign change in mean per-face
    transport (qbar), maximising |qbar_north| + |qbar_south| subject to
    |h_north - h_south| <= 0.5 m

Depth-matching is the binding constraint -- it is what makes a north/south
difference lateral rather than a depth artifact -- and the transport term
picks the pair that actually carries the exchange. Run on pc_lp the rule
returns p=2 and p=11, i.e. it reproduces the pair chosen by hand, which is
why it is trusted on pc_lj and pc_cp. The selection is printed and written to
a CSV; -faces overrides it.

WHAT IS PLOTTED
  forcing, full width
    daily tidal range with the spring/neap label and the moon events
    sea surface height at the reference section
    Skagit River discharge -- USGS-gauged, so a feature here is a real 2025
      event and not the day-of-year climatology that 29 of the 32 sources use
    wind speed at Penn Cove with direction arrows every 6 h
    the along-cove wind component, positive INTO the cove (mouth -> head),
      which is the part that can drive a surface flow up or down the axis
  per section, one column each, mouth on the left
    salinity   surface (solid) and bottom (dashed), both points
    velocity   same, positive INTO the cove
    bottom DO  north and south

DO is bottom-only on purpose: surface DO is a photosynthesis signal and says
little about the water being exchanged, while bottom DO is the quantity that
matters for the cove.

SIGNS ARE VERIFIED PER SECTION, not assumed -- qnet is correlated against
d(ssh)/dt for each section separately and the flood sign is reported. Note u
is the section-NORMAL component only, so |u| is a lower bound on true speed.

Runs on the mac from the local extractions_avg.

run 20260806_pc_sections_series.py
run 20260806_pc_sections_series.py --t0 2025.10.01 --t1 2025.10.31
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('--ds0', default='2024.01.01')
p.add_argument('--ds1', default='2025.12.31')
p.add_argument('--t0', default='2025.09.01', help='first day to plot')
p.add_argument('--t1', default='2025.10.31', help='last day to plot')
p.add_argument('--sects', default='pc_lp,pc_cp', help='mouth first')
p.add_argument('--ref', default='pc_lp', help='section defining tidal phase')
p.add_argument('--faces', default='', help='override, e.g. pc_lp:2/11,pc_lj:2/5')
p.add_argument('--dhmax', type=float, default=0.5,
               help='depth mismatch [m] to try first within a north/south '
                    'pair; relaxed in 0.25 m steps if nothing else qualifies')
p.add_argument('--layout', default='full', choices=['full', 'alt'],
               help="'full' = every variable at every section; 'alt' = the "
                    "forcing (minus wind speed), bottom DO at the inner "
                    "section, and outer-section velocity split into its own "
                    "surface and bottom panels")
p.add_argument('--center', default='pair', choices=['pair', 'section'],
               help="'pair' = face nearest the midpoint of the north/south "
                    "pair (default); 'section' = nearest the section's own "
                    "mid-latitude, which need not be the same place")
p.add_argument('--minsep', type=float, default=0.55,
               help='minimum north-south separation as a fraction of the '
                    "section's full width")
p.add_argument('--agg', default='none', choices=['none', 'monthly'],
               help="'monthly' = calendar-month means of every series, with a "
                    "+/-1 sd band for the within-month spread")
p.add_argument('--show', default='lp', choices=['lp', 'both'],
               help="'lp' = Godin lowpass only (default), 'both' = lowpass "
                    "over the raw hourly")
p.add_argument('--tz', default='America/Los_Angeles')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
in_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_pc_sections_series'
Lfun.make_dir(out_dir)

SECTS = [s.strip() for s in args.sects.split(',')]
SLABEL = {'pc_lp': 'pc_lp  (mouth)', 'pc_lj': 'pc_lj  (mid-cove)',
          'pc_cp': 'pc_cp  (inner)'}
PCOLOR = {'north': '#0072B2', 'south': '#D55E00', 'center': '#009E73'}
# north and south are a depth-MATCHED pair; the centre face is chosen on
# position alone, so where it is also much deeper -- pc_lp's centre is 26.8 m
# against the flanks' 19.8/20.0 -- a centre-vs-flank difference mixes depth
# with lateral position. That is flagged per panel rather than hidden, by
# comparing the centre depth against the flanks and saying so when it differs
# by more than the pair's own tolerance. That matters most for bottom DO,
# where a deeper centre samples water that has had longer to draw down and
# will read low for reasons that have nothing to do with north vs south.
VPTS = ['north', 'center', 'south']
CPTS = {'salt': VPTS, 'vel': VPTS, 'do': VPTS}
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
PHCOLOR = {'spring': '#4565e8', 'neap': '#7f7f7f', 'transition': '#9c9c9c'}
PHALPHA = {'spring': 0.10, 'neap': 0.10, 'transition': 0.0}
O2_MMOL_TO_MGL = 32.0 / 1000.0
RAW = args.show == 'both'      # draw the hourly trace under the lowpass?
MON = args.agg == 'monthly'    # collapse everything to calendar-month means?


def agg(ix, y):
    """Series as plotted: itself, or its calendar-month mean and spread.

    At monthly resolution the mean is plotted at the MIDDLE of its month, not
    at the month boundary, because a month-start stamp reads as if the value
    belonged to the first day. The band is +/-1 sd of the values that went
    into each mean, which for velocity and DO is most of the story -- a month
    whose mean is near zero because it swung hard both ways is not the same
    as a month that sat still, and the mean alone cannot tell them apart.
    """
    if not MON:
        return pd.DatetimeIndex(ix), np.asarray(y, dtype=float), None
    ser = pd.Series(np.asarray(y, dtype=float), index=pd.DatetimeIndex(ix))
    g = ser.resample('MS')
    m, sd = g.mean(), g.std()
    mid = m.index + (m.index.to_series().diff().shift(-1).fillna(
        pd.Timedelta(days=30)) / 2)
    return pd.DatetimeIndex(mid), m.values, sd.values


def draw(ax, ix, y, ls='-', lw=2.2, color='k', label=None, alpha=1.0,
         band=True):
    """Plot a series through agg(), with the monthly spread band if asked."""
    x, v, sd = agg(ix, y)
    mk = 'o' if MON else None
    ax.plot(x, v, ls, lw=lw, color=color, label=label, alpha=alpha,
            marker=mk, ms=4)
    if MON and band and sd is not None:
        ax.fill_between(x, v - sd, v + sd, color=color, alpha=0.13, lw=0)
    return x, v
PAD_D = 5

t0 = pd.Timestamp(args.t0.replace('.', '-')).tz_localize(args.tz)
t1 = (pd.Timestamp(args.t1.replace('.', '-')) + pd.Timedelta(days=1)
      ).tz_localize(args.tz)


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def to_local(ix):
    return ix.tz_localize('UTC').tz_convert(args.tz).tz_localize(None)


# ------------------------------------------------------------ ssh / qnet ---
fn_flux = tef2 / ('hourly_flux_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag))
dflux = xr.open_dataset(fn_flux)
SSH = {sn: dflux.ssh.sel(sect=sn).to_pandas() for sn in SECTS}
QNET = {sn: dflux.qnet.sel(sect=sn).to_pandas() for sn in SECTS}
dflux.close()

FSIGN = {}
for sn in SECTS:
    r = np.corrcoef(QNET[sn].values, np.gradient(SSH[sn].values))[0, 1]
    FSIGN[sn] = (-1.0 if r < 0 else 1.0, r)
    print('%-6s flood check: corr(qnet, d(ssh)/dt) = %+.2f -> flood is q %s 0'
          % (sn, r, '<' if r < 0 else '>'))

# ------------------------------------------------- pick the six points ---
# structure_*.nc is the only place the face coordinates and the time-mean
# per-face transport live; the extraction itself has neither
dstr = xr.open_dataset(tef2 / ('structure_%s_%s_%s.nc' % (args.ds0, args.ds1,
                                                          gctag)))
OVR = {}
for tok in [s for s in args.faces.split(',') if s]:
    sn, ff = tok.split(':')
    OVR[sn] = [int(v) for v in ff.split('/')]

PICK = {}
rows = []
for sn in SECTS:
    lat = dstr['%s_lat' % sn].values
    lon = dstr['%s_lon' % sn].values
    h = dstr['%s_h' % sn].values
    qbar = dstr['%s_qbar' % sn].values.sum(axis=0)     # sum over z, per face

    if sn in OVR:
        kN, kS = OVR[sn]
        why = 'set by -faces'
    else:
        # One face from each side of the transport sign change, depth-matched
        # AND far enough apart to be a north/south contrast rather than two
        # neighbouring cells. Depth matching alone put pc_cp's pair three
        # faces apart in the middle of the section -- technically matched,
        # useless as a lateral comparison -- because the deep middle faces
        # were the only ones within 0.5 m of each other. Separation is
        # therefore a hard requirement and the depth tolerance is what gives,
        # relaxed in 0.25 m steps only until something qualifies, with the
        # tolerance actually used reported per section.
        iN = np.where(qbar < 0)[0]
        iS = np.where(qbar > 0)[0]
        span = lat.max() - lat.min()
        need = args.minsep * span
        dh, kN, kS = args.dhmax, None, None
        while kN is None and dh <= span * 1e9:
            best = -np.inf
            for a in iN:
                for b in iS:
                    if lat[a] - lat[b] < need:     # north of south, far enough
                        continue
                    if abs(h[a] - h[b]) > dh:
                        continue
                    score = abs(qbar[a]) + abs(qbar[b])
                    if score > best:
                        best, kN, kS = score, a, b
            if kN is None:
                dh += 0.25
                if dh > h.max():
                    raise SystemExit('%s: no pair at least %.0f%% of the '
                                     'section apart' % (sn, 100 * args.minsep))
        why = ('max |qbar| sum, >= %.0f%% of section apart, |dh| <= %.2f m'
               % (100 * args.minsep, dh))
        print('  %-6s pair is %.0f%% of the section apart, |dh| = %.2f m'
              % (sn, 100 * (lat[kN] - lat[kS]) / span, abs(h[kN] - h[kS])))
    # centre = face nearest the section's mid-latitude, excluding the two
    # already chosen; at both sections this lands on the qbar sign change,
    # i.e. the hinge of the lateral exchange
    # The centre of the SECTION and the midpoint of the chosen PAIR are not
    # the same latitude whenever the pair is not symmetric about the section.
    # At pc_lp the section runs to 48.2474 but the north point is at 48.2438,
    # so the section centre sits ~310 m north of halfway between the flanks
    # and the marker reads as misplaced. Bisecting the pair is what a
    # north/centre/south triplet implies, so that is the default.
    ctr = (0.5 * (lat[kN] + lat[kS]) if args.center == 'pair' else lat.mean())
    # ties (two faces equidistant) broken toward the one whose depth is
    # closest to the flanks', since the centre is only interpretable to the
    # extent it is comparable to them
    cand = np.array([i for i in range(len(lat)) if i not in (kN, kS)])
    dlat = np.abs(lat[cand] - ctr)
    tied = cand[dlat <= dlat.min() + 1e-9]
    kC = int(tied[np.argmin(np.abs(h[tied] - 0.5 * (h[kN] + h[kS])))])
    PICK[sn] = dict(north=kN, south=kS, center=kC, lat=lat, lon=lon, h=h,
                    qbar=qbar)
    for nm, k in [('north', kN), ('center', kC), ('south', kS)]:
        rows.append(dict(sect=sn, point=nm, face=int(k), lat=lat[k],
                         lon=lon[k], h=h[k], qbar=qbar[k],
                         rule=('nearest the %s midpoint' % (
                             'north/south pair' if args.center == 'pair'
                             else 'section')
                             if nm == 'center' else why)))
SEL = pd.DataFrame(rows)
dstr.close()
print('\nthe %d points:' % (3 * len(SECTS)))
print(SEL.round(4).to_string(index=False))
SEL.to_csv(out_dir / 'point_selection.csv', index=False, float_format='%.4f')

# ---------------------------------------------------------------- load ---
D = {}
for sn in SECTS:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    tt = to_local(pd.to_datetime(ds.time.values))
    keep = np.where((tt >= (t0 - pd.Timedelta(days=PAD_D)).tz_localize(None))
                    & (tt < (t1 + pd.Timedelta(days=PAD_D)).tz_localize(None)))[0]
    kk = [PICK[sn][nm] for nm in VPTS]
    sub = ds.isel(time=keep, p=kk)
    salt = sub.salt.values
    oxy = sub.oxygen.values * O2_MMOL_TO_MGL
    area = sub.DZ.values * sub.dd.values[np.newaxis, np.newaxis, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = FSIGN[sn][0] * np.where(area > 0, sub.q.values / area, np.nan)
    ds.close()
    idx = tt[keep]
    d = {}
    for j, nm in enumerate(VPTS):
        d[(nm, 's_top')] = salt[:, -1, j]
        d[(nm, 's_bot')] = salt[:, 0, j]
        d[(nm, 'u_top')] = u[:, -1, j]
        d[(nm, 'u_bot')] = u[:, 0, j]
        d[(nm, 'o_bot')] = oxy[:, 0, j]
    for k in list(d):
        d[k + ('lp',)] = godin(d[k])
    D[sn] = dict(idx=idx, v=d)

idx = D[SECTS[0]]['idx']
show = (idx >= t0.tz_localize(None)) & (idx < t1.tz_localize(None))
ssh = pd.Series(SSH[args.ref].values, index=to_local(SSH[args.ref].index))
ssh = ssh[(ssh.index >= idx[show][0]) & (ssh.index <= idx[show][-1])]
print('\nwindow %s to %s local, %d hourly steps'
      % (idx[show][0], idx[show][-1], show.sum()))

# ------------------------------------------------------------- forcing ---
# Skagit: one of only 3 gauged sources in this forcing, so it carries real
# 2025 hydrology rather than a repeating day-of-year climatology
RV = pd.read_pickle(Ldir['LOo'] / 'DM_outs' / '20260806_river_hydrographs'
                    / ('river_daily_%s_%s_%s.p' % ('trapsN00', args.ds0,
                                                   args.ds1)))
Qr = RV['Q']['skagit']
Qr = Qr[(Qr.index >= idx[show][0].normalize())
        & (Qr.index <= idx[show][-1])]
riv_src = RV['prov'].get('skagit', '?')

WD = pd.read_pickle(Ldir['LOo'] / 'DM_outs' / '20260806_wind'
                    / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))['W']
WD = WD.copy()
WD.index = to_local(WD.index)
WD = WD[(WD.index >= idx[show][0]) & (WD.index <= idx[show][-1])]
AX = (-0.9789, -0.2043)              # cove axis, mouth -> head (from tef2)
w_along = WD.u_pc.values * AX[0] + WD.v_pc.values * AX[1]
print('Skagit %s: %.0f to %.0f m3/s.  wind |U| max %.1f m/s'
      % (riv_src, Qr.min(), Qr.max(), WD.spd_pc.max()))

# ---------------------------------------------------------- tidal phase ---
P = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20260806_tidal_phase'
                / 'phase_daily.csv', index_col='date_local', parse_dates=True)
Pw = P.loc[(P.index >= idx[show][0].normalize()) & (P.index <= idx[show][-1])]
spans = []
ph = Pw.model_phase.values
cut = np.where(ph[1:] != ph[:-1])[0] + 1
for a, b in zip(np.r_[0, cut], np.r_[cut, len(ph)]):
    spans.append((Pw.index[a], Pw.index[b - 1] + pd.Timedelta(days=1), ph[a],
                  Pw.range_m.values[a:b].mean()))

E = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20260806_tidal_phase'
                / 'phase_events.csv', parse_dates=['time_local'])
E = E[((E.cycle == 'synodic') | E.event.str.contains('perigee|apogee'))
      & (E.time_local >= idx[show][0]) & (E.time_local <= idx[show][-1])]
EVENTS = list(zip(E.time_local, E.event))

# ------------------------------------------------------------------ plot ---
# EVERYTHING IN ONE COLUMN on one shared time axis. Three columns side by
# side would put the sections at different places on the page, and the whole
# point is to drop a vertical line anywhere and read the forcing and all six
# points off it at once. So the sections stack, mouth at the top, and the
# panels share x. Salinity panels share y with each other and velocity panels
# with each other, so amplitudes are comparable section to section by eye.
NS = len(SECTS)
ALT = args.layout == 'alt'
OUTER, INNER = SECTS[0], SECTS[-1]

# The row order IS the figure, so it is declared once as a spec and the rest
# of the code just walks it. 'alt' drops wind speed (the along-cove component
# is the part that can drive a flow), drops salinity, and splits the outer
# section's velocity into separate surface and bottom panels -- on a shared
# axis the bed is five times smaller and reads as a flat line, which is
# exactly the signal this layout exists to show.
if ALT:
    SPEC = [('range', None, 0.40), ('ssh', None, 0.80), ('riv', None, 0.70),
            ('wal', None, 0.60),
            ('do', INNER, 1.10),
            ('vel_top', OUTER, 1.10), ('vel_bot', OUTER, 1.10)]
else:
    SPEC = ([('range', None, 0.40), ('ssh', None, 0.80), ('riv', None, 0.70),
             ('wsp', None, 0.80), ('wal', None, 0.50)]
            + [('salt', sn, 1.05) for sn in SECTS]
            + [('vel', sn, 1.05) for sn in SECTS]
            + [('do', sn, 1.05) for sn in SECTS])

HR = [r[2] for r in SPEC]
fig = plt.figure(figsize=(16, 1.85 * sum(HR)), layout='constrained')
gs = fig.add_gridspec(len(HR), 1, height_ratios=HR)
AXS, ALL = {}, []
ax_rng = ax_ssh = ax_riv = ax_wsp = ax_wal = None
for r, (kind, sn, _) in enumerate(SPEC):
    # in 'full', like-for-like panels share a y axis so sections compare by
    # eye; in 'alt' each velocity panel needs its own scale, which is the
    # whole reason they are separate panels
    share = None
    if not ALT and sn is not None and (kind, SECTS[0]) in AXS:
        share = AXS[(kind, SECTS[0])]
    ax = fig.add_subplot(gs[r, 0], sharex=(ax_rng if r else None),
                         sharey=share)
    if r == 0:
        ax_rng = ax
    if kind == 'ssh':
        ax_ssh = ax
    elif kind == 'riv':
        ax_riv = ax
    elif kind == 'wsp':
        ax_wsp = ax
    elif kind == 'wal':
        ax_wal = ax
    elif sn is not None:
        AXS[(kind, sn)] = ax
    ALL.append(ax)

# The spring/neap shading, the moon-event lines and the vertical gridlines
# were being stamped on all twelve panels, which turned the figure into a
# lattice with the data behind it. They are context, so they live on the two
# context panels -- the tidal range and the subtidal sea level -- plus Skagit,
# and every data panel gets horizontal gridlines only.
CTX = [] if MON else [a for a in [ax_rng, ax_ssh, ax_riv] if a is not None]
for ax in ALL:
    if ax in CTX:
        for a, b, phase, rng in spans:
            if PHALPHA.get(phase, 0) > 0:
                ax.axvspan(a, b, color=PHCOLOR[phase], alpha=PHALPHA[phase],
                           lw=0, zorder=0)
        for te, elab in EVENTS:
            ax.axvline(te, color='0.35', lw=0.8, ls=':', zorder=1)
        ax.grid(**GRID)
    else:
        ax.grid(axis='y', **GRID)

# ---- 0  daily tidal range
if MON:
    draw(ax_rng, Pw.index, Pw.range_m.values, color='#404040', lw=1.8)
    ax_rng.set_ylim(0, np.nanmax(Pw.range_m.values) * 1.15)
else:
    rr = np.r_[Pw.range_m.values, Pw.range_m.values[-1]]
    te = list(Pw.index) + [Pw.index[-1] + pd.Timedelta(days=1)]
    ax_rng.step(te, rr, where='post', color='#404040', lw=1.8)
    ax_rng.set_ylim(0, np.nanmax(rr) * 1.55)
y0r, y1r = ax_rng.get_ylim()
for a, b, phase, rng in ([] if MON else spans):
    if (min(b, idx[show][-1]) - max(a, idx[show][0])) >= pd.Timedelta(days=3):
        ax_rng.text(max(a, idx[show][0]), y0r, ' %s' % phase, ha='left',
                    va='bottom', fontsize=8.5, fontweight='bold',
                    color=PHCOLOR.get(phase, '0.4'))
for te_, elab in ([] if MON else EVENTS):
    ax_rng.text(te_, y1r, ' ' + elab, rotation=90, ha='left', va='top',
                fontsize=7, color='0.35')
ax_rng.set_ylabel('daily tidal\nrange (m)')

# ---- 1  sea surface height
# Godin-lowpassed ssh is SUBTIDAL sea level -- the tide is already accounted
# for by the range panel above, so what is left here is the setup/setdown that
# the wind and the shelf impose, tens of cm rather than metres.
if RAW:
    ax_ssh.plot(ssh.index, ssh.values, '-', lw=0.5, color='0.6', alpha=0.6)
draw(ax_ssh, ssh.index, godin(ssh.values), lw=2.0, color='k')
ax_ssh.axhline(0, color='0.5', lw=0.8)
ax_ssh.set_ylabel('subtidal sea\nlevel (m)')
if not RAW:
    ax_ssh.text(0.004, 0.94, 'Godin lowpass -- tide removed; the range panel '
                'above carries the tidal amplitude', transform=ax_ssh.transAxes,
                fontsize=8, va='top', color='0.35')

# ---- 2  Skagit discharge
if MON:
    draw(ax_riv, Qr.index, Qr.values, color='#2e7d32', lw=1.8)
else:
    ax_riv.fill_between(Qr.index, 0, Qr.values, step='post', color='#2e7d32',
                        alpha=0.20, lw=0)
    ax_riv.step(Qr.index, Qr.values, where='post', color='#2e7d32', lw=1.8)
ax_riv.set_ylabel('Skagit R\n(m$^3$ s$^{-1}$)')
ax_riv.set_ylim(0, np.nanmax(Qr.values) * (1.15 if not MON else 0.9))
# no Godin here on purpose: the river forcing is stored daily, so it carries
# no tidal band for a 71-hour filter to remove
ax_riv.text(0.004, 0.92, 'daily, %s -- a real 2025 hydrograph, not climatology.'
            '  already subtidal, so no Godin filter applied' % riv_src,
            transform=ax_riv.transAxes, fontsize=8, va='top', color='#1b5e20')

# ---- wind speed, with direction arrows (absent in the 'alt' layout,
#      which keeps only the along-cove component)
if ax_wsp is not None:
    if RAW:
        ax_wsp.plot(WD.index, WD.spd_pc.values, '-', lw=0.5, color='0.55',
                    alpha=0.8)
    wsp_lp = godin(WD.spd_pc.values)
    draw(ax_wsp, WD.index, wsp_lp, lw=2.0, color='k')
    ymax = ((np.nanmax(WD.spd_pc.values) if RAW else np.nanmax(wsp_lp))
            * (1.32 if not MON else 1.6))
    ax_wsp.set_ylim(0, ymax)
    u_lp, v_lp = godin(WD.u_pc.values), godin(WD.v_pc.values)
    if MON:
        xq, uq, _ = agg(WD.index, u_lp)
        _, vq, _ = agg(WD.index, v_lp)
        qscale = 30
    else:
        sl = slice(0, None, 12)                                     # every 12 h
        xq, uq, vq = WD.index[sl], u_lp[sl], v_lp[sl]
        qscale = 90
    ax_wsp.quiver(xq, np.full(len(xq), ymax * 0.88), uq, vq, color='#1f4e79',
                  scale=qscale, width=0.0018, headwidth=4, headlength=5,
                  alpha=0.9)
    # below the arrow band, above the speed trace, so it sits in the empty middle
    ax_wsp.text(0.004, 0.62, 'Penn Cove wind; arrows are the mean wind VECTOR, '
                'pointing the way it blows TOWARD' if MON else
                'Penn Cove wind, Godin lowpass; arrows every 12 h '
                'point the way the (lowpassed) wind blows TOWARD',
                transform=ax_wsp.transAxes, fontsize=8, va='center',
                color='#1f4e79')
    ax_wsp.set_ylabel('wind speed\n(m s$^{-1}$)')

# ---- 4  along-cove wind
wl = godin(w_along)
ax_wal.axhline(0, color='0.4', lw=0.9)
xw, wlm, _ = agg(WD.index, wl)
if RAW:
    ax_wal.plot(WD.index, w_along, '-', lw=0.4, color='0.6', alpha=0.7)
draw(ax_wal, WD.index, wl, lw=2.0, color='#6a3d9a', band=False)
ax_wal.fill_between(xw, 0, wlm, where=wlm >= 0, color='#6a3d9a',
                    alpha=0.20, lw=0, interpolate=True)
ax_wal.fill_between(xw, 0, wlm, where=wlm < 0, color='#b15928',
                    alpha=0.20, lw=0, interpolate=True)
ax_wal.set_ylabel('along-cove\nwind (m s$^{-1}$)')
ax_wal.text(0.004, 0.93, 'up-cove (mouth $\\rightarrow$ head)',
            transform=ax_wal.transAxes, fontsize=8, va='top', color='#6a3d9a')
ax_wal.text(0.004, 0.07, 'down-cove (head $\\rightarrow$ mouth)',
            transform=ax_wal.transAxes, fontsize=8, va='bottom',
            color='#b15928')

# ---- 5,6,7  the six points
ROWDEF = {
    'salt': ([('s_top', '-', 'surface'), ('s_bot', '--', 'bottom')],
             'salinity\n(g kg$^{-1}$)'),
    'vel': ([('u_top', '-', 'surface'), ('u_bot', '--', 'bottom')],
            'velocity into\ncove (m s$^{-1}$)'),
    'vel_top': ([('u_top', '-', 'surface')],
                'SURFACE velocity\ninto cove (m s$^{-1}$)'),
    'vel_bot': ([('u_bot', '-', 'bottom')],
                'BOTTOM velocity\ninto cove (m s$^{-1}$)'),
    'do': ([('o_bot', '-', 'bottom')], 'bottom DO\n(mg L$^{-1}$)')}
CPTS['vel_top'] = CPTS['vel_bot'] = VPTS

for (vn, sn) in [(k, v) for k, v, _ in SPEC if v is not None]:
        varlist, ylab = ROWDEF[vn]
        c = SECTS.index(sn)
        ax = AXS[(vn, sn)]
        v = D[sn]['v']
        for nm in CPTS[vn]:
            for key, ls, dlab in varlist:
                if RAW:
                    ax.plot(idx[show], v[(nm, key)][show], ls, lw=0.5,
                            alpha=0.22, color=PCOLOR[nm])
                # monthly means come off the UNFILTERED hourly series: a
                # month is far longer than the 71-hour filter, so lowpassing
                # first would change nothing but would lose the ends
                src = v[(nm, key)] if MON else v[(nm, key, 'lp')]
                draw(ax, idx[show], src[show], ls=ls, color=PCOLOR[nm],
                     label='%s %s' % (nm, dlab))
        if vn.startswith('vel'):
            ax.axhline(0, color='0.4', lw=0.9)
        if vn == 'do':
            for thr, tl in [(2.0, 'hypoxic (2 mg L$^{-1}$)'),
                            (5.0, '5 mg L$^{-1}$')]:
                lo, hi = ax.get_ylim()
                if lo < thr < hi:
                    ax.axhline(thr, color='#c00000', lw=0.9, ls='-.',
                               alpha=0.7)
                    if c == 0 or ALT:
                        ax.text(idx[show][0], thr, ' ' + tl, fontsize=7.5,
                                color='#c00000', va='bottom')
        kN, kS, kC = (PICK[sn]['north'], PICK[sn]['south'],
                      PICK[sn]['center'])
        hh, la = PICK[sn]['h'], PICK[sn]['lat']
        ax.set_ylabel('%s\n%s' % (SLABEL.get(sn, sn).split()[1], ylab))
        lab = '%s   north p=%d (h %.1f m), south p=%d (h %.1f m)' % (
            SLABEL.get(sn, sn), kN, hh[kN], kS, hh[kS])
        if 'center' in CPTS[vn]:
            dh = abs(hh[kC] - 0.5 * (hh[kN] + hh[kS]))
            lab += ',  centre p=%d (h %.1f m%s)' % (
                kC, hh[kC], ', %.1f m deeper -- NOT depth-matched' % dh
                if dh > args.dhmax else ', depth-matched')
        ax.text(0.004, 0.94, lab,
                transform=ax.transAxes, fontsize=8.5, va='top', color='0.25',
                fontweight='bold')
        if c == 0 or ALT:
            ax.legend(fontsize=7.5, ncol=4, loc='lower left', framealpha=0.85)


for ax in ALL:
    ax.set_xlim(idx[show][0], idx[show][-1])
for ax in ALL[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)

ndays = (idx[show][-1] - idx[show][0]).total_seconds() / 86400
ax_last = ALL[-1]
if MON or ndays > 120:
    # day ticks on a year read as arbitrary dates (Jan 13, Jan 31, Feb 18);
    # months are the unit the eye is actually using at this length
    ax_last.xaxis.set_major_locator(MonthLocator())
    ax_last.xaxis.set_major_formatter(DateFormatter('%b %Y' if ndays > 400
                                                    else '%b'))
else:
    ax_last.xaxis.set_major_locator(DayLocator(interval=max(1, int(ndays // 20))))
    ax_last.xaxis.set_major_formatter(DateFormatter('%b %d'))
plt.setp(ax_last.get_xticklabels(), rotation=45, ha='right')
ax_last.set_xlabel('local time (%s)' % args.tz)

fig.suptitle('%s -- Penn Cove, %d points across %d section%s, '
             '%s to %s\n'
             'one shared time axis: read any vertical line through the forcing '
             'and every point.  blue = north, orange = south.  '
             'solid = surface, dashed = bottom.  %s'
             % (args.gtx, 3 * NS, NS, '' if NS == 1 else 's',
                args.t0, args.t1,
                'CALENDAR-MONTH MEANS, band = +/-1 sd within the month.'
                if MON else 'lowpass over raw hourly.' if RAW else
                'EVERY series is Godin-lowpassed (Skagit is daily already).'),
             fontsize=13)

fn = out_dir / ('sections_series%s_%s_%s%s.png'
                % ('_alt' if ALT else '', args.t0, args.t1,
                   '_monthly' if MON else ('' if RAW else '_godin')))
fig.savefig(fn, dpi=180, bbox_inches='tight')
print('\nsaved %s' % fn)

# ------------------------------------------------------------------ data ---
out = pd.DataFrame(index=idx[show])
out.index.name = 'time_local'
# Over a window that spans the DST fall-back, converting UTC to local time
# repeats one hour, so these series carry duplicate index labels and pandas
# refuses to reindex on them. Keep the first of each label -- the pair are an
# hour apart in UTC but land on the same local stamp, and for a subtidal
# figure either one is the same answer.
def on_idx(ser):
    ser = ser[~ser.index.duplicated(keep='first')]
    return ser.reindex(pd.DatetimeIndex(idx[show])).values


out['ssh_%s' % args.ref] = on_idx(ssh)
out['wind_spd_pc'] = on_idx(WD.spd_pc)
out['wind_along_cove'] = on_idx(pd.Series(w_along, index=WD.index))
out['skagit_m3s'] = Qr.reindex(pd.DatetimeIndex(idx[show]).normalize()).values
for sn in SECTS:
    for nm in VPTS:
        for key in ['s_top', 's_bot', 'u_top', 'u_bot', 'o_bot']:
            out['%s_%s_%s' % (sn, nm, key)] = D[sn]['v'][(nm, key)][show]
out.to_csv(out_dir / ('sections_series_%s_%s.csv' % (args.t0, args.t1)),
           float_format='%.5f')

rows = []
for sn in SECTS:
    for nm in VPTS:
        for key in ['s_top', 's_bot', 'u_top', 'u_bot', 'o_bot']:
            a = D[sn]['v'][(nm, key)][show]
            lp = D[sn]['v'][(nm, key, 'lp')][show]
            rows.append(dict(sect=sn, point=nm, var=key, mean=np.nanmean(a),
                             min=np.nanmin(a), max=np.nanmax(a),
                             tidal_rms=np.sqrt(np.nanmean((a - lp) ** 2)),
                             subtidal_range=np.nanmax(lp) - np.nanmin(lp)))
R = pd.DataFrame(rows)
R.to_csv(out_dir / ('sections_stats_%s_%s.csv' % (args.t0, args.t1)),
         index=False, float_format='%.5f')
print('\n' + R.round(3).to_string(index=False))
