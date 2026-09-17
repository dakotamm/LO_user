"""
Along-channel salinity gradient ds/dx in Penn Cove, from the wb1 model output.

x runs MOUTH -> HEAD along the line joining the pc section centroids
(pc_lp = mouth, pc_lj = mid-cove, pc_cp = head), so

    ds/dx > 0  means the HEAD is saltier than the mouth
    ds/dx < 0  means the head is fresher, the textbook estuarine sense

Three ways of saying the same thing are carried, because they are not
interchangeable:

    dsdx_*        least-squares slope of the three section means against x.
                  The single number, but it averages the cove's two halves.
    dsdx_*_outer  (mid - mouth)/dx  and
    dsdx_*_inner  (head - mid)/dx. The inner half responds several times
                  harder than the outer one, so a fitted slope hides the
                  signal -- always look at the halves too.
    ds_*          the raw head-minus-mouth difference, g/kg, undivided.

and each is computed in three layers: 'top' (upper HLAY m), 'bot' (lower
HLAY m) and 'bar' (depth mean).

THE ONE THING NOT TO DO: never quote a single depth-mean ds/dx as "the"
gradient. Penn Cove has no river -- fresh water arrives from OUTSIDE, at the
mouth, as Saratoga Passage water -- so the surface and bottom gradients have
OPPOSITE signs and the depth mean is the small residual of two larger numbers.
What the along-channel structure actually is, is a gradient in STRATIFICATION:
the cove mixes vertically toward its closed end.

Layer means use PARTIAL cells: a sigma cell straddling the HLAY level
contributes only the fraction of itself inside the layer. Without that the
"gradient" would partly be a gradient in layer thickness, since h differs
across and between sections and the cells breathe with the tide.

Everything is Godin lowpassed before differencing. The hourly (unfiltered)
depth-mean gradient is carried too, and drawn behind the subtidal one, because
its size relative to the subtidal gradient is the tidal-advection term that
matters for the Chen et al. (2012) comparison -- but it is not a gradient
anyone should quote.

Reads the hourly section extractions,
    LO_output/extract/[gtx]/tef2/extractions_avg_[ds0]_[ds1]/[sn].nc
plus structure_[ds0]_[ds1]_[coll].nc for the section centroids. Paths resolve
through Ldir, so this runs on apogee, where the output lives.

Writes to LO_output/DM_outs/20260916_dsdx/:
    dsdx_series_[ds0]_[ds1].csv   daily series, small enough to carry back
    dsdx_series_[ds0]_[ds1].png   the time series
    dsdx_profile_[ds0]_[ds1].png  mean s vs x, and the seasonal cycle

run 20260916_dsdx.py
"""
import argparse
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('-0', '--ds0', default='2024.01.01')
p.add_argument('-1', '--ds1', default='2025.12.31')
p.add_argument('--sects', default='pc_lp,pc_lj,pc_cp', help='MOUTH FIRST')
p.add_argument('--hlay', type=float, default=3.0,
               help='thickness (m) of the surface and bottom layers')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
in_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260916_dsdx'
Lfun.make_dir(out_dir)

SECTS = [s.strip() for s in args.sects.split(',') if s.strip()]
SLAB = {'pc_lp': 'mouth', 'pc_lj': 'mid-cove', 'pc_cp': 'head'}
SLAB = {sn: SLAB.get(sn, sn) for sn in SECTS}
CB = dict(blue='#0072B2', orange='#D55E00', green='#009E73', red='#CC0000',
          purple='#7B3294', yellow='#E69F00', pink='#CC79A7', grey='#7f7f7f')
SC = {sn: c for sn, c in zip(SECTS, [CB['blue'], CB['green'], CB['orange'],
                                     CB['purple'], CB['pink']])}
LAY = [('top', CB['blue'], 'surface (top %.0f m)' % args.hlay),
       ('bot', CB['orange'], 'bottom (bed %.0f m)' % args.hlay),
       ('bar', 'k', 'depth-mean')]
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


# ------------------------------------------------------------- geometry ---
# The axis is defined by the sections themselves, since that is the line the
# differences below are actually taken along.
dstr = xr.open_dataset(tef2 / ('structure_%s_%s_%s.nc'
                               % (args.ds0, args.ds1, args.coll)))
CEN = {sn: (float(np.mean(dstr['%s_lon' % sn].values)),
            float(np.mean(dstr['%s_lat' % sn].values))) for sn in SECTS}
dstr.close()
lat0 = np.mean([c[1] for c in CEN.values()])
COS = np.cos(np.deg2rad(lat0))


def xy_km(lo, la):
    return lo * COS * 111.32, la * 111.32


x0, y0 = xy_km(*CEN[SECTS[0]])
xN, yN = xy_km(*CEN[SECTS[-1]])
axl = np.hypot(xN - x0, yN - y0)
ax_, ay_ = (xN - x0) / axl, (yN - y0) / axl          # unit vector, mouth->head
X = {}
for sn in SECTS:
    xs, ys = xy_km(*CEN[sn])
    X[sn] = (xs - x0) * ax_ + (ys - y0) * ay_
xv = np.array([X[sn] for sn in SECTS])

print('--- along-channel axis, %s -> %s ---' % (SLAB[SECTS[0]], SLAB[SECTS[-1]]))
print('unit vector (%.4f, %.4f), i.e. %.0f deg true'
      % (ax_, ay_, np.rad2deg(np.arctan2(ax_, ay_)) % 360))
for sn in SECTS:
    print('  %-7s (%-8s) x = %5.2f km' % (sn, SLAB[sn], X[sn]))

# ------------------------------------------------- per-section reduction ---
print('\n--- reducing sections ---')
R = {}
for sn in SECTS:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    tt = pd.DatetimeIndex(ds.time.values)
    dd = ds.dd.values[None, None, :]                  # (1,1,p) face width
    DZ = ds.DZ.values                                 # (t,z,p)
    salt = ds.salt.values.astype(float)

    cum_hi = np.cumsum(DZ, axis=1)                    # height above the bed
    cum_lo = cum_hi - DZ
    H = cum_hi[:, -1:, :]
    d_hi = H - cum_hi                                 # depth below the surface
    w = {'bot': np.clip(args.hlay - cum_lo, 0, DZ) / DZ,
         'top': np.clip(args.hlay - d_hi, 0, DZ) / DZ,
         'bar': np.ones_like(DZ)}

    d = {}
    for nm in ['top', 'bot', 'bar']:
        A = dd * DZ * w[nm]                           # face area in the layer
        d['s_' + nm] = ((salt * A).sum(axis=(1, 2)) / A.sum(axis=(1, 2)))
    d['A'] = (dd * DZ).sum(axis=(1, 2))
    R[sn] = pd.DataFrame(d, index=tt)
    print('  %-7s %2d faces, area %.3f km2, mean H %.1f m'
          % (sn, ds.sizes['p'], R[sn].A.mean() / 1e6, float(H.mean())))
    ds.close()

TT = R[SECTS[0]].index
for sn in SECTS:
    assert R[sn].index.equals(TT), 'section time axes differ'

# --------------------------------------------------------- the gradients ---
S = pd.DataFrame(index=TT)
for sn in SECTS:
    for nm in ['top', 'bot', 'bar']:
        S['%s_s_%s' % (sn, nm)] = godin(R[sn]['s_' + nm].values)
    S['%s_dstrat' % sn] = S['%s_s_bot' % sn] - S['%s_s_top' % sn]

den = ((xv - xv.mean()) ** 2).sum()
for nm in ['top', 'bot', 'bar']:
    M = np.column_stack([S['%s_s_%s' % (sn, nm)].values for sn in SECTS])
    S['dsdx_' + nm] = ((M - M.mean(axis=1, keepdims=True))
                       * (xv - xv.mean())).sum(axis=1) / den
    # head minus mouth, both undivided and divided by the cove's length
    S['ds_' + nm] = S['%s_s_%s' % (SECTS[-1], nm)] - S['%s_s_%s' % (SECTS[0], nm)]
    S['dsdx_%s_endpt' % nm] = S['ds_' + nm] / (xv[-1] - xv[0])
    # and by half
    for a, b, half in [(0, 1, 'outer'), (1, 2, 'inner')]:
        if b < len(SECTS):
            S['dsdx_%s_%s' % (nm, half)] = (
                (S['%s_s_%s' % (SECTS[b], nm)] - S['%s_s_%s' % (SECTS[a], nm)])
                / (xv[b] - xv[a]))

# the hourly (tide included) depth-mean gradient, for scale only
Mh = np.column_stack([R[sn]['s_bar'].values for sn in SECTS])
S['dsdx_bar_hourly'] = ((Mh - Mh.mean(axis=1, keepdims=True))
                        * (xv - xv.mean())).sum(axis=1) / den

D = S.resample('1D').mean()
D.to_csv(out_dir / ('dsdx_series_%s_%s.csv' % (args.ds0, args.ds1)))

print('\n--- time-mean gradients, %s to %s (g kg-1 km-1) ---'
      % (TT[0].date(), TT[-1].date()))
print('%-12s %9s %9s %9s %9s %9s %9s'
      % ('layer', 'fit', 'endpt', 'outer', 'inner', 'ds[g/kg]', 'rms'))
for nm, _, lab in LAY:
    cols = ['dsdx_' + nm, 'dsdx_%s_endpt' % nm, 'dsdx_%s_outer' % nm,
            'dsdx_%s_inner' % nm, 'ds_' + nm]
    v = [S[c].mean() for c in cols]
    print('%-12s %+9.3f %+9.3f %+9.3f %+9.3f %+9.3f %9.3f'
          % (nm, v[0], v[1], v[2], v[3], v[4],
             np.sqrt(np.nanmean(S['dsdx_' + nm] ** 2))))
print('  sign convention: + = the head is SALTIER than the mouth')
print('  surface and bottom have opposite signs, so the depth-mean "fit" is a'
      ' residual, not the gradient')
print('\nstratification (s_bot - s_top), g/kg: '
      + ', '.join('%s %.2f' % (SLAB[sn], S['%s_dstrat' % sn].mean())
                  for sn in SECTS))
print('hourly depth-mean ds/dx: mean %+.3f, rms %.3f -- the tide swings it '
      '%.1fx its own subtidal rms'
      % (S.dsdx_bar_hourly.mean(), np.sqrt(np.nanmean(S.dsdx_bar_hourly ** 2)),
         np.sqrt(np.nanmean(S.dsdx_bar_hourly ** 2))
         / np.sqrt(np.nanmean(S.dsdx_bar ** 2))))

# ======================================================= figure 1: series ===
sub = S.dropna(subset=['dsdx_bar'])
fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                        layout='constrained')

ax = axs[0]
# drawn head first so the mouth ends up on top: the mouth and mid-cove series
# sit almost exactly on each other and whichever is drawn last is the one you
# can see
for sn in SECTS[::-1]:
    ax.plot(sub.index, sub['%s_s_bar' % sn], lw=1.2, color=SC[sn],
            label='%s (x = %.2f km)' % (SLAB[sn], X[sn]))
h_, l_ = ax.get_legend_handles_labels()
ax.legend(h_[::-1], l_[::-1], fontsize=8, ncol=len(SECTS), loc='lower left')
ax.set_ylabel('depth-mean salinity\n(g kg$^{-1}$), subtidal')
ax.set_title('Penn Cove along-channel salinity gradient, %s, %s to %s'
             % (args.gtx, args.ds0, args.ds1), fontsize=11)

ax = axs[1]
for nm, c_, lab in LAY:
    ax.plot(sub.index, sub['ds_' + nm], lw=1.3, color=c_, label=lab)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('$\\Delta$s, head $-$ mouth\n(g kg$^{-1}$)')
ax.legend(fontsize=8, ncol=3, loc='upper left')

ax = axs[2]
ax.plot(sub.index, sub['dsdx_bar_hourly'], lw=0.4, color='0.75',
        label='depth-mean, hourly (tide in)')
for nm, c_, lab in LAY:
    ax.plot(sub.index, sub['dsdx_' + nm], lw=1.3, color=c_, label=lab)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('ds/dx, 3-section fit\n(g kg$^{-1}$ km$^{-1}$)')
ax.legend(fontsize=8, ncol=4, loc='upper left')

ax = axs[3]
for half, c_, lab in [('outer', CB['purple'], '%s $\\to$ %s'
                       % (SLAB[SECTS[0]], SLAB[SECTS[1]])),
                      ('inner', CB['green'], '%s $\\to$ %s'
                       % (SLAB[SECTS[1]], SLAB[SECTS[-1]]))]:
    c = 'dsdx_bar_' + half
    if c in sub:
        ax.plot(sub.index, sub[c], lw=1.3, color=c_,
                label='%s (rms %.3f)' % (lab, np.sqrt(np.nanmean(sub[c] ** 2))))
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('depth-mean ds/dx by half\n(g kg$^{-1}$ km$^{-1}$)')
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

for a in axs:
    a.grid(**GRID)
fig.text(0.01, -0.025,
         'x positive %s $\\to$ %s, so ds/dx > 0 means a SALTIER head.\n'
         'Surface and bottom have opposite signs: this is a gradient in '
         'stratification, not a horizontal front -- Penn Cove has no river, '
         'so its fresh water enters at the mouth.'
         % (SLAB[SECTS[0]], SLAB[SECTS[-1]]),
         fontsize=8, color='0.3', va='top')
fn = out_dir / ('dsdx_series_%s_%s.png' % (args.ds0, args.ds1))
fig.savefig(fn, dpi=170, bbox_inches='tight', transparent=True)
plt.close(fig)
print('\nSaved ' + str(fn))

# ===================================================== figure 2: profile ===
# Bands are drawn as an opaque colour blended toward white rather than with
# alpha: these figures are saved with transparent=True, and a semi-transparent
# patch over a transparent background renders differently depending on what
# composites the PNG.
def pale(c, f=0.18):
    r, g, b = mcolors.to_rgb(c)
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


fig, axs = plt.subplots(1, 1, figsize=(7, 5.5), layout='constrained')
axs = [axs]

# salinity RELATIVE TO THE MOUTH. Plotting absolute salinity here is
# useless -- the seasonal range at any one section is ~5 g/kg, thirty times the
# 0.2-0.3 g/kg difference between the mouth and the head, so the along-channel
# structure is invisible inside its own band. Referencing every section to the
# mouth at the same hour removes the seasonal cycle the sections share.
ax = axs[0]
for nm, c_, lab in LAY:
    A = np.column_stack([(S['%s_s_%s' % (sn, nm)] - S['%s_s_%s' % (SECTS[0], nm)]).values
                         for sn in SECTS])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = np.nanmean(A, axis=0)
        lo = np.nanpercentile(A, 25, axis=0)
        hi = np.nanpercentile(A, 75, axis=0)
    ax.fill_between(xv, lo, hi, color=pale(c_), lw=0, zorder=1)
    ax.plot(xv, m, 'o-', color=c_, lw=1.8, ms=6, zorder=3,
            label='%s, ds/dx = %+.3f' % (lab, S['dsdx_' + nm].mean()))
ax.axhline(0, color='0.5', lw=0.8, zorder=2)
for sn in SECTS:
    ax.axvline(X[sn], color='0.85', lw=0.8, zorder=0)
    ax.annotate(SLAB[sn], (X[sn], 1.0), xycoords=('data', 'axes fraction'),
                xytext=(0, -10), textcoords='offset points', fontsize=8,
                color='0.4', ha='center')
ax.set_xlabel('x, %s $\\to$ %s (km)' % (SLAB[SECTS[0]], SLAB[SECTS[-1]]))
ax.set_ylabel('salinity minus the %s value (g kg$^{-1}$)' % SLAB[SECTS[0]])
ax.set_title('time-mean salinity along the cove, referenced to the %s\n'
             '(band = interquartile range of the subtidal series; %s mean is '
             '%.2f g kg$^{-1}$)'
             % (SLAB[SECTS[0]], SLAB[SECTS[0]],
                S['%s_s_bar' % SECTS[0]].mean()), fontsize=10)
ax.legend(fontsize=8, loc='best')
ax.grid(**GRID)

fn = out_dir / ('dsdx_profile_%s_%s.png' % (args.ds0, args.ds1))
fig.savefig(fn, dpi=170, bbox_inches='tight', transparent=True)
plt.close(fig)
print('Saved ' + str(fn))

# ============================================== figure 3: year by year ======
# The seasonal cycle WITHOUT pooling: every day of the subtidal record plotted
# against its day of year, one line per calendar year. A monthly climatology of
# a two-year run is a false average -- with n = 2 per month, a single freshet or
# a single wind event is half the "climatology", and the November-December
# collapse below turns out to be much bigger in one year than the other. Only
# the day-resolved lines show that.
YRS = sorted(set(D.index.year))
# deliberately NOT the layer colours (blue/orange/black) or the half colours
# (purple/green) used in the other figures, so a line's colour never means two
# different things across this output directory
YC = {y: c for y, c in zip(YRS, ['#4C4C4C', CB['red'], CB['yellow'],
                                 CB['pink'], CB['blue']])}
MON0 = [pd.Timestamp('2001-%02d-01' % m).dayofyear for m in range(1, 13)]

print('\n--- annual means, no pooling (g kg-1 km-1) ---')
print('%-6s %8s %8s %8s %7s' % ('year', 'top', 'bot', 'bar', 'ndays'))
for y in YRS:
    g = D[D.index.year == y]
    print('%-6d %+8.3f %+8.3f %+8.3f %7d'
          % (y, g.dsdx_top.mean(), g.dsdx_bot.mean(), g.dsdx_bar.mean(),
             g.dsdx_bar.notna().sum()))

fig, axs = plt.subplots(len(LAY), 1, figsize=(13, 10), sharex=True,
                        layout='constrained')
for ax, (nm, c_, lab) in zip(axs, LAY):
    for y in YRS:
        g = D[D.index.year == y]
        v = g['dsdx_' + nm]
        if v.notna().sum() < 10:
            continue
        ax.plot(g.index.dayofyear, v.values, lw=1.3, color=YC[y],
                label='%d (mean %+.3f)' % (y, v.mean()))
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_ylabel('%s\nds/dx (g kg$^{-1}$ km$^{-1}$)' % lab)
    ax.legend(fontsize=8, ncol=len(YRS), loc='best')
    ax.grid(**GRID)
axs[0].set_title('ds/dx day by day, each year on its own -- %s, %s to %s'
                 % (args.gtx, args.ds0, args.ds1), fontsize=11)
axs[-1].set_xlim(1, 366)
axs[-1].set_xticks(MON0)
axs[-1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                         'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axs[-1].set_xlabel('day of year')
fn = out_dir / ('dsdx_byyear_%s_%s.png' % (args.ds0, args.ds1))
fig.savefig(fn, dpi=170, bbox_inches='tight', transparent=True)
plt.close(fig)
print('Saved ' + str(fn))
