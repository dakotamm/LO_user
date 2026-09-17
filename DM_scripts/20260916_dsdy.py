"""
Cross-channel salinity gradient ds/dy AT EACH SECTION, from the wb1 model
output. The lateral sibling of 20260916_dsdx.py, built to be read beside it --
same run, same sections, same layers, same year-by-year plots, rotated 90 deg.

y is the cross-channel coordinate, POSITIVE TOWARD THE NORTH SHORE: the
right-hand normal (ay, -ax) of the along-channel vector (ax, ay) that
20260916_dsdx.py defines. Each pc section is a constant-longitude N-S line, so
y is very nearly latitude, and the script asserts corr(y, lat) = +1 rather
than trusting it.

  ***  SIGN WARNING  ***
  LO_user/extract/tef2/reduce_wind_cove.py and 20260806_wind_characterize.py
  both label their cross-cove component "positive toward the north shore" and
  are BACKWARDS -- they build it as (-u ay + v ax), the left-hand normal of a
  westward vector, which points SOUTH. This script uses the right-hand normal
  and prints the check. Don't compare signs with those two without flipping.

Unlike the along-channel case, a single number per section is not automatically
meaningful here, so three versions of the same gradient are carried:

    dy      width-weighted least-squares slope of the per-face layer mean
            against y. The headline number.
    dy_h    the partial slope on y from a regression on [1, y, h] -- the
            lateral gradient AT FIXED DEPTH.
    ns      the difference between a DEPTH-MATCHED pair of faces, one on each
            side of the mean transport sign change (the pairing rule from
            20260806_pc_sections_series.py). Confound-free, and the number to
            quote when the straight line is a poor fit.

and the weighted R2 of the straight line is carried alongside each, because a
slope only summarises the lateral structure when that structure is a ramp.

  THE DEPTH CONFOUND IS THE WHOLE PROBLEM. A section is a channel
  cross-section: depth is correlated with lateral position (corr(y,h) = -0.55
  at pc_lp), so a raw BOTTOM-layer ds/dy is partly a statement about which
  faces are deep rather than about north versus south. At pc_lp the raw bottom
  slope is about -0.23 g/kg/km and the h-controlled one about -0.02: ~90% of it
  is geometry. The surface layer is nearly free of this and the two agree
  there. Never quote a raw bottom ds/dy.

  AND AT THE HEAD THE LINE IS THE WRONG MODEL. Mean per-face transport across
  pc_lp and pc_lj changes sign once (in on the north side, out on the south --
  the lateral exchange those sections are known for), but across pc_cp it
  changes sign three times. A straight line through that means little; the R2
  printed per section is what says so, and the per-face profile figure is what
  you read instead.

Layer means use PARTIAL cells, exactly as in 20260916_dsdx.py.

Reads LO_output/extract/[gtx]/tef2/extractions_avg_[ds0]_[ds1]/[sn].nc plus
structure_[ds0]_[ds1]_[coll].nc (per-face lon, lat, h, dd, qbar) and
hourly_flux_... (for the flood sign check). Paths resolve through Ldir, so this
runs on apogee.

Writes to LO_output/DM_outs/20260916_dsdy/:
    dsdy_series_[ds0]_[ds1].csv   daily series, all three versions
    dsdy_series_[ds0]_[ds1].png   the full record, one row per section
    dsdy_byyear_[ds0]_[ds1].png   day of year, one line per year
    dsdy_profile_[ds0]_[ds1].png  per-face salinity and h across each section

run 20260916_dsdy.py
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
p.add_argument('--dhmax', type=float, default=0.5,
               help='initial depth tolerance (m) for the N/S pair')
p.add_argument('--minsep_face', type=float, default=0.4,
               help='minimum N-S separation of the pair, as a fraction of '
                    'section width')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
in_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260916_dsdy'
Lfun.make_dir(out_dir)

SECTS = [s.strip() for s in args.sects.split(',') if s.strip()]
_SL = {'pc_lp': 'mouth', 'pc_lj': 'mid-cove', 'pc_cp': 'head'}
SLAB = {sn: _SL.get(sn, sn) for sn in SECTS}
CB = dict(blue='#0072B2', orange='#D55E00', green='#009E73', red='#CC0000',
          purple='#7B3294', yellow='#E69F00', pink='#CC79A7', grey='#7f7f7f')
LAY = [('top', CB['blue'], 'surface (top %.0f m)' % args.hlay),
       ('bot', CB['orange'], 'bottom (bed %.0f m)' % args.hlay),
       ('bar', 'k', 'depth-mean')]
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def pale(c, f=0.18):
    """An opaque colour blended toward white -- see 20260916_dsdx.py for why
    these figures do not use alpha."""
    r, g, b = mcolors.to_rgb(c)
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def wls(X, w):
    """Weighted least-squares projector and hat matrix, so a weighted R2 can
    be formed. beta = data @ P.T."""
    Wm = np.diag(w)
    P = np.linalg.solve(X.T @ Wm @ X, X.T @ Wm)
    return P, X @ P


# ------------------------------------------------------------- geometry ---
dstr = xr.open_dataset(tef2 / ('structure_%s_%s_%s.nc'
                               % (args.ds0, args.ds1, gctag)))
CEN = {sn: (float(np.mean(dstr['%s_lon' % sn].values)),
            float(np.mean(dstr['%s_lat' % sn].values))) for sn in SECTS}
lat0 = np.mean([c[1] for c in CEN.values()])
COS = np.cos(np.deg2rad(lat0))


def xy_km(lo, la):
    return lo * COS * 111.32, la * 111.32


x0, y0 = xy_km(*CEN[SECTS[0]])
xN, yN = xy_km(*CEN[SECTS[-1]])
axl = np.hypot(xN - x0, yN - y0)
ax_, ay_ = (xN - x0) / axl, (yN - y0) / axl          # mouth -> head
cx_, cy_ = ay_, -ax_                                 # right-hand normal
print('--- axes ---')
print('along  (%.4f, %.4f)  %s -> %s' % (ax_, ay_, SLAB[SECTS[0]],
                                         SLAB[SECTS[-1]]))
print('cross  (%.4f, %.4f)  positive toward the NORTH shore' % (cx_, cy_))
print('a due-north wind (u=0, v=1) projects to %+.3f here and to %+.3f with '
      'the\n(-u ay + v ax) form in reduce_wind_cove.py -- those call southward'
      ' "north".' % (cy_, ax_))

# ------------------------------------------------------ flood sign check ---
dflux = xr.open_dataset(tef2 / ('hourly_flux_%s_%s_%s.nc'
                                % (args.ds0, args.ds1, gctag)))
SGN = {}
for sn in SECTS:
    q = dflux.qnet.sel(sect=sn).values
    z = dflux.ssh.sel(sect=sn).values
    SGN[sn] = -1.0 if np.corrcoef(q, np.gradient(z))[0, 1] < 0 else 1.0
dflux.close()

# ------------------------------------------- per-face y, h, the N/S pair ---
print('\n--- cross-section geometry, and the depth-matched N/S pair ---')
GEO, PAIR = {}, {}
for sn in SECTS:
    lon = dstr['%s_lon' % sn].values
    lat = dstr['%s_lat' % sn].values
    h = dstr['%s_h' % sn].values
    dd = dstr['%s_dd' % sn].values
    qbar = dstr['%s_qbar' % sn].values.sum(axis=0)
    xs, ys = xy_km(lon, lat)
    y = (xs - xs.mean()) * cx_ + (ys - ys.mean()) * cy_        # km, + = north
    rl = np.corrcoef(y, lat)[0, 1]
    assert rl > 0.999, '%s: y is not northward (corr with lat %.3f)' % (sn, rl)
    GEO[sn] = dict(y=y, h=h, dd=dd, qbar=qbar, lat=lat)

    # one face each side of the transport sign change, depth-matched and far
    # enough apart to be a lateral contrast -- the rule from
    # 20260806_pc_sections_series.py, kept identical so the two agree
    iN, iS = np.where(qbar < 0)[0], np.where(qbar > 0)[0]
    span = y.max() - y.min()
    need = args.minsep_face * span
    dh, kN, kS = args.dhmax, None, None
    while kN is None and dh <= span * 1e9:
        best = -np.inf
        for a in iN:
            for b in iS:
                if y[a] - y[b] < need or abs(h[a] - h[b]) > dh:
                    continue
                if abs(qbar[a]) + abs(qbar[b]) > best:
                    best, kN, kS = abs(qbar[a]) + abs(qbar[b]), a, b
        if kN is None:
            dh += 0.25
    PAIR[sn] = (kN, kS, dh)
    nsign = int(np.sum(np.diff(np.sign(qbar)) != 0))
    print('  %-6s width %.2f km, corr(y,h) = %+.2f, h %.1f-%.1f m, qbar '
          'changes sign %d time(s)'
          % (sn, dd.sum() / 1e3, np.corrcoef(y, h)[0, 1], h.min(), h.max(),
             nsign))
    print('         N/S pair p=%d (lat %.4f, h %.1f) vs p=%d (lat %.4f, h '
          '%.1f), tol %.2f m'
          % (kN, lat[kN], h[kN], kS, lat[kS], h[kS], dh))
    print('         qbar per face, north -> south: %s'
          % ' '.join('%+.0f' % v for v in qbar))
dstr.close()

# ------------------------------------------------- per-section reduction ---
print('\n--- reducing sections (top/bottom %.1f m layers, per FACE) ---'
      % args.hlay)
S = pd.DataFrame()
PROF = {}
for sn in SECTS:
    g = GEO[sn]
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    tt = pd.DatetimeIndex(ds.time.values)
    DZ = ds.DZ.values
    salt = ds.salt.values.astype(float)
    dd3 = g['dd'][None, None, :]

    cum_hi = np.cumsum(DZ, axis=1)
    cum_lo = cum_hi - DZ
    H = cum_hi[:, -1:, :]
    w = {'bot': np.clip(args.hlay - cum_lo, 0, DZ) / DZ,
         'top': np.clip(args.hlay - (H - cum_hi), 0, DZ) / DZ,
         'bar': np.ones_like(DZ)}

    F = {}
    for nm in ['top', 'bot', 'bar']:
        A = dd3 * DZ * w[nm]                         # (t,z,p)
        F[nm] = (salt * A).sum(axis=1) / A.sum(axis=1)   # (t,p) face means
    ds.close()

    # the two designs: y alone, and y with depth controlled
    ones = np.ones_like(g['y'])
    Py, Hy = wls(np.column_stack([ones, g['y']]), g['dd'])
    Pyh, _ = wls(np.column_stack([ones, g['y'], g['h']]), g['dd'])
    wsum = g['dd'].sum()

    if S.empty:
        S = pd.DataFrame(index=tt)
    kN, kS, _ = PAIR[sn]
    PROF[sn] = {}
    for nm in ['top', 'bot', 'bar']:
        M = F[nm]                                    # (t,p)
        S['%s_%s_dy' % (sn, nm)] = godin(M @ Py.T[:, 1])
        S['%s_%s_dy_h' % (sn, nm)] = godin(M @ Pyh.T[:, 1])
        fit = M @ Hy.T
        mu = (M * g['dd']).sum(axis=1) / wsum
        ssr = ((M - fit) ** 2 * g['dd']).sum(axis=1)
        sst = ((M - mu[:, None]) ** 2 * g['dd']).sum(axis=1)
        S['%s_%s_r2' % (sn, nm)] = godin(1 - ssr / np.maximum(sst, 1e-12))
        S['%s_%s_ns' % (sn, nm)] = godin(M[:, kN] - M[:, kS])
        # per-face profile, as an anomaly from the section's width-weighted
        # mean at the same hour: the seasonal cycle is common to every face
        # and thirty times bigger than the lateral structure
        Anom = godin(M - mu[:, None])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            PROF[sn][nm] = dict(mean=np.nanmean(Anom, axis=0),
                                lo=np.nanpercentile(Anom, 25, axis=0),
                                hi=np.nanpercentile(Anom, 75, axis=0))
    print('  %-6s %2d faces' % (sn, len(g['y'])))

D = S.resample('1D').mean()
D.to_csv(out_dir / ('dsdy_series_%s_%s.csv' % (args.ds0, args.ds1)))

# ------------------------------------------------------------- the table ---
print('\n--- time-mean cross-channel gradients (g kg-1 km-1), + = the NORTH '
      'side is saltier ---')
print('%-9s %-5s %9s %9s %9s %8s'
      % ('sect', 'layer', 'ds/dy', 'on h', 'N-S[g/kg]', 'R2'))
for sn in SECTS:
    for nm, _, _ in [('top', 0, 0), ('bot', 0, 0), ('bar', 0, 0)]:
        print('%-9s %-5s %+9.3f %+9.3f %+9.3f %8.2f'
              % (SLAB[sn] if nm == 'top' else '', nm,
                 S['%s_%s_dy' % (sn, nm)].mean(),
                 S['%s_%s_dy_h' % (sn, nm)].mean(),
                 S['%s_%s_ns' % (sn, nm)].mean(),
                 S['%s_%s_r2' % (sn, nm)].mean()))
print("  'on h' is the partial slope from a fit on [1, y, h]: the lateral")
print('  gradient at fixed depth. Where it and ds/dy disagree, the raw slope')
print('  is mostly bathymetry. A low R2 means the lateral structure is not a')
print('  ramp -- read the profile figure and the N-S pair instead.')

print('\n--- annual means, no pooling: surface / bottom-on-h (g kg-1 km-1) ---')
YRS = sorted(set(D.index.year))
print('%-7s %s' % ('sect', ' '.join('%17d' % y for y in YRS)))
for sn in SECTS:
    cells = []
    for y in YRS:
        gy = D[D.index.year == y]
        cells.append('%+8.3f %+8.3f' % (gy['%s_top_dy' % sn].mean(),
                                        gy['%s_bot_dy_h' % sn].mean()))
    print('%-7s %s' % (SLAB[sn], ' '.join(cells)))

# ====================================================== figure 1: series ===
sub = S.dropna(subset=['%s_top_dy' % SECTS[0]])
fig, axs = plt.subplots(len(SECTS), 1, figsize=(14, 4 * len(SECTS)),
                        sharex=True, layout='constrained')
for ax, sn in zip(np.atleast_1d(axs), SECTS):
    ax.plot(sub.index, sub['%s_top_dy' % sn], lw=1.3, color=CB['blue'],
            label='surface, raw')
    ax.plot(sub.index, sub['%s_bot_dy' % sn], lw=1.0, color=pale(CB['orange'],
                                                                 0.55),
            label='bottom, RAW (depth confounded)')
    ax.plot(sub.index, sub['%s_bot_dy_h' % sn], lw=1.3, color=CB['orange'],
            label='bottom, h controlled')
    ax.plot(sub.index, sub['%s_bar_dy_h' % sn], lw=1.1, color='k',
            label='depth-mean, h controlled')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_ylabel('%s\nds/dy (g kg$^{-1}$ km$^{-1}$)' % SLAB[sn])
    ax.legend(fontsize=8, ncol=4, loc='upper left')
    ax.grid(**GRID)
np.atleast_1d(axs)[0].set_title(
    'Penn Cove cross-channel salinity gradient at each section, %s, %s to %s'
    % (args.gtx, args.ds0, args.ds1), fontsize=11)
axl_ = np.atleast_1d(axs)[-1]
axl_.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axl_.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.text(0.01, -0.02,
         'y positive toward the NORTH shore, so ds/dy < 0 means a fresher '
         'north side.\nThe raw bottom slope is drawn pale because depth is '
         'correlated with lateral position; the h-controlled one is the '
         'lateral gradient at fixed depth, and is the one to quote.',
         fontsize=8, color='0.3', va='top')
fn = out_dir / ('dsdy_series_%s_%s.png' % (args.ds0, args.ds1))
fig.savefig(fn, dpi=170, bbox_inches='tight', transparent=True)
plt.close(fig)
print('\nSaved ' + str(fn))

# ================================================== figure 2: year by year ===
YC = {y: c for y, c in zip(YRS, ['#4C4C4C', CB['red'], CB['yellow'],
                                 CB['pink'], CB['blue']])}
MON0 = [pd.Timestamp('2001-%02d-01' % m).dayofyear for m in range(1, 13)]
COLS = [('top_dy', 'surface, raw'), ('bot_dy_h', 'bottom, h controlled'),
        ('bar_dy_h', 'depth-mean, h controlled')]

fig, axs = plt.subplots(len(SECTS), len(COLS),
                        figsize=(5.2 * len(COLS), 3.2 * len(SECTS)),
                        sharex=True, squeeze=False, layout='constrained')
for i, sn in enumerate(SECTS):
    for j, (c_, lab) in enumerate(COLS):
        ax = axs[i][j]
        for y in YRS:
            gy = D[D.index.year == y]
            v = gy['%s_%s' % (sn, c_)]
            if v.notna().sum() < 10:
                continue
            ax.plot(gy.index.dayofyear, v.values, lw=1.1, color=YC[y],
                    label='%d (%+.3f)' % (y, v.mean()))
        ax.axhline(0, color='0.5', lw=0.8)
        ax.grid(**GRID)
        if i == 0:
            ax.set_title(lab, fontsize=10)
        if j == 0:
            ax.set_ylabel('%s\nds/dy (g kg$^{-1}$ km$^{-1}$)' % SLAB[sn])
        ax.legend(fontsize=7, ncol=len(YRS), loc='best')
        ax.set_xlim(1, 366)
        ax.set_xticks(MON0)
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O',
                            'N', 'D'])
for j in range(len(COLS)):
    axs[-1][j].set_xlabel('day of year')
fig.suptitle('ds/dy day by day, each year on its own -- %s' % args.gtx,
             fontsize=11)
fn = out_dir / ('dsdy_byyear_%s_%s.png' % (args.ds0, args.ds1))
fig.savefig(fn, dpi=170, bbox_inches='tight', transparent=True)
plt.close(fig)
print('Saved ' + str(fn))

# ============================================== figure 3: face profiles ===
# What the slope is a summary OF. Read this before quoting any ds/dy: at the
# head the profile is not a ramp, and the grey depth curve behind it is the
# confound the h-controlled slope removes.
fig, axs = plt.subplots(1, len(SECTS), figsize=(5.0 * len(SECTS), 5.2),
                        squeeze=False, layout='constrained')
for j, sn in enumerate(SECTS):
    ax = axs[0][j]
    g = GEO[sn]
    kN, kS, _ = PAIR[sn]
    a2 = ax.twinx()
    a2.plot(g['y'], -g['h'], lw=1.2, color='0.75', zorder=0)
    a2.fill_between(g['y'], -g['h'], -g['h'].max() * 1.05, color='0.93',
                    lw=0, zorder=0)
    a2.set_ylabel('bed depth (m, grey)', color='0.55', fontsize=9)
    a2.tick_params(axis='y', colors='0.55', labelsize=8)
    for nm, c_, lab in LAY:
        P = PROF[sn][nm]
        ax.fill_between(g['y'], P['lo'], P['hi'], color=pale(c_), lw=0,
                        zorder=1)
        ax.plot(g['y'], P['mean'], 'o-', color=c_, lw=1.7, ms=5, zorder=3,
                label='%s (R$^2$ %.2f)' % (lab, S['%s_%s_r2' % (sn, nm)].mean()))
    ax.axhline(0, color='0.5', lw=0.8, zorder=2)
    for k, mk in [(kN, 'N'), (kS, 'S')]:
        ax.annotate(mk, (g['y'][k], 0.955), xycoords=('data', 'axes fraction'),
                    fontsize=9, color=CB['purple'], ha='center', weight='bold',
                    bbox=dict(fc='white', ec='none', pad=0.6))
    ax.set_zorder(a2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel('y, south $\\to$ NORTH (km)')
    if j == 0:
        ax.set_ylabel('face salinity minus the section mean (g kg$^{-1}$)')
    ax.set_title('%s (%s)\nds/dy surf %+.3f, bot %+.3f (on h %+.3f)'
                 % (SLAB[sn], sn, S['%s_top_dy' % sn].mean(),
                    S['%s_bot_dy' % sn].mean(),
                    S['%s_bot_dy_h' % sn].mean()), fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(**GRID)
fig.suptitle('time-mean lateral salinity structure across each section '
             '(band = interquartile range; N/S = the depth-matched pair)',
             fontsize=11)
fn = out_dir / ('dsdy_profile_%s_%s.png' % (args.ds0, args.ds1))
fig.savefig(fn, dpi=170, bbox_inches='tight', transparent=True)
plt.close(fig)
print('Saved ' + str(fn))
