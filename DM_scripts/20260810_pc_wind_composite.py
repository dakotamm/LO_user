"""
What the Penn Cove circulation actually looks like under up-cove vs down-cove wind.

20260810_pc_turning_wind.py established the numbers: up-cove wind weakens the
lateral limb by ~33 m3/s per m/s at a one-day lag, which retracts the fixed
transport contours ~0.5 km toward the mouth, while the normalised decay shape
(x50) barely moves. This script draws the maps behind those numbers, so the
mechanism can be seen rather than inferred from three correlation
coefficients.

COMPOSITING RULE. Days are classified by the 30-day anomaly of the along-cove
wind on the PREVIOUS day (the +1 d lag the correlation analysis found), not by
the wind that same day, and not by the raw wind -- the raw along-cove wind and
the cove's circulation share a large seasonal cycle, and compositing on it
would draw a picture of summer minus winter. Thresholds are +/- 1 sd of that
anomaly, with a near-zero class kept as a reference so the two extremes can be
read against something rather than only against each other.

WHAT THE PANELS TEST. The difference map (up minus down) is the wind's own
signature. It is drawn three ways -- depth-averaged, top third, bottom third --
because the mechanism proposed for the weakening is that up-cove wind drives a
competing along-cove overturning (surface in, deep out) that eats into the
lateral gyre. If that is right, the top-third and bottom-third difference maps
must show opposing along-cove flow. If they instead show the same thing, the
wind is simply spinning the lateral gyre down and the overturning story is
wrong.

Cells are stippled where the difference clears +/- 2 standard errors of the
difference, so what is being read is separated from what is noise.

Everything is on the daily Godin-filtered (subtidal) fields from
20260807_pc_turning_reduce.py -- the tidal band carries 86% of the turning
point's variance and would otherwise dominate any composite.

Runs on the mac.
run 20260810_pc_wind_composite.py
run 20260810_pc_wind_composite.py -sd 1.5 -lag 2
"""
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-f', '--fn', default='turning_his_wb1_t0_xn11abbur00_'
                                     '2024.01.01_2025.12.31.p')
p.add_argument('-tz', default='America/Los_Angeles')
p.add_argument('-lag', default=1, type=int,
               help='days the wind leads the response')
p.add_argument('-sd', default=1.0, type=float,
               help='threshold in standard deviations of the wind anomaly')
p.add_argument('-qabs', default=150.0, type=float,
               help='fixed transport contour to mark, m3/s')
args = p.parse_args()
warnings.simplefilter('ignore', RuntimeWarning)

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20260807_pc_turning'
wind_dir = Ldir['LOo'] / 'DM_outs' / '20260806_wind'

C_UP, C_DOWN, C_MID = '#0072B2', '#D55E00', '#6b6b6b'
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
ROLL = 30

D = pd.read_pickle(out_dir / args.fn)
UM, VM = D['UM'], D['VM']
lonc, lonr, latr = D['lon_u'][0], D['lon_rho'], D['lat_rho']
iu = list(D['iu_glob'])
im = iu.index(D['mouth_iu'])
cove, h = D['cove'], D['h']
KMDEG = 111.32 * np.cos(np.deg2rad(np.nanmean(D['lat_u'])))
nface = UM.sum(axis=0)
COL = np.where(nface >= 4)[0]
idx = pd.DatetimeIndex(D['time']).tz_localize('UTC').tz_convert(args.tz)

# --- the same recirculating decomposition the metrics are built on ----------
zbar = float(D['T'].zeta.mean())
h_u = 0.5 * (h[:, :-1] + h[:, 1:])
A_FACE = np.where(UM, D['DYU'] * (h_u + zbar), np.nan)
W_FACE = A_FACE / np.nansum(A_FACE, axis=0)[None, :]
QU2_raw = np.where(UM[None], D['QU2'], np.nan)
QU2 = QU2_raw - np.nansum(QU2_raw, axis=1)[:, None, :] * W_FACE[None]
QV2 = np.where(VM[None], D['QV2'], np.nan)
qbar = np.nanmean(QU2, axis=0)
JS = np.array([int(np.nanargmax([-np.nansum(qbar[j:, c])
                                 for j in range(qbar.shape[0])]))
               for c in range(qbar.shape[1])])
NMASK = np.arange(QU2.shape[1])[None, :, None] >= JS[None, None, :]
Q_N = np.nansum(np.where(NMASK, QU2, 0), axis=1)          # (nt, nu)


def x_at(qn, T):
    """Longitude where the limb first falls below a fixed transport."""
    P = -np.sign(qn[im]) * (-qn)
    if not np.isfinite(P[im]) or P[im] < T:
        return lonc[im]
    for k in range(len(COL) - 1, 0, -1):
        c, cw = COL[k], COL[k - 1]
        if P[c] >= T > P[cw]:
            return lonc[c] + (P[c] - T) / (P[c] - P[cw]) * (lonc[cw] - lonc[c])
    return lonc[COL[0]]


# ------------------------------------------------------------------- wind ---
Wd = pd.read_csv(wind_dir / 'daily_wind.csv')
Wd.index = pd.to_datetime(Wd[Wd.columns[0]], utc=True).dt.tz_convert(args.tz)
al = Wd.along_pc
anom = al - al.rolling(ROLL, center=True, min_periods=ROLL // 2).mean()
# the response on day t follows the wind on day t-lag
wind = anom.reindex(idx.floor('D') - pd.Timedelta(days=args.lag)).values
sd = np.nanstd(wind)

CLS = {'down-cove': wind < -args.sd * sd,
       'near zero': np.abs(wind) < 0.5 * sd,
       'up-cove': wind > args.sd * sd}
# the limb as an anomaly too: up- and down-cove wind days are both drawn from
# the windy half of the year, so a RAW composite of either against the calm
# class carries a season with it. The up-minus-down difference does not.
limb_raw = -Q_N[:, im]
limb_an = pd.Series(limb_raw, index=idx)
limb_an = (limb_an - limb_an.rolling(ROLL, center=True,
                                     min_periods=ROLL // 2).mean()).values
print('along-cove wind anomaly sd = %.2f m/s, lag %d d' % (sd, args.lag))
print('  %-10s %5s %10s %10s %10s %9s'
      % ('class', 'days', 'wind anom', 'limb', 'limb anom', 'x@%.0f' % args.qabs))
for nm, m in CLS.items():
    print('  %-10s %5d %+10.2f %10.0f %+10.0f %+9.2f'
          % (nm, m.sum(), np.nanmean(wind[m]), np.nanmean(limb_raw[m]),
             np.nanmean(limb_an[m]),
             (x_at(np.nanmean(Q_N[m], axis=0), args.qabs) - lonc[im]) * KMDEG))
print('  limb is m3/s, wind m/s, x in km west of the mouth. Compare up-cove '
      'with down-cove:\n  the calm class sits in a different part of the year '
      'and its raw limb is not a fair baseline.')


def vel(mask, key_u, key_v):
    """Composite depth-averaged velocity at rho points, m/s, and its SE."""
    Du = 0.5 * (h[:, :-1] + h[:, 1:]) + zbar
    Dv = 0.5 * (h[:-1, :] + h[1:, :]) + zbar
    fu = np.where(UM[None], D[key_u], np.nan) / (D['DYU'] * Du)[None]
    fv = np.where(VM[None], D[key_v], np.nan) / (D['DXV'] * Dv)[None]
    if key_u != 'QU2':                     # a third of the column, a third of
        fu, fv = fu * 3, fv * 3            # the area: rescale to a velocity

    def to_rho(au, av):
        u = np.full(h.shape, np.nan)
        v = np.full(h.shape, np.nan)
        u[:, 1:-1] = 0.5 * (np.nan_to_num(au[:, :-1]) + np.nan_to_num(au[:, 1:]))
        v[1:-1, :] = 0.5 * (np.nan_to_num(av[:-1, :]) + np.nan_to_num(av[1:, :]))
        u[~cove], v[~cove] = np.nan, np.nan
        return u, v

    n = max(int(mask.sum()), 1)
    um, vm = to_rho(np.nanmean(fu[mask], axis=0), np.nanmean(fv[mask], axis=0))
    us, vs = to_rho(np.nanstd(fu[mask], axis=0) / np.sqrt(n),
                    np.nanstd(fv[mask], axis=0) / np.sqrt(n))
    return um, vm, us, vs


# ------------------------------------------------------------------- plot ---
fig = plt.figure(figsize=(16.5, 9.5), layout='constrained')
gs = fig.add_gridspec(2, 3)
SC = 0.30                                   # common quiver scale, m/s


def draw_map(ax, u, v, title, cmax=0.02, se=None, ref=True):
    pc = ax.pcolormesh(lonr, latr, np.where(cove, u, np.nan), cmap='RdBu_r',
                       vmin=-cmax, vmax=cmax, shading='nearest')
    plt.colorbar(pc, ax=ax, label='eastward velocity (m s$^{-1}$)')
    q = ax.quiver(lonr, latr, u, v, scale=SC, width=0.0028, color='k')
    if ref:
        ax.quiverkey(q, 0.86, 0.06, 0.02, '2 cm s$^{-1}$', labelpos='E',
                     fontproperties=dict(size=8))
    if se is not None:
        sig = np.hypot(u, v) > 2 * np.hypot(*se)
        ax.plot(lonr[sig], latr[sig], '.', ms=1.2, color='0.25')
    ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('longitude')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    return pc


COMP = {}
for j, nm in enumerate(['down-cove', 'up-cove']):
    m = CLS[nm]
    um, vm, us, vs = vel(m, 'QU2', 'QV2')
    COMP[nm] = (um, vm, us, vs)
    ax = fig.add_subplot(gs[0, j])
    draw_map(ax, um, vm, '%s wind, %d days\nlimb %.0f m$^3$ s$^{-1}$ '
             '(anomaly %+.0f)' % (nm, m.sum(), np.nanmean(limb_raw[m]),
                                  np.nanmean(limb_an[m])))
    if j == 0:
        ax.set_ylabel('latitude')
    for nm2, c in [('down-cove', C_DOWN), ('up-cove', C_UP)]:
        ax.axvline(x_at(np.nanmean(Q_N[CLS[nm2]], axis=0), args.qabs),
                   color=c, lw=2, ls='-' if nm2 == nm else ':')

ax = fig.add_subplot(gs[0, 2])
du = COMP['up-cove'][0] - COMP['down-cove'][0]
dv = COMP['up-cove'][1] - COMP['down-cove'][1]
dse = (np.hypot(COMP['up-cove'][2], COMP['down-cove'][2]),
       np.hypot(COMP['up-cove'][3], COMP['down-cove'][3]))
draw_map(ax, du, dv, 'up minus down: the wind\'s own signature\n'
                     'dots = clears 2 SE of the difference', cmax=0.02, se=dse)

for j, (ku, kv, lab) in enumerate([('QU2_top', 'QV2_top', 'top third'),
                                   ('QU2_bot', 'QV2_bot', 'bottom third')]):
    a = vel(CLS['up-cove'], ku, kv)
    b = vel(CLS['down-cove'], ku, kv)
    ax = fig.add_subplot(gs[1, j])
    draw_map(ax, a[0] - b[0], a[1] - b[1],
             'up minus down, %s' % lab, cmax=0.03,
             se=(np.hypot(a[2], b[2]), np.hypot(a[3], b[3])))
    if j == 0:
        ax.set_ylabel('latitude')

# --- the along-cove profiles the maps are summarised by
ax = fig.add_subplot(gs[1, 2])
for nm, c in [('down-cove', C_DOWN), ('near zero', C_MID), ('up-cove', C_UP)]:
    m = CLS[nm]
    q = np.nanmean(Q_N[m], axis=0)[COL]
    se = (np.nanstd(Q_N[m], axis=0) / np.sqrt(m.sum()))[COL]
    ax.plot(lonc[COL], q, '-', color=c, lw=2, label='%s (%d d)' % (nm, m.sum()))
    ax.fill_between(lonc[COL], q - 2 * se, q + 2 * se, color=c, alpha=0.15, lw=0)
ax.axhline(0, color='0.5', lw=0.8)
ax.axhline(-args.qabs, color='0.4', lw=1, ls=':')
ax.text(lonc[COL[0]], -args.qabs, ' %.0f m$^3$ s$^{-1}$' % args.qabs,
        fontsize=7, va='bottom', color='0.3')
for nm, c in [('down-cove', C_DOWN), ('up-cove', C_UP)]:
    ax.axvline(x_at(np.nanmean(Q_N[CLS[nm]], axis=0), args.qabs), color=c,
               lw=1.5, ls='--')
ax.set_xlabel('longitude')
ax.set_ylabel('north-limb transport (m$^3$ s$^{-1}$)')
ax.set_title('the limb along the cove\nband = +/- 2 SE; dashed = the %.0f '
             'm$^3$ s$^{-1}$ contour' % args.qabs, fontsize=10)
ax.grid(**GRID)
ax.legend(fontsize=8)

fig.suptitle('Penn Cove subtidal circulation composited on the along-cove wind '
             'anomaly (%s sd, %d-day lag) -- %s'
             % (args.sd, args.lag, D['info']['gtx']), fontsize=12)
fn = out_dir / 'fig7_wind_composite.png'
fig.savefig(fn, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn)

# ---------------------------------------------------------------- numbers ---
rows = []
for nm, m in CLS.items():
    q = np.nanmean(Q_N[m], axis=0)
    for ku, lab in [('QU2', 'depth-averaged'), ('QU2_top', 'top third'),
                    ('QU2_bot', 'bottom third')]:
        f = np.where(UM[None], D[ku], np.nan)
        rows.append(dict(
            wind=nm, layer=lab, n_days=int(m.sum()),
            mouth_net=float(np.nanmean(np.nansum(f[m], axis=1)[:, im])),
            mouth_north=float(np.nanmean(
                np.nansum(np.where(NMASK, f[m], 0), axis=1)[:, im])),
            limb=float(-q[im]), limb_anom=float(np.nanmean(limb_an[m]))))
S = pd.DataFrame(rows)
S.to_csv(out_dir / 'wind_composite_transports.csv', index=False,
         float_format='%.2f')
print('\nmouth transports by wind class (m3/s, + = out of the cove):')
print(S.to_string(index=False, float_format=lambda v: '%.1f' % v))
print('\nmouth_net is the whole section, mouth_north the north half. In the '
      'top/bottom rows a\nnet that flips sign between the layers IS the '
      'along-cove overturning the wind drives.')
