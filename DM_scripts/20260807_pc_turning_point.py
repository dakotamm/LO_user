"""
Where along Penn Cove does the inflow turn around, and what moves that point?

Reads the pickle written by 20260807_pc_turning_reduce.py and answers the
question in longitude: scanning west from the mouth, how far does the
inflowing limb get before it has crossed to the south side and started back
out, and how does that longitude vary with the tide, the season, and the wind.

HOW THE TURNING POINT IS DEFINED, AND WHY NOT BY A ZERO CROSSING.
Split every cross-cove section at the latitude where the mean row transport
changes sign, giving the north-limb transport Q_N(x) (negative = into the
cove) and the south limb Q_S(x). The obvious definition of the turning point
is where Q_N(x) crosses zero, and in the wb1 run IT USUALLY DOES NOT: Q_N
weakens westward to a pinch off Coupeville and then grows again, because the
inner cove runs its own smaller circulation of the same sense. A zero crossing
would be NaN most days and would throw away the fact that 90% of the inflow
has already turned by then.

So the primary metric is a PENETRATION longitude, which needs no zero:

    x_f  =  the longitude, scanning west from the mouth, where |Q_N| has
            first fallen to (1 - f) of its value at the mouth.

x50 is where half the inflowing limb has turned south and headed back out;
x90 is the effective head of the limb. Both are linearly interpolated between
columns, so they move continuously rather than hopping cell to cell. The zero
crossing is still reported (x_zero) for the days it exists, and so is the
longitude of the pinch (x_min), which is a property of the geometry more than
of the flow.

    dQ_N/dx = -V_split(x)

is continuity along the north half: the along-cove decay of the limb IS the
cross-cove transport, so x50 can be read equivalently as the median longitude
of the southward crossing. That is computed too (x_Vcent) as an independent
check on the same number.

LATERAL OR VERTICAL? The same treatment is applied to the bottom third against
the top third of the water column, giving x50_bot for a classical estuarine
exchange. Which decomposition the cove actually runs is settled per column by
the recirculating transport it accounts for, E_lat vs E_vert, in figure 1 --
not assumed.

WHEN THE METRIC IS UNDEFINED. On days when the limb at the mouth is weak or
reversed (Q_N >= -QMIN) there is no inflowing limb to follow and x_f is NaN.
Those days are counted and plotted, not silently dropped -- a reversed gyre is
part of the variation being asked about.

Everything subtidal is on Godin-filtered daily fields. If the pickle also
carries the hourly transports, the same metrics are computed hour by hour and
composited on the ebb-flood cycle (figure 4), which is a different question:
not where the residual limb ends, but how the turning point sweeps back and
forth within a tidal cycle.

Runs on the mac.
run 20260807_pc_turning_point.py
run 20260807_pc_turning_point.py -f turning_his_wb1_r0_xn11b_2017.09.04_2017.09.18.p
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.signal import find_peaks
from scipy.stats import t as tdist

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-f', '--fn', default='turning_his_wb1_t0_xn11abbur00_'
                                      '2024.01.01_2025.12.31.p')
p.add_argument('-tz', default='America/Los_Angeles')
p.add_argument('-qmin', default=50.0, type=float,
               help='m3/s of north-limb inflow at the mouth below which the '
                    'turning point is undefined')
p.add_argument('-nface', default=4, type=int,
               help='skip columns with fewer than this many wet faces')
p.add_argument('--no-utide', dest='utide', action='store_false',
               help='skip the harmonic analysis of the turning point position')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20260807_pc_turning'
D = pd.read_pickle(out_dir / args.fn)
INFO = D['info']
GTX = INFO['gtx']

C_IN, C_OUT = '#0072B2', '#D55E00'          # inflow / outflow, colourblind safe
C_LAT, C_VERT = '#009E73', '#CC79A7'        # lateral / vertical decomposition
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FLEV = [0.5, 0.9]                           # penetration fractions to track


def savefig(fig, name):
    fn = out_dir / name
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    print('saved %s' % fn)
    plt.close(fig)


# ------------------------------------------------------------------ setup ---
UM, VM = D['UM'], D['VM']
lonc = D['lon_u'][0]                        # one longitude per u-face column
lon_rho_c = D['lon_rho'][0]
latu = D['lat_u'][:, 0]
iu, mouth_iu = D['iu_glob'], D['mouth_iu']
nface = UM.sum(axis=0)
COL = np.where(nface >= args.nface)[0]      # usable columns, west to east
im = int(list(iu).index(mouth_iu))
KMDEG = 111.32 * np.cos(np.deg2rad(np.nanmean(latu)))
km = (lonc - lonc[im]) * KMDEG              # negative = west of the mouth
print('%s (%s, %s to %s)\n%d columns with >= %d faces, mouth at lon %.4f'
      % (GTX, INFO['source'], INFO['date0'], INFO['date1'], len(COL),
         args.nface, lonc[im]))
if INFO.get('stokes_missing'):
    print('** this pickle came from lowpassed.nc: the transports are missing '
          'the tidal Stokes term **')
print('continuity at the mouth in the reduce: r = %+.3f, %.0f%% of dV/dt '
      'unexplained' % (INFO.get('continuity_r', np.nan),
                       100 * INFO.get('continuity_residual', np.nan)))


def mask_rows(a):
    """(nt, nj, nu) with non-cove faces NaN rather than the nansum's zero."""
    return np.where(UM[None, :, :], a, np.nan)


def split_rows(qbar):
    """Row index that splits each column into an inflowing north half and an
    outflowing south half: the row that maximises the north-half inflow, which
    is exactly where the mean row transport changes sign."""
    nj, nu = qbar.shape
    js = np.zeros(nu, int)
    for c in range(nu):
        cum = np.array([-np.nansum(qbar[j:, c]) for j in range(nj)])
        js[c] = int(np.nanargmax(cum)) if np.isfinite(cum).any() else 0
    return js


def limbs(Q2, js):
    """(north, south) half transports per column, (nt, nu)."""
    m = np.arange(Q2.shape[1])[None, :, None] >= js[None, None, :]
    return (np.nansum(np.where(m, Q2, 0), axis=1),
            np.nansum(np.where(~m, Q2, 0), axis=1))


# --- the recirculating (exchange) part of every section --------------------
# The turning point has to mean the same thing hour by hour as it does day by
# day, or the tidal and subtidal excursions are not comparable. Raw hourly
# transports will not do that: at peak flood the whole section runs into the
# cove, there is no north limb to follow, and x50 goes undefined (it was NaN
# on 31% of hours). So the net through each column is removed first,
#
#     q'(j) = q(j) - Q_net * A(j) / A_tot,          sum_j q'(j) = 0
#
# leaving the part that circulates. Subtidally Q_net ~ 0 and this changes
# almost nothing; at peak flood it is the whole point. Face areas are the
# record-mean ones (h_u + mean zeta): the tide moves them ~10%, which is
# second order in a term whose job is to be removed.
h_u = 0.5 * (D['h'][:, :-1] + D['h'][:, 1:])
A_FACE = np.where(UM, D['DYU'] * (h_u + float(D['T'].zeta.mean())), np.nan)
W_FACE = A_FACE / np.nansum(A_FACE, axis=0)[None, :]


def recirc(Q2):
    """Row transports with the column's net removed, distributed by area."""
    net = np.nansum(Q2, axis=1)
    return Q2 - net[:, None, :] * W_FACE[None, :, :]


QU2_raw = mask_rows(D['QU2'])
QU2 = recirc(QU2_raw)
qbar = np.nanmean(QU2, axis=0)
JS = split_rows(qbar)
lat_split = np.array([0.5 * (latu[JS[c]] + latu[max(JS[c] - 1, 0)])
                      for c in range(len(lonc))])

Q_N, Q_S = limbs(QU2, JS)
Q_N_raw, _ = limbs(QU2_raw, JS)
Q_TOT = np.nansum(QU2_raw, axis=1)                  # the net, kept as it is
# the vertical bands carry exactly a third of the face area each, so the same
# removal is just the column net split three ways
Q_BOT = np.nansum(mask_rows(D['QU2_bot']), axis=1) - Q_TOT / 3
Q_TOP = np.nansum(mask_rows(D['QU2_top']), axis=1) - Q_TOT / 3

# How much transport circulates per column, across the section (lateral) and
# down it (vertical). Whichever is larger is the circulation the cove actually
# runs. E_VERT is left on the raw per-level transports -- the per-level area
# shares are not stored to remove the net properly -- which is fair here
# because these are the subtidal fields, where the net is small anyway.
E_LAT = 0.5 * np.nansum(np.abs(QU2), axis=1)
E_VERT = 0.5 * np.nansum(np.abs(D['QUkx']), axis=1)

# cross-cove transport through the split line, on rho columns
QV2 = np.where(VM[None, :, :], D['QV2'], np.nan)
js_rho = np.zeros(len(lon_rho_c), int)
for i in range(len(lon_rho_c)):
    cs = [c for c in (i - 1, i) if 0 <= c < len(JS)]
    js_rho[i] = int(np.round(np.mean([JS[c] for c in cs])))
V_SPLIT = np.array([QV2[:, min(js_rho[i] - 1, QV2.shape[1] - 1), i]
                    for i in range(len(lon_rho_c))]).T      # (nt, ni)

time = D['time']
idx = pd.DatetimeIndex(time).tz_localize('UTC').tz_convert(args.tz)


# -------------------------------------------------------- turning metrics ---
def penetration(qn, f):
    """(longitude, status) where the mouth's limb has decayed to (1-f) of its
    strength, scanning west. Linear in longitude between columns.

    The limb is whichever half is flowing IN at the mouth, not the north half
    by assumption. Hourly the lateral exchange reverses for part of the cycle
    (38 of 360 hours in the 2017 test) and a north-half-only definition throws
    those away; taking the sign from the mouth follows the mirrored gyre
    instead and keeps the series continuous. The sign is reported separately
    so gyre and reversed periods stay distinguishable.

    Three outcomes, all of them information:
      ok        the limb decayed inside the cove
      censored  it never did -- it reaches the head, so the head longitude is
                returned rather than a NaN, which would drop exactly the times
                the limb goes deepest and bias the excursion low
      weak      |limb| < qmin: the exchange is passing through zero and there
                is no limb to locate. Genuinely undefined.
    """
    q0 = qn[im]
    if not np.isfinite(q0) or abs(q0) < args.qmin:
        return np.nan, 'weak'
    P = -np.sign(q0) * (-qn)                   # positive at the mouth
    target = (1 - f) * P[im]
    for k in range(len(COL) - 1, 0, -1):
        c, cw = COL[k], COL[k - 1]
        if P[c] <= target and P[cw] > target:
            w = (target - P[c]) / (P[cw] - P[c])
            return lonc[c] + w * (lonc[cw] - lonc[c]), 'ok'
    return lonc[COL[0]], 'censored'


def pen_series(QN, f):
    """penetration() over a (nt, nu) stack -> (longitudes, statuses)."""
    out = [penetration(q, f) for q in QN]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def zero_west(qn):
    """First sign change of Q_N scanning west from the mouth (NaN if none)."""
    if not np.isfinite(qn[im]) or qn[im] >= 0:
        return np.nan
    for k in range(len(COL) - 1, 0, -1):
        c, cw = COL[k], COL[k - 1]
        if qn[c] < 0 <= qn[cw]:
            w = (0 - qn[c]) / (qn[cw] - qn[c])
            return lonc[c] + w * (lonc[cw] - lonc[c])
    return np.nan


def v_centroid(vs):
    """Median longitude of the southward crossing transport, weighted."""
    w = np.clip(-vs, 0, None)
    w = np.where(np.isfinite(w), w, 0)
    if w.sum() <= 0:
        return np.nan
    o = np.argsort(lon_rho_c)
    cw = np.cumsum(w[o]) / w.sum()
    return float(np.interp(0.5, cw, lon_rho_c[o]))


def metrics(qn, qb, vs):
    r = dict(QN_mouth=qn[im], QS_mouth=np.nan, x_zero=zero_west(qn),
             x_min=lonc[COL[int(np.nanargmin(np.abs(qn[COL])))]],
             x_Vcent=v_centroid(vs), limb_sign=-np.sign(qn[im]))
    for f in FLEV:
        r['x%d' % (100 * f)], r['status%d' % (100 * f)] = penetration(qn, f)
        r['x%d_bot' % (100 * f)] = penetration(qb, f)[0]
    return r


X = pd.DataFrame([metrics(Q_N[n], Q_BOT[n], V_SPLIT[n])
                  for n in range(len(idx))], index=idx)
X['QS_mouth'] = Q_S_raw[:, im]
X['Q_net_mouth'] = Q_TOT[:, im]
X['E_lat_mouth'] = E_LAT[:, im]
X['E_vert_mouth'] = E_VERT[:, im]
X['limb'] = -X.QN_mouth                       # positive = inflow on the north
X['state'] = np.where(X.limb > args.qmin, 'gyre',
                      np.where(X.limb < -args.qmin, 'reversed', 'weak'))
for c in [k for k in X.columns if k.startswith('x')]:
    X[c.replace('x', 'km', 1)] = (X[c] - lonc[im]) * KMDEG

X['km50_raw'] = ([penetration(q, 0.5) for q in Q_N_raw] - lonc[im]) * KMDEG
print('\nremoving the net changes the subtidal turning point by a median of '
      '%.3f km\n  (that is the sanity check on the recirculating definition -- '
      'it should be small here\n  and large hourly)'
      % (X.km50 - X.km50_raw).abs().median())
print('\nstate of the mouth limb: %s'
      % ', '.join('%s %.0f%%' % (k, 100 * v)
                  for k, v in X.state.value_counts(normalize=True).items()))
print('turning point (deg lon, and km west of the mouth):')
for f in FLEV:
    k = 'x%d' % (100 * f)
    print('  %s  median %.4f (%+.2f km), IQR %.4f-%.4f, defined %.0f%% of days'
          % (k, X[k].median(), X['km%d' % (100 * f)].median(),
             X[k].quantile(0.25), X[k].quantile(0.75),
             100 * X[k].notna().mean()))
print('  x_zero exists on %.0f%% of days; the pinch x_min sits at %.4f'
      % (100 * X.x_zero.notna().mean(), X.x_min.median()))

# ------------------------------------------------------------- drivers ---
Dd = pd.DataFrame(index=idx.floor('D'))
Dd['limb'] = X.limb.values
for f in FLEV:
    Dd['km%d' % (100 * f)] = X['km%d' % (100 * f)].values

def local_days(a):
    """Any date column -> local midnight, tz-aware. The wind csv carries an
    offset and the phase csv does not, so both are forced through UTC."""
    ix = pd.to_datetime(pd.Index(a), utc=True)
    return ix.tz_convert(args.tz).floor('D')


def join_csv(fn, cols, note=''):
    if not fn.is_file():
        return
    G = pd.read_csv(fn)
    G.index = local_days(G[G.columns[0]])
    keep = [c for c in cols if c in G.columns]
    if keep:
        Dd[keep] = G[keep].reindex(Dd.index)
        print('joined %s%s' % (fn.name, note))


base = Ldir['LOo'] / 'DM_outs'
join_csv(base / '20260806_tidal_phase' / 'phase_daily.csv',
         ['range_smooth_m', 'days_from_syzygy', 'model_phase'])
join_csv(base / '20260806_wind' / 'daily_wind.csv',
         ['along_pc', 'cross_pc', 'spd_pc', 'u_pc', 'v_pc'],
         ' (along_pc > 0 blows up-cove)')
fn_r = base / '20260806_river_hydrographs' / 'daily_flow.csv'
if fn_r.is_file():
    R = pd.read_csv(fn_r)
    R.index = local_days(R[R.columns[0]])
    Dd['river'] = R.select_dtypes('number').sum(axis=1).reindex(Dd.index)
    print('joined %s' % fn_r.name)
Dd['month'] = Dd.index.month


def anom(s, w=30):
    return s - s.rolling(w, center=True, min_periods=w // 2).mean()


def eff_corr(x, y):
    """Pearson r with the p-value corrected for serial correlation."""
    g = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(g) < 12:
        return np.nan, np.nan, np.nan, 0
    r = g.x.corr(g.y)
    slope = np.polyfit(g.x, g.y, 1)[0]
    r1x, r1y = g.x.autocorr(1), g.y.autocorr(1)
    ne = float(np.clip(len(g) * (1 - r1x * r1y) / (1 + r1x * r1y), 3, len(g)))
    ts = r * np.sqrt((ne - 2) / max(1e-12, 1 - r ** 2))
    return r, slope, 2 * (1 - tdist.cdf(abs(ts), ne - 2)), ne


# ================================================== 1. ALONG-COVE STRUCTURE ==
fig = plt.figure(figsize=(15, 12), layout='constrained')
gs = fig.add_gridspec(3, 2, height_ratios=[1.9, 1, 1])

# --- map: mean depth-averaged velocity, with the turning longitudes on it
ax = fig.add_subplot(gs[0, 0])
h, cove = D['h'], D['cove']
zbar = float(D['T'].zeta.mean())
Du = 0.5 * (h[:, :-1] + h[:, 1:]) + zbar
Dv = 0.5 * (h[:-1, :] + h[1:, :]) + zbar
ubar_m = np.nanmean(QU2, axis=0) / (D['DYU'] * Du)
vbar_m = np.nanmean(QV2, axis=0) / (D['DXV'] * Dv)
u_rho = np.full(h.shape, np.nan)
v_rho = np.full(h.shape, np.nan)
u_rho[:, 1:-1] = 0.5 * (np.nan_to_num(ubar_m[:, :-1]) + np.nan_to_num(ubar_m[:, 1:]))
v_rho[1:-1, :] = 0.5 * (np.nan_to_num(vbar_m[:-1, :]) + np.nan_to_num(vbar_m[1:, :]))
u_rho[~cove], v_rho[~cove] = np.nan, np.nan
spd = np.hypot(u_rho, v_rho)
lonr, latr = D['lon_rho'], D['lat_rho']
pc = ax.pcolormesh(lonr, latr, np.where(cove, u_rho, np.nan), cmap='RdBu_r',
                   vmin=-0.02, vmax=0.02, shading='nearest')
plt.colorbar(pc, ax=ax, label='mean eastward velocity (m s$^{-1}$)\n'
                              'red = out of the cove')
ax.quiver(lonr, latr, u_rho, v_rho, scale=0.35, width=0.0022, color='k')
ax.plot(lonc[COL], lat_split[COL], '-', color='0.35', lw=1.4,
        label='north/south split')
for f, ls in zip(FLEV, ['-', '--']):
    xf = X['x%d' % (100 * f)].median()
    ax.axvline(xf, color=C_IN, ls=ls, lw=2,
               label='median x%d = %.4f' % (100 * f, xf))
ax.plot([lonc[im]] * 2, [latu.min(), latu.max()], '-', color=C_OUT, lw=2.5,
        label='mouth (pc_lp)')
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('mean depth-averaged circulation\nwater enters along the north '
             'shore and leaves along the south', fontsize=10)
ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
ax.xaxis.set_major_locator(plt.MaxNLocator(5))

# --- the limbs
ax = fig.add_subplot(gs[0, 1])
for y, c, lab in [(Q_N, C_IN, 'north half (inflow)'),
                  (Q_S, C_OUT, 'south half (outflow)'),
                  (Q_TOT, '0.4', 'net')]:
    m = np.nanmean(y, axis=0)[COL]
    lo, hi = (np.nanpercentile(y[:, COL], 25, axis=0),
              np.nanpercentile(y[:, COL], 75, axis=0))
    ax.plot(lonc[COL], m, '-', lw=2, color=c, label=lab)
    ax.fill_between(lonc[COL], lo, hi, color=c, alpha=0.15, lw=0)
ax.axhline(0, color='0.5', lw=0.8)
for f, ls in zip(FLEV, ['-', '--']):
    ax.axvline(X['x%d' % (100 * f)].median(), color=C_IN, ls=ls, lw=1.5)
ax.set_xlabel('longitude')
ax.set_ylabel('transport (m$^3$ s$^{-1}$)')
ax.set_title('the two limbs along the cove\nband = interquartile range over '
             'the record', fontsize=10)
ax.grid(**GRID)
ax.legend(fontsize=8)

# --- lateral vs vertical
ax = fig.add_subplot(gs[1, 0])
ax.plot(lonc[COL], np.nanmean(E_LAT, axis=0)[COL], '-', lw=2, color=C_LAT,
        label='lateral (across the section)')
ax.plot(lonc[COL], np.nanmean(E_VERT, axis=0)[COL], '-', lw=2, color=C_VERT,
        label='vertical (down the section)')
ax.set_xlabel('longitude')
ax.set_ylabel('recirculating transport (m$^3$ s$^{-1}$)')
ratio = np.nanmean(E_LAT, axis=0)[COL] / np.nanmean(E_VERT, axis=0)[COL]
ax.set_title('which circulation is the cove running?\nlateral / vertical = '
             '%.1f at the mouth, %.1f in the mid cove'
             % (ratio[-1], np.median(ratio)), fontsize=10)
ax.grid(**GRID)
ax.legend(fontsize=8)

# --- cross-cove transport: the turning itself
ax = fig.add_subplot(gs[1, 1])
vm = np.nanmean(V_SPLIT, axis=0)
ax.bar(lon_rho_c, vm, width=0.9 * np.diff(lon_rho_c).mean(),
       color=np.where(vm < 0, C_IN, C_OUT))
ax.axhline(0, color='0.5', lw=0.8)
for f, ls in zip(FLEV, ['-', '--']):
    ax.axvline(X['x%d' % (100 * f)].median(), color=C_IN, ls=ls, lw=1.5)
ax.axvline(X.x_Vcent.median(), color='k', ls=':', lw=1.6,
           label='median crossing = %.4f' % X.x_Vcent.median())
ax.set_xlabel('longitude')
ax.set_ylabel('northward transport across the split (m$^3$ s$^{-1}$)')
ax.set_title('where the water actually crosses the cove\n'
             'blue = southward, i.e. the turn', fontsize=10)
ax.grid(**GRID)
ax.legend(fontsize=8)

# --- the vertical version of the same question
ax = fig.add_subplot(gs[2, 0])
for y, c, lab in [(Q_BOT, '#4565e8', 'bottom third'),
                  (Q_TOP, '#e04256', 'top third')]:
    ax.plot(lonc[COL], np.nanmean(y, axis=0)[COL], '-', lw=2, color=c,
            label=lab)
ax.axhline(0, color='0.5', lw=0.8)
ax.axvline(X['x50_bot'].median(), color='#4565e8', lw=1.5)
ax.set_xlabel('longitude')
ax.set_ylabel('transport (m$^3$ s$^{-1}$)')
ax.set_title('the vertical decomposition, for comparison\nx50_bot = %.4f'
             % X['x50_bot'].median(), fontsize=10)
ax.grid(**GRID)
ax.legend(fontsize=8)

# --- the vertical structure of every section, side by side
ax = fig.add_subplot(gs[2, 1])
Zk = np.nanmean(D['QUkx'], axis=0)[:, COL]
vk = np.nanpercentile(np.abs(Zk), 99)
pk = ax.pcolormesh(lonc[COL], np.arange(Zk.shape[0]), Zk, cmap='RdBu_r',
                   vmin=-vk, vmax=vk, shading='nearest')
plt.colorbar(pk, ax=ax, label='transport per level (m$^3$ s$^{-1}$)')
for f, ls in zip(FLEV, ['-', '--']):
    ax.axvline(X['x%d' % (100 * f)].median(), color='k', ls=ls, lw=1.5)
ax.set_xlabel('longitude')
ax.set_ylabel('sigma level (0 = bed)')
ax.set_title('vertical structure of each section\n'
             'row-summed, so this is the vertical exchange only', fontsize=10)

fig.suptitle('%s -- Penn Cove along-cove structure. The blue/black lines are '
             'where half (solid) and 90%% (dashed) of the inflowing limb has '
             'turned around.' % GTX, fontsize=12)
savefig(fig, 'fig1_along_cove_structure.png')

# ========================================================= 2. HOW IT MOVES ==
fig, axs = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')

A = axs[0][0]
qn = Q_N[:, COL].T
vmax = np.nanpercentile(np.abs(qn), 98)
pcm = A.pcolormesh(idx, lonc[COL], qn, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   shading='nearest')
plt.colorbar(pcm, ax=A, label='north-limb transport (m$^3$ s$^{-1}$)')
for f, c in zip(FLEV, ['k', '0.45']):
    A.plot(idx, X['x%d' % (100 * f)].rolling(5, center=True).mean(), '-',
           color=c, lw=1.0, label='x%d (5-day)' % (100 * f))
A.set_ylabel('longitude')
A.set_title('north-limb transport along the cove, day by day\n'
            'blue = inflow; the lines are where it has decayed', fontsize=10)
A.legend(fontsize=8, loc='lower left')
A.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

B = axs[0][1]
for f, c in zip(FLEV, [C_IN, C_LAT]):
    k = 'km%d' % (100 * f)
    B.plot(idx, X[k], '.', ms=2, alpha=0.3, color=c)
    B.plot(idx, X[k].rolling(15, center=True).mean(), '-', lw=1.8, color=c,
           label='x%d, 15-day mean' % (100 * f))
B.axhline(X.km50.median(), color=C_IN, ls=':', lw=1)
B.set_ylabel('km west of the mouth')
B.set_title('the turning point through the record\n%.0f%% of days have a '
            'defined turning point' % (100 * X.km50.notna().mean()),
            fontsize=10)
B.grid(**GRID)
B.legend(fontsize=8)
B.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

C = axs[1][0]
for f, c in zip(FLEV, [C_IN, C_LAT]):
    v = X['km%d' % (100 * f)].dropna()
    C.hist(v, bins=25, histtype='step', lw=2, color=c, density=True,
           label='x%d: median %+.2f km, sd %.2f' % (100 * f, v.median(),
                                                    v.std()))
C.set_xlabel('km west of the mouth')
C.set_ylabel('density')
C.set_title('how far it ranges', fontsize=10)
C.grid(**GRID)
C.legend(fontsize=8)

E = axs[1][1]
mon = np.arange(1, 13)
E2 = E.twinx()
g = Dd.groupby('month')['limb']
E2.plot(mon, g.mean().reindex(mon).values, '-', color='0.65', lw=1.5)
E2.set_ylabel('mouth limb transport (m$^3$ s$^{-1}$)', color='0.5')
E2.tick_params(axis='y', colors='0.5')
for f, c in zip(FLEV, [C_IN, C_LAT]):
    k = 'km%d' % (100 * f)
    g = Dd.groupby('month')[k]
    E.errorbar(mon, g.mean().reindex(mon).values,
               yerr=(g.std() / np.sqrt(g.count())).reindex(mon).values * 2,
               fmt='o-', color=c, lw=1.8, ms=4, capsize=2, label='x%d' % (100 * f))
E.set_zorder(E2.get_zorder() + 1)
E.patch.set_visible(False)
E.set_xticks(mon)
E.set_xlabel('month')
E.set_ylabel('km west of the mouth')
E.set_title('seasonal cycle (grey = limb strength)\n+/- 2 standard errors',
            fontsize=10)
E.grid(**GRID)
E.legend(fontsize=8, loc='upper left')

fig.suptitle('%s -- where the Penn Cove inflow turns around, and how that '
             'longitude moves' % GTX, fontsize=12)
savefig(fig, 'fig2_turning_point_variation.png')

# ====================================================== 3. WHAT MOVES IT ====
drivers = [(c, lab) for c, lab in
           [('range_smooth_m', 'tidal range (m)'),
            ('along_pc', 'up-cove wind (m s$^{-1}$)'),
            ('river', 'total river flow (m$^3$ s$^{-1}$)')] if c in Dd]
if drivers:
    fig, axs = plt.subplots(2, len(drivers), figsize=(5 * len(drivers), 8.5),
                            squeeze=False, layout='constrained')
    rows = []
    for j, (cn, lab) in enumerate(drivers):
        for row, (yk, ylab) in enumerate([('km50', 'x50 (km west of mouth)'),
                                          ('limb', 'limb transport (m3/s)')]):
            ax = axs[row][j]
            # 30-day anomalies: the fortnightly and event response is a small
            # part of a large seasonal range, so a raw scatter is mostly season
            x, y = anom(Dd[cn]), anom(Dd[yk])
            r, slope, pv, ne = eff_corr(x, y)
            ax.plot(x, y, '.', ms=3, alpha=0.35, color=C_LAT)
            gg = pd.DataFrame({'x': x, 'y': y}).dropna()
            if len(gg) > 12:
                xx = np.linspace(gg.x.min(), gg.x.max(), 10)
                ax.plot(xx, np.polyval(np.polyfit(gg.x, gg.y, 1), xx), '-',
                        color='k', lw=2)
            r0, s0, p0, n0 = eff_corr(Dd[cn], Dd[yk])
            rows.append(dict(driver=cn, response=yk, detrended='30 d anomaly',
                             r=r, slope=slope, p=pv, n_eff=ne))
            rows.append(dict(driver=cn, response=yk, detrended='none', r=r0,
                             slope=s0, p=p0, n_eff=n0))
            ax.axhline(0, color='0.5', lw=0.8)
            ax.axvline(0, color='0.5', lw=0.8)
            ax.set_xlabel('%s anomaly' % lab)
            ax.set_ylabel(ylab if j == 0 else '')
            ax.set_title('r = %+.2f, %.3g per unit (p = %.1e, n_eff = %.0f)\n'
                         'raw r = %+.2f' % (r, slope, pv, ne, r0), fontsize=9)
            ax.grid(**GRID)
    fig.suptitle('%s -- what moves the turning point. 30-day anomalies, '
                 'p-values corrected for serial correlation.' % GTX,
                 fontsize=12)
    savefig(fig, 'fig3_drivers.png')
    RR = pd.DataFrame(rows)
    RR.to_csv(out_dir / 'turning_drivers.csv', index=False, float_format='%.5f')
    print('\ndrivers of the turning point:')
    print(RR.round(3).to_string(index=False))

# =========================================================== 4. TIDAL CYCLE ==
if 'QU2_h' in D:
    QU2h_raw = np.where(UM[None, :, :], D['QU2_h'], np.nan)
    QU2h = recirc(QU2h_raw)               # same definition as the daily fields
    QNh, _ = limbs(QU2h, JS)
    QNh_raw, _ = limbs(QU2h_raw, JS)
    zeta_h = D['zeta_h']
    idh = pd.DatetimeIndex(D['time_h']).tz_localize('UTC').tz_convert(args.tz)
    i_hw, _ = find_peaks(zeta_h, distance=8)
    hw = idh[i_hw].tz_convert('UTC').tz_localize(None).values.astype('datetime64[ns]')
    al = idh.tz_convert('UTC').tz_localize(None).values.astype('datetime64[ns]')
    k = np.clip(np.searchsorted(hw, al), 1, len(hw) - 1)
    dp = (al - hw[k - 1]) / np.timedelta64(1, 'h')
    dn = (al - hw[k]) / np.timedelta64(1, 'h')
    hrs = np.where(np.abs(dp) <= np.abs(dn), dp, dn)

    Xh = pd.DataFrame(dict(
        hrs=hrs, zeta=zeta_h,
        x50=[penetration(QNh[n], 0.5) for n in range(len(idh))],
        x90=[penetration(QNh[n], 0.9) for n in range(len(idh))],
        x50_raw=[penetration(QNh_raw[n], 0.5) for n in range(len(idh))],
        QN=QNh[:, im]), index=idh)
    if 'QU2_bot_h' in D:                      # only present with -hourly_all
        Qnet_h = np.nansum(QU2h_raw, axis=1)
        QBh = np.nansum(np.where(UM[None, :, :], D['QU2_bot_h'], np.nan),
                        axis=1) - Qnet_h / 3
        Xh['x50_bot'] = [penetration(QBh[n], 0.5) for n in range(len(idh))]
    for c in [k for k in Xh.columns if k.startswith('x')]:
        Xh[c.replace('x', 'km', 1)] = (Xh[c] - lonc[im]) * KMDEG
    print('\nhourly turning point: defined on %.0f%% of hours with the net '
          'removed,\n  %.0f%% without -- that is why the metric is defined on '
          'the recirculating flow'
          % (100 * Xh.x50.notna().mean(), 100 * Xh.x50_raw.notna().mean()))
    BINS = np.arange(-6.5, 6.6, 1.0)
    CTR = 0.5 * (BINS[:-1] + BINS[1:])
    Xh['bin'] = pd.cut(Xh.hrs, BINS, labels=CTR)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')
    A = axs[0]
    for b, cmap in [(CTR, plt.cm.twilight)]:
        for n, c in enumerate(b):
            sel = Xh['bin'] == c
            if sel.sum() == 0:
                continue
            A.plot(lonc[COL], np.nanmean(QNh[sel.values][:, COL], axis=0), '-',
                   color=cmap(n / len(b)), lw=1.6,
                   label='%+.0f h' % c if n % 3 == 0 else None)
    A.axhline(0, color='0.5', lw=0.8)
    A.set_xlabel('longitude')
    A.set_ylabel('north-limb transport (m$^3$ s$^{-1}$)')
    A.set_title('the profile through the tidal cycle\n(hours from high water)',
                fontsize=10)
    A.grid(**GRID)
    A.legend(fontsize=7, ncol=2)
    A.xaxis.set_major_locator(plt.MaxNLocator(5))

    B = axs[1]
    g = Xh.groupby('bin', observed=False)['QN']
    m, se = g.mean(), g.std() / np.sqrt(g.count())
    B.plot(CTR, m.values, '-', color=C_IN, lw=2)
    B.fill_between(CTR, m - 2 * se, m + 2 * se, color=C_IN, alpha=0.2, lw=0)
    B.axhline(0, color='0.5', lw=0.8)
    B.axvline(0, color='0.5', lw=0.8)
    B.set_xlabel('hours from high water')
    B.set_ylabel('north-limb transport at the mouth (m$^3$ s$^{-1}$)')
    B.set_title('the limb through the cycle', fontsize=10)
    B.grid(**GRID)

    Cx = axs[2]
    g = Xh.groupby('bin', observed=False)['x50']
    m, se = g.mean(), g.std() / np.sqrt(g.count())
    kmb = (m.values - lonc[im]) * KMDEG
    Cx.plot(CTR, kmb, '-', color=C_LAT, lw=2)
    Cx.fill_between(CTR, kmb - 2 * se.values * KMDEG, kmb + 2 * se.values * KMDEG,
                    color=C_LAT, alpha=0.2, lw=0)
    Cx.axvline(0, color='0.5', lw=0.8)
    Cx.set_xlabel('hours from high water')
    Cx.set_ylabel('x50 (km west of the mouth)')
    Cx.set_title('does the turning point sweep with the tide?\n'
                 'defined on %.0f%% of hours' % (100 * Xh.x50.notna().mean()),
                 fontsize=10)
    Cx.grid(**GRID)

    fig.suptitle('%s -- the same metric hour by hour, composited on the '
                 'ebb-flood cycle' % GTX, fontsize=12)
    savefig(fig, 'fig4_tidal_cycle.png')

    # ================================================= 5. HOW FAR IT SHIFTS ==
    # The question this figure exists for: how many km does the turning point
    # move back and forth, and how much of that is within a tidal cycle versus
    # between one week and the next. Both come off the SAME hourly series, cut
    # by a Godin filter, so the two numbers are directly comparable.
    KMCOLS = [c for c in ['km50', 'km90', 'km50_bot'] if c in Xh]
    KMLAB = {'km50': 'x50 (lateral)', 'km90': 'x90 (lateral)',
             'km50_bot': 'x50 (bottom third)'}
    KMCOL = {'km50': C_IN, 'km90': C_LAT, 'km50_bot': C_VERT}

    band = {}
    for c in KMCOLS:
        # short gaps get interpolated so the filter has something to chew on;
        # anything longer stays NaN and is reported
        s = Xh[c].interpolate(limit=6, limit_area='inside')
        lp = pd.Series(zfun.lowpass(s.values.astype(float), f='godin'),
                       index=Xh.index)
        band[c] = dict(raw=Xh[c], filled=s, sub=lp, tid=s - lp,
                       gap=float(Xh[c].isna().mean()))

    # per tidal cycle (high water to high water): how far does it swing?
    cyc = []
    for a, b in zip(i_hw[:-1], i_hw[1:]):
        v = Xh[KMCOLS[0]].values[a:b]
        v = v[np.isfinite(v)]
        if len(v) >= 8:
            cyc.append(dict(t=idh[a], excursion=np.ptp(v), mean=v.mean(),
                            zeta_range=np.ptp(zeta_h[a:b])))
    CY = pd.DataFrame(cyc)
    if len(CY):
        CY['date'] = CY.t.dt.floor('D')
        CY = CY.merge(Dd[['range_smooth_m']].rename_axis('date').reset_index(),
                      on='date', how='left')

    fig, axs = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')

    # --- a window of the record, so the two timescales can be seen at once
    A = axs[0][0]
    n_show = min(len(Xh), 30 * 24)
    sl = slice(len(Xh) // 2 - n_show // 2, len(Xh) // 2 + n_show // 2)
    c0 = KMCOLS[0]
    A.plot(Xh.index[sl], band[c0]['raw'].values[sl], '-', color='0.62', lw=0.8,
           label='hourly')
    A.plot(Xh.index[sl], band[c0]['sub'].values[sl], '-', color=C_IN, lw=2.4,
           label='Godin (subtidal)')
    A2 = A.twinx()
    A2.plot(Xh.index[sl], zeta_h[sl], '-', color='#c9b458', lw=0.7, alpha=0.8)
    A2.set_ylabel('cove mean zeta (m)', color='#a08c2e')
    A2.tick_params(axis='y', colors='#a08c2e')
    A.set_zorder(A2.get_zorder() + 1)
    A.patch.set_visible(False)
    A.set_ylabel('%s\nkm west of the mouth' % KMLAB[c0])
    A.set_title('%d days of the record: the turning point moves on both '
                'timescales\n(gold = tide)' % (n_show // 24), fontsize=10)
    A.grid(**GRID)
    A.legend(fontsize=8, loc='upper left')

    # --- how much of the movement is in each band
    B = axs[0][1]
    rows = []
    for c in KMCOLS:
        d_ = band[c]
        ok = np.isfinite(d_['sub'])
        vt, vs = np.var(d_['tid'][ok]), np.var(d_['sub'][ok])
        rows.append(dict(metric=c, sd_total=float(np.nanstd(d_['filled'][ok])),
                         sd_tidal=float(np.sqrt(vt)),
                         sd_subtidal=float(np.sqrt(vs)),
                         frac_tidal=float(vt / (vt + vs)),
                         frac_subtidal=float(vs / (vt + vs)),
                         p5_95_tidal=float(np.nanpercentile(d_['tid'][ok], 95)
                                           - np.nanpercentile(d_['tid'][ok], 5)),
                         p5_95_subtidal=float(np.nanpercentile(d_['sub'][ok], 95)
                                              - np.nanpercentile(d_['sub'][ok], 5)),
                         undefined_frac=d_['gap']))
    V = pd.DataFrame(rows)
    xb = np.arange(len(V))
    B.bar(xb - 0.19, V.sd_tidal, 0.36, color='#4565e8', label='tidal band')
    B.bar(xb + 0.19, V.sd_subtidal, 0.36, color='#e04256', label='subtidal band')
    for i, r in V.iterrows():
        B.text(i - 0.19, r.sd_tidal, '%.0f%%' % (100 * r.frac_tidal),
               ha='center', va='bottom', fontsize=8)
        B.text(i + 0.19, r.sd_subtidal, '%.0f%%' % (100 * r.frac_subtidal),
               ha='center', va='bottom', fontsize=8)
    B.set_xticks(xb)
    B.set_xticklabels([KMLAB[c] for c in V.metric], fontsize=8)
    B.set_ylabel('standard deviation of position (km)')
    B.set_title('how far it shifts, by timescale\nlabels = share of the '
                'variance', fontsize=10)
    B.grid(**GRID)
    B.legend(fontsize=8)

    # --- the swing within one tidal cycle, against the size of that tide
    Cc = axs[1][0]
    if len(CY) and CY.range_smooth_m.notna().any():
        r, slope, pv, ne = eff_corr(CY.range_smooth_m, CY.excursion)
        Cc.plot(CY.range_smooth_m, CY.excursion, '.', ms=4, alpha=0.4,
                color=C_IN)
        g = CY[['range_smooth_m', 'excursion']].dropna()
        xx = np.linspace(g.range_smooth_m.min(), g.range_smooth_m.max(), 10)
        Cc.plot(xx, np.polyval(np.polyfit(g.range_smooth_m, g.excursion, 1), xx),
                '-', color='k', lw=2,
                label='r = %+.2f, %.2f km per m of range\n(p = %.1e, n_eff = %.0f)'
                      % (r, slope, pv, ne))
        Cc.set_xlabel('smoothed daily tidal range (m)')
        Cc.legend(fontsize=8)
    else:
        Cc.hist(CY.excursion.dropna() if len(CY) else [], bins=20,
                color=C_IN, alpha=0.8)
        Cc.set_xlabel('excursion within one tidal cycle (km)')
    Cc.set_ylabel('excursion within one tidal cycle (km)'
                  if len(CY) and CY.range_smooth_m.notna().any() else 'cycles')
    Cc.set_title('the swing within a single cycle\nmedian %.2f km over %d '
                 'cycles' % (CY.excursion.median() if len(CY) else np.nan,
                             len(CY)), fontsize=10)
    Cc.grid(**GRID)

    # --- constituents of the position itself
    E = axs[1][1]
    done = False
    if args.utide and (Xh.index[-1] - Xh.index[0]).days >= 30:
        try:
            import utide
            tnum = pd.to_datetime(Xh.index.tz_convert('UTC').tz_localize(None)
                                  .to_pydatetime())
            MAIN = ['M2', 'S2', 'N2', 'K1', 'O1', 'P1']
            wid = 0.8 / len(KMCOLS)
            urows = []
            for kk, c in enumerate(KMCOLS):
                y = band[c]['filled'].values.astype(float)
                sol = utide.solve(tnum, y, lat=48.23, method='ols',
                                  conf_int='none', nodal=True, trend=False,
                                  verbose=False)
                amp = dict(zip(sol.name, sol.A))
                E.bar(np.arange(len(MAIN)) + kk * wid,
                      [amp.get(m, np.nan) for m in MAIN], wid,
                      color=KMCOL[c], label=KMLAB[c])
                urows += [dict(metric=c, constituent=m, amplitude_km=amp.get(m))
                          for m in MAIN]
            pd.DataFrame(urows).to_csv(out_dir / 'turning_utide.csv',
                                       index=False, float_format='%.4f')
            E.set_xticks(np.arange(len(MAIN)) + 0.4 - wid / 2)
            E.set_xticklabels(MAIN)
            E.set_ylabel('amplitude of the position (km)')
            E.set_title('tidal constituents OF THE TURNING POINT\n'
                        'how many km each constituent moves it', fontsize=10)
            E.legend(fontsize=8)
            E.grid(**GRID)
            done = True
        except Exception as ex:
            print('utide on the turning point failed: %s' % ex)
    if not done:
        for c in KMCOLS:
            E.hist(band[c]['tid'].dropna(), bins=40, histtype='step', lw=1.8,
                   density=True, color=KMCOL[c], label='%s, tidal' % KMLAB[c])
            E.hist(band[c]['sub'].dropna() - band[c]['sub'].mean(), bins=40,
                   histtype='step', lw=1.8, ls='--', density=True,
                   color=KMCOL[c], label='%s, subtidal' % KMLAB[c])
        E.set_xlabel('displacement from the mean position (km)')
        E.set_ylabel('density')
        E.set_title('the two bands side by side\n(record too short for utide)',
                    fontsize=10)
        E.legend(fontsize=7)
        E.grid(**GRID)

    fig.suptitle('%s -- how far the turning point shifts, and on what '
                 'timescale. One hourly series, cut by a Godin filter.' % GTX,
                 fontsize=12)
    savefig(fig, 'fig5_excursion.png')

    if len(CY):
        V['cycle_excursion_median'] = CY.excursion.median()
        V['cycle_excursion_p90'] = CY.excursion.quantile(0.9)
    V.to_csv(out_dir / 'turning_excursion_summary.csv', index=False,
             float_format='%.4f')
    print('\nHOW FAR THE TURNING POINT SHIFTS (km):')
    print(V.to_string(index=False, float_format=lambda v: '%.3f' % v))
    print('  sd_tidal is the swing within the tidal band, sd_subtidal the '
          'week-to-week wandering;\n  frac_* are their shares of the total '
          'variance of the position.')
    if len(CY):
        print('  within a single high-water-to-high-water cycle the median '
              'swing is %.2f km (p90 %.2f).'
              % (CY.excursion.median(), CY.excursion.quantile(0.9)))
        CY.to_csv(out_dir / 'turning_cycle_excursion.csv', index=False,
                  float_format='%.4f')
    Xh.drop(columns='bin').to_csv(out_dir / 'turning_point_hourly.csv',
                                  float_format='%.5f')

# ------------------------------------------------------------------ tables ---
X.to_csv(out_dir / 'turning_point_daily.csv', float_format='%.5f')
prof = pd.DataFrame(dict(
    iu=iu[COL], lon=lonc[COL], km_from_mouth=km[COL], nface=nface[COL],
    lat_split=lat_split[COL],
    QN_mean=np.nanmean(Q_N, axis=0)[COL], QS_mean=np.nanmean(Q_S, axis=0)[COL],
    Qnet_mean=np.nanmean(Q_TOT, axis=0)[COL],
    Qbot_mean=np.nanmean(Q_BOT, axis=0)[COL],
    Qtop_mean=np.nanmean(Q_TOP, axis=0)[COL],
    E_lat=np.nanmean(E_LAT, axis=0)[COL], E_vert=np.nanmean(E_VERT, axis=0)[COL],
    QN_sd=np.nanstd(Q_N, axis=0)[COL]))
prof.to_csv(out_dir / 'along_cove_profile.csv', index=False,
            float_format='%.4f')
Dd.to_csv(out_dir / 'turning_point_with_drivers.csv', float_format='%.5f')

print('\nmean along-cove profile (transports in m3/s):')
print(prof.to_string(index=False, float_format=lambda v: '%.1f' % v,
                     formatters={c: (lambda v: '%.4f' % v)
                                 for c in ['lon', 'lat_split']}))
print('\nsaved figures and tables to\n  %s' % out_dir)
