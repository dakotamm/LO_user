"""
pc_lj (Lovejoy) on a set of 2024 lunar days: flood, ebb and residual velocity, one figure for a slide.
Columns flood | ebb | residual; one line per day of the set (-set):
    seasons     the seasonal baselines DJF, MAM, JJA, SON (JJA = the main baseline day)
    springneap  the main baseline, spring and neap
    tropeq      the main baseline, tropic and equatorial
    tropseason  the seasonal tropic days DJF, MAM, JJA, SON
    wind        the main baseline (calm), up-cove and down-cove wind
    strat       the main baseline, high and low stratification

ROW 1  cross-channel averaged profile, U_b = Q_b / A_b in 1 m bins of depth below the moving surface
    (as 20261110_pc_midchannel_u_timeseries.py), averaged over the class hours by transport:
    U_b = sum_t Q_b / sum_t A_b. A bin is drawn only where it is wet in every hour of the class.
ROW 2  across-channel profile, the depth-averaged u of each face vs latitude (north at the top),
    averaged over the class hours by transport: sum_t sum_k u dz / sum_t sum_k dz.
CLASSES  flood = hours with the section-total Q > 0 (into the cove), ebb = Q < 0, residual = all hours
    of the lunar day (24-25 h; one lunar day spans the M2 and the lunar diurnal tide, so the plain mean removes them).
SIGN  + = into the cove (west); x axes reversed so + is drawn to the LEFT (section-script convention).
DAYS  from 20261113_pc_candidate_days (2024, tide order fEFe as the baseline), t0 <= t < t1:
    seasons     n = 1 of each season in seasonal_baselines.csv (calm, season-median D2/D1/drho)
    springneap  picks.csv: baseline, spring (hand pick 2024-09-02 07:00) and neap, all within 9 days
    tropeq      picks.csv: baseline, tropic (2024-08-01 04:00) and equatorial (2024-09-07 21:00, the only one passing)
    tropseason  n = 1 of each season in seasonal_tropic.csv (STEP 9: D1 rank near 5/6 of the season, calm,
                season-median D2 and drho)
    wind        picks.csv: baseline (calm), up-cove (2024-03-28 08:00, the only 2024 day with the tide order;
                cross stress +0.058 Pa > along +0.018) and down-cove (2024-07-02 03:00)
    strat       picks.csv: baseline, high strat (2024-03-20 17:00) and low strat (2024-10-05 20:00 = the SON baseline)
    Hourly ocean_his snapshots from the pc_cove box.
ARROWS  (off for now, ARROWS = False; drawn on the main-baseline line when on) small arrows from u = 0 to the curve at every ZSTEP-th bin (row 1) and every face (row 2), blue = in,
    red = out; none where |u| < UMIN. Each layer (contiguous same-sign run whose peak is at least AMIN of the
    axis limit) is labelled IN / OUT past its peak.
All six panels share one symmetric x scale.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
OUTPUT LO_output/DM_outs/20261113_pc_baseline_profiles/[set]_profiles.png

run 20261113_pc_baseline_profiles.py [-set seasons|springneap|tropeq|tropseason|wind|strat]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-set', default='seasons', choices=['seasons', 'springneap', 'tropeq', 'tropseason', 'wind', 'strat'])
args = parser.parse_args()

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                 # section-positive (east) is out of the cove
FACES = {                                 # u faces in the pc_cove box, north to south; west to east
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (Lovejoy)', 'pc_cp': 'pc_cp (Coupeville)'}
SN = 'pc_lj'
SETS = {                                  # label: (csv in 20261113_pc_candidate_days, case in that csv)
    'seasons': {'DJF': ('seasonal_baselines.csv', 'DJF baseline'), 'MAM': ('seasonal_baselines.csv', 'MAM baseline'),
                'JJA': ('seasonal_baselines.csv', 'JJA baseline'), 'SON': ('seasonal_baselines.csv', 'SON baseline')},
    'springneap': {'baseline': ('picks.csv', 'baseline'), 'spring': ('picks.csv', 'spring'), 'neap': ('picks.csv', 'neap')},
    'tropeq': {'baseline': ('picks.csv', 'baseline'), 'tropic': ('picks.csv', 'tropic'), 'equatorial': ('picks.csv', 'equatorial')},
    'tropseason': {sea: ('seasonal_tropic.csv', sea + ' tropic') for sea in ['DJF', 'MAM', 'JJA', 'SON']},
    'wind': {'baseline': ('picks.csv', 'baseline'), 'up-cove': ('picks.csv', 'up-cove'), 'down-cove': ('picks.csv', 'down-cove')},
    'strat': {'baseline': ('picks.csv', 'baseline'), 'high strat': ('picks.csv', 'high strat'), 'low strat': ('picks.csv', 'low strat')},
}
BASE = {'seasons': 'JJA', 'springneap': 'baseline', 'tropeq': 'baseline', 'tropseason': 'JJA', 'wind': 'baseline', 'strat': 'baseline'}[args.set]    # the main baseline day of each set
CASES = list(SETS[args.set])
COL = {'DJF': '#4477AA', 'MAM': '#228833', 'JJA': '#EE6677', 'SON': '#CCBB44',   # one colour per day
       'baseline': 'k', 'spring': '#CC3311', 'neap': '#33BBEE', 'tropic': '#EE7733', 'equatorial': '#009988',
       'up-cove': '#EE3377', 'down-cove': '#0077BB', 'high strat': '#AA3377', 'low strat': '#66CCEE'}
ARROWS = False                            # in/out arrows on the JJA (main baseline) line
COL_IN, COL_OUT = '#2166AC', '#B2182B'   # layer arrows: blue = entering the cove, red = leaving
EDGES = np.arange(0, 31, 1.0)             # profile bins, depth below the surface [m]
AMIN = 0.1                                # no IN/OUT label where a layer's peak |u| < 10% of the x limit
UMIN = 0.3                                # no arrow where |u| < 0.3 cm/s (shorter than its head)
ZSTEP = 2                                 # row 1: an arrow on every 2nd 1 m bin

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
sel_dir = Ldir['LOo'] / 'DM_outs' / '20261113_pc_candidate_days'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261113_pc_baseline_profiles'
out_dir.mkdir(parents=True, exist_ok=True)

DAY = {}
for cs, (csv, case) in SETS[args.set].items():
    D = pd.read_csv(sel_dir / csv, parse_dates=['t0', 't1'])
    DAY[cs] = D[(D.case == case) & (D.n == 1)].iloc[0]
    print('%-8s lunar day %s to %s UTC' % (cs, DAY[cs].t0, DAY[cs].t1))

# ------------------------------------------------------------------ hourly transports of one section
def binned_QA(u, zw, dy):
    """Q_b and A_b (t, b) in the EDGES bins of depth below the moving surface, top bin first: each cell's
    transport split among the bins it overlaps (sum_b Q_b = Q exactly)."""
    dz = np.diff(zw, axis=1)
    q = u * dz * dy
    dw = zw[:, -1:, :] - zw                                          # depth below the surface of each interface
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    assert d_bot.max() < EDGES[-1]
    nb = len(EDGES) - 1
    Qb, Ab = np.zeros((len(u), nb)), np.zeros((len(u), nb))
    for b in range(nb):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None)
        Qb[:, b] = (q * ov / dz).sum(axis=(1, 2))
        Ab[:, b] = (ov * dy).sum(axis=(1, 2))
    assert np.allclose(Qb.sum(axis=1), q.sum(axis=(1, 2)))
    return Qb, Ab


ds = xr.open_dataset(in_fn)
TA = pd.DatetimeIndex(ds.ocean_time.values)
XI_U, ETA = FACES[SN]['XI_U'], FACES[SN]['ETA']
R = {}
for cs in CASES:
    it = np.flatnonzero((TA >= DAY[cs].t0) & (TA < DAY[cs].t1))
    sl = dict(ocean_time=slice(it[0], it[-1] + 1))
    u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA, **sl).to_numpy().astype(float)                  # (t, k, p) + into the cove
    zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().mean(axis=-1)     # (t, w, p) at the face
    dy = (1 / ds.pn.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy()).mean(axis=-1)      # (p,)
    lat = ds.lat_u.isel(eta_u=ETA, xi_u=XI_U).to_numpy()                                      # north to south
    dz = np.diff(zw, axis=1)
    Qb, Ab = binned_QA(u, zw, dy)
    R[cs] = dict(Qb=Qb, Ab=Ab, dy=dy, lat=lat,
                 Q=Qb.sum(axis=1),                                   # (t,) section-total Q, + into the cove
                 udz=(u * dz).sum(axis=1), H=dz.sum(axis=1))         # (t, p) per face: sum_k u dz and the water depth
    print('  %s %s %d h: Q %+.0f to %+.0f m3/s' % (cs, SN, len(it), R[cs]['Q'].min(), R[cs]['Q'].max()))
ds.close()

# ------------------------------------------------------------------ flood / ebb / residual averages
CLASSES = ['flood', 'ebb', 'residual']
for cs, r in R.items():
    M = {'flood': r['Q'] > 0, 'ebb': r['Q'] < 0, 'residual': np.ones(len(r['Q']), bool)}
    wet = r['Ab'] >= 0.5 * r['dy'].mean()                            # (t, b) bin wetted by at least half a face
    r['prof'], r['face'] = {}, {}
    for C in CLASSES:
        m = M[C]
        Ub = r['Qb'][m].sum(axis=0) / np.maximum(r['Ab'][m].sum(axis=0), 1e-12)
        r['prof'][C] = np.where(wet[m].all(axis=0), Ub, np.nan)      # (b,) top bin first
        r['face'][C] = r['udz'][m].sum(axis=0) / r['H'][m].sum(axis=0)   # (p,) north to south
        print('  %-4s %-8s %2d h: profile %+5.1f to %+5.1f cm/s, faces %+5.1f to %+5.1f cm/s'
              % (cs, C, m.sum(), 100 * np.nanmin(r['prof'][C]), 100 * np.nanmax(r['prof'][C]),
                 100 * r['face'][C].min(), 100 * r['face'][C].max()))

# ------------------------------------------------------------------ figure: 2 rows x flood | ebb | residual
def layer_arrows(ax, u, y, umin, step):
    """Small arrows from u = 0 to the curve at every step-th point (blue = in, red = out, none where |u| < UMIN),
    plus one IN (+) / OUT (-) label per layer (contiguous same-sign run of u, NaN breaks a run) past its peak."""
    for k in range(0, len(u), step):
        if np.isfinite(u[k]) and abs(u[k]) >= UMIN:
            ax.annotate('', xy=(u[k], y[k]), xytext=(0, y[k]),
                        arrowprops=dict(arrowstyle='-|>', color=COL_IN if u[k] > 0 else COL_OUT, lw=1, mutation_scale=8,
                                        shrinkA=0, shrinkB=0))
    ok = np.isfinite(u)
    sgn = np.where(ok, np.sign(u), 0)
    brk = np.flatnonzero(np.diff(sgn) != 0) + 1
    for run in np.split(np.arange(len(u)), brk):
        if sgn[run[0]] == 0:
            continue
        k = run[np.argmax(np.abs(u[run]))]
        if abs(u[k]) < umin:
            continue
        color = COL_IN if u[k] > 0 else COL_OUT
        ax.annotate('IN' if u[k] > 0 else 'OUT', xy=(u[k], y[k]), xytext=(-6 if u[k] > 0 else 6, 0),   # + is drawn LEFT
                    textcoords='offset points', ha='right' if u[k] > 0 else 'left', va='center',
                    color=color, fontweight='bold', fontsize=11)


plt.rcParams.update({'font.size': 13})
zc = 0.5 * (EDGES[1:] + EDGES[:-1])                                  # bin centres, depth below the surface
allu = np.concatenate([R[cs][key][C] for cs in CASES for key in ['prof', 'face'] for C in CLASSES])
XL = np.ceil(2 * 1.3 * 100 * np.nanmax(np.abs(allu))) / 2            # one symmetric x limit [cm/s] for every panel
zbot = max(zc[np.isfinite(np.vstack([R[cs]['prof'][C] for C in CLASSES])).any(axis=0)].max() for cs in CASES) + 1
lat = R[BASE]['lat']                                                # same faces every day
dlat = np.abs(np.diff(lat)).mean()

fig, axs = plt.subplots(2, 3, figsize=(13.33, 7.5), layout='constrained', sharex=True)
for j, C in enumerate(CLASSES):
    for cs in CASES:
        r = R[cs]
        lab = '%s  %s' % (cs, DAY[cs].t0.strftime('%Y-%m-%d')) + ('  (baseline)' if args.set == 'seasons' and cs == BASE else '') + ('  (calm)' if args.set == 'wind' and cs == BASE else '')
        up, uf = 100 * r['prof'][C], 100 * r['face'][C]
        axs[0, j].plot(up, zc, '-', color=COL[cs], lw=2.5, label=lab)   # row 1: cross-channel average
        axs[1, j].plot(uf, lat, '-', color=COL[cs], lw=2.5)             # row 2: depth average per face
        if ARROWS and cs == BASE:
            layer_arrows(axs[0, j], up, zc, AMIN * XL, ZSTEP)
            layer_arrows(axs[1, j], uf, lat, AMIN * XL, 1)
    axs[0, j].set_ylim(zbot, 0)
    axs[0, j].set_title({'flood': 'Flood', 'ebb': 'Ebb', 'residual': 'Residual (lunar-day mean)'}[C], fontweight='bold')
    axs[1, j].set_ylim(lat.min() - dlat, lat.max() + dlat)          # north at the top
    axs[1, j].set_xlabel('u [cm s$^{-1}$]   \u2190 IN | OUT \u2192')
    for i in range(2):
        ax = axs[i, j]
        ax.axvline(0, color='0.4', lw=0.8)
        ax.grid(color='lightgray', linestyle='--', alpha=0.5)
axs[0, 0].set_xlim(XL, -XL)                                          # + into the cove drawn LEFT (shared by all panels)
axs[0, 0].set_ylabel('cross-channel average\ndepth below surface [m]')
axs[1, 0].set_ylabel('depth average\nlatitude [\u00b0N]   S \u2193 N \u2191')
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=4, fontsize=11, frameon=False)
fig.suptitle('%s, %s lunar days 2024' % (NAME[SN], {'seasons': 'seasonal baseline', 'springneap': 'baseline, spring and neap',
                                           'tropeq': 'baseline, tropic and equatorial',
                                           'tropseason': 'seasonal tropic',
                                           'wind': 'calm baseline, up-cove and down-cove wind',
                                           'strat': 'baseline, high and low stratification'}[args.set]))

fn = out_dir / ('%s_profiles.png' % args.set)
fig.savefig(fn, dpi=300, transparent=True)
plt.close(fig)
print('saved %s' % fn)
