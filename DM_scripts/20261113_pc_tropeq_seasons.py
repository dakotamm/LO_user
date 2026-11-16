"""
Penn Cove tropic/equatorial tides by season: what the tide looks like at each end of
the D1 axis, and how that changes through the year. Built step by step.

Input (wb1_t0_xn11abbur00, 2024-2025, hh:00 his snapshots):
    LO_output/DM_outs/20261112_pc_ind_var_ts/ind_var_ts.p
        D2, D1      semidiurnal / diurnal envelopes of the pc_lp tidal transport [m3/s]
        Q_tide      harmonic tidal transport at pc_lp, + = out of the cove (ebb) [m3/s]
        eta_tide    harmonic tidal sea level at pc_lp [m]

STEP 1: find every tropic and equatorial time = the maxima and minima of hourly D1.
    D1 cycles at the tropic month (13.66 d), so the extremes are found with a minimum
    spacing of 10 d (find_peaks on D1 and on -D1). Extremes within 2 d of the record
    ends are dropped (hilbert end effects).
    Each extreme gets the season of its date (DJF, MAM, JJA, SON) and the D2 at that
    hour, so the spring/neap phase it falls on is kept (the two cycles beat at 180 d).

run 20261113_pc_tropeq_seasons.py
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
SEASONS = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
C_SEA = {'DJF': '#0072B2', 'MAM': '#009E73', 'JJA': '#E69F00', 'SON': '#CC79A7'}
EDGE = 48

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
in_dir = Ldir['LOo'] / 'DM_outs'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261113_pc_tropeq_seasons'
Lfun.make_dir(out_dir)

di = pd.read_pickle(in_dir / '20261112_pc_ind_var_ts' / 'ind_var_ts.p')
T = di.index
D1 = di.D1.values
D2 = di.D2.values
qf = -di.Q_tide.values                  # + = flood (into the cove)
eta = di.eta_tide.values
season_of = {m: s for s, ms in SEASONS.items() for m in ms}

rows = []
for kind, sgn in [('tropic', 1), ('equatorial', -1)]:
    k, _ = find_peaks(sgn * D1, distance=10 * 24)
    k = k[(k >= EDGE) & (k < len(T) - EDGE)]
    for kk in k:
        rows.append(dict(kind=kind, k=kk, time=T[kk], season=season_of[T[kk].month], D1=D1[kk], D2=D2[kk]))
ext = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
ext['D1_D2'] = ext.D1 / ext.D2

print('STEP 1: %d tropic, %d equatorial' % ((ext.kind == 'tropic').sum(), (ext.kind == 'equatorial').sum()))
print('spacing between tropics: median %.2f d' % (ext[ext.kind == 'tropic'].time.diff().dt.total_seconds() / 86400).median())
print(ext.groupby(['kind', 'season'])[['D1', 'D2', 'D1_D2']].agg(['count', 'mean', 'min', 'max']).round(2).to_string())

"""
STEP 2: one lunar day around each extreme, and what the tide does in it.
    Window: starts at the flood start (zero up-crossing of qf) before the GREATER flood
    nearest the extreme (the biggest qf within +-18 h of it), and runs NW = 25 h
    (hours 0-25, one lunar day of 24.84 h plus the next flood start). Starting every
    window at the greater flood lines the days up by the mixed tide's order, so the
    overlays in STEP 4 compare shapes, not phase.
    Within hours 0-24: floods = runs of qf > 0, ebbs = runs of qf < 0; a run is kept
    if it starts at hour 0-23 (the run starting near hour 25 is the next day's).
        nF, nE          number of floods / ebbs (1 = the lesser one never reversed Q)
        QFg, QFl        greater / lesser peak flood qf [m3/s] (QFl = 0 if nF = 1)
        QEg, QEl        greater / lesser peak ebb |qf|
        VFg, VFl        greater / lesser flood volume, sum of qf * 3600 [m3]
        rF, rE, rV      lesser / greater: QFl/QFg, QEl/QEg, VFl/VFg (1 = equal, 0 = one)
        HH, LH, HL, LL  higher high, lower high, higher low, lower low eta_tide [m]: the
                        local maxima / minima at hours -3 to 21 (a flood start is low
                        water, so the day's lows sit at its edges; this 25 h span holds
                        each high and low of one lunar day once)
        Rg, Rl          greater / lesser range: max and min of (high - next low) [m]
    Saved: extremes.csv
"""
NW = 25

def runs(x):
    """(start, stop) index pairs of the runs of True in x."""
    d = np.diff(np.r_[0, x.astype(int), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))

up = np.nonzero((qf[:-1] <= 0) & (qf[1:] > 0))[0] + 1
mets = []
for _, e in ext.iterrows():
    kp = e.k - 18 + np.argmax(qf[e.k - 18:e.k + 19])            # greater flood peak
    a = up[up <= kp][-1]
    q = qf[a:a + NW + 1]
    h = eta[a - 3:a + NW + 4]                                    # hour -3 at index 0
    fl = [q[i:j].max() for i, j in runs(q > 0) if i <= 23]
    eb = [-q[i:j].min() for i, j in runs(q < 0) if i <= 23]
    vf = [q[i:j].sum() * 3600 for i, j in runs(q > 0) if i <= 23]
    fl, eb, vf = sorted(fl)[::-1], sorted(eb)[::-1], sorted(vf)[::-1]
    ext_i = lambda f: [i for i in range(1, len(h) - 1) if f(h[i], h[i - 1]) and f(h[i], h[i + 1])]
    hi = ext_i(lambda x, y: x > y)
    lo = ext_i(lambda x, y: x < y)
    hi1 = [i for i in hi if i <= 24]                            # hours -3 to 21
    lo1 = [i for i in lo if i <= 24]
    H = sorted(h[hi1])[::-1]
    L = sorted(h[lo1])
    rng = [h[i] - h[[j for j in lo if j > i][0]] for i in hi1]
    mets.append(dict(a=a, t0=T[a], nF=len(fl), nE=len(eb),
                     QFg=fl[0], QFl=fl[1] if len(fl) > 1 else 0, QEg=eb[0], QEl=eb[1] if len(eb) > 1 else 0,
                     VFg=vf[0], VFl=vf[1] if len(vf) > 1 else 0,
                     HH=H[0], LH=H[1] if len(H) > 1 else np.nan, HL=L[-1] if len(L) > 1 else np.nan, LL=L[0],
                     Rg=max(rng), Rl=min(rng) if len(rng) > 1 else np.nan))
ext = pd.concat([ext, pd.DataFrame(mets)], axis=1)
ext['rF'] = ext.QFl / ext.QFg
ext['rE'] = ext.QEl / ext.QEg
ext['rV'] = ext.VFl / ext.VFg
ext.to_csv(out_dir / 'extremes.csv', index=False, float_format='%.4g')

print('\nSTEP 2: share of days with only one flood (lesser never reverses):')
print((ext.nF == 1).groupby([ext.kind, ext.season]).mean().round(2).unstack().to_string())
print(ext.groupby(['kind', 'season'])[['QFg', 'QFl', 'QEg', 'QEl', 'rF', 'rE', 'rV', 'Rg', 'Rl']].mean().round(2).to_string())
print('saved %s' % (out_dir / 'extremes.csv'))

"""
STEP 3: figure of where the extremes fall in the record.
    a  hourly D1 (black) and D2 (grey), tropics (up triangles) and equatorials (down)
       coloured by season
    b  D1 / D2 at each extreme: the diurnal share of the tidal transport, which sets
       how unequal the two floods are (the D1 and D2 cycles beat at 180 d, so a tropic
       can land on a spring or a neap)
    Figure: tropeq_series.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MK = {'tropic': '^', 'equatorial': 'v'}

fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True, layout='constrained')
axs[0].plot(T, D2, color='0.6', lw=0.8, label='D2 (spring/neap)')
axs[0].plot(T, D1, color='k', lw=0.8, label='D1 (tropic/equatorial)')
for _, e in ext.iterrows():
    axs[0].plot(e.time, e.D1, MK[e.kind], color=C_SEA[e.season], ms=7, mec='k', mew=0.4)
    axs[1].plot(e.time, e.D1_D2, MK[e.kind], color=C_SEA[e.season], ms=7, mec='k', mew=0.4)
for s, c in C_SEA.items():
    axs[1].plot([], [], 's', color=c, label=s)
for kind, m in MK.items():
    axs[1].plot([], [], m, color='w', mec='k', label=kind)
axs[0].set_ylabel('envelope [m$^3$ s$^{-1}$]')
axs[1].set_ylabel('D1 / D2 at the extreme')
axs[0].legend(loc='upper left', fontsize=8, ncol=2)
axs[1].legend(loc='upper left', fontsize=8, ncol=6)
for ax in axs:
    ax.grid(alpha=0.3)
axs[0].set_title('Penn Cove mouth (pc_lp) tidal transport envelopes, tropic/equatorial extremes by season (%s)' % GTAGEX)
fig.savefig(out_dir / 'tropeq_series.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'tropeq_series.png'))

"""
STEP 4: figure of the tide itself on those days, by season.
    Columns = seasons; rows = flood transport qf (+ = into the cove) and eta_tide on the
    tropic days, then the same on the equatorial days. Every day is a thin line from its
    STEP 2 window start (the flood start before the greater flood), the season mean is
    thick. Rows share their y axis across seasons.
    Figure: tropeq_shapes.png
"""
HR = np.arange(-3, NW + 4)

fig, axs = plt.subplots(4, 4, figsize=(15, 11), sharex=True, sharey='row', layout='constrained')
for r, (kind, vn) in enumerate([('tropic', 'qf'), ('tropic', 'eta'), ('equatorial', 'qf'), ('equatorial', 'eta')]):
    x = qf if vn == 'qf' else eta
    for c, s in enumerate(SEASONS):
        ax = axs[r, c]
        ee = ext[(ext.kind == kind) & (ext.season == s)]
        Y = np.array([x[a - 3:a + NW + 4] for a in ee.a])
        for y in Y:
            ax.plot(HR, y, color=C_SEA[s], lw=0.6, alpha=0.6)
        ax.plot(HR, Y.mean(axis=0), color='k', lw=2)
        ax.axhline(0, color='0.5', lw=0.5)
        ax.axvline(0, color='0.5', lw=0.5, ls='--')
        ax.grid(alpha=0.3)
        if r == 0:
            ax.set_title(s, fontsize=12)
        ax.text(0.02, 0.97, '%s, n = %d\nD1 %.0f, D2 %.0f' % (kind, len(ee), ee.D1.mean(), ee.D2.mean()),
                transform=ax.transAxes, va='top', fontsize=8)
    axs[r, 0].set_ylabel(('flood transport [m$^3$ s$^{-1}$]\n+ = into the cove' if vn == 'qf' else '$\\eta$ tide [m]') + '\n' + kind)
fig.supxlabel('hours from the flood start before the greater flood')
fig.suptitle('Penn Cove mouth (pc_lp) tide on tropic and equatorial lunar days by season; thin = each day, thick = mean (%s)' % GTAGEX)
fig.savefig(out_dir / 'tropeq_shapes.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'tropeq_shapes.png'))

"""
STEP 5: figure of the STEP 2 numbers by season, tropic vs equatorial.
    Each panel: one column of points per season, tropic left and equatorial right of
    the season tick, each day a dot, the mean a black bar.
        D1, D2, D1/D2                       forcing on the day
        QFg, QFl, QEg, QEl                  greater / lesser peak flood and ebb
        rF, rE, rV                          lesser / greater flood peak, ebb peak, flood volume
        Rg, Rl                              greater / lesser range
    Figure: tropeq_metrics.png
"""
PAN = [('D1', 'D1 [m$^3$ s$^{-1}$]'), ('D2', 'D2 [m$^3$ s$^{-1}$]'), ('D1_D2', 'D1 / D2'),
       ('QFg', 'greater flood peak [m$^3$ s$^{-1}$]'), ('QFl', 'lesser flood peak [m$^3$ s$^{-1}$]'), ('rF', 'lesser / greater flood peak'),
       ('QEg', 'greater ebb peak [m$^3$ s$^{-1}$]'), ('QEl', 'lesser ebb peak [m$^3$ s$^{-1}$]'), ('rE', 'lesser / greater ebb peak'),
       ('Rg', 'greater range [m]'), ('Rl', 'lesser range [m]'), ('rV', 'lesser / greater flood volume')]
DX = {'tropic': -0.17, 'equatorial': 0.17}

fig, axs = plt.subplots(4, 3, figsize=(12, 13), sharex=True, layout='constrained')
for ax, (vn, lab) in zip(axs.flat, PAN):
    for c, s in enumerate(SEASONS):
        for kind, dx in DX.items():
            y = ext[(ext.kind == kind) & (ext.season == s)][vn]
            ax.plot(c + dx + np.random.default_rng(c).uniform(-0.05, 0.05, len(y)), y, MK[kind], color=C_SEA[s], ms=5, mec='k', mew=0.3, alpha=0.8)
            ax.plot([c + dx - 0.1, c + dx + 0.1], [y.mean()] * 2, color='k', lw=2)
    ax.set_ylabel(lab)
    ax.grid(alpha=0.3)
for ax in axs[:, 2]:
    ax.set_ylim(bottom=0)
axs[0, 0].plot([], [], MK['tropic'], color='w', mec='k', label='tropic (left)')
axs[0, 0].plot([], [], MK['equatorial'], color='w', mec='k', label='equatorial (right)')
axs[0, 0].legend(fontsize=8)
for ax in axs[-1]:
    ax.set_xticks(range(4), list(SEASONS))
fig.suptitle('Penn Cove mouth (pc_lp) tropic vs equatorial lunar days by season (%s)' % GTAGEX)
fig.savefig(out_dir / 'tropeq_metrics.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'tropeq_metrics.png'))
