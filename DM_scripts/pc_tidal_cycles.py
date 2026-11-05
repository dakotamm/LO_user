"""
Spring / neap and tropic / equatorial phases of the Penn Cove tide, hour by hour,
shared by the 20261103_pc_* section scripts (their -cycle option).

DEFINITION. One utide fit (OLS, nodal, no trend; the 11 constituents of
20261008_tidal_ellipses.py + S1) to the section-total transport through the
MOUTH, Q(t) = int u dA at pc_lp, from the pc_cove box (hourly ocean_his
snapshots). The cove floods and ebbs together, so one reference serves all
three sections.
    semidiurnal band  D2 = M2 + S2 + N2 + K2, reconstructed  -> envelope |hilbert(D2)|
    diurnal band      D1 = K1 + O1 + P1 + Q1, reconstructed  -> envelope |hilbert(D1)|
S1 is left out of D1: it is mostly the daily wind (20261027_pc_diurnal_S1.py).
spring = top third of hours by the D2 envelope, neap = bottom third;
tropic = top third by the D1 envelope, equatorial = bottom third
(terciles, as in 20261027_tidal_ellipses_windowed.py; the middle third is in neither).
The terciles are taken WITHIN EACH CALENDAR MONTH OF EACH YEAR (2026-11-03), not over
the whole record: the envelopes have a semiannual modulation (D1: K1-P1 beat, strongest
tropic tides at the solstices; D2: S2-K2, strongest springs near the equinoxes), so a
single record-wide threshold put 49-53% of Jun / Dec hours but only 4-8% of Mar / Sep
hours in 'tropic' and compared seasons instead of tides. Per-month terciles give every
month exactly a third of its hours in each class, so each half is compared within the
same season: 'the stronger third of the tides at that time of year'.
The D2 envelope of the transport is the prism, not the sea-level range: sea level
weights the two bands equally and its envelope wanders +-2-3 d off the true
spring-neap cycle with a 180 d beat ([[springneap-define-on-qprism]]).

FORCING STATES (2026-11-04). The same mechanism also splits by the forcing
classes of pc_forcing_states.py, so every script's -cycle option takes
    strat  'high strat' / 'low strat'  record-wide top / bottom third of the
                                       Godin-filtered drho (bottom - top sigma0)
    wind   'up-cove' / 'down-cove'     Godin-filtered along-cove wind STRESS that ROMS
                                       applied, > +0.01 / < -0.01 Pa (weak hours in neither)
    windhs 'up-cove, high strat' / 'down-cove, high strat'   the wind split within high-strat hours
    windls 'up-cove, low strat' / 'down-cove, low strat'     the wind split within low-strat hours
windhs and windls together are the four strat x wind cases, each figure comparing the
two wind directions at fixed stratification (so wind is not confounded with strat).
    cross  'northward' / 'southward'   cross-cove wind stress > +0.01 / < -0.01 Pa (+ toward the north shore)
    stratcalm 'high strat, calm' / 'low strat, calm'   the strat split within CALM hours (|along| and
                                       |cross| stress both <= 0.01 Pa): stratification without wind
These are not tidal, but they live here so the scripts need no other change.

USE
    from pc_tidal_cycles import cycle_masks, expand_phases
    C = cycle_masks(time)            # dict of boolean arrays aligned to `time` (any hourly axis)
    PHASES, MASK, COLOR, LS = expand_phases(PHASES, MASK, COLOR, 'springneap', C)

CACHE  LO_output/DM_outs/20261103_pc_tidal_cycles/mouth_Q_utide.p (the fit),
       and cycles.png (the envelopes and the tercile split, for checking).
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import utide
from scipy.signal import hilbert

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
CONS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'MS4', 'M6', 'S1']
D2 = ['M2', 'S2', 'N2', 'K2']
D1 = ['K1', 'O1', 'P1', 'Q1']
CYCLES = {'springneap': ('spring', 'neap'), 'tropeq': ('tropic', 'equatorial'),
          'strat': ('high strat', 'low strat'), 'wind': ('up-cove', 'down-cove'),
          'windhs': ('up-cove, high strat', 'down-cove, high strat'),
          'windls': ('up-cove, low strat', 'down-cove, low strat'),
          'cross': ('northward', 'southward'),
          'stratcalm': ('high strat, calm', 'low strat, calm')}
LAT = 48.23

Ldir = Lfun.Lstart(gridname='wb1')
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
cyc_dir = Ldir['LOo'] / 'DM_outs' / '20261103_pc_tidal_cycles'


def mouth_fit():
    """utide coefficients of the transport through pc_lp (cached)."""
    fn = cyc_dir / 'mouth_Q_utide.p'
    if fn.exists():
        return pickle.load(open(fn, 'rb'))
    cyc_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(box_fn)
    ETA, XI_U = np.arange(19, 7, -1), 30   # pc_lp u-faces, north to south
    u = -ds.u.isel(xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)
    zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy().mean(axis=-1)
    dy = (1 / ds.pn.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy()).mean(axis=-1)
    T = ds.ocean_time.values
    ds.close()
    Q = (u * np.diff(zw, axis=1) * dy[None, None, :]).sum(axis=(1, 2))
    coef = utide.solve(T, Q, lat=LAT, method='ols', conf_int='linear', constit=CONS, nodal=True, trend=False, verbose=False)
    pickle.dump(coef, open(fn, 'wb'))
    return coef


def envelopes(time):
    """Hourly D2 and D1 envelopes of the mouth transport [m3/s] at `time`."""
    coef = mouth_fit()
    out = {}
    for name, cons in [('D2', D2), ('D1', D1)]:
        x = utide.reconstruct(time, coef, constit=cons, verbose=False).h
        out[name] = np.abs(hilbert(x))
    return out


def cycle_masks(time):
    """spring / neap / tropic / equatorial boolean arrays aligned to `time`: the top / bottom
    third of each envelope, ranked within each calendar month of each year."""
    E = envelopes(time)
    ym = pd.DatetimeIndex(time).to_period('M')
    rank = {b: pd.Series(E[b]).groupby(ym).rank(pct=True).to_numpy() for b in ['D2', 'D1']}   # 0-1 within the month
    out = {'spring': rank['D2'] > 2 / 3, 'neap': rank['D2'] <= 1 / 3,
           'tropic': rank['D1'] > 2 / 3, 'equatorial': rank['D1'] <= 1 / 3, 'env': E}
    from pc_forcing_states import state_masks     # stratification and wind-stress classes
    F = state_masks(time)
    out.update({'high strat': F['strat_high'], 'low strat': F['strat_low'],
                'up-cove': F['wind_up'], 'down-cove': F['wind_down'],
                'northward': F['cross_north'], 'southward': F['cross_south'],
                'calm': F['wind_calm'],
                'high strat, calm': F['strat_high'] & F['wind_calm'], 'low strat, calm': F['strat_low'] & F['wind_calm']})
    for w in ['up', 'down']:              # the four strat x wind cases
        for st in ['high', 'low']:
            out['%s-cove, %s strat' % (w, st)] = F['wind_' + w] & F['strat_' + st]
    return out


def expand_phases(PHASES, MASK, COLOR, cycle, C):
    """Split every phase into its high / low half of the cycle ('flood' -> 'flood, spring',
    'flood, neap'). Returns new PHASES, MASK, COLOR and line styles LS (high solid, low dashed)."""
    hi, lo = CYCLES[cycle]
    P, M, Cl, LS = [], {}, {}, {}
    for ph in PHASES:
        for half, ls in [(hi, '-'), (lo, '--')]:
            name = '%s, %s' % (ph, half)
            P.append(name)
            M[name] = MASK[ph] & C[half]
            Cl[name] = COLOR[ph]
            LS[name] = ls
    return P, M, Cl, LS


if __name__ == '__main__':
    # check figure: the two envelopes over the record, with the tercile split
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ds = xr.open_dataset(box_fn)
    T = ds.ocean_time.values
    ds.close()
    C = cycle_masks(T)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout='constrained')
    for ax, band, hi, lo, lab in [(axs[0], 'D2', 'spring', 'neap', 'semidiurnal (M2 S2 N2 K2)'),
                                  (axs[1], 'D1', 'tropic', 'equatorial', 'diurnal (K1 O1 P1 Q1)')]:
        e = C['env'][band]
        ax.plot(T, e, color='0.5', lw=0.5)
        ax.plot(T[C[hi]], e[C[hi]], '.', ms=1, color='#D55E00', label='%s (top third of its month)' % hi)
        ax.plot(T[C[lo]], e[C[lo]], '.', ms=1, color='#0072B2', label='%s (bottom third of its month)' % lo)
        ax.set_ylabel('envelope of mouth Q [m$^3$ s$^{-1}$]')
        ax.set_title(lab + ' envelope of the transport through pc_lp', fontsize=10)
        ax.legend(fontsize=8, loc='upper right', markerscale=8)
        ax.grid(True, alpha=0.3)
        print('%s: %s %.0f%% of hours, %s %.0f%%' % (band, hi, 100 * C[hi].mean(), lo, 100 * C[lo].mean()))
    print('overlap: spring & tropic %.0f%% of hours, spring & equatorial %.0f%%'
          % (100 * (C['spring'] & C['tropic']).mean(), 100 * (C['spring'] & C['equatorial']).mean()))
    m = pd.DatetimeIndex(T).month
    print('share of each calendar month in each class [%%]:\n%s' % pd.DataFrame(
        {k: pd.Series(C[k]).groupby(m).mean() * 100 for k in ['spring', 'neap', 'tropic', 'equatorial']}).round(0).astype(int).T.to_string())
    fig.savefig(cyc_dir / 'cycles.png', dpi=200, transparent=True)
    print('saved ' + str(cyc_dir / 'cycles.png'))
