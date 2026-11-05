"""
Stratification and along-cove wind states of Penn Cove, hour by hour, for
compositing the 20261103_pc_* section scripts by forcing (as pc_tidal_cycles.py
does by tidal cycle).

STRATIFICATION  drho = sigma0(bottom) - sigma0(top) [kg/m3], TEOS-10 from the tef2
    section top / bottom salinity and temperature (strat_*_wb1_pc1.nc: hourly
    hour-means, width-weighted across faces), averaged over pc_cp, pc_lj, pc_lp
    (hourly corr between them >= 0.98), Godin-filtered so tidal straining does
    not flip the class within a tide.
    high = top third of the record, low = bottom third (the middle third is neither).

ALONG-COVE WIND STRESS (the classes)  tau = sustr * ax + svstr * ay [Pa], the
    surface stress ROMS actually applied (BULK_FLUXES / COARE, ocean_his, from the
    pc_cove_mix box), averaged over the cove's water faces (u faces west of the
    mouth face, v faces in rho columns <= 30), on the cove axis (ax, ay) =
    (-0.9789, -0.2043), mouth -> head, so + = pushing toward the HEAD (up-cove),
    the sign of w_along / along_pc in the earlier wind work. Computed hourly and
    THEN Godin-filtered (stress is quadratic in the wind, so the stress of a
    filtered wind would miss e.g. the sea-breeze hours), which also removes the
    sea breeze cycle (S1) and the 00 UTC atm00 seam.
    up-cove = tau > +TTH, down-cove = tau < -TTH, weak = |tau| <= TTH, TTH = 0.01 Pa
    (~3 m/s of steady wind); the direction is set by the sign, not by a share of
    the record. Check: corr with a Large & Pond (1981) stress from the cove-mean
    WRF wind 0.95 hourly, ratio of std 1.10.
    Cached in DM_outs/20261104_pc_forcing_states/tau_cove.npz.

CROSS-COVE WIND STRESS (2026-11-04)  tau_cr = -sustr * ay... i.e. the stress on the right-hand
    normal of the axis, (ay, -ax) = (-0.2043, +0.9789), so + = pushing toward the NORTH shore
    (the corrected sign, [[wcross-sign-bug]]), same cove faces, hourly then Godin.
    northward = tau_cr > +TTH, southward = tau_cr < -TTH (cross_weak between).
    wind_calm = |tau_al| <= TTH AND |tau_cr| <= TTH: no appreciable stress either way (38% of hours).

ALONG-COVE WIND (reference only, not used for the classes)  W = u_pc ax + v_pc ay
    [m/s], cove-mean WRF wind (20260806_wind_reduce.py, atm00, hourly), Godin-filtered.

Both are put on the hourly time axis of the caller (nearest hour). Godin leaves
35 h at each end undefined; those hours are in no class.

USE
    from pc_forcing_states import state_masks
    S = state_masks(time)       # dict of boolean arrays: strat_high, strat_low, wind_up, wind_down, wind_weak,
                                # cross_north, cross_south, cross_weak (wind_* / cross_* are classes of wind STRESS)

OUTPUT (run this file)  LO_output/DM_outs/20261104_pc_forcing_states/forcing_states.png
"""
import numpy as np
import pandas as pd
import xarray as xr
import gsw

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head (reduce_wind_cove.py)
TTH = 0.01                                # along-cove stress threshold [Pa] for up- / down-cove
LON, LAT = -122.69, 48.23

Ldir = Lfun.Lstart(gridname='wb1')
strat_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'tef2' / ('strat_%s_%s_wb1_pc1.nc' % (DS0, DS1))
wind_fn = Ldir['LOo'] / 'DM_outs' / '20260806_wind' / ('wind_hourly_atm00_%s_%s.p' % (DS0, DS1))
mix_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261104_pc_forcing_states'


def cove_stress():
    """Hourly cove-mean along- and cross-cove surface stress [Pa] that ROMS applied (cached)."""
    fn = out_dir / 'tau_cove.npz'
    if fn.exists():
        z = np.load(fn)
        return pd.DatetimeIndex(z['t']), z['tau'], z['tau_cr']
    out_dir.mkdir(parents=True, exist_ok=True)
    m = xr.open_dataset(mix_fn)
    mask = m.mask_rho.values == 1
    mu = mask[:, :-1] & mask[:, 1:]
    mu[:, 30:] = False                    # cove u faces: west of the mouth face (box xi_u 30)
    mv = mask[:-1, :] & mask[1:, :]
    mv[:, 31:] = False                    # cove v faces: rho columns <= 30
    tx = np.nanmean(m.sustr.values[:, mu], axis=1)
    ty = np.nanmean(m.svstr.values[:, mv], axis=1)
    t = m.ocean_time.values
    m.close()
    tau = tx * AXIS[0] + ty * AXIS[1]                  # along, + toward the head
    tau_cr = tx * AXIS[1] - ty * AXIS[0]               # cross, + toward the north shore
    np.savez(fn, t=t, tau=tau, tau_cr=tau_cr)
    return pd.DatetimeIndex(t), tau, tau_cr


def series():
    """Hourly drho (3-section mean, and per section), along-cove wind stress and wind, raw and Godin-filtered."""
    s = xr.open_dataset(strat_fn).sel(sect=SECTS)
    t_s = pd.DatetimeIndex(s.time.values)                                   # hour-means, stamped hh:30
    sig = lambda S, T: gsw.sigma0(gsw.SA_from_SP(S, 0, LON, LAT), gsw.CT_from_pt(gsw.SA_from_SP(S, 0, LON, LAT), T))
    dr = sig(s.s_bot.values, s.t_bot.values) - sig(s.s_top.values, s.t_top.values)   # (t, sect)
    s.close()
    W = pd.read_pickle(wind_fn)['W']                                       # hourly, UTC (tz-naive)
    w = (W.u_pc * AXIS[0] + W.v_pc * AXIS[1]).to_numpy()
    t_t, tau, tau_cr = cove_stress()
    return dict(t_s=t_s, drho_sect=dr, drho=dr.mean(axis=1), drho_lp=zfun.lowpass(dr.mean(axis=1), f='godin'),
                t_w=pd.DatetimeIndex(W.index), wind=w, wind_lp=zfun.lowpass(w, f='godin'),
                t_t=t_t, tau=tau, tau_lp=zfun.lowpass(tau, f='godin'), tau_cr=tau_cr, tau_cr_lp=zfun.lowpass(tau_cr, f='godin'))


def classify(X):
    """Classes on each series' own hours."""
    q = np.nanpercentile(X['drho_lp'], [100 / 3, 200 / 3])
    ok_s, ok_t = np.isfinite(X['drho_lp']), np.isfinite(X['tau_lp'])
    C = dict(strat_high=ok_s & (X['drho_lp'] > q[1]), strat_low=ok_s & (X['drho_lp'] <= q[0]), q=q,
             wind_up=ok_t & (X['tau_lp'] > TTH), wind_down=ok_t & (X['tau_lp'] < -TTH),
             wind_weak=ok_t & (np.abs(X['tau_lp']) <= TTH),
             cross_north=ok_t & (X['tau_cr_lp'] > TTH), cross_south=ok_t & (X['tau_cr_lp'] < -TTH),
             cross_weak=ok_t & (np.abs(X['tau_cr_lp']) <= TTH))
    C['wind_calm'] = C['wind_weak'] & C['cross_weak']        # weak stress in BOTH directions
    return C


def nearest(t_from, t_to):
    """Index into t_from of the nearest hour to each time in t_to."""
    a, b = t_from.asi8, pd.DatetimeIndex(t_to).asi8
    i = np.clip(np.searchsorted(a, b), 1, len(a) - 1)
    return np.where(np.abs(a[i - 1] - b) <= np.abs(a[i] - b), i - 1, i)


def state_masks(time):
    """strat_high / strat_low / wind_up / wind_down / wind_weak boolean arrays aligned to `time`."""
    X = series()
    C = classify(X)
    i_s, i_w = nearest(X['t_s'], time), nearest(X['t_t'], time)
    out = {k: C[k][i_s] for k in ['strat_high', 'strat_low']}
    out.update({k: C[k][i_w] for k in ['wind_up', 'wind_down', 'wind_weak', 'cross_north', 'cross_south', 'cross_weak', 'wind_calm']})
    return out


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    X = series()
    C = classify(X)
    COL = {'strat_high': '#D55E00', 'strat_low': '#0072B2', 'wind_up': '#009E73', 'wind_down': '#CC79A7',
           'cross_north': '#E69F00', 'cross_south': '#56B4E9'}
    LAB = {'strat_high': 'high stratification (top third)', 'strat_low': 'low stratification (bottom third)',
           'wind_up': 'up-cove stress (> +%g Pa, toward the head)' % TTH, 'wind_down': 'down-cove stress (< -%g Pa, toward the mouth)' % TTH,
           'cross_north': 'northward stress (> +%g Pa, toward the north shore)' % TTH, 'cross_south': 'southward stress (< -%g Pa, toward the south shore)' % TTH}

    def pale(c, f=0.3):
        """Opaque mix of colour c with white (alpha would turn solid on the transparent background in some viewers)."""
        rgb = np.array(matplotlib.colors.to_rgb(c))
        return tuple(f * rgb + (1 - f))

    def shade(ax, t, m, key):
        ax.fill_between(t, 0, 1, where=m, transform=ax.get_xaxis_transform(), color=pale(COL[key]), lw=0, label=LAB[key], zorder=0)

    fig, axs = plt.subplots(5, 1, figsize=(14, 16), layout='constrained', gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.8]})
    ax = axs[0]
    for k in ['strat_high', 'strat_low']:
        shade(ax, X['t_s'], C[k], k)
    ax.plot(X['t_s'], X['drho'], color='0.7', lw=0.3)
    ax.plot(X['t_s'], X['drho_lp'], color='k', lw=1, label='Godin-filtered')
    for q in C['q']:
        ax.axhline(q, color='k', lw=0.8, ls='--')
    ax.set_ylabel(r'$\Delta\sigma_0$ bottom - top [kg m$^{-3}$]')
    ax.set_title(r'stratification: $\sigma_0$(bottom) - $\sigma_0$(top), mean of pc_cp, pc_lj, pc_lp (grey hourly; dashed: tercile limits %.2f, %.2f)' % tuple(C['q']), fontsize=10)
    ax.legend(fontsize=8, loc='upper left', ncol=3)
    ax = axs[1]
    for k in ['wind_up', 'wind_down']:
        shade(ax, X['t_t'], C[k], k)
    ax.plot(X['t_t'], X['tau'], color='0.7', lw=0.3)
    ax.plot(X['t_t'], X['tau_lp'], color='k', lw=1, label='Godin-filtered')
    for y in [-TTH, 0, TTH]:
        ax.axhline(y, color='k', lw=0.8, ls='--' if y else '-')
    ax.set_ylim(*np.nanpercentile(X['tau'], [0.5, 99.5]))
    ax.set_ylabel('along-cove stress [Pa]\n+ toward the head')
    ax.set_title('along-cove wind STRESS that ROMS applied (sustr, svstr), cove mean; + = up-cove. These are the wind classes.', fontsize=10)
    ax.legend(fontsize=8, loc='lower left', ncol=3)
    ax = axs[2]
    for k in ['cross_north', 'cross_south']:
        shade(ax, X['t_t'], C[k], k)
    ax.plot(X['t_t'], X['tau_cr'], color='0.7', lw=0.3)
    ax.plot(X['t_t'], X['tau_cr_lp'], color='k', lw=1, label='Godin-filtered')
    for y in [-TTH, 0, TTH]:
        ax.axhline(y, color='k', lw=0.8, ls='--' if y else '-')
    ax.set_ylim(*np.nanpercentile(X['tau_cr'], [0.5, 99.5]))
    ax.set_ylabel('cross-cove stress [Pa]\n+ toward the north shore')
    ax.set_title('cross-cove wind STRESS that ROMS applied, cove mean; + = toward the north shore. These are the cross-wind classes.', fontsize=10)
    ax.legend(fontsize=8, loc='lower left', ncol=3)
    ax = axs[3]
    for k in ['wind_up', 'wind_down']:
        shade(ax, X['t_t'], C[k], k)
    ax.plot(X['t_w'], X['wind'], color='0.7', lw=0.3)
    ax.plot(X['t_w'], X['wind_lp'], color='k', lw=1, label='Godin-filtered')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylim(*np.nanpercentile(X['wind'], [0.5, 99.5]))
    ax.set_ylabel('along-cove wind [m s$^{-1}$]\n+ toward the head')
    ax.set_title('along-cove wind (cove-mean WRF, atm00), for reference; shading = the stress classes above', fontsize=10)
    for a in axs[:4]:
        a.set_xlim(X['t_s'][0], X['t_s'][-1])
        a.grid(True, alpha=0.3)
    ax = axs[4]
    keys = ['strat_high', 'strat_low', 'wind_up', 'wind_down', 'cross_north', 'cross_south']
    share = pd.DataFrame({k: pd.Series(C[k], index=X['t_s'] if k.startswith('strat') else X['t_t']).groupby(lambda t: t.month).mean() * 100 for k in keys})
    x = np.arange(1, 13)
    for n, k in enumerate(keys):
        ax.bar(x + (n - 2.5) * 0.14, share[k], 0.14, color=COL[k], label=LAB[k])
    ax.axhline(100 / 3, color='k', lw=0.8, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_ylim(0, 115)                   # room for the legend above the bars
    ax.set_ylabel('share of the month\'s hours [%]')
    ax.set_title('when each class happens: share of each calendar month in each class (2024-2025; dotted = one third)', fontsize=10)
    ax.legend(fontsize=8, loc='upper center', ncol=3)
    ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('Penn Cove forcing states, %s, %s to %s' % (GTAGEX, DS0, DS1), fontsize=11)
    fig.savefig(out_dir / 'forcing_states.png', dpi=200, transparent=True)
    plt.close(fig)

    print('share of hours: ' + ', '.join('%s %.0f%%' % (k, 100 * C[k].mean()) for k in ['strat_high', 'strat_low', 'wind_up', 'wind_down', 'wind_weak', 'cross_north', 'cross_south', 'cross_weak']))
    print('corr(Godin along, Godin cross stress) = %.2f' % np.corrcoef(*[x[np.isfinite(X['tau_lp'])] for x in (X['tau_lp'], X['tau_cr_lp'])])[0, 1])
    print('monthly share [%]:\n' + share.round(0).astype(int).T.to_string())
    i_w = nearest(X['t_t'], X['t_s'])
    print('wind class within each stratification class [% of its hours]:')
    for s_ in ['strat_high', 'strat_low']:
        m = C[s_]
        print('  %-10s up-cove %3.0f%%, weak %3.0f%%, down-cove %3.0f%%'
              % (s_, *(100 * C[w][i_w][m].mean() for w in ['wind_up', 'wind_weak', 'wind_down'])))
    print('saved ' + str(out_dir / 'forcing_states.png'))
