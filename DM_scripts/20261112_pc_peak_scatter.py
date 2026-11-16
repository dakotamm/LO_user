"""
Penn Cove scatter plots of one variable against another, on just the daily maximum
flood and maximum ebb hours, built step by step.

Inputs (wb1_t0_xn11abbur00, 2024-2025):
    LO_output/DM_outs/20261112_pc_ind_var_ts/ind_var_ts.p       hh:00 his snapshots, + Q_tide
    LO_output/DM_outs/20261112_pc_res_var_ts/res_var_ts.p       hh:00 his snapshots
    LO_output/DM_outs/20261112_pc_res_var_ts/res_var_ts_avg.p   hh:30 hour means

STEP 1: pick the daily maximum flood and maximum ebb hours.
    Flood/ebb from Q_tide (ind_var_ts.p STEP 6), the harmonic prediction of the
    transport through the pc_lp mouth column, every constituent but S1, on the hh:00
    index. Its sign is flipped here so + = into the cove = flood. The astronomical
    tide alone, so wind and subtidal flow do not move which hours are picked.
    Half-cycles as in 20261111_pc_metric_drivers.py: runs of one sign of Q, runs
    shorter than MIN_RUN = 3 h (slack jitter) absorbed into the run before them, the
    first and last (partial) runs dropped. Each half-cycle's peak = its max |Q| hour.
    "Daily max" = the greater half-cycle of each lunar day, by non-maximum suppression
    on the flood (ebb) peaks: go through them largest |Q| first, keep a peak unless
    it is within +/- WIN = 18 h of one already kept. The lesser half-cycle sits ~12 h
    from the greater one and the next greater one ~25 h away, so the window removes the
    lesser of each pair but never a neighbouring lunar day.
    Two simpler rules fail:
      - keep a peak >= the neighbouring flood (ebb) peaks: on diurnal days the lesser
        half-cycle shrinks below MIN_RUN and vanishes, two greater ones become
        neighbours, and one is dropped (2024-12-23 22:30 ebb, -2004 m3/s).
      - keep a peak that is the largest within +/- WIN: near equatorial tides, when
        successive peaks creep upward, each is knocked out by a bigger one not itself
        kept, and the drops chain (no ebb kept 2024-10-12 10:30 to 10-15 02:30).
    Not UTC calendar days: in winter the greater ebb peaks near 00 UTC, so it often
    falls just outside a day (e.g. 2024-12-25 23:30 and 2024-12-27 00:30), and
    2024-12-26 got its lesser ebb (-185 m3/s) as its "max".
    The hour-mean variables (res_var_ts_avg.p, hh:30) are put on the hh:00 index as
    the mean of the two hours bracketing it (hh-1:30 and hh:30). The first and last
    hh:00 have only one side and stay NaN (never picked: partial half-cycles).
    Saved to LO_output/DM_outs/20261112_pc_peak_scatter/peaks.p, one row per picked
    hour: time (hh:00), phase ('flood' / 'ebb'), Q (the sign-flipped Q_tide), then
    every variable.
    Check figure peaks_check.png: Q_tide over Dec 2024 (the month with the calendar-day
    failure) with Qnet_pc_lp and the picked hours.

run 20261112_pc_peak_scatter.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
MIN_RUN = 3
WIN = 18

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
in_dir = Ldir['LOo'] / 'DM_outs'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261112_pc_peak_scatter'
Lfun.make_dir(out_dir)
out_fn = out_dir / 'peaks.p'

di = pd.read_pickle(in_dir / '20261112_pc_ind_var_ts' / 'ind_var_ts.p')
dr = pd.read_pickle(in_dir / '20261112_pc_res_var_ts' / 'res_var_ts.p')
da = pd.read_pickle(in_dir / '20261112_pc_res_var_ts' / 'res_var_ts_avg.p')

# hh:30 hour means -> hh:00
assert (di.index == dr.index).all()
dh = pd.concat([di, dr], axis=1)
ah = pd.DataFrame(np.nan, index=dh.index, columns=da.columns)
assert (da.index == dh.index[:-1] + pd.Timedelta(minutes=30)).all()
ah.iloc[1:-1] = 0.5 * (da.values[:-1] + da.values[1:])
dall = pd.concat([dh, ah], axis=1)
dall.insert(0, 'Q', -dall['Q_tide'])                    # + = flood

# half-cycles
def runs(x):
    """[(start, stop), ...] of constant x (stop exclusive)"""
    k = np.flatnonzero(np.diff(x) != 0) + 1
    return list(zip(np.r_[0, k], np.r_[k, len(x)]))

q = dall['Q'].values
sg = np.sign(q)
for a, b in runs(sg):                                    # short runs -> 0, then take the previous sign
    if b - a < MIN_RUN:
        sg[a:b] = 0
sg = pd.Series(sg).replace(0, np.nan).ffill().bfill().to_numpy()
HC = runs(sg)[1:-1]                                      # the first and last runs are partial
hc_sign = np.array([sg[a] for a, b in HC])
hc_pk = np.array([a + np.argmax(np.abs(q[a:b])) for a, b in HC])    # peak hour of each half-cycle

# greater half-cycle of each lunar day (k is the hour index, so |k - k[i]| is in hours)
rows = []
for phase, sgn in [('flood', 1), ('ebb', -1)]:
    k = hc_pk[hc_sign == sgn]
    a = np.abs(q[k])
    keep = np.zeros(len(k), dtype=bool)
    for i in np.argsort(-a):                             # largest first
        if not keep[np.abs(k - k[i]) <= WIN].any():
            keep[i] = True
    d = dall.iloc[k[keep]].copy()
    d.insert(0, 'phase', phase)
    dt = np.diff(d.index).astype('timedelta64[m]').astype(float) / 60
    print('%-5s %d half-cycles, %d kept; spacing median %.1f h, 5-95%% %.1f-%.1f h; |Q| at pick: mean %.0f, min %.0f, max %.0f m3/s'
          % (phase, len(k), keep.sum(), np.median(dt), *np.percentile(dt, [5, 95]), a[keep].mean(), a[keep].min(), a[keep].max()))
    rows.append(d)
dp = pd.concat(rows).rename_axis('time').reset_index()
dp.to_pickle(out_fn)
print('saved %s (%d rows, %d columns)' % (out_fn, len(dp), dp.shape[1]))

# check figure
t0, t1 = pd.Timestamp('2024-12-01'), pd.Timestamp('2025-01-01')
w = (dall.index >= t0) & (dall.index < t1)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dall.index[w], dall['Qnet_pc_lp'][w], '-', color='0.75', lw=0.8, label='Qnet_pc_lp (model)')
ax.plot(dall.index[w], dall['Q'][w], '-', color='0.3', lw=0.8, label='-Q_tide')
for phase, c in [('flood', 'tab:blue'), ('ebb', 'tab:red')]:
    x = dp[(dp.phase == phase) & (dp.time >= t0) & (dp.time < t1)]
    ax.plot(x.time, x['Q'], 'o', color=c, ms=5, label='daily max ' + phase)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlim(t0, t1)
ax.set_ylabel('Q at pc_lp [m$^3$ s$^{-1}$], + = flood')
ax.legend(loc='upper right', ncol=4, fontsize=8)
ax.set_title('Daily max flood and ebb hours at pc_lp, Dec 2024')
fig.tight_layout()
fig.savefig(out_dir / 'peaks_check.png', dpi=150, transparent=True)
plt.close(fig)

"""
STEP 2: scatter of each independent variable against each response, on the daily max
flood hours and the daily max ebb hours separately.
    Independent (ind_var_ts.p, columns):
        Q           the sign-flipped Q_tide at the picked hour (+ = flood), so on the
                    ebb hours a stronger ebb is more negative               [m3/s]
        eta_tide    tidal sea level at the mouth                             [m]
        D2, D1      spring/neap and tropic/equatorial envelopes              [m3/s]
        tau_al      along-cove wind stress, + = up-cove                      [Pa]
        tau_cr      cross-cove wind stress, + = toward the north shore       [Pa]
        tau_ew      east-west wind stress over upper Saratoga, + = east      [Pa]
        tau_ns      north-south wind stress over upper Saratoga, + = north   [Pa]
        Tair        air temperature over Penn Cove                           [degC]
        drho_<sect> top-bottom density difference at each section            [kg/m3]
    IND is every ind_var_ts.p column (Q_tide as Q), so columns added there show up
    here; IND_ORDER only sets the order of the ones it names.
    Responses (rows), grouped as in 20261112_pc_tidal_scatter.py; one figure per
    phase, group and section (rows = that group's <var>_<sect>), plus one for the
    whole-cove variables:
        u      umax, uin, uout                        [m/s]
        Q      Qin, Qout, Qnet                        [m3/s], + = in
        dir    Usurf, Ubot, Unorth, Usouth            [m/s], + = in
        share  ain, Hin, Hout, Win, Wout              [0-1]
        pos    zin, zout, yin, yout                   sig (bed -1, surface 0), y (south 0, north 1)
        exch   rex, flay                              [0-1]
        count  nlay, nlat, npin, npout
        cove   drhoS_pc, drhoT_pc, drhodx_top, drhodx_bot, drhodx_bar, phi_pc
        bb     back bay (west of pc_cp), 20261112_pc_res_var_ts.py STEPS 7-8:
               Ut_ed      eddy ring depth-mean azimuthal speed, + = ccw        [m/s]
               fccw_ed    share of ring cells turning ccw                      [0-1]
               fcol_ed    share of the water column turning ccw                [0-1]
               UbotN_bb   bottom-half along-cove u, north half, + = to the head [m/s]
               UbotS_bb   the same, south half
    Panels share x down each column and y across each row; grey lines at x = 0 and y = 0
    (the axes stretch to include 0). Each is annotated with the
    Pearson r and n (pairs with a NaN dropped).
    Figures: <phase>/<group>_<sect>.png, <phase>/cove.png and <phase>/bb.png
    Table: corr_<phase>.csv, r of every response (rows) with every independent (columns).
"""
IND_ORDER = ['Q', 'eta_tide', 'D2', 'D1', 'tau_al', 'tau_cr', 'tau_ew', 'tau_ns', 'Tair', 'drho_pc_cp', 'drho_pc_lj', 'drho_pc_lp']
ind_cols = ['Q'] + [c for c in di.columns if c != 'Q_tide']
IND = [c for c in IND_ORDER if c in ind_cols] + [c for c in ind_cols if c not in IND_ORDER]
IND_UNITS = {'Q': 'm$^3$ s$^{-1}$, + = flood', 'eta_tide': 'm', 'D2': 'm$^3$ s$^{-1}$', 'D1': 'm$^3$ s$^{-1}$',
             'tau_al': 'Pa, + = up-cove', 'tau_cr': 'Pa, + = north', 'tau_ew': 'Pa, + = east', 'tau_ns': 'Pa, + = north',
             'Tair': '$^\\circ$C', 'drho_pc_cp': 'kg m$^{-3}$', 'drho_pc_lj': 'kg m$^{-3}$', 'drho_pc_lp': 'kg m$^{-3}$'}
print('IND (%d): %s' % (len(IND), ', '.join(IND)))
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
GROUPS = {'u': ['umax', 'uin', 'uout'], 'Q': ['Qin', 'Qout', 'Qnet'], 'dir': ['Usurf', 'Ubot', 'Unorth', 'Usouth'],
          'share': ['ain', 'Hin', 'Hout', 'Win', 'Wout'], 'pos': ['zin', 'zout', 'yin', 'yout'],
          'exch': ['rex', 'flay'], 'count': ['nlay', 'nlat', 'npin', 'npout']}
COVE = ['drhoS_pc', 'drhoT_pc', 'drhodx_top', 'drhodx_bot', 'drhodx_bar', 'phi_pc']
BB = ['Ut_ed', 'fccw_ed', 'fcol_ed', 'UbotN_bb', 'UbotS_bb']
COLORS = {'flood': 'tab:blue', 'ebb': 'tab:red'}

def scatter_grid(d, resp, phase, title, fn):
    """Columns = IND (x), rows = resp (y), one point per picked hour, r and n in each panel."""
    fig, axs = plt.subplots(len(resp), len(IND), figsize=(2.1 * len(IND) + 1, 1.9 * len(resp) + 0.8),
                            sharex='col', sharey='row', layout='constrained', squeeze=False)
    for r, yv in enumerate(resp):
        for c, xv in enumerate(IND):
            ax = axs[r, c]
            ok = d[xv].notna() & d[yv].notna()
            ax.plot(d[xv][ok], d[yv][ok], '.', color=COLORS[phase], ms=2, alpha=0.5)
            ax.text(0.03, 0.97, 'r = %+.2f\nn = %d' % (np.corrcoef(d[xv][ok], d[yv][ok])[0, 1], ok.sum()),
                    transform=ax.transAxes, va='top', fontsize=7,
                    bbox=dict(facecolor='w', alpha=0.7, lw=0, pad=1))
            ax.axhline(0, color='0.5', lw=0.5)
            ax.axvline(0, color='0.5', lw=0.5)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel(yv, fontsize=8)
            if r == len(resp) - 1:
                ax.set_xlabel('%s\n[%s]' % (xv, IND_UNITS.get(xv, '')), fontsize=8)
    fig.suptitle('%s, daily max %s hours (n = %d)' % (title, phase, len(d)))
    fig.savefig(fn, dpi=150, transparent=True)
    plt.close(fig)

for phase in ['flood', 'ebb']:
    d = dp[dp.phase == phase]
    pdir = out_dir / phase
    Lfun.make_dir(pdir)
    for g, vars in GROUPS.items():
        for sn in SECTS:
            scatter_grid(d, [vn + '_' + sn for vn in vars], phase, '%s at %s' % (g, sn), pdir / ('%s_%s.png' % (g, sn)))
    scatter_grid(d, COVE, phase, 'whole cove', pdir / 'cove.png')
    scatter_grid(d, BB, phase, 'back bay', pdir / 'bb.png')
    resp = [vn + '_' + sn for g in GROUPS.values() for sn in SECTS for vn in g] + COVE + BB
    cr = pd.DataFrame({xv: [d[xv].corr(d[yv]) for yv in resp] for xv in IND}, index=resp)
    cr.to_csv(out_dir / ('corr_%s.csv' % phase), float_format='%.3f')
    print('%s: %d figures in %s; corr_%s.csv' % (phase, len(GROUPS) * len(SECTS) + 2, pdir, phase))

"""
STEP 3: heat maps of r^2 (Pearson, from corr_<phase>.csv of STEP 2) of every response
against every independent variable, one figure per phase, so all of them show at once.
    Panels: pc_cp, pc_lj, pc_lp (rows = the section's responses, in the STEP 2 group
    order, groups split by white lines), the whole cove and, under it, the back bay;
    columns = IND.
    Each cell shows r^2 to 2 decimals; colour 0-1. r^2 drops the sign: the sign is in
    corr_<phase>.csv and the STEP 2 scatters.
    Figures: r2_<phase>.png
"""
SVARS = [vn for g in GROUPS.values() for vn in g]
GEDGE = np.cumsum([len(g) for g in GROUPS.values()])[:-1] - 0.5

def r2_panel(ax, r2, rows, edges=()):
    im = ax.imshow(r2, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    for (i, j), v in np.ndenumerate(r2):
        if np.isfinite(v):
            ax.text(j, i, '%.2f' % v, ha='center', va='center', fontsize=6, color='k' if v > 0.6 else 'w')
    for e in edges:
        ax.axhline(e, color='w', lw=1.5)
    ax.set_xticks(range(len(IND)))
    ax.set_xticklabels(IND, rotation=90, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    return im

for phase in ['flood', 'ebb']:
    cr = pd.read_csv(out_dir / ('corr_%s.csv' % phase), index_col=0)
    fig = plt.figure(figsize=(1.2 * len(IND) + 4, 10), layout='constrained')
    gs = fig.add_gridspec(len(SVARS), len(SECTS) + 2, width_ratios=[1] * len(SECTS) + [0.35, 1])   # spacer for the cove labels
    for c, sn in enumerate(SECTS):
        ax = fig.add_subplot(gs[:, c])
        im = r2_panel(ax, cr.loc[[vn + '_' + sn for vn in SVARS], IND].values ** 2, SVARS, GEDGE)
        ax.set_title(sn)
        if c > 0:
            ax.set_yticklabels([])                       # same rows as pc_cp
    ax = fig.add_subplot(gs[:len(COVE), -1])
    r2_panel(ax, cr.loc[COVE, IND].values ** 2, COVE)
    ax.set_title('whole cove')
    ax = fig.add_subplot(gs[len(COVE) + 5:len(COVE) + 5 + len(BB), -1])         # gap for the cove x labels
    r2_panel(ax, cr.loc[BB, IND].values ** 2, BB)
    ax.set_title('back bay')
    cb = fig.colorbar(im, ax=fig.axes, shrink=0.5, pad=0.01)
    cb.set_label('r$^2$')
    fig.suptitle('r$^2$ of each response with each independent variable, daily max %s hours (n = %d)'
                 % (phase, (dp.phase == phase).sum()))
    fig.savefig(out_dir / ('r2_%s.png' % phase), dpi=150, transparent=True)
    plt.close(fig)
    print('saved %s' % (out_dir / ('r2_%s.png' % phase)))
