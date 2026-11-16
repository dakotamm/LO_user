"""
Penn Cove scatter plots of the Godin-filtered independent variables against the
Godin-filtered response variables, built step by step. The subtidal version of
20261112_pc_peak_scatter.py.

Inputs (wb1_t0_xn11abbur00, 2024-2025):
    LO_output/DM_outs/20261113_pc_ind_var_godin/ind_var_godin.p  Godin, hh:00
    LO_output/DM_outs/20261112_pc_res_var_ts/res_var_ts.p        hourly hh:00 his snapshots
    LO_output/DM_outs/20261112_pc_res_var_ts/res_var_ts_avg.p    hourly hh:30 hour means

STEP 1: Godin filter the response variables and take one value a day.
    The hour-mean variables (res_var_ts_avg.p, hh:30) are put on the hh:00 index as
    the mean of the two hours bracketing it (hh-1:30 and hh:30), as in
    20261112_pc_peak_scatter.py, so every variable shares the ind_var_godin.p index.
    The first and last hh:00 have only one side and stay NaN.
    zin, zout, yin, yout are NaN in the few hours with no inflow (or no outflow) at a
    section; those gaps are filled by linear interpolation in time before filtering,
    since one NaN would blank the 71 hours of Godin output around it.
    Then zfun.lowpass(f='godin') on every column, as in 20261113_pc_ind_var_godin.py.
    Godin-filtered Qin, Qout, umax, uin, uout are the subtidal means of rectified
    tidal quantities (e.g. Godin Qin ~ the mean gross inflow), not residual flows;
    Qnet, Usurf, Ubot, ... keep their sign and are residuals.
    Daily subsample: every SUB = 24 hours at HR = 12:00 UTC (as in
    20261111_pc_metric_drivers.py). The Godin output varies over ~3 days, so the
    hourly points are not independent; SUB = 1 keeps every hour.
    Saved to LO_output/DM_outs/20261113_pc_godin_scatter/godin_daily.p, one row per
    day: every ind_var_godin.p column, then every response.

run 20261113_pc_godin_scatter.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
SUB = 24
HR = 12

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
in_dir = Ldir['LOo'] / 'DM_outs'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261113_pc_godin_scatter'
Lfun.make_dir(out_dir)
out_fn = out_dir / 'godin_daily.p'

di = pd.read_pickle(in_dir / '20261113_pc_ind_var_godin' / 'ind_var_godin.p')
dr = pd.read_pickle(in_dir / '20261112_pc_res_var_ts' / 'res_var_ts.p')
da = pd.read_pickle(in_dir / '20261112_pc_res_var_ts' / 'res_var_ts_avg.p')

# fill the short NaN gaps (hours with no in- or outflow) before anything else
gap = da.columns[da.isna().any()]
for vn in gap:
    n = da[vn].isna()
    runs = n.groupby((~n).cumsum()).sum()
    print('%-11s %3d NaN hours, longest gap %d h -> interpolated' % (vn, n.sum(), runs.max()))
da = da.interpolate(method='time', limit_direction='both')

# hh:30 hour means -> hh:00
assert (di.index == dr.index).all()
ah = pd.DataFrame(np.nan, index=dr.index, columns=da.columns)
assert (da.index == dr.index[:-1] + pd.Timedelta(minutes=30)).all()
ah.iloc[1:-1] = 0.5 * (da.values[:-1] + da.values[1:])
resp = pd.concat([dr, ah], axis=1)

# Godin filter every response (the hh:30 columns carry one NaN at each end; lowpass pads 35 h anyway)
rg = pd.DataFrame(index=resp.index)
for vn in resp.columns:
    rg[vn] = zfun.lowpass(np.nan_to_num(resp[vn].values.astype(float)), f='godin')

dg = pd.concat([di, rg], axis=1)
assert dg.index[0].hour == 0
dd = dg.iloc[HR::SUB]                                    # hh = HR, every SUB hours
dd = dd.dropna(how='all', subset=rg.columns)
assert dd.notna().all().all()

print('%d independent, %d responses; %d samples (every %d h at %02d:00), %s to %s'
      % (di.shape[1], rg.shape[1], len(dd), SUB, HR, str(dd.index[0])[:10], str(dd.index[-1])[:10]))
print('%-12s %12s %12s' % ('', 'hourly std', 'Godin std'))
for vn in ['umax_pc_lj', 'Qin_pc_lj', 'Qnet_pc_lj', 'Usurf_pc_lj', 'flay_pc_lj', 'phi_pc', 'Ut_ed']:
    print('%-12s %12.4g %12.4g' % (vn, resp[vn].std(), rg[vn].std()))

dd.to_pickle(out_fn)
print('saved %s (%d rows, %d columns)' % (out_fn, len(dd), dd.shape[1]))

"""
STEP 2: scatter of each Godin-filtered independent variable against each Godin-filtered
response, one point per day (STEP 1).
    Independent (ind_var_godin.p, columns):
        D2, D1      spring/neap and tropic/equatorial envelopes              [m3/s]
        tau_al      along-cove wind stress, + = up-cove                      [Pa]
        tau_cr      cross-cove wind stress, + = toward the north shore       [Pa]
        tau_ew      east-west wind stress over upper Saratoga, + = east      [Pa]
        tau_ns      north-south wind stress over upper Saratoga, + = north   [Pa]
        Tair        air temperature over Penn Cove                           [degC]
        drho_<sect> top-bottom density difference at each section            [kg/m3]
    IND is every ind_var_godin.p column; IND_ORDER only sets the order of the ones it
    names.
    Responses (rows), grouped as in 20261112_pc_peak_scatter.py; one figure per group
    and section (rows = that group's <var>_<sect>), plus one for the whole-cove,
    eddy and back-bay variables:
        u      umax, uin, uout                        [m/s]
        Q      Qin, Qout, Qnet                        [m3/s], + = in
        dir    Usurf, Ubot, Unorth, Usouth            [m/s], + = in
        share  ain, Hin, Hout, Win, Wout              [0-1]
        pos    zin, zout, yin, yout                   sig (bed -1, surface 0), y (south 0, north 1)
        exch   rex, flay                              [0-1]
        count  nlay, nlat, npin, npout                (Godin: the mean count)
        cove   every response that is not <var>_<sect>: drhoS_pc, drhoT_pc,
               drhodx_top, drhodx_bot, drhodx_bar, phi_pc, Ut_ed, fccw_ed, fcol_ed,
               UbotN_bb, UbotS_bb (bottom-half along-cove velocity, north / south
               back bay, + = toward the head) [m/s]; columns added to res_var_ts.p
               later show up here
    Panels share x down each column and y across each row. Each is annotated with the
    Pearson r and n.
    Figures: scatter/<group>_<sect>.png and scatter/cove.png
    Table: corr.csv, r of every response (rows) with every independent (columns).
"""
IND_ORDER = ['D2', 'D1', 'tau_al', 'tau_cr', 'tau_ew', 'tau_ns', 'Tair', 'drho_pc_cp', 'drho_pc_lj', 'drho_pc_lp']
IND = [c for c in IND_ORDER if c in di.columns] + [c for c in di.columns if c not in IND_ORDER]
IND_UNITS = {'D2': 'm$^3$ s$^{-1}$', 'D1': 'm$^3$ s$^{-1}$',
             'tau_al': 'Pa, + = up-cove', 'tau_cr': 'Pa, + = north', 'tau_ew': 'Pa, + = east', 'tau_ns': 'Pa, + = north',
             'Tair': '$^\\circ$C', 'drho_pc_cp': 'kg m$^{-3}$', 'drho_pc_lj': 'kg m$^{-3}$', 'drho_pc_lp': 'kg m$^{-3}$'}
print('IND (%d): %s' % (len(IND), ', '.join(IND)))

SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
GROUPS = {'u': ['umax', 'uin', 'uout'], 'Q': ['Qin', 'Qout', 'Qnet'], 'dir': ['Usurf', 'Ubot', 'Unorth', 'Usouth'],
          'share': ['ain', 'Hin', 'Hout', 'Win', 'Wout'], 'pos': ['zin', 'zout', 'yin', 'yout'],
          'exch': ['rex', 'flay'], 'count': ['nlay', 'nlat', 'npin', 'npout']}
SRESP = [vn + '_' + sn for g in GROUPS.values() for sn in SECTS for vn in g]
COVE = [c for c in rg.columns if c not in SRESP]
print('COVE (%d): %s' % (len(COVE), ', '.join(COVE)))
COLOR = 'tab:purple'

def scatter_grid(d, resp, title, fn):
    """Columns = IND (x), rows = resp (y), one point per day, r and n in each panel."""
    fig, axs = plt.subplots(len(resp), len(IND), figsize=(2.1 * len(IND) + 1, 1.9 * len(resp) + 0.8),
                            sharex='col', sharey='row', layout='constrained', squeeze=False)
    for r, yv in enumerate(resp):
        for c, xv in enumerate(IND):
            ax = axs[r, c]
            ax.plot(d[xv], d[yv], '.', color=COLOR, ms=2, alpha=0.5)
            ax.text(0.03, 0.97, 'r = %+.2f\nn = %d' % (np.corrcoef(d[xv], d[yv])[0, 1], len(d)),
                    transform=ax.transAxes, va='top', fontsize=7,
                    bbox=dict(facecolor='w', alpha=0.7, lw=0, pad=1))
            ax.axhline(0, color='0.5', lw=0.5)                 # zero lines; they also pull 0 into the axis limits
            ax.axvline(0, color='0.5', lw=0.5)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel(yv, fontsize=8)
            if r == len(resp) - 1:
                ax.set_xlabel('%s\n[%s]' % (xv, IND_UNITS.get(xv, '')), fontsize=8)
    fig.suptitle('%s, Godin-filtered, one point per day at %02d:00 (n = %d)' % (title, HR, len(d)))
    fig.savefig(fn, dpi=150, transparent=True)
    plt.close(fig)

sdir = out_dir / 'scatter'
Lfun.make_dir(sdir)
for g, vars in GROUPS.items():
    for sn in SECTS:
        scatter_grid(dd, [vn + '_' + sn for vn in vars], '%s at %s' % (g, sn), sdir / ('%s_%s.png' % (g, sn)))
scatter_grid(dd, COVE, 'whole cove and back bay', sdir / 'cove.png')

RESP = SRESP + COVE
cr = pd.DataFrame({xv: [dd[xv].corr(dd[yv]) for yv in RESP] for xv in IND}, index=RESP)
cr.to_csv(out_dir / 'corr.csv', float_format='%.3f')
print('%d figures in %s; corr.csv' % (len(GROUPS) * len(SECTS) + 1, sdir))

"""
STEP 3: heat map of r^2 (Pearson, from corr.csv of STEP 2) of every response against
every independent variable, so all of them show at once.
    Panels: pc_cp, pc_lj, pc_lp (rows = the section's responses, in the STEP 2 group
    order, groups split by white lines) and the whole cove; columns = IND.
    Each cell shows r^2 to 2 decimals; colour 0-1. r^2 drops the sign: the sign is in
    corr.csv and the STEP 2 scatters.
    Figure: r2.png
"""
SVARS = [vn for g in GROUPS.values() for vn in g]
GEDGE = np.cumsum([len(g) for g in GROUPS.values()])[:-1] - 0.5

def r2_panel(ax, r2, rows, edges=()):
    im = ax.imshow(r2, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    for (i, j), v in np.ndenumerate(r2):
        ax.text(j, i, '%.2f' % v, ha='center', va='center', fontsize=6, color='k' if v > 0.6 else 'w')
    for e in edges:
        ax.axhline(e, color='w', lw=1.5)
    ax.set_xticks(range(len(IND)))
    ax.set_xticklabels(IND, rotation=90, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    return im

cr = pd.read_csv(out_dir / 'corr.csv', index_col=0)
fig = plt.figure(figsize=(1.2 * len(IND) + 4, 10), layout='constrained')
gs = fig.add_gridspec(len(SVARS), len(SECTS) + 2, width_ratios=[1] * len(SECTS) + [0.35, 1])   # spacer for the cove labels
for c, sn in enumerate(SECTS):
    ax = fig.add_subplot(gs[:, c])
    im = r2_panel(ax, cr.loc[[vn + '_' + sn for vn in SVARS], IND].values ** 2, SVARS, GEDGE)
    ax.set_title(sn)
    if c > 0:
        ax.set_yticklabels([])                           # same rows as pc_cp
ax = fig.add_subplot(gs[:len(COVE), -1])
r2_panel(ax, cr.loc[COVE, IND].values ** 2, COVE)
ax.set_title('whole cove and back bay')
cb = fig.colorbar(im, ax=fig.axes, shrink=0.5, pad=0.01)
cb.set_label('r$^2$')
fig.suptitle('r$^2$ of each Godin-filtered response with each Godin-filtered independent variable (n = %d days)' % len(dd))
fig.savefig(out_dir / 'r2.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'r2.png'))
