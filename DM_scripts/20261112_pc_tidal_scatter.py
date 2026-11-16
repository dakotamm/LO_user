"""
Penn Cove scatter plots of the response variables against the tidal transport Q_tide,
every hour, to show how each varies through the tidal cycle. Built step by step.

Inputs (wb1_t0_xn11abbur00, 2024-2025):
    LO_output/DM_outs/20261112_pc_ind_var_ts/ind_var_ts.p       hh:00 his snapshots
    LO_output/DM_outs/20261112_pc_res_var_ts/res_var_ts.p       hh:00 his snapshots
    LO_output/DM_outs/20261112_pc_res_var_ts/res_var_ts_avg.p   hh:30 hour means

STEP 1: read the independent and response variables.
    Q_tide  [m3/s]  astronomical tidal transport through the pc_lp mouth (harmonic
            prediction, no wind or subtidal signal; 20261112_pc_ind_var_ts.py STEP 6).
            Flipped here so + = into the cove = flood, the sign of Qin/Qnet in
            res_var_ts_avg.p (ind_var_ts.p has + = east, out of the cove).
    The responses stay on their own time bases, in two frames:
        dh  hh:00 (17545 hours): Q_tide + every res_var_ts.p column, exact times
        da  hh:30 (17544 hours): Q_tide + every res_var_ts_avg.p column, with Q_tide
            the mean of the two snapshots bracketing the hour (hh:00 and hh+1:00), the
            trapezoid estimate of the hour mean, as in 20261112_pc_peak_scatter.py.
    Only Q_tide is moved (a smooth harmonic sum: the trapezoid mean of an M2 sinusoid
    is within ~2% of the true hour mean); no response is resampled.

run 20261112_pc_tidal_scatter.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import pandas as pd

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
XV = 'Q_tide'

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
in_dir = Ldir['LOo'] / 'DM_outs'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261112_pc_tidal_scatter'
Lfun.make_dir(out_dir)

di = pd.read_pickle(in_dir / '20261112_pc_ind_var_ts' / 'ind_var_ts.p')
dr = pd.read_pickle(in_dir / '20261112_pc_res_var_ts' / 'res_var_ts.p')
dra = pd.read_pickle(in_dir / '20261112_pc_res_var_ts' / 'res_var_ts_avg.p')

x = -di[XV]                                              # + = flood

# hh:00 frame
assert (x.index == dr.index).all()
dh = pd.concat([x, dr], axis=1)

# hh:30 frame
xa = pd.Series(0.5 * (x.values[:-1] + x.values[1:]), name=XV,
               index=x.index[:-1] + pd.Timedelta(minutes=30))
assert (xa.index == dra.index).all()
da = pd.concat([xa, dra], axis=1)

print('%s (+ = flood): std %.0f, min %+.0f, max %+.0f m3/s' % (XV, x.std(), x.min(), x.max()))
print('dh hh:00: %d hours, %d responses' % (len(dh), dh.shape[1] - 1))
print('da hh:30: %d hours, %d responses' % (len(da), da.shape[1] - 1))
print('sign check: r(%s, Qnet_pc_lp) = %+.3f on hh:30' % (XV, np.corrcoef(da[XV], da['Qnet_pc_lp'])[0, 1]))

"""
STEP 2: scatter of the maximum section velocities against Q_tide, every hh:00 hour
(dh) of one tidal (lunar) day from 2024-08-31 00:00 (T0 <= time < T1 = T0 + 24.84 h,
25 hours, both floods and both ebbs), drawn as a spaghetti line through the hours in
time order, coloured by time (hours since T0, the same colour scale in every figure),
the variables of 20261112_pc_res_var_ts.py STEP 1:
    umax   max |u| over the section, flood or ebb               [m/s]
    uin    max of the inflow (westward) speed, + = into the cove
    uout   max of the outflow (eastward) speed, + = out of the cove
Rows are the sections head -> mouth (pc_cp, pc_lj, pc_lp), columns the variables;
all panels share both axes.
    Figure: tidal_scatter_u.png
"""
U_VARS = ['umax', 'uin', 'uout']
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
T0 = pd.Timestamp('2024-08-31')
T1 = T0 + pd.Timedelta(hours=24.84)
TITLE = '%s + one tidal day (24.84 h)' % T0.strftime('%Y-%m-%d %H:%M')

CMAP = 'viridis'
NORM = plt.Normalize(0, (T1 - T0) / pd.Timedelta(hours=1))

def spaghetti(ax, d, vn, t0=T0, norm=NORM):
    """d[vn] against d[XV], joined in time order, each segment and point coloured by hours since t0."""
    thr = ((d.index - t0) / pd.Timedelta(hours=1)).values
    xy = np.c_[d[XV].values, d[vn].values]
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(seg, cmap=CMAP, norm=norm, lw=1)
    lc.set_array(0.5 * (thr[:-1] + thr[1:]))                # segment colour = its midpoint time
    ax.add_collection(lc)
    return ax.scatter(xy[:, 0], xy[:, 1], c=thr, cmap=CMAP, norm=norm, s=8, lw=0, zorder=3)

def time_cbar(fig, axs, sc):
    cb = fig.colorbar(sc, ax=axs, shrink=0.8, pad=0.02)
    cb.set_label('hours since %s' % T0.strftime('%Y-%m-%d %H:%M'))

dw = dh[(dh.index >= T0) & (dh.index < T1)]
print('STEP 2: %s to %s, %d hours' % (T0, T1, len(dw)))

fig, axs = plt.subplots(len(SECTS), len(U_VARS), figsize=(12, 9), sharex=True, sharey=True, layout='constrained')
for r, sn in enumerate(SECTS):
    for c, vn in enumerate(U_VARS):
        ax = axs[r, c]
        sc = spaghetti(ax, dw, vn + '_' + sn)
        ax.axhline(0, color='0.5', lw=0.5)
        ax.axvline(0, color='0.5', lw=0.5)
        ax.grid(alpha=0.3)
        if r == 0:
            ax.set_title(vn)
        if c == 0:
            ax.set_ylabel(sn + '\n[m s$^{-1}$]')
        if r == len(SECTS) - 1:
            ax.set_xlabel(XV + ' [m$^3$ s$^{-1}$], + = flood')
fig.suptitle(TITLE)
time_cbar(fig, axs, sc)
fig.savefig(out_dir / 'tidal_scatter_u.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'tidal_scatter_u.png'))

"""
STEP 3: scatter of the inflow, outflow and net section transports against Q_tide, every
hh:30 hour (da, Q_tide as the hour mean) of the same tidal day as STEP 2, as a spaghetti line, the variables of
20261112_pc_res_var_ts.py STEP 2 (hour-mean Huon, + = into the cove):
    Qin    sum of the elements flowing in                       [m3/s]
    Qout   sum of the elements flowing out (<= 0)
    Qnet   Qin + Qout, the whole-section transport
Rows are the sections head -> mouth, columns the variables; all panels share both axes.
    Figure: tidal_scatter_Q.png
"""
Q_VARS = ['Qin', 'Qout', 'Qnet']

dwa = da[(da.index >= T0) & (da.index < T1)]
print('STEP 3: %s to %s, %d hours' % (T0, T1, len(dwa)))

fig, axs = plt.subplots(len(SECTS), len(Q_VARS), figsize=(12, 9), sharex=True, sharey=True, layout='constrained')
for r, sn in enumerate(SECTS):
    for c, vn in enumerate(Q_VARS):
        ax = axs[r, c]
        sc = spaghetti(ax, dwa, vn + '_' + sn)
        ax.axhline(0, color='0.5', lw=0.5)
        ax.axvline(0, color='0.5', lw=0.5)
        ax.grid(alpha=0.3)
        if r == 0:
            ax.set_title(vn)
        if c == 0:
            ax.set_ylabel(sn + '\n[m$^3$ s$^{-1}$], + = in')
        if r == len(SECTS) - 1:
            ax.set_xlabel(XV + ' [m$^3$ s$^{-1}$], + = flood')
fig.suptitle(TITLE)
time_cbar(fig, axs, sc)
fig.savefig(out_dir / 'tidal_scatter_Q.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'tidal_scatter_Q.png'))

"""
STEP 4: scatter of the along-cove density gradient against Q_tide, every hh:00 hour (dh)
of the same tidal day as STEP 2, as a spaghetti line, the variables of 20261112_pc_res_var_ts.py STEP 4:
    drhodx_top   head minus mouth sigma0 at the top level, / 3.8 km    [kg/m3/km]
    drhodx_bot   the same at the bottom level
    drhodx_bar   the same for the depth-averaged sigma0
    + = denser toward the head
All panels share both axes.
    Figure: tidal_scatter_drhodx.png
"""
X_VARS = ['drhodx_top', 'drhodx_bot', 'drhodx_bar']

fig, axs = plt.subplots(1, len(X_VARS), figsize=(14, 4.5), sharex=True, sharey=True, layout='constrained')
for c, vn in enumerate(X_VARS):
    ax = axs[c]
    sc = spaghetti(ax, dw, vn)
    ax.axhline(0, color='0.5', lw=0.5)
    ax.axvline(0, color='0.5', lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_title(vn)
    ax.set_xlabel(XV + ' [m$^3$ s$^{-1}$], + = flood')
axs[0].set_ylabel('[kg m$^{-3}$ km$^{-1}$], + = denser at head')
fig.suptitle(TITLE)
time_cbar(fig, axs, sc)
fig.savefig(out_dir / 'tidal_scatter_drhodx.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'tidal_scatter_drhodx.png'))

"""
STEP 5: spaghetti of the whole-cove potential energy anomaly against Q_tide, every hh:00
hour (dh) of the same tidal day as STEP 2, the variable of 20261112_pc_res_var_ts.py STEP 5:
    phi_pc   area-weighted cove mean of phi, the energy per volume needed to mix the
             column completely (0 = mixed, + = stable)            [J/m3]
    Figure: tidal_scatter_phi.png
"""
fig, ax = plt.subplots(figsize=(6.5, 4.5), layout='constrained')
sc = spaghetti(ax, dw, 'phi_pc')
ax.axhline(0, color='0.5', lw=0.5)
ax.axvline(0, color='0.5', lw=0.5)
ax.grid(alpha=0.3)
ax.set_title('phi_pc')
ax.set_xlabel(XV + ' [m$^3$ s$^{-1}$], + = flood')
ax.set_ylabel('[J m$^{-3}$], + = stable')
fig.suptitle(TITLE)
time_cbar(fig, ax, sc)
fig.savefig(out_dir / 'tidal_scatter_phi.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'tidal_scatter_phi.png'))

"""
STEP 6: spaghetti of the flow-shape metrics against Q_tide, every hh:30 hour (da, Q_tide as
the hour mean) of the same tidal day as STEP 2, the variables of 20261112_pc_res_var_ts.py
STEP 6 (+ = into the cove), grouped by its numbering into five figures. In each, rows
are the sections head -> mouth, columns the metrics; all panels share both axes.
    tidal_scatter_dir.png     1-2  Usurf, Ubot, Unorth, Usouth     mean u over the top /
                                   bottom row, north / south face column       [m/s]
    tidal_scatter_share.png   3-5  ain, Hin, Hout, Win, Wout       share of section area
                                   flowing in; layered thickness and lateral width
                                   shares of in- and outflow                   [0-1]
    tidal_scatter_pos.png     6    zin, zout, yin, yout            |q|-weighted mean sig
                                   (-1 bed, 0 surface) and y (0 south, 1 north) of the
                                   in- and outflow
    tidal_scatter_exch.png    7-8  rex, flay                       exchange ratio (0 one
                                   way, 1 pure exchange); layered share of the
                                   counterflow variance                        [0-1]
    tidal_scatter_count.png   9    nlay, nlat, npin, npout         layers in Qz(z), lateral
                                   bands in Qp(p), inflow / outflow patches
"""
def section_grid(d, vars, ylabel, fn):
    """Rows = sections head -> mouth, columns = vars, spaghetti of d[<var>_<sect>] against Q_tide."""
    fig, axs = plt.subplots(len(SECTS), len(vars), figsize=(3.2 * len(vars) + 1.5, 9), sharex=True, sharey=True,
                            layout='constrained', squeeze=False)
    for r, sn in enumerate(SECTS):
        for c, vn in enumerate(vars):
            ax = axs[r, c]
            sc = spaghetti(ax, d, vn + '_' + sn)
            ax.axhline(0, color='0.5', lw=0.5)
            ax.axvline(0, color='0.5', lw=0.5)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(vn)
            if c == 0:
                ax.set_ylabel(sn + '\n' + ylabel)
            if r == len(SECTS) - 1:
                ax.set_xlabel(XV + ' [m$^3$ s$^{-1}$], + = flood')
    fig.suptitle(TITLE)
    time_cbar(fig, axs, sc)
    fig.savefig(out_dir / fn, dpi=150, transparent=True)
    plt.close(fig)
    print('saved %s' % (out_dir / fn))

section_grid(dwa, ['Usurf', 'Ubot', 'Unorth', 'Usouth'], '[m s$^{-1}$], + = in', 'tidal_scatter_dir.png')
section_grid(dwa, ['ain', 'Hin', 'Hout', 'Win', 'Wout'], 'share', 'tidal_scatter_share.png')
section_grid(dwa, ['zin', 'zout', 'yin', 'yout'], 'sig (bed -1, surface 0)\ny (south 0, north 1)', 'tidal_scatter_pos.png')
section_grid(dwa, ['rex', 'flay'], '[0-1]', 'tidal_scatter_exch.png')
section_grid(dwa, ['nlay', 'nlat', 'npin', 'npout'], 'count', 'tidal_scatter_count.png')

"""
STEP 7: the same spaghetti plots for the candidate lunar days of
20261113_pc_candidate_days.py: the chosen day of each case in picks.csv (its STEP 6:
the best order-matched day, or a hand pick; baseline, spring, neap, tropic, equatorial,
high / low strat, up- / down-cove wind), each day t0 <= time < t1 (flood start -> flood start, 24-26 h).
    pc_lj only for the section variables; drhodx_* and phi_pc are whole-cove.
    One figure per variable group of STEPS 2-6: rows = the days, columns = the
    variables; all panels share both axes. Colour = hours since the day's t0, on one
    0-26 h scale, so a colour is the same point in the day in every row.
    hh:00 variables from dh, hh:30 variables from da (Q_tide as the hour mean), as before.
    Figures: cand_lj_<group>.png
"""
CAND_FN = in_dir / '20261113_pc_candidate_days' / 'picks.csv'
SN = 'pc_lj'
NORM_D = plt.Normalize(0, 26)

cand = pd.read_csv(CAND_FN, parse_dates=['t0', 't1'])
print('STEP 7: %d candidate days' % len(cand))
for i, row in cand.iterrows():
    print('    %-11s %s to %s (n = %d)' % (row.case, row.t0, row.t1, row.n))

def day_grid(d, vars, sfx, ylabel, fn, where, sharey=True):
    """Rows = candidate days, columns = vars, spaghetti of d[<var><sfx>] against Q_tide."""
    fig, axs = plt.subplots(len(cand), len(vars), figsize=(3.2 * len(vars) + 2.5, 2.1 * len(cand)), sharex=True, sharey=sharey,
                            layout='constrained', squeeze=False)
    for r, row in cand.iterrows():
        dd = d[(d.index >= row.t0) & (d.index < row.t1)]
        for c, vn in enumerate(vars):
            ax = axs[r, c]
            sc = spaghetti(ax, dd, vn + sfx, t0=row.t0, norm=NORM_D)
            ax.axhline(0, color='0.5', lw=0.5)
            ax.axvline(0, color='0.5', lw=0.5)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(vn)
            if c == 0:
                ax.set_ylabel('%s\n%s\n%s' % (row.case, row.t0.strftime('%Y-%m-%d %H:%M'), ylabel), fontsize=8)
            if r == len(cand) - 1:
                ax.set_xlabel(XV + ' [m$^3$ s$^{-1}$], + = flood')
    fig.suptitle('Candidate lunar days, %s' % where)
    cb = fig.colorbar(sc, ax=axs, shrink=0.4, pad=0.02)
    cb.set_label("hours since the day's t0 (flood start)")
    fig.savefig(out_dir / fn, dpi=150, transparent=True)
    plt.close(fig)
    print('saved %s' % (out_dir / fn))

sfx = '_' + SN
day_grid(dh, U_VARS, sfx, '[m s$^{-1}$]', 'cand_lj_u.png', SN)
day_grid(da, Q_VARS, sfx, '[m$^3$ s$^{-1}$], + = in', 'cand_lj_Q.png', SN)
day_grid(dh, X_VARS, '', '[kg m$^{-3}$ km$^{-1}$]', 'cand_lj_drhodx.png', 'whole cove')
day_grid(dh, ['phi'], '_pc', '[J m$^{-3}$]', 'cand_lj_phi.png', 'whole cove')
day_grid(da, ['Usurf', 'Ubot', 'Unorth', 'Usouth'], sfx, '[m s$^{-1}$], + = in', 'cand_lj_dir.png', SN)
day_grid(da, ['ain', 'Hin', 'Hout', 'Win', 'Wout'], sfx, 'share', 'cand_lj_share.png', SN)
day_grid(da, ['zin', 'zout', 'yin', 'yout'], sfx, 'sig / y', 'cand_lj_pos.png', SN)
day_grid(da, ['rex', 'flay'], sfx, '[0-1]', 'cand_lj_exch.png', SN)
day_grid(da, ['nlay', 'nlat', 'npin', 'npout'], sfx, 'count', 'cand_lj_count.png', SN)

"""
STEP 8: the new back-bay response variables of 20261112_pc_res_var_ts.py, all hh:00 (dh),
for the tidal day of STEP 2 and for the candidate days of STEP 7, as before.
    eddy (its STEP 7), the ring of fastest counter-clockwise swirl west of pc_cp:
        Ut_ed     ring-mean azimuthal velocity, depth mean, + = counter-clockwise  [m/s]
        fccw_ed   fraction of ring cells turning counter-clockwise                 [0-1]
        fcol_ed   fraction of the water column turning counter-clockwise           [0-1]
        Different units, so each column has its own y axis (shared down the column).
    back bay (its STEP 8), along-cove velocity in the bottom half of the column, north
    and south halves of the back bay, + = toward the head (into the cove):
        UbotN_bb, UbotS_bb                                                         [m/s]
        Both share the y axis.
    Figures: tidal_scatter_eddy.png, tidal_scatter_bb.png (tidal day of STEP 2);
             cand_eddy.png, cand_bb.png (candidate days)
"""
ED_VARS = ['Ut_ed', 'fccw_ed', 'fcol_ed']
BB_VARS = ['UbotN_bb', 'UbotS_bb']

def one_row(d, vars, ylabels, fn, sharey):
    """One row, columns = vars, spaghetti of d[var] against Q_tide over the STEP 2 tidal day."""
    fig, axs = plt.subplots(1, len(vars), figsize=(4.5 * len(vars) + 1, 4.5), sharex=True, sharey=sharey, layout='constrained')
    for c, vn in enumerate(vars):
        ax = axs[c]
        sc = spaghetti(ax, d, vn)
        ax.axhline(0, color='0.5', lw=0.5)
        ax.axvline(0, color='0.5', lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_title(vn)
        ax.set_xlabel(XV + ' [m$^3$ s$^{-1}$], + = flood')
        if ylabels[c]:
            ax.set_ylabel(ylabels[c])
    fig.suptitle(TITLE)
    time_cbar(fig, axs, sc)
    fig.savefig(out_dir / fn, dpi=150, transparent=True)
    plt.close(fig)
    print('saved %s' % (out_dir / fn))

one_row(dw, ED_VARS, ['[m s$^{-1}$], + = counter-clockwise', 'fraction of ring', 'fraction of column'], 'tidal_scatter_eddy.png', False)
one_row(dw, BB_VARS, ['[m s$^{-1}$], + = toward head', ''], 'tidal_scatter_bb.png', True)
day_grid(dh, ED_VARS, '', '[m s$^{-1}$] / [0-1]', 'cand_eddy.png', 'back-bay eddy', sharey='col')
day_grid(dh, BB_VARS, '', '[m s$^{-1}$], + = toward head', 'cand_bb.png', 'back bay, bottom half')

"""
STEP 9: the tidal hysteresis of the baseline day only, pc_lj only, every section variable
on one page. Baseline = the 'baseline' row of picks.csv (STEP 7), t0 <= time < t1.
    Rows = the variable groups of STEPS 2, 3 and 6 (u, Q, dir, share, pos, exch, count),
    columns = the variables of the group; each row shares its y axis (one unit per row),
    all panels share the x axis. Colour = hours since t0, as in STEP 7.
    Figure: baseline_lj.png
"""
LJ_GROUPS = [(dh, U_VARS, '[m s$^{-1}$]'),
             (da, Q_VARS, '[m$^3$ s$^{-1}$], + = in'),
             (da, ['Usurf', 'Ubot', 'Unorth', 'Usouth'], '[m s$^{-1}$], + = in'),
             (da, ['ain', 'Hin', 'Hout', 'Win', 'Wout'], 'share'),
             (da, ['zin', 'zout', 'yin', 'yout'], 'sig / y'),
             (da, ['rex', 'flay'], '[0-1]'),
             (da, ['nlay', 'nlat', 'npin', 'npout'], 'count')]
NCOL = max(len(g[1]) for g in LJ_GROUPS)
base = cand[cand.case == 'baseline'].iloc[0]
print('STEP 9: baseline %s to %s' % (base.t0, base.t1))

fig, axs = plt.subplots(len(LJ_GROUPS), NCOL, figsize=(3.2 * NCOL + 1.5, 2.3 * len(LJ_GROUPS)), sharex=True,
                        layout='constrained', squeeze=False)
for r, (d, vars, ylabel) in enumerate(LJ_GROUPS):
    dd = d[(d.index >= base.t0) & (d.index < base.t1)]
    for c in range(NCOL):
        ax = axs[r, c]
        if c >= len(vars):
            ax.axis('off')
            continue
        if c > 0:
            ax.sharey(axs[r, 0])
        sc = spaghetti(ax, dd, vars[c] + sfx, t0=base.t0, norm=NORM_D)
        ax.axhline(0, color='0.5', lw=0.5)
        ax.axvline(0, color='0.5', lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_title(vars[c], fontsize=10)
        if c == 0:
            ax.set_ylabel(ylabel)
        if r == len(LJ_GROUPS) - 1 or c >= len(LJ_GROUPS[r + 1][1]):     # nothing below: label the x axis here
            ax.set_xlabel(XV + ' [m$^3$ s$^{-1}$], + = flood')
            ax.tick_params(labelbottom=True)
fig.suptitle('Baseline lunar day %s UTC, %s' % (base.t0.strftime('%Y-%m-%d %H:%M'), SN))
cb = fig.colorbar(sc, ax=axs, shrink=0.4, pad=0.02)
cb.set_label("hours since the day's t0 (flood start)")
fig.savefig(out_dir / 'baseline_lj.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'baseline_lj.png'))

"""
STEP 10: the first row of STEP 9 on its own: umax, uin, uout at pc_lj over the baseline
day, sharing both axes.
    Figure: baseline_lj_u.png
"""
dd = dh[(dh.index >= base.t0) & (dh.index < base.t1)]
fig, axs = plt.subplots(1, len(U_VARS), figsize=(13, 4.5), sharex=True, sharey=True, layout='constrained')
for c, vn in enumerate(U_VARS):
    ax = axs[c]
    sc = spaghetti(ax, dd, vn + sfx, t0=base.t0, norm=NORM_D)
    ax.axhline(0, color='0.5', lw=0.5)
    ax.axvline(0, color='0.5', lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_title(vn)
    ax.set_xlabel(XV + ' [m$^3$ s$^{-1}$], + = flood')
axs[0].set_ylabel('[m s$^{-1}$]')
fig.suptitle('Baseline lunar day %s UTC, %s' % (base.t0.strftime('%Y-%m-%d %H:%M'), SN))
cb = fig.colorbar(sc, ax=axs, shrink=0.8, pad=0.02)
cb.set_label("hours since the day's t0 (flood start)")
fig.savefig(out_dir / 'baseline_lj_u.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'baseline_lj_u.png'))

"""
STEP 11: STEP 10 with the 2024 wind cases of picks.csv added (up-cove and down-cove wind,
the other axes held at the baseline's), all on the same panels: umax, uin, uout at pc_lj
against Q_tide, each day t0 <= time < t1.
    Coloured by case, not by time (three loops in one panel); the open circle marks
    each day's t0 (flood start), so the direction round each loop shows.
    Figure: baseline_wind_lj_u.png
"""
WIND_CASES = {'baseline': 'k', 'up-cove': 'tab:blue', 'down-cove': 'tab:red'}

fig, axs = plt.subplots(1, len(U_VARS), figsize=(13, 4.5), sharex=True, sharey=True, layout='constrained')
for case, col in WIND_CASES.items():
    row = cand[cand.case == case].iloc[0]
    dd = dh[(dh.index >= row.t0) & (dh.index < row.t1)]
    for c, vn in enumerate(U_VARS):
        ax = axs[c]
        ax.plot(dd[XV], dd[vn + sfx], '-o', color=col, lw=1, ms=2.5,
                label='%s, %s' % (case, row.t0.strftime('%Y-%m-%d %H:%M')))
        ax.plot(dd[XV].iloc[0], dd[vn + sfx].iloc[0], 'o', mfc='none', mec=col, ms=9, mew=1.2)
for c, vn in enumerate(U_VARS):
    ax = axs[c]
    ax.axhline(0, color='0.5', lw=0.5)
    ax.axvline(0, color='0.5', lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_title(vn)
    ax.set_xlabel(XV + ' [m$^3$ s$^{-1}$], + = flood')
axs[0].set_ylabel('[m s$^{-1}$]')
fig.legend(*axs[0].get_legend_handles_labels(), fontsize=9, loc='outside lower center', ncol=len(WIND_CASES))
fig.suptitle('Baseline and 2024 wind lunar days, %s (open circle = t0, flood start)' % SN)
fig.savefig(out_dir / 'baseline_wind_lj_u.png', dpi=150, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'baseline_wind_lj_u.png'))
