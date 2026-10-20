"""
TEF vs Eulerian -dQ/ds at the Penn Cove mouth (pc_lp), averaged over each month
of 2025 -- the monthly version of MacCready (2011) Fig. 3, which he averages
over one spring-neap cycle (our record-mean version is
pc_lp_dQds_tef_vs_eu.png from 20261013_maccready2011_eu_vs_tef_avg.py).

    TEF       -dQ/ds  = < q(s_i, t) > / ds   hourly transport binned by the hourly
                                              salinity, then Godin filtered
    Eulerian  -dQ^Eu/ds (3.10)               filtered (u0 + u1) dA0 binned by the
                                              filtered salinity s0 + s1
Both are hourly, 1000 bins over 0-36 g/kg, positive = inflow (into the cove).
Each month is the mean over its valid hours.

SOURCE (-src)
    avg   hourly-AVERAGE ROMS output (ocean_avg Huon, hh:30), the _avg scripts.
          Closes the cove volume and salt budgets.  (default)
    snap  hourly SNAPSHOT output (ocean_his u, on the hour), the original scripts.
          Sub-hourly motion aliases into the subtidal band: the snapshot section
          transport has ~6x the true subtidal variance (cove volume budget r = 0.24).

Monthly TEF terms are averaged as in MacCready (2011) section 4: Q_in, Q_out
directly, salinities flux weighted (4.1), [[s_in]] = [[Q_in s_in]] / [[Q_in]].

INPUTS (run first)
    avg:  20261013_maccready2011_sec3_avg.py, 20261013_maccready2011_eu_vs_tef_avg.py
    snap: 20261013_maccready2011_sec3.py,     20261013_maccready2011_eu_vs_tef.py
    -> pc_lp_sec3.nc (qbar, ok) and pc_lp_eu_vs_tef.nc (qEu, Q_in/out, F_in/out)

OUTPUT LO_output/DM_outs/20261020_pc_dQds_monthly_2025_avg/    (avg)
       LO_output/DM_outs/20261020_pc_dQds_monthly_2025_snap/   (snap)
    pc_lp_dQds_monthly_2025.png        12 panels, shared axes
    pc_lp_monthly_2025.csv             monthly Q_in, Q_out, s_in, s_out, Ds, hours
    by_month/pc_lp_dQds_2025-MM.png    one figure per month, own axis limits

run 20261020_pc_dQds_monthly_2025.py -src avg
run 20261020_pc_dQds_monthly_2025.py -src snap
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
parser.add_argument('-src', choices=['avg', 'snap'], default='avg')
args = parser.parse_args()

SN = 'pc_lp'
YEAR = 2025
IN_SUF = {'avg': '_avg', 'snap': ''}[args.src]
OUT_SUF = {'avg': '_avg', 'snap': '_snap'}[args.src]
SRC_LAB = {'avg': 'hourly-average output', 'snap': 'hourly snapshots'}[args.src]

Ldir = Lfun.Lstart(gridname='wb1')
out_root = Ldir['LOo'] / 'DM_outs'
fn3 = out_root / ('20261013_maccready2011_sec3' + IN_SUF) / (SN + '_sec3.nc')
fnc = out_root / ('20261013_maccready2011_eu_vs_tef' + IN_SUF) / (SN + '_eu_vs_tef.nc')
out_dir = out_root / ('20261020_pc_dQds_monthly_%d' % YEAR + OUT_SUF)
month_dir = out_dir / 'by_month'
month_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ load
d3 = xr.open_dataset(fn3)
dc = xr.open_dataset(fnc)
assert (d3.time.values == dc.time.values).all(), 'sec3 and eu_vs_tef time axes differ'
time = pd.DatetimeIndex(d3.time.values)
qT = d3.qbar.to_numpy().astype(float)     # (t, NS) TEF bin transports [m3/s]
qE = dc.qEu.to_numpy().astype(float)      # (t, NS) Eulerian bin transports
sbins = d3.sbin.to_numpy()
dsb = d3.sedge.values[1] - d3.sedge.values[0]
ok = d3.ok.to_numpy() & dc.ok.to_numpy()
terms = {k: dc[k].to_numpy() for k in
         ['Q_in_tef', 'Q_out_tef', 'F_in_tef', 'F_out_tef',
          'Q_in_eu', 'Q_out_eu', 'F_in_eu', 'F_out_eu']}
d3.close(), dc.close()

# ------------------------------------------------------------------ monthly means
months = range(1, 13)
prof = {}                                 # month -> (TEF, Eulerian) mean -dQ/ds
rows = []
for mo in months:
    w = ok & (time.year == YEAR) & (time.month == mo)
    prof[mo] = (qT[w].mean(axis=0) / dsb, qE[w].mean(axis=0) / dsb)
    r = {'month': '%d-%02d' % (YEAR, mo), 'hours': int(w.sum())}
    for suf in ['tef', 'eu']:
        Qi, Qo = terms['Q_in_' + suf][w].mean(), terms['Q_out_' + suf][w].mean()
        si = terms['F_in_' + suf][w].mean() / Qi          # (4.1)
        so = terms['F_out_' + suf][w].mean() / Qo
        r.update({'Q_in_' + suf: Qi, 'Q_out_' + suf: Qo,
                  's_in_' + suf: si, 's_out_' + suf: so, 'Ds_' + suf: si - so})
    rows.append(r)
df = pd.DataFrame(rows).set_index('month')
df.to_csv(out_dir / (SN + '_monthly_%d.csv' % YEAR), float_format='%.4f')
print(SN + ', ' + SRC_LAB + ', monthly means over valid hours, positive into the cove')
print(df.to_string(float_format=lambda x: '%8.2f' % x))


def s_range(curves, frac=0.01):
    """Salinity range where any curve exceeds frac of its own max |value|."""
    keep = np.zeros(len(sbins), bool)
    for c in curves:
        keep |= np.abs(c) > frac * np.abs(c).max()
    i = np.nonzero(keep)[0]
    return np.floor(sbins[i[0]]) - 0.5, np.ceil(sbins[i[-1]]) + 0.5


def draw(ax, mo):
    T, E = prof[mo]
    ax.plot(T, sbins, color='k', lw=0.8,
            label='TEF ($Q_{in}$ %.0f)' % df.loc['%d-%02d' % (YEAR, mo), 'Q_in_tef'])
    ax.plot(E, sbins, color='0.6', lw=0.8,
            label='Eulerian ($Q_{in}$ %.0f)' % df.loc['%d-%02d' % (YEAR, mo), 'Q_in_eu'])
    ax.axvline(0, color='k', lw=0.6)
    ax.grid(True, alpha=0.3)


# ------------------------------------------------------------------ one figure per month
for mo in months:
    fig, ax = plt.subplots(figsize=(6, 6))
    draw(ax, mo)
    ax.set_ylim(*s_range(prof[mo]))
    ax.set_xlabel(r'mean $-\partial Q/\partial s$ [m$^3$ s$^{-1}$ (g kg$^{-1}$)$^{-1}$]'
                  + '\n+ inflow (into cove)')
    ax.set_ylabel(r'$s$ [g kg$^{-1}$]')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(SN + ': TEF vs Eulerian, %d-%02d (%d h, %s)'
                 % (YEAR, mo, df.loc['%d-%02d' % (YEAR, mo), 'hours'], SRC_LAB), fontsize=10)
    fig.tight_layout()
    fig.savefig(month_dir / (SN + '_dQds_%d-%02d.png' % (YEAR, mo)), dpi=200,
                transparent=True)
    plt.close(fig)

# ------------------------------------------------------------------ 12-panel figure, shared axes
lo = min(s_range(prof[mo])[0] for mo in months)
hi = max(s_range(prof[mo])[1] for mo in months)
xm = max(np.abs(np.concatenate(prof[mo])).max() for mo in months)
fig, axs = plt.subplots(3, 4, figsize=(14, 12), sharex=True, sharey=True)
for mo, ax in zip(months, axs.flat):
    draw(ax, mo)
    ax.set_title(pd.Timestamp(YEAR, mo, 1).strftime('%b %Y'), fontsize=10)
    ax.legend(loc='lower right', fontsize=7)
ax.set_ylim(lo, hi)
ax.set_xlim(-1.05 * xm, 1.05 * xm)
for ax in axs[:, 0]:
    ax.set_ylabel(r'$s$ [g kg$^{-1}$]')
for ax in axs[-1, :]:
    ax.set_xlabel(r'mean $-\partial Q/\partial s$' + '\n'
                  + r'[m$^3$ s$^{-1}$ (g kg$^{-1}$)$^{-1}$], + inflow')
fig.suptitle(SN + ': TEF (black) vs Eulerian (grey) $-\\partial Q/\\partial s$, monthly '
             'means, %d (%s, 1000 bins)' % (YEAR, SRC_LAB))
fig.tight_layout()
fig.savefig(out_dir / (SN + '_dQds_monthly_%d.png' % YEAR), dpi=200, transparent=True)
plt.close(fig)
print('\nsaved to ' + str(out_dir))
