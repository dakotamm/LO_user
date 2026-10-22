"""
Oxygen exchange and oxygen budgets along Penn Cove, from the three tef2 sections
pc_lp (mouth), pc_lj (middle) and pc_cp (inner), wb1_t0_xn11abbur00, hourly-
average output. Combines the per-section outputs of 20261020_pc_do_tef.py.

SEGMENTS (tef2 segments, mean volume)
    outer   pc_lp_m   between pc_lp and pc_lj   0.066 km3
    middle  pc_cp_p   between pc_lj and pc_cp   0.055 km3
    inner   pc_cp_m   landward of pc_cp         0.086 km3

Each section's fluxes are positive INTO the cove (landward), so the flux INTO a
segment is the flux through its seaward section minus the flux through its
landward section:
    outer:  G(pc_lp) - G(pc_lj)     middle:  G(pc_lj) - G(pc_cp)     inner:  G(pc_cp)
and the same for the volume transport Q = < int u dA >.

BUDGETS per segment (ocean_his snapshots on the hour, forward difference onto
the hh:30 flux clock, then Godin filter):
    volume  dV/dt = Q_into                       (check: residual should be ~0)
    oxygen  dO/dt = G_into + R
R = net interior oxygen source: photosynthesis - respiration - sediment demand,
plus air-sea exchange (not separated). R < 0 is net consumption. R / V is the
segment-mean rate in mmol m-3 per day.

Also per section: TEF Q_in, O_in, O_out, and the flushing time V/Q_in of the
whole volume landward of it. Monthly values are means over valid hours, with
O_in, O_out flux weighted (MacCready 2011, 4.1).

INPUTS (run first, for each -sn in pc_lp pc_lj pc_cp)
    20261020_pc_do_tef.py -src avg -sn [sn]  ->  20261020_pc_do_tef_avg/[sn]_do_tef.nc
    LO_output/extract/[gtagex]/tef2/segments_[dates]_wb1_pc1_trapsN00.nc

OUTPUT LO_output/DM_outs/20261021_pc_do_sections_avg/
    pc_do_sections_record.csv       record means (2024-2025) per section / segment
    pc_do_sections_monthly_2025.csv monthly 2025
    pc_do_sections_monthly_2025.png Q_in, O_in/O_out, flushing time, R/V by segment

run 20261021_pc_do_sections.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
YEAR = 2025
K = 0.032                 # mmol O2 -> g, and mmol m-3 -> mg/L
DT = 3600.0
SECTS = ['pc_lp', 'pc_lj', 'pc_cp']                          # seaward to landward
SEG = {'outer': ('pc_lp_m', 'pc_lp', 'pc_lj'),               # segment, seaward, landward
       'middle': ('pc_cp_p', 'pc_lj', 'pc_cp'),
       'inner': ('pc_cp_m', 'pc_cp', None)}
LANDWARD = {'pc_lp': ['pc_lp_m', 'pc_cp_p', 'pc_cp_m'],      # volume landward of each section
            'pc_lj': ['pc_cp_p', 'pc_cp_m'],
            'pc_cp': ['pc_cp_m']}
COL = {'pc_lp': 'k', 'pc_lj': 'tab:blue', 'pc_cp': 'tab:orange',
       'outer': 'k', 'middle': 'tab:blue', 'inner': 'tab:orange'}

Ldir = Lfun.Lstart(gridname='wb1')
in_dir = Ldir['LOo'] / 'DM_outs' / '20261020_pc_do_tef_avg'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261021_pc_do_sections_avg'
out_dir.mkdir(parents=True, exist_ok=True)


def lp(x):
    """Tidal low-pass <>: Godin filter on axis 0, hourly (NaN at the ends)."""
    return zfun.lowpass(x, f='godin')


# ------------------------------------------------------------------ load sections
S = {}
for sn in SECTS:
    d = xr.open_dataset(in_dir / (sn + '_do_tef.nc'))
    S[sn] = {k: d[k].to_numpy() for k in
             ['Q_in_tef', 'Q_out_tef', 'G_in_tef', 'G_out_tef', 'G', 'G_E', 'G_T', 'ok']}
    S[sn]['time'] = d.time.to_numpy()
    d.close()
time = S['pc_lp']['time']
for sn in SECTS:
    assert (S[sn]['time'] == time).all(), sn + ' time axis differs'
    S[sn]['Q'] = S[sn]['Q_in_tef'] + S[sn]['Q_out_tef']       # < int u dA >
ok = np.logical_and.reduce([S[sn]['ok'] for sn in SECTS])

# ------------------------------------------------------------------ segments: storage
sg = xr.open_dataset(Ldir['LOo'] / 'extract' / GTAGEX / 'tef2'
                     / ('segments_' + DS0 + '_' + DS1 + '_wb1_pc1_trapsN00.nc'))
assert (sg.time.values[:-1] + np.timedelta64(30, 'm') == time).all(), \
    'segment snapshots do not bracket the flux hours'
V, dVdt, dOdt = {}, {}, {}
for seg in sg.seg.values:
    v = sg.volume.sel(seg=seg).to_numpy()
    o = (sg.oxygen * sg.volume).sel(seg=seg).to_numpy()      # mmol
    V[seg] = lp(0.5 * (v[1:] + v[:-1]))
    dVdt[seg] = lp(np.diff(v) / DT)
    dOdt[seg] = lp(np.diff(o) / DT)
sg.close()
ok &= np.logical_and.reduce([np.isfinite(dOdt[s]) for s in dOdt])

# ------------------------------------------------------------------ per-segment budgets
B = {}
for name, (seg, sea, land) in SEG.items():
    Qin = S[sea]['Q'] - (S[land]['Q'] if land else 0)
    Gin = S[sea]['G'] - (S[land]['G'] if land else 0)
    R = dOdt[seg] - Gin
    B[name] = {'V': V[seg], 'Q_into': Qin, 'vol_res': dVdt[seg] - Qin,
               'G_into': Gin, 'dOdt': dOdt[seg], 'R': R, 'R_rate': R / V[seg] * 86400}
for sn in SECTS:
    S[sn]['V_land'] = sum(V[s] for s in LANDWARD[sn])

# ------------------------------------------------------------------ statistics
tt = pd.DatetimeIndex(time)


def stats(w):
    """Means over mask w: per section (TEF) and per segment (budget)."""
    m = lambda x: np.mean(x[w])
    r = {'hours': int(w.sum())}
    for sn in SECTS:
        D = S[sn]
        Qi = m(D['Q_in_tef'])
        r.update({sn + ' Q_in': Qi,
                  sn + ' O_in': m(D['G_in_tef']) / Qi * K,
                  sn + ' O_out': m(D['G_out_tef']) / m(D['Q_out_tef']) * K,
                  sn + ' G': m(D['G']) * K, sn + ' G_E': m(D['G_E']) * K,
                  sn + ' G_T': m(D['G_T']) * K,
                  sn + ' V/Q_in [d]': m(D['V_land']) / Qi / 86400})
    for name, D in B.items():
        r.update({name + ' vol_res': m(D['vol_res']), name + ' G_into': m(D['G_into']) * K,
                  name + ' dO/dt': m(D['dOdt']) * K, name + ' R': m(D['R']) * K,
                  name + ' R/V': m(D['R_rate'])})
    return r


rec = pd.Series(stats(ok))
mon = pd.DataFrame({'%d-%02d' % (YEAR, mo): stats(ok & (tt.year == YEAR) & (tt.month == mo))
                    for mo in range(1, 13)}).T
mon.index.name = 'month'
rec.to_csv(out_dir / 'pc_do_sections_record.csv', header=['value'], float_format='%.4f')
mon.to_csv(out_dir / ('pc_do_sections_monthly_%d.csv' % YEAR), float_format='%.4f')

print('units: Q m3/s; O mg/L; G, dO/dt, R g O2/s; R/V mmol m-3 d-1; + into the cove')
print('\nvolume closure, max |dV/dt - Q_into| per segment [m3/s]: '
      + ', '.join('%s %.3f' % (n, np.max(np.abs(B[n]['vol_res'][ok]))) for n in B))
print('\nRECORD MEANS %s to %s (%d h)\n' % (DS0, DS1, rec['hours']))
sec_tab = pd.DataFrame({sn: [rec[sn + k] for k in
                             [' Q_in', ' O_in', ' O_out', ' G', ' G_E', ' G_T', ' V/Q_in [d]']]
                        for sn in SECTS},
                       index=['Q_in', 'O_in', 'O_out', 'G', 'G_E', 'G_T', 'V/Q_in [d]'])
print('sections (TEF, landward volume):')
print(sec_tab.to_string(float_format=lambda x: '%9.2f' % x))
seg_tab = pd.DataFrame({n: [rec[n + k] for k in [' G_into', ' dO/dt', ' R', ' R/V']]
                        for n in B}, index=['G_into', 'dO/dt', 'R', 'R/V'])
print('\nsegment oxygen budgets:')
print(seg_tab.to_string(float_format=lambda x: '%9.2f' % x))
print('\nMONTHLY %d' % YEAR)
cols = ([sn + ' Q_in' for sn in SECTS] + [sn + ' O_in' for sn in SECTS]
        + [sn + ' O_out' for sn in SECTS] + [sn + ' V/Q_in [d]' for sn in SECTS]
        + [n + ' R/V' for n in B])
print(mon[cols].to_string(float_format=lambda x: '%6.2f' % x))

# ------------------------------------------------------------------ figure
x = np.arange(1, 13)
fig, axs = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
for sn in SECTS:
    axs[0].plot(x, mon[sn + ' Q_in'], 'o-', color=COL[sn], label=sn)
    axs[1].plot(x, mon[sn + ' O_in'], '^-', color=COL[sn], label=sn + r' $O_{in}$')
    axs[1].plot(x, mon[sn + ' O_out'], 'v--', color=COL[sn], mfc='w',
                label=sn + r' $O_{out}$')
    axs[2].plot(x, mon[sn + ' V/Q_in [d]'], 'o-', color=COL[sn],
                label='landward of ' + sn)
for n, (seg, _, _) in SEG.items():
    axs[3].plot(x, mon[n + ' R/V'], 's-', color=COL[n], label='%s (%s)' % (n, seg))
axs[3].axhline(0, color='k', lw=0.6)
axs[0].set_ylabel(r'TEF $Q_{in}$ [m$^3$ s$^{-1}$]')
axs[1].set_ylabel(r'DO [mg L$^{-1}$]')
axs[2].set_ylabel(r'flushing time $V/Q_{in}$ [d]')
axs[3].set_ylabel(r'$R/V$ [mmol m$^{-3}$ d$^{-1}$]' + '\n' + r'net O$_2$ source ($<0$ = sink)')
for a in axs:
    a.grid(True, alpha=0.3)
    a.legend(loc='best', fontsize=8, ncol=3)
axs[-1].set_xticks(x)
axs[-1].set_xticklabels([pd.Timestamp(YEAR, m, 1).strftime('%b') for m in x])
axs[0].set_title('Penn Cove sections and segments, monthly means %d '
                 '(hourly-average output, TEF), + into the cove' % YEAR)
fig.tight_layout()
fig.savefig(out_dir / ('pc_do_sections_monthly_%d.png' % YEAR), dpi=200, transparent=True)
plt.close(fig)
print('\nsaved to ' + str(out_dir))
