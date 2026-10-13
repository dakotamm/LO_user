"""
Volume and salt budgets for Penn Cove landward of pc_lp, closing the MacCready
(2011) Section 2 and Section 3 flux terms against the change in what the cove
holds. wb1_t0_xn11abbur00, hourly throughout.

CONTROL VOLUME  the three tef2 segments landward of pc_lp:
    pc_lp_m + pc_cp_p + pc_cp_m      (0.210 km3)
pc_lp is its only open boundary. The two cove outfalls (Coupeville STP and Penn
Cove WWTP, 0.009 m3/s together, zero salt) are neglected.

STORAGE, from the segment extraction (ocean_his snapshots on the hour, the
same clock as the pc_cove box that the section fluxes come from):
    V(t) = sum_seg volume                       [m3]
    S(t) = sum_seg salt * volume                [g/kg m3]  (salt is volume-mean)
tidally averaged and differentiated hourly:  dV/dt = d<V>/dt,  dS/dt = d<S>/dt
(centred difference; the filter is linear, so d<X>/dt = <dX/dt>).

VOLUME BUDGET
    dV/dt = Q,      Q = < int u dA > = u0 A0 = -Q_R            (Section 2)

SALT BUDGET
    dS/dt = F + F_surf
    F_surf = < sum_seg salt_surf * area * EminusP >   as in LO tracer_budget.py
with three versions of the section salt flux F:
    Section 2   F_R + F_E + F_T          (MacCready 2.6-2.7)
    TEF         F_in + F_out             (3.7)
    Eulerian    F_R + F_E                (= F^Eu_in + F^Eu_out, (3.10); no F_T)
residual = dS/dt - F - F_surf

Record means of a derivative are (end - start) / duration, so the mean
storage term is set entirely by how much the cove's salt content changed.
The cumulative plot integrates each flux in time and compares it with the
actual change in salt content since the first valid hour.

SIGN  positive = into Penn Cove.

INPUTS (run first)
    20261013_maccready2011_sec2.py      -> pc_lp_sec2.nc
    20261013_maccready2011_eu_vs_tef.py -> pc_lp_eu_vs_tef.nc
    LO_output/extract/[gtagex]/tef2/segments_[dates]_wb1_pc1_trapsN00.nc

OUTPUT LO_output/DM_outs/20261013_pc_budgets/

run 20261013_pc_budgets.py
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
SN = 'pc_lp'
SEGS = ['pc_lp_m', 'pc_cp_p', 'pc_cp_m']
DT = 3600.0                               # s between snapshots

Ldir = Lfun.Lstart(gridname='wb1')
out_root = Ldir['LOo'] / 'DM_outs'
fn2 = out_root / '20261013_maccready2011_sec2' / (SN + '_sec2.nc')
fnc = out_root / '20261013_maccready2011_eu_vs_tef' / (SN + '_eu_vs_tef.nc')
fns = (Ldir['LOo'] / 'extract' / GTAGEX / 'tef2'
       / ('segments_' + DS0 + '_' + DS1 + '_wb1_pc1_trapsN00.nc'))
out_dir = out_root / '20261013_pc_budgets'
out_dir.mkdir(parents=True, exist_ok=True)


def lp(x):
    """Tidal low-pass <>: Godin filter on axis 0, hourly (NaN at the ends)."""
    return zfun.lowpass(x, f='godin')


# ------------------------------------------------------------------ load
d2 = xr.open_dataset(fn2)
dc = xr.open_dataset(fnc)
sg = xr.open_dataset(fns).sel(seg=SEGS)
time = d2.time.to_numpy()
assert (time == dc.time.values).all() and (time == sg.time.values).all(), \
    'time axes differ'
Q = (d2.u0 * d2.A0).to_numpy()            # < int u dA >
F_R, F_E, F_T = d2.F_R.to_numpy(), d2.F_E.to_numpy(), d2.F_T.to_numpy()
F_tef = (dc.F_in_tef + dc.F_out_tef).to_numpy()
ok = d2.ok.to_numpy() & dc.ok.to_numpy()
V_seg = sg.volume.to_numpy()              # (t, seg)
S_seg = (sg.salt * sg.volume).to_numpy()
surf_seg = (sg.salt_surf * sg.area * sg.EminusP).to_numpy()
d2.close(), dc.close(), sg.close()
print('cove volume %.4f km3 (mean of snapshots)' % (V_seg.sum(axis=1).mean() / 1e9))

# ------------------------------------------------------------------ storage
V = lp(V_seg.sum(axis=1))
S = lp(S_seg.sum(axis=1))
dVdt = np.gradient(V, DT)
dSdt = np.gradient(S, DT)
F_surf = lp(surf_seg.sum(axis=1))
ok = ok & np.isfinite(dVdt) & np.isfinite(dSdt) & np.isfinite(F_surf)

# ------------------------------------------------------------------ budgets
fluxes = {'Section 2: F_R + F_E + F_T': F_R + F_E + F_T,
          'TEF: F_in + F_out': F_tef,
          'Eulerian: F_R + F_E': F_R + F_E}
vres = dVdt - Q


def m(x):
    return np.mean(x[ok])


def rms(x):
    return np.sqrt(np.mean(x[ok]**2))


print('\n' + SN + ' / Penn Cove, %d valid hours, positive into the cove' % ok.sum())
print('\nVOLUME [m3/s]')
print(pd.DataFrame({'mean': [m(dVdt), m(Q), m(vres)],
                    'std': [np.std(dVdt[ok]), np.std(Q[ok]), np.std(vres[ok])]},
                   index=['dV/dt', 'Q = <int u dA>', 'residual dV/dt - Q']
                   ).to_string(float_format=lambda x: '%10.3f' % x))
print('corr(dV/dt, Q) = %.4f' % np.corrcoef(dVdt[ok], Q[ok])[0, 1])

rows = {'dS/dt': [m(dSdt), np.std(dSdt[ok]), np.nan],
        'F_surf': [m(F_surf), np.std(F_surf[ok]), np.nan]}
for lab, F in fluxes.items():
    r = dSdt - F - F_surf
    rows[lab] = [m(F), np.std(F[ok]), np.corrcoef(dSdt[ok], F[ok])[0, 1]]
    rows['   residual dS/dt - F - F_surf'.ljust(32) + '(' + lab.split(':')[0] + ')'] = \
        [m(r), np.std(r[ok]), np.nan]
print('\nSALT [g/kg m3/s]')
print(pd.DataFrame(rows, index=['mean', 'std', 'corr with dS/dt']).T
      .to_string(float_format=lambda x: '%10.3f' % x))

# what the mean storage term means for the cove's salinity
T_s = ok.sum() * DT
Vm = m(V)
print('\nover the %.0f valid days: change in salt content %.3e g/kg m3'
      ' = %.3f g/kg in volume-mean salinity' % (T_s / 86400, m(dSdt) * T_s,
                                               m(dSdt) * T_s / Vm))
for lab, F in fluxes.items():
    print('  implied by %-28s %.3e g/kg m3 = %7.3f g/kg'
          % (lab, m(F + F_surf) * T_s, m(F + F_surf) * T_s / Vm))

# ------------------------------------------------------------------ save
xr.Dataset({'V': ('time', V), 'S': ('time', S), 'dVdt': ('time', dVdt),
            'dSdt': ('time', dSdt), 'Q': ('time', Q), 'F_surf': ('time', F_surf),
            'F_sec2': ('time', F_R + F_E + F_T), 'F_tef': ('time', F_tef),
            'F_eu': ('time', F_R + F_E), 'ok': ('time', ok)},
           coords={'time': time},
           attrs={'control_volume': ' + '.join(SEGS) + ', landward of ' + SN,
                  'sign': 'positive into Penn Cove',
                  'units': 'V m3; S g/kg m3; dVdt, Q m3/s; dSdt, F g/kg m3/s'}
           ).to_netcdf(out_dir / 'pc_budgets.nc')

# ------------------------------------------------------------------ plot: hourly budgets
cols = {'Section 2: F_R + F_E + F_T': 'tab:blue', 'TEF: F_in + F_out': 'tab:orange',
        'Eulerian: F_R + F_E': 'tab:green'}
fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
a = axs[0]
a.plot(time, dVdt, color='k', lw=1, label=r'$dV/dt$')
a.plot(time, Q, color='tab:red', lw=0.6, label=r'$Q=\langle\int u\,dA\rangle$')
a.set_ylabel(r'volume [m$^3$ s$^{-1}$]')
a = axs[1]
a.plot(time, dSdt, color='k', lw=1, label=r'$dS/dt$')
for lab, F in fluxes.items():
    a.plot(time, F + F_surf, color=cols[lab], lw=0.6, label=lab + r' $+F_{surf}$')
a.set_ylabel(r'salt [g kg$^{-1}$ m$^3$ s$^{-1}$]')
a = axs[2]
for lab, F in fluxes.items():
    a.plot(time, dSdt - F - F_surf, color=cols[lab], lw=0.6,
           label=lab.split(':')[0] + ' (mean %.1f)' % m(dSdt - F - F_surf))
a.set_ylabel('salt residual\n' + r'$dS/dt-F-F_{surf}$')
for a in axs:
    a.axhline(0, color='k', lw=0.6)
    a.legend(loc='upper left', fontsize=8, ncol=4)
    a.grid(True, alpha=0.3)
axs[0].set_title('Penn Cove (landward of ' + SN + '): volume and salt budgets, hourly low-passed')
fig.tight_layout()
fig.savefig(out_dir / 'pc_budgets_timeseries.png', dpi=200, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ plot: cumulative salt
i0 = np.argmax(ok)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(time[ok], (S - S[i0])[ok] / Vm, color='k', lw=1.4,
        label=r'actual: $(S-S_0)/\bar V$')
for lab, F in fluxes.items():
    cum = np.cumsum(np.where(ok, F + F_surf, 0)) * DT
    ax.plot(time[ok], (cum - cum[i0])[ok] / Vm, color=cols[lab], lw=1,
            label=lab + r' $+F_{surf}$, integrated')
ax.axhline(0, color='k', lw=0.6)
ax.set_ylabel(r'change in cove-mean salinity [g kg$^{-1}$]')
ax.legend(loc='lower left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title('Penn Cove salt content since the first valid hour, actual vs integrated fluxes')
fig.tight_layout()
fig.savefig(out_dir / 'pc_salt_cumulative.png', dpi=200, transparent=True)
plt.close(fig)
print('\nsaved to ' + str(out_dir))
