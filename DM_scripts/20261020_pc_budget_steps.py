"""
Penn Cove volume / salt / DO budgets, built one step at a time.
wb1_t0_xn11abbur00, 2024-2025, hourly.

CONTROL VOLUME  tef2 segments pc_lp_m + pc_cp_p + pc_cp_m: everything landward
of the mouth section pc_lp, which is its only open boundary.
SIGN  positive = into Penn Cove (pc_lp's own positive direction is east = out).

STEP 1  hourly volume budget, no filtering
    storage  V(t) from the segment extraction: ocean_his SNAPSHOTS at hh:00
    flux     Q(t) = sum over pc_lp faces of hour-mean Huon (ocean_avg), at hh:30
    Q is the mean across the hour bracketed by two snapshots, so
        dV/dt(n) = [V(n+1) - V(n)] / dt   should equal   Q(n)
    exactly, apart from freshwater sources inside the cove (two outfalls,
    ~0.009 m3/s) and any surface E-P.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SN = 'pc_lp'
SIGN = -1
SEGS = ['pc_lp_m', 'pc_cp_p', 'pc_cp_m']
DT = 3600.0

Ldir = Lfun.Lstart(gridname='wb1')
tef = Ldir['LOo'] / 'extract' / GTAGEX / 'tef2'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261020_pc_budget_steps'
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ storage
sg = xr.open_dataset(tef / ('segments_' + DS0 + '_' + DS1 + '_wb1_pc1_trapsN00.nc')).sel(seg=SEGS)
t_snap = sg.time.to_numpy()                          # hh:00, N+1 snapshots
V = sg.volume.sum('seg').to_numpy()                  # m3
EmP = (sg.EminusP * sg.area).sum('seg').to_numpy()   # m3/s, + = net evaporation
sg.close()

# ------------------------------------------------------------------ flux
ds = xr.open_dataset(tef / ('extractions_avg_' + DS0 + '_' + DS1) / (SN + '.nc'))
t_flux = ds.time.to_numpy()                          # hh:30, N hours
Q = SIGN * ds.q.sum(dim=('z', 'p')).to_numpy()       # m3/s into the cove
ds.close()

assert (t_snap[:-1] + np.timedelta64(30, 'm') == t_flux).all(), 'clocks do not interleave'

# ------------------------------------------------------------------ budget
dVdt = np.diff(V) / DT                               # m3/s, on the flux clock
res = dVdt - Q

print('snapshots %d (%s .. %s), flux hours %d' % (len(t_snap), t_snap[0], t_snap[-1], len(t_flux)))
print('cove volume  mean %.4f km3, range %.4f to %.4f' % (V.mean() / 1e9, V.min() / 1e9, V.max() / 1e9))
print('Q            std %8.1f m3/s, max |Q| %8.1f' % (Q.std(), np.abs(Q).max()))
print('dV/dt        std %8.1f m3/s' % dVdt.std())
print('residual     mean %+.4f, std %.4f, max |res| %.4f m3/s' % (res.mean(), res.std(), np.abs(res).max()))
print('corr(dV/dt, Q) = %.8f' % np.corrcoef(dVdt, Q)[0, 1])
print('E-P (area-weighted, cove)  mean %+.2e m3/s, max |.| %.2e' % (np.nanmean(EmP), np.nanmax(np.abs(EmP))))

# ------------------------------------------------------------------ figure: three days
w = slice(24 * 200, 24 * 203)
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
a = axs[0]
a.plot(t_flux[w], Q[w], 'k-', lw=1.5, label=r'$Q$ at pc_lp (hour mean)')
a.plot(t_flux[w], dVdt[w], 'o', color='tab:orange', ms=3, label=r'$\Delta V/\Delta t$ (snapshots)')
a.set_ylabel(r'm$^3$ s$^{-1}$')
a.legend(loc='upper right')
a = axs[1]
a.plot(t_flux[w], res[w], 'k-', lw=1)
a.set_ylabel(r'$dV/dt - Q$ [m$^3$ s$^{-1}$]')
for a in axs:
    a.grid(True, alpha=0.3)
axs[0].set_title('Penn Cove step 1: hourly volume budget, + into cove')
fig.tight_layout()
fig.savefig(out_dir / 'step1_volume_hourly.png', dpi=200, transparent=True)
plt.close(fig)
print('saved to ' + str(out_dir))
