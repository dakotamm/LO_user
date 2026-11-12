"""
The Fr_f axis of Geyer & MacCready (2014) Fig. 6 for one Penn Cove section, wb1_t0_xn11abbur00, 2024-2025.

    Fr_f = U_R / c_0,    U_R = Q_R / A,    c_0 = (beta g s_ocn H)^(1/2)

U_R is the net velocity the river would drive through the section; c_0 the fastest possible
two-layer frontal speed. A, H and s_ocn are read from 20261109_gm14_M_axis.py's summary so both
axes use the same numbers.

Penn Cove has no river, so Q_R is defined two ways:

  (a) LOCAL   the freshwater the model actually adds landward of the section. In t0 the cove
              outfalls (COUPEVILLE STP + PENN COVE WWTP, 0.009 m3/s) never enter (LwSrc = F in
              every wb1 run), so the only local freshwater is net precipitation P - E over segments
              pc_cp_m + pc_cp_p, which ROMS applies as a virtual salt flux (no volume).
              Q_R = -sum (E-P) * area. The outfall total is drawn as a reference only.
  (b) EFFECTIVE (from stratification)  invert the scaling GM14 use for their mixing line,
              ds/s_ocn = alpha Fr_f^(2/3) (MacCready 1999; alpha = 3.4 from Geyer 2010), with the
              section's tidally averaged top-bottom salinity difference ds:
              Fr_eff = (ds / (alpha s_ocn))^(3/2).
              This is where Penn Cove's stratification WOULD place it if it were generated
              locally by a river; it is a diagnostic, not a forcing.

The GM14 boxes span seasonal / event changes in river flow, so each Fr_f is reduced to monthly
means; the box is the range of the monthly values, the point the record median.

INPUT  LO_output/extract/[gtagex]/tef2/segments_[ds0]_[ds1]_wb1_pc1_trapsN00.nc   (hourly E-P, area)
       LO_output/extract/[gtagex]/tef2/extractions_avg_[ds0]_[ds1]/[sect].nc       (salt, dd)
       LO_output/DM_outs/20261109_gm14_M_axis/M_summary_[sect].csv                   (A, H, s_ocn)
OUTPUT LO_output/DM_outs/20261109_gm14_Frf_axis/
    Frf_monthly_[sect].csv   monthly Q_R(a), Fr_f(a), ds, Fr_eff
    Frf_summary_[sect].csv   c_0 and the median / min / max of each Fr_f

run 20261109_gm14_Frf_axis.py -sect pc_lj
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', type=str, default='wb1_t0_xn11abbur00')
parser.add_argument('-sect', type=str, default='pc_lj')
args = parser.parse_args()

DS0, DS1 = '2024.01.01', '2025.12.31'
BETA, G = 7.7e-4, 9.81
ALPHA = 3.4                                # ds/s_ocn = alpha Fr_f^(2/3)
Q_OUTFALL = 0.008 + 0.001                  # m3/s, COUPEVILLE STP + PENN COVE WWTP (not in t0; reference)
LANDWARD = {'pc_lj': ['pc_cp_m', 'pc_cp_p'],    # wb1_pc1 segments inside each section
            'pc_cp': ['pc_cp_m'],
            'pc_lp': ['pc_cp_m', 'pc_cp_p', 'pc_lp_m']}[args.sect]

Ldir = Lfun.Lstart(gridname='wb1')
tef = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
seg_fn = tef / ('segments_%s_%s_wb1_pc1_trapsN00.nc' % (DS0, DS1))
ext_fn = tef / ('extractions_avg_%s_%s' % (DS0, DS1)) / (args.sect + '.nc')
summ_fn = Ldir['LOo'] / 'DM_outs' / '20261109_gm14_M_axis' / ('M_summary_%s.csv' % args.sect)
out_dir = Ldir['LOo'] / 'DM_outs' / '20261109_gm14_Frf_axis'
out_dir.mkdir(parents=True, exist_ok=True)

Ms = pd.read_csv(summ_fn, index_col=0).loc['median']
A, H, s_ocn = Ms.A, Ms.H, Ms.s_ocn
c0 = np.sqrt(BETA * G * s_ocn * H)
print('%s: A = %.3e m2, H = %.2f m, s_ocn = %.2f  ->  c_0 = (beta g s_ocn H)^1/2 = %.2f m/s'
      % (args.sect, A, H, s_ocn, c0))

# ------------------------------------------------------------------ (a) local freshwater: net precipitation
sg = xr.open_dataset(seg_fn).sel(seg=LANDWARD)
Ts = pd.DatetimeIndex(sg.time.values)
Qr = pd.Series(-(sg.EminusP.values * sg.area.values).sum(axis=1), index=Ts)     # m3/s, + = net rain
Qr = Qr[Qr.index < pd.Timestamp(DS1) + pd.Timedelta(days=1)]                   # drop the 2026-01-01 00:00 stub
area = sg.area.values.mean(axis=0).sum()
sg.close()
Qr_mon = Qr.resample('MS').mean()
Fr_loc = Qr_mon / A / c0
print('\n(a) P - E over %s (%.2f km2): mean %.4f m3/s, hourly max %.2f' % ('+'.join(LANDWARD), area / 1e6, Qr.mean(), Qr.max()))
print('    monthly Q_R %.4f - %.4f m3/s; %d of %d months negative (E > P)'
      % (Qr_mon.min(), Qr_mon.max(), (Qr_mon < 0).sum(), len(Qr_mon)))
print('    Fr_f = Q_R/(A c_0): record mean %.1e, wettest month %.1e   (outfalls, not in t0: %.1e)'
      % (Qr.mean() / A / c0, Fr_loc.max(), Q_OUTFALL / A / c0))
print('    monthly median Fr_f %.1e' % Fr_loc.median())

# ------------------------------------------------------------------ (b) effective Fr_f from stratification
ds = xr.open_dataset(ext_fn)
T = pd.DatetimeIndex(ds.time.values)
dd = ds.dd.values
s_top = (ds.salt.values[:, -1, :] * dd).sum(axis=1) / dd.sum()     # top layer, width-weighted
s_bot = (ds.salt.values[:, 0, :] * dd).sum(axis=1) / dd.sum()      # bottom layer
ds.close()
dS = pd.Series(zfun.lowpass(s_bot - s_top, f='godin'), index=T)     # tidally averaged top-bottom difference
dS_mon = dS.resample('MS').mean()
Fr_eff = (dS_mon / (ALPHA * s_ocn)) ** 1.5
print('\n(b) ds = s_bot - s_top (Godin): mean %.2f g/kg, monthly %.2f - %.2f  (ds/s_ocn %.3f - %.3f)'
      % (dS.mean(), dS_mon.min(), dS_mon.max(), dS_mon.min() / s_ocn, dS_mon.max() / s_ocn))
print('    Fr_eff = (ds/(alpha s_ocn))^1.5: monthly median %.1e, range %.1e - %.1e'
      % (Fr_eff.median(), Fr_eff.min(), Fr_eff.max()))
print('    i.e. the Q_R a local river would need: median %.0f m3/s (range %.0f - %.0f)'
      % (Fr_eff.median() * A * c0, Fr_eff.min() * A * c0, Fr_eff.max() * A * c0))

# ------------------------------------------------------------------ save
mon = pd.DataFrame(dict(Q_R_local=Qr_mon, Fr_local=Fr_loc, dS=dS_mon, Fr_eff=Fr_eff))
mon.to_csv(out_dir / ('Frf_monthly_%s.csv' % args.sect))
S = pd.DataFrame({'local': [Fr_loc.median(), Fr_loc.min(), Fr_loc.max()],
                  'effective': [Fr_eff.median(), Fr_eff.min(), Fr_eff.max()],
                  'outfalls': [Q_OUTFALL / A / c0] * 3}, index=['median', 'min', 'max']).T
S['c0'], S['A'], S['H'], S['s_ocn'] = c0, A, H, s_ocn
S.to_csv(out_dir / ('Frf_summary_%s.csv' % args.sect))

# ------------------------------------------------------------------ check figure
C_LOC, C_EFF, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
FLOOR = 1e-8
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhspan(1e-4, 1, color='#f0efec', lw=0)
ax.text(Fr_eff.index[0], 3e-1, 'GM14 Fig. 6 axis range (Fr$_f$ 1e-4 - 1)', fontsize=8, color=MUTED, va='center')
ax.axhline(Q_OUTFALL / A / c0, color=MUTED, lw=1, ls='--')
ax.text(Fr_eff.index[0], Q_OUTFALL / A / c0 * 1.4, 'cove outfalls 0.009 m$^3$/s (not in t0)', fontsize=8, color=MUTED)
ax.plot(Fr_eff.index, Fr_eff, color=C_EFF, lw=2, marker='o', ms=5, label=r'(b) effective, $(\delta s/\alpha s_{ocn})^{3/2}$')
pos = Fr_loc > 0
ax.plot(Fr_loc.index[pos], Fr_loc[pos], color=C_LOC, lw=0, marker='o', ms=6, label='(a) local, net precipitation')
ax.plot(Fr_loc.index[~pos], np.full((~pos).sum(), FLOOR * 1.5), color=C_LOC, lw=0, marker='o', ms=6, mfc='none',
        label='(a) net-evaporative month (Fr$_f$ < 0)')
ax.set_yscale('log')
ax.set_ylim(FLOOR, 1)
ax.set_ylabel(r'$Fr_f = Q_R / (A\,c_0)$')
ax.set_title(r'%s monthly Fr$_f$;  A = %.2e m$^2$, $c_0$ = %.2f m/s' % (args.sect, A, c0), loc='left', fontsize=10)
ax.legend(loc='center right', fontsize=8, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#e5e4e0', lw=0.5)
fig.tight_layout()
fn = out_dir / ('Frf_axis_%s.png' % args.sect)
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(fn))
