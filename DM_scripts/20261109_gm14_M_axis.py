"""
The M axis of Geyer & MacCready (2014) Fig. 6 for one Penn Cove section, wb1_t0_xn11abbur00, 2024-2025.

    M^2 = C_D U_T^2 / (omega N_0 H^2),    N_0 = (beta g s_ocn / H)^(1/2)          (GM14 Eq. 20)

M^2 is the ratio of the tidal period to the time a bottom boundary layer needs to mix up through
a water column stratified at N_0 (Eq. 13 with N_inf -> N_0).

STEPS (one block each below)
    geometry   W = sum of face widths, A(t) = sum DZ dd, H = <A>/W (section-mean depth)
    U_T        Ubar(t) = Q(t)/A(t) from hour-mean transport; tidal part u' = Ubar - Godin(Ubar);
               U_T per lunar day = sqrt(2) rms(u') (the amplitude of a sinusoid with that rms)
    spring/neap  qprism (bulk_avg, daily) fortnightly maxima / minima; U_T on those days
    s_ocn      time-mean bottom-layer salinity at the section (the saltiest water the cove sees)
    M          from the above, per lunar day; spring and neap values give the box in M

INPUT  LO_output/extract/[gtagex]/tef2/extractions_avg_[ds0]_[ds1]/[sect].nc   (hour means, hh:30)
       LO_output/extract/[gtagex]/tef2/bulk_avg_[ds0]_[ds1]/[sect].nc          (daily qprism)
OUTPUT LO_output/DM_outs/20261109_gm14_M_axis/
    M_daily_[sect].csv     per lunar day: U_T, U_T of the fastest column, M, M_col, spring/neap flag
    M_summary_[sect].csv   H, W, s_ocn, N_0, and U_T / M at neap, median, spring

run 20261109_gm14_M_axis.py -sect pc_lj -omega M2
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', type=str, default='wb1_t0_xn11abbur00')
parser.add_argument('-sect', type=str, default='pc_lj')
parser.add_argument('-omega', type=str, default='M2', choices=['M2', 'K1'])
args = parser.parse_args()

DS0, DS1 = '2024.01.01', '2025.12.31'
CD = 2.5e-3                                # bottom drag coefficient (GM14 value)
BETA = 7.7e-4                              # haline contraction, 1/(g/kg)
G = 9.81
OMEGA = {'M2': 2 * np.pi / (12.4206 * 3600), 'K1': 2 * np.pi / (23.9345 * 3600)}[args.omega]
LUNAR_DAY = 24.8412                        # hours; U_T window (one diurnal + two semidiurnal cycles)
SIGN = -1                                  # wb1_pc1 + is east = out of the cove; flip so + = into the cove

Ldir = Lfun.Lstart(gridname='wb1')
tef = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
ext_fn = tef / ('extractions_avg_%s_%s' % (DS0, DS1)) / (args.sect + '.nc')
bulk_fn = tef / ('bulk_avg_%s_%s' % (DS0, DS1)) / (args.sect + '.nc')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261109_gm14_M_axis'
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ geometry
ds = xr.open_dataset(ext_fn)
T = pd.DatetimeIndex(ds.time.values)                       # hour means, stamped hh:30
dd = ds.dd.values                                          # face widths [m], (p,)
h = ds.h.values                                            # static depth per face [m], (p,)
DZ = ds.DZ.values                                          # (t, z, p) [m]
q = SIGN * ds.q.values                                     # (t, z, p) transport per cell [m3/s], + into the cove
salt = ds.salt.values                                      # (t, z, p)
ds.close()

W = dd.sum()
A_t = (DZ * dd).sum(axis=(1, 2))                           # (t,) cross-section area incl. zeta
H = A_t.mean() / W
print('%s: %d faces, %d hours %s -> %s' % (args.sect, len(dd), len(T), T[0], T[-1]))
print('W = %.0f m   <A> = %.3e m2 (static sum h dd = %.3e)   A range %.3e - %.3e'
      % (W, A_t.mean(), (h * dd).sum(), A_t.min(), A_t.max()))
print('H = <A>/W = %.2f m   (face depths %.1f - %.1f m)' % (H, h.min(), h.max()))

# ------------------------------------------------------------------ tidal velocity U_T
Q = q.sum(axis=(1, 2))                                     # (t,) section transport [m3/s]
Ubar = Q / A_t                                             # section-mean velocity [m/s]
u_tide = Ubar - zfun.lowpass(Ubar, f='godin')              # tidal band (Godin leaves ~35 h NaN at each end)

A_col = (DZ * dd).sum(axis=1)                              # (t, p) area of each face column
U_col = q.sum(axis=1) / A_col                              # (t, p) column depth-mean velocity
u_col_tide = U_col - np.column_stack([zfun.lowpass(U_col[:, p], f='godin') for p in range(len(dd))])

# consecutive lunar days from the first hour; a day with any NaN (the Godin ends) is dropped
iday = np.floor((T - T[0]) / pd.Timedelta(hours=LUNAR_DAY)).astype(int)
rows = []
for d in np.unique(iday):
    m = iday == d
    if m.sum() < 24 or np.isnan(u_tide[m]).any():
        continue
    ut_col = np.sqrt(2) * np.sqrt(np.mean(u_col_tide[m] ** 2, axis=0))      # (p,)
    rows.append(dict(time=T[m][0] + (T[m][-1] - T[m][0]) / 2,
                     U_T=np.sqrt(2) * np.sqrt(np.mean(u_tide[m] ** 2)),
                     U_peak=np.abs(u_tide[m]).max(),
                     U_T_col=ut_col.max(), p_col=int(ut_col.argmax())))
D = pd.DataFrame(rows).set_index('time')
print('\n%d lunar days' % len(D))
print('U_T (section mean, sqrt2 rms): median %.3f m/s, 10-90%% %.3f - %.3f' % ((D.U_T.median(),) + tuple(D.U_T.quantile([.1, .9]))))
print('peak |u\'| per day          : median %.3f m/s  (peak / U_T = %.2f)' % (D.U_peak.median(), (D.U_peak / D.U_T).median()))
print('U_T fastest column         : median %.3f m/s  (x%.2f section mean; column p=%s most often, h = %.1f m)'
      % (D.U_T_col.median(), (D.U_T_col / D.U_T).median(), D.p_col.mode()[0], h[D.p_col.mode()[0]]))

# ------------------------------------------------------------------ spring / neap from qprism
b = xr.open_dataset(bulk_fn)
qp = pd.Series(b.qprism.values, index=pd.DatetimeIndex(b.time.values))   # daily, Godin-based
b.close()
i_sp, _ = find_peaks(qp.values, distance=10)
i_np, _ = find_peaks(-qp.values, distance=10)
D['phase'] = ''
for idx, lab in ((i_sp, 'spring'), (i_np, 'neap')):
    for t in qp.index[idx]:
        k = np.abs(D.index - t).argmin()
        if abs(D.index[k] - t) < pd.Timedelta(hours=LUNAR_DAY / 2):
            D.iloc[k, D.columns.get_loc('phase')] = lab
print('\nqprism extrema: %d springs, %d neaps  (mean spacing %.2f / %.2f d)'
      % (len(i_sp), len(i_np), np.diff(i_sp).mean(), np.diff(i_np).mean()))
for lab in ('neap', 'spring'):
    s = D.U_T[D.phase == lab]
    print('U_T %-6s: %.3f +- %.3f m/s  (n=%d)   fastest column %.3f' % (lab, s.mean(), s.std(), len(s),
                                                                    D.U_T_col[D.phase == lab].mean()))
print('spring/neap U_T ratio %.2f' % (D.U_T[D.phase == 'spring'].mean() / D.U_T[D.phase == 'neap'].mean()))

# ------------------------------------------------------------------ s_ocn, N_0, M
s_bot = (salt[:, 0, :] * dd).sum(axis=1) / W               # (t,) bottom layer (z index 0), width-weighted
s_ocn = s_bot.mean()
N0 = np.sqrt(BETA * G * s_ocn / H)
Mfun = lambda U, s=s_ocn, om=OMEGA: np.sqrt(CD * U ** 2 / (om * np.sqrt(BETA * G * s / H) * H ** 2))
D['M'] = Mfun(D.U_T.values)
D['M_col'] = Mfun(D.U_T_col.values)

print('\ns_ocn = <bottom-layer salinity> = %.2f g/kg  (range %.2f - %.2f)' % (s_ocn, s_bot.min(), s_bot.max()))
print('N_0 = (beta g s_ocn / H)^1/2 = %.4f 1/s   (c_0 = N_0 H = %.2f m/s)' % (N0, N0 * H))
print('omega = %s = %.4e 1/s' % (args.omega, OMEGA))
summ = {}
for lab, sel in (('neap', D.phase == 'neap'), ('median', None), ('spring', D.phase == 'spring')):
    U = D.U_T.median() if sel is None else D.U_T[sel].mean()
    Uc = D.U_T_col.median() if sel is None else D.U_T_col[sel].mean()
    summ[lab] = dict(U_T=U, M=float(Mfun(U)), U_T_col=Uc, M_col=float(Mfun(Uc)))
    print('M %-6s = %.4f   (U_T %.3f m/s)    fastest column M = %.4f (U_T %.3f)' % (lab, Mfun(U), U, Mfun(Uc), Uc))
Um = summ['median']['U_T']
print('sensitivity at the median: s_ocn = 33.5 -> M = %.4f;  omega = %s -> M = %.4f'
      % (Mfun(Um, s=33.5), 'K1' if args.omega == 'M2' else 'M2',
         Mfun(Um, om=OMEGA * (12.4206 / 23.9345 if args.omega == 'M2' else 23.9345 / 12.4206))))
print('GM14 Fig 6 axis starts at M = 0.15; their Puget Sound box is M = 0.15 - 0.33')

# ------------------------------------------------------------------ save
D.to_csv(out_dir / ('M_daily_%s.csv' % args.sect))
S = pd.DataFrame(summ).T
for k, v in dict(H=H, W=W, A=A_t.mean(), s_ocn=s_ocn, N0=N0, omega=OMEGA, CD=CD).items():
    S[k] = v
S.to_csv(out_dir / ('M_summary_%s.csv' % args.sect))

# ------------------------------------------------------------------ check figure
C_SEC, C_COL, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
fig, axs = plt.subplots(3, 1, figsize=(9, 9), gridspec_kw=dict(height_ratios=[1, 1, 1.2]))

ax = axs[0]                                                # one month of the tidal velocity and its envelope
t0, t1 = pd.Timestamp('2024-08-01'), pd.Timestamp('2024-09-01')
m = (T >= t0) & (T < t1)
ax.plot(T[m], 100 * u_tide[m], color=MUTED, lw=0.8, label="u' (hourly, section mean)")
Dm = D[(D.index >= t0) & (D.index < t1)]
for sgn in (1, -1):
    ax.step(Dm.index, sgn * 100 * Dm.U_T, where='mid', color=C_SEC, lw=2, label=r'$\pm U_T$ per lunar day' if sgn > 0 else None)
ax.axhline(0, color=MUTED, lw=0.5)
ax.set_ylabel("u' [cm/s], + into cove")
ax.set_title('(a) %s tidal velocity, Aug 2024' % args.sect, loc='left', fontsize=10, pad=14)
ax.legend(loc='lower right', bbox_to_anchor=(1, 1), fontsize=8, frameon=False, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax = axs[1]                                                # daily U_T record with spring / neap picks
ax.plot(D.index, 100 * D.U_T, color=C_SEC, lw=1.5, label='section mean')
ax.plot(D.index, 100 * D.U_T_col, color=C_COL, lw=1.5, label='fastest column')
for lab, mk in (('spring', '^'), ('neap', 'v')):
    s = D[D.phase == lab]
    ax.plot(s.index, 100 * s.U_T, ls='none', marker=mk, ms=6, mfc='none', mec=INK, label=lab + ' (qprism)')
ax.set_ylabel(r'$U_T$ [cm/s]')
ax.set_title(r'(b) $U_T=\sqrt{2}\,$rms(u$^\prime$) per lunar day', loc='left', fontsize=10, pad=14)
ax.legend(loc='lower right', bbox_to_anchor=(1, 1), fontsize=8, frameon=False, ncol=4)

ax = axs[2]                                                # M against the GM14 axis
ax.axhspan(0.15, 3.0, color='#f0efec', lw=0)
ax.axhspan(0.15, 0.33, color='#cde2fb', lw=0)
ax.text(D.index[5], 0.22, 'GM14 Puget Sound box (M 0.15-0.33)', fontsize=8, color=MUTED, va='center')
ax.text(D.index[5], 1.2, 'GM14 Fig. 6 axis range (M 0.15-3.0)', fontsize=8, color=MUTED, va='center')
ax.plot(D.index, D.M, color=C_SEC, lw=1.5, label='section mean')
ax.plot(D.index, D.M_col, color=C_COL, lw=1.5, label='fastest column')
for lab, ls in (('neap', ':'), ('spring', '--')):
    ax.axhline(summ[lab]['M'], color=C_SEC, lw=1, ls=ls)
    ax.text(D.index[-1], summ[lab]['M'], ' %s %.3f' % (lab, summ[lab]['M']), fontsize=8, color=INK, va='center')
ax.set_yscale('log')
ax.set_ylim(0.01, 3.0)
ax.set_ylabel('M')
ax.set_title(r'(c) $M=(C_D U_T^2/\omega N_0 H^2)^{1/2}$, $\omega$=%s, H=%.1f m, $s_{ocn}$=%.1f'
             % (args.omega, H, s_ocn), loc='left', fontsize=10)
ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=2)

for ax in axs:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e5e4e0', lw=0.5)
fig.tight_layout()
fn = out_dir / ('M_axis_%s_%s.png' % (args.sect, args.omega))
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(fn))
