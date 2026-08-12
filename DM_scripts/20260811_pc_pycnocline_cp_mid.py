"""
Pycnocline depth at cp_mid, the Coupeville mooring inside Penn Cove, 2024-2025.

The single-point version of 20260811_pc_pycnocline.py, which did the same thing
width-averaged across the pc_lp and pc_cp TEF sections. cp_mid is the midpoint
of the pc_cp line -- rho cell (j=219, i=53), h = 20.6 m, job 'pc4' in
LO_user/extract/moor/job_lists.py -- so this is the same water the section
average was describing, without the width average on top of it. It is also the
same station as the bottom-DO series in 20260811_pc_cp_mid_bottom_DO.py, so the
two series can be read against each other directly.

METHOD. Potential density sigma0 from gsw (SP -> SA at the station's lon/lat,
pt -> CT) on the 30 sigma levels, then the buoyancy frequency on the 29
interfaces between them,

    N2_k = -(g/rho0) * (sigma_{k+1} - sigma_k) / (z_k+1 - z_k)      [s-2]

Each interface is then weighted by the buoyancy JUMP across it,
w_k = max(N2_k, 0) * dz_k -- the actual density step it carries, so only stable
interfaces vote and the answer does not shift with sigma-level spacing. That
turns the profile into a distribution of density contrast over depth, and "the
pycnocline depth" is a statistic of that distribution. Three of them:

    z_50    MEDIAN depth of the buoyancy      the primary series -- see below
    z_pyc   w-weighted centroid (mean)        dragged deep by the tail
    z_max   depth of the single largest N2    the classic definition, quantized
                                              to the 30 levels
    h_pyc   w-weighted std about z_pyc        pycnocline THICKNESS
    z_25, z_75                                the middle half of the buoyancy

plus N2_max and dsigma = sigma_bot - sigma_surf. Depths are below the
INSTANTANEOUS free surface, positive down, so this is comparable to a CTD cast
rather than to a fixed z.

WHICH DEPTH TO QUOTE: z_50. The three definitions disagree by 2.6 m here, so
the choice is not cosmetic, and the script prints the comparison that settles
it. The N2 profile is ONE broad peak, not two -- with a prominence threshold at
a third of the profile's own maximum, 97 % of days have a single peak, and the
mean profile (panel e) rises to a maximum near 4.4 m and decays monotonically
to the bed with no secondary maximum. But it is strongly SKEWED: skewness +0.57,
positive on 88 % of days, with 72 % of the column's buoyancy jump sitting BELOW
the peak. That is why the centroid lands 2.6 m deeper than the peak.

Because it is unimodal, all three depths describe the same feature and none of
them is landing in well-mixed water -- the usual objection to a centroid does
not apply. What separates them is noise. z_max moves 3.05 m from one day to the
next against a total sd of 3.57 m, so ~73 % of its variance is jitter from
being pinned to the 30 sigma levels; it is not usable as a series. z_50 and
z_pyc are both smooth (day-to-day sd 1.19 and 0.93 m) and carry an identical
signal, r = +0.99. z_50 wins on interpretation alone: it sits 1.05 m shallower,
nearer the gradient the eye picks out in panel a, because a median ignores how
far the deep tail extends while a centroid does not.

SOURCE. The pc4 extraction is run with -lt lowpass, so each sample is one
Godin-filtered field per day stamped 12:00 UTC -- the tide is already gone and
there is nothing left to filter. If you re-run it with -lt hourly the script
notices the 1 h spacing on its own and adds the raw series behind its own Godin
average; nothing below needs changing. Same convention as the bottom-DO script.

WHEN THERE IS NO PYCNOCLINE. A depth computed from a nearly uniform column is
meaningless -- the centroid of noise sits mid-column by construction. Days with
dsigma < 0.5 kg m-3 are flagged and drawn hollow rather than silently plotted.

WHAT IT SAYS (2024-2025, wb1_t0_xn11abbur00)
The pycnocline sits at z_50 = 7.09 m, 0.35 of the 20.3 m column, sd 2.32 m,
with the middle half of the buoyancy spread over 4.42-10.63 m. It is a broad
gradient, not a sharp interface: h_pyc averages 4.42 m in a 20 m column.

The width-averaged pc_cp section gave 7.82 m on the centroid definition against
8.14 m here, so the single point is 0.3 m deeper than the line it sits on --
close enough that simplifying to the mooring costs almost nothing, which is the
useful result of that comparison.

Seasonally it is deepest in March (9.40 m) and shallowest in July (5.35 m):
winter mixing drives it down, summer heating pins it near the surface. Depth
and strength are only weakly linked (corr(z_50, dsigma) -0.12, with N2_max
-0.40); a deeper pycnocline is a broader one (corr with h_pyc +0.38).

The column is essentially never unstratified -- dsigma < 0.5 kg m-3 on 1 of
729 days (2024-03-02, next to the deepest pycnocline of the record, 15.9 m on
2024-03-03) -- so the flag almost never bites and panel b reads straight
through. The event that sets the scale of panel c is mid-December 2025, when
surface salinity falls to 8.9 and dsigma hits 16.2 kg m-3. That is a real
freshet in the model, not a numerical artifact.

CAVEATS. Vertical resolution is the model's 30 sigma levels, so z_50 is a
layer-scale estimate, not fine structure. sigma0 ignores the pressure
dependence of density, which is the right call for a 21 m column. One point is
not the cove: the section version is the one to quote for a cove-wide number,
and the two differ wherever the cove is laterally structured.

run 20260811_pc_pycnocline_cp_mid.py
run 20260811_pc_pycnocline_cp_mid.py -year 2025
run 20260811_pc_pycnocline_cp_mid.py -sn lp_mid
"""
import argparse
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-job', default='pc4', type=str)
p.add_argument('-sn', default='cp_mid', type=str)
p.add_argument('-0', '--ds0', default='2024.01.02', type=str,
               help='start of the EXTRACTION (part of the filename)')
p.add_argument('-1', '--ds1', default='2025.12.30', type=str,
               help='end of the EXTRACTION (part of the filename)')
p.add_argument('-year', default='all', type=str,
               help="calendar year to plot, or 'all' for the whole record")
p.add_argument('-dsig', default=0.5, type=float,
               help='dsigma [kg m-3] below which the column counts as unstratified')
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gtagex.split('_')[0])
moor_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'moor'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_pycnocline_cp_mid'
Lfun.make_dir(out_dir)

CB = dict(blue='#0072B2', red='#CC0000', green='#009E73', orange='#D55E00',
          purple='#CC79A7', grey='#7f7f7f')
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 13
G, RHO0 = 9.81, 1025.0
# cp_mid orange, matching 20260811_pc_cp_mid_bottom_DO.py and the pc_cp section
SC = {'cp_mid': CB['orange'], 'lp_mid': CB['blue'], 'M5': CB['purple']}
COL = SC.get(args.sn, CB['orange'])

# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
stem = '%s_%s_%s.nc' % (args.sn, args.ds0, args.ds1)
cands = [moor_dir / args.job / stem, moor_dir / stem]
fn = next((c for c in cands if c.is_file()), None)
if fn is None:
    print('No mooring file for %s. Looked for:' % args.sn)
    for c in cands:
        print('  ' + str(c))
    print('\nExtract on apogee, where the run lives, using the LO driver:\n'
          '  cd ~/LO/extract/moor\n'
          '  python multi_mooring_driver.py -gtx %s -ro 2 -0 %s -1 %s '
          '-lt lowpass -job %s -get_all True -Nproc 100 > %s.log &'
          % (args.gtagex, args.ds0, args.ds1, args.job, args.job))
    sys.exit(1)
print('reading ' + str(fn))

ds = xr.open_dataset(fn)
tt = pd.to_datetime(ds.ocean_time.values)
z_rho = ds.z_rho.values                                # (time, 30)
z_w = ds.z_w.values                                    # (time, 31)
zeta = ds.zeta.values
h = float(ds.h)
lon, lat = float(ds.lon_rho), float(ds.lat_rho)
SP = ds.salt.values.astype(float)
pt = ds.temp.values.astype(float)
ds.close()

dt_h = np.median(np.diff(tt.values)) / np.timedelta64(1, 'h')
hourly = bool(np.isclose(dt_h, 1.0, atol=0.01))
print('%s at (%.6f, %.6f), h = %.1f m; %d samples at %.1f h, %s to %s'
      % (args.sn, lon, lat, h, len(tt), dt_h, tt[0].date(), tt[-1].date()))
print('input is %s' % ('hourly -- Godin filter applied below'
                       if hourly else 'already Godin-lowpassed (one field/day)'))

# ---------------------------------------------------------------------------
# density structure and the pycnocline
# ---------------------------------------------------------------------------
pres = gsw.p_from_z(z_rho, lat)
SA = gsw.SA_from_SP(SP, pres, lon, lat)
CT = gsw.CT_from_pt(SA, pt)
sig = gsw.sigma0(SA, CT)                               # (time, 30)

dzi = np.diff(z_rho, axis=1)                           # interface thickness
N2 = -(G / RHO0) * np.diff(sig, axis=1) / dzi          # (time, 29)
d_int = zeta[:, None] - z_w[:, 1:-1]                   # depth below surface, + down

w = np.clip(N2, 0.0, None) * dzi                       # buoyancy jump, stable only
wsum = w.sum(axis=1)
ok = wsum > 0
z_pyc = np.where(ok, (w * d_int).sum(axis=1) / np.where(ok, wsum, 1.0), np.nan)
h_pyc = np.where(ok, np.sqrt((w * (d_int - z_pyc[:, None]) ** 2).sum(axis=1)
                             / np.where(ok, wsum, 1.0)), np.nan)
skew = np.where(ok, ((w * (d_int - z_pyc[:, None]) ** 3).sum(axis=1)
                     / np.where(ok, wsum, 1.0)) / h_pyc ** 3, np.nan)
kmax = N2.argmax(axis=1)
ti = np.arange(len(tt))
z_max = np.where(ok, d_int[ti, kmax], np.nan)
N2_max = np.where(ok, N2[ti, kmax], np.nan)

# Quantiles of the same buoyancy-jump distribution. z_50 is the primary series:
# it carries the identical signal to the centroid (r = 0.99) but is not dragged
# down by the deep tail, and unlike z_max it is continuous rather than pinned to
# the 30 sigma levels. See the docstring for why that matters here.
sh = slice(None, None, -1)                             # reorder shallow -> deep
dq, wq = d_int[:, sh], w[:, sh]
cdf = np.cumsum(wq, axis=1) / np.where(ok, wsum, 1.0)[:, None]
z_25, z_50, z_75 = (np.full(len(tt), np.nan) for _ in range(3))
for i in np.flatnonzero(ok):
    z_25[i], z_50[i], z_75[i] = np.interp([0.25, 0.5, 0.75], cdf[i], dq[i])

df = pd.DataFrame(index=tt)
df.index.name = 'time_utc'
df['z_50'] = z_50                                      # primary
df['z_25'] = z_25
df['z_75'] = z_75
df['z_pyc'] = z_pyc
df['h_pyc'] = h_pyc
df['bskew'] = skew
df['z_max'] = z_max
df['N2_max'] = N2_max
df['dsigma'] = sig[:, 0] - sig[:, -1]                  # bottom minus surface
df['sig_surf'] = sig[:, -1]
df['sig_bot'] = sig[:, 0]
df['s_surf'] = SP[:, -1]
df['s_bot'] = SP[:, 0]
df['H'] = h + zeta
df['z_pyc_norm'] = df.z_pyc / df.H
df['strat'] = df.dsigma >= args.dsig

# Only hourly input needs a Godin filter; a lowpass extraction is already one.
# Filter the FULL record so windowing to a year does not blank its ends.
for c in ['z_50', 'z_25', 'z_75', 'z_pyc', 'h_pyc', 'z_max', 'N2_max',
          'dsigma', 'z_pyc_norm']:
    df[c + '_lp'] = zfun.lowpass(df[c].to_numpy(dtype=float),
                                 f='godin') if hourly else df[c]

df.to_csv(out_dir / ('pycnocline_%s_%s.csv' % (args.sn, args.gtagex)),
          float_format='%.5f')

if args.year.lower() == 'all':
    span_lbl, tag = '2024-2025', 'all'
else:
    yr = int(args.year)
    span_lbl, tag = str(yr), str(yr)
    keep = df.index.year == yr
    if not keep.any():
        print('*** no samples in %d -- the extraction covers %s to %s'
              % (yr, args.ds0, args.ds1))
        sys.exit(1)
    df, N2, d_int, sig = df[keep], N2[keep], d_int[keep], sig[keep]
    tt = df.index

# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
print('\n--- pycnocline at %s, %s (depths in m below the surface) ---'
      % (args.sn, span_lbl))
print('z_50    mean %.2f, median %.2f, 10th-90th %.2f-%.2f, sd %.2f'
      % (df.z_50.mean(), df.z_50.median(), df.z_50.quantile(0.1),
         df.z_50.quantile(0.9), df.z_50.std()))
print('        as a fraction of the column: %.3f H (mean H %.1f m)'
      % ((df.z_50 / df.H).mean(), df.H.mean()))
print('        buoyancy IQR %.2f to %.2f m (width %.2f m)'
      % (df.z_25.mean(), df.z_75.mean(), (df.z_75 - df.z_25).mean()))
print('h_pyc   mean %.2f  -- pycnocline thickness' % df.h_pyc.mean())

# The three definitions head to head. A depth series is only useful if its
# day-to-day scatter is small against the signal it is meant to show, so the
# noise ratio is the number that decides which one to quote.
print('\n--- three definitions of "the" depth ---')
print('%-16s %6s %6s %9s %9s' % ('', 'mean', 'sd', 'd-to-d sd', 'noise/sd'))
for nm, c in [('z_max  (peak)', 'z_max'), ('z_50   (median)', 'z_50'),
              ('z_pyc  (centroid)', 'z_pyc')]:
    v = df[c].dropna()
    dd_ = np.diff(v.values).std()
    print('%-16s %6.2f %6.2f %9.2f %9.2f'
          % (nm, v.mean(), v.std(), dd_, dd_ / v.std()))
print('corr(z_50, z_pyc) %+.3f | corr(z_50, z_max) %+.3f'
      % (df.z_50.corr(df.z_pyc), df.z_50.corr(df.z_max)))
print('distribution skewness: mean %+.2f, positive on %.0f%% of samples '
      '(> 0 = tail toward the bed)'
      % (df.bskew.mean(), 100 * (df.bskew > 0).mean()))
print('N2_max  mean %.4f, max %.4f s-2' % (df.N2_max.mean(), df.N2_max.max()))
print('dsigma  mean %.2f, min %.2f, max %.2f kg m-3'
      % (df.dsigma.mean(), df.dsigma.min(), df.dsigma.max()))
n_un = int((~df.strat).sum())
print('unstratified (dsigma < %.1f): %d of %d days (%.1f%%)'
      % (args.dsig, n_un, len(df), 100 * n_un / len(df)))

# Is the profile actually two-peaked, or just skewed? That is what decides
# whether a single depth means anything at all: a centroid between two separate
# pycnoclines would sit in well-mixed water, but a centroid inside one broad
# skewed gradient is real water. Count peaks with a prominence threshold set as
# a fraction of the profile's own maximum, so the test scales with how
# stratified the day is.
print('\n--- is the profile bimodal? peaks per profile ---')
N2s = N2[:, ::-1]
for pf in [0.10, 0.20, 0.33]:
    npk = np.ones(len(df), int)
    for i in range(len(df)):
        pr = N2s[i]
        mx = np.nanmax(pr)
        if not np.isfinite(mx) or mx <= 0:
            continue
        pk, _ = find_peaks(pr, prominence=pf * mx)
        npk[i] = max(len(pk), 1)
    print('  prominence > %2.0f%% of max N2: 1 peak %3.0f%% | 2 peaks %3.0f%% '
          '| 3+ %3.0f%%' % (100 * pf, 100 * (npk == 1).mean(),
                            100 * (npk == 2).mean(), 100 * (npk >= 3).mean()))
above = np.array([w[i][d_int[i] < z_max[i]].sum() for i in range(len(df))])
print('  buoyancy jump above the peak %.0f%%, below it %.0f%% -- a top-heavy '
      'gradient with a deep tail' % (100 * (above / wsum[:len(df)]).mean(),
                                     100 * (1 - above / wsum[:len(df)]).mean()))

print('\nshallowest %s (%.2f m) | deepest %s (%.2f m)'
      % (df.z_50.idxmin().date(), df.z_50.min(),
         df.z_50.idxmax().date(), df.z_50.max()))
i = df.dsigma.idxmax()
print('peak stratification %s: dsigma %.1f kg m-3, surface salinity %.1f, '
      'z_50 %.1f m, N2_max %.3f s-2'
      % (i.date(), df.dsigma[i], df.s_surf[i], df.z_50[i], df.N2_max[i]))

print('\n--- what the depth tracks ---')
print('corr(z_50, dsigma) %+.2f | corr(z_50, N2_max) %+.2f | '
      'corr(z_50, h_pyc) %+.2f'
      % (df.z_50.corr(df.dsigma), df.z_50.corr(df.N2_max),
         df.z_50.corr(df.h_pyc)))

MO = df.groupby(df.index.month)[['z_50', 'z_pyc', 'z_max', 'h_pyc', 'N2_max',
                                 'dsigma']].mean()
MO.index.name = 'month'
print('\n--- monthly means ---')
print(MO.round(3).to_string())
MO.to_csv(out_dir / ('pycnocline_monthly_%s.csv' % args.sn),
          float_format='%.5f')

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=0.42, wspace=0.4)

# a. the whole density structure, with the pycnocline drawn on it
ax = fig.add_subplot(gs[0, :])
zmesh = np.nanmean(d_int, axis=0)
pc = ax.pcolormesh(tt, zmesh, np.clip(N2, 1e-6, None).T, shading='nearest',
                   cmap='viridis', norm=LogNorm(vmin=1e-4, vmax=5e-2))
ax.plot(tt, df.z_50, color='w', lw=1.8)
ax.plot(tt, df.z_50, color='k', lw=0.9, label='$z_{50}$')
ax.plot(tt, df.z_25, color='w', lw=0.9, ls='--', alpha=0.8)
ax.plot(tt, df.z_75, color='w', lw=0.9, ls='--', alpha=0.8,
        label='$z_{25}$, $z_{75}$ (buoyancy IQR)')
ax.invert_yaxis()
ax.set_ylabel('depth below\nsurface [m]', fontsize=FS)
ax.set_title('a. $N^2$ structure at %s, with the pycnocline on it '
             '(dashed = middle half of the buoyancy)' % args.sn,
             fontsize=FS, loc='left')
ax.legend(fontsize=FS - 4, ncol=2, frameon=False, loc='lower left',
          labelcolor='w')
cb = fig.colorbar(pc, ax=ax, pad=0.01, aspect=18)
cb.set_label('$N^2$ [s$^{-2}$]', fontsize=FS - 2)

# b. the series itself
ax = fig.add_subplot(gs[1, :])
ax.plot(tt, df.z_max, color=CB['grey'], lw=0.7, alpha=0.65,
        label='$z_{max}$, peak $N^2$ (quantized, %.0f%% of its variance is '
              'day-to-day jitter)'
              % (100 * (np.diff(df.z_max.dropna().values).std()
                        / df.z_max.std()) ** 2))
ax.plot(tt, df.z_pyc, color=CB['purple'], lw=1.0, alpha=0.9,
        label='$z_{pyc}$, centroid')
ax.plot(tt, df.z_50, color=COL, lw=1.7, label='$z_{50}$, median (use this one)')
un = df[~df.strat]
if len(un):
    ax.plot(un.index, un.z_50, 'o', mfc='none', mec=CB['red'], ms=5,
            label='unstratified, $\\Delta\\sigma_0$ < %.1f' % args.dsig)
ax.axhline(df.H.mean(), color='k', lw=0.8, ls=':')
ax.text(0.998, df.H.mean(), 'bed, %.1f m ' % df.H.mean(), ha='right',
        va='bottom', fontsize=FS - 5, transform=ax.get_yaxis_transform())
ax.invert_yaxis()
ax.set_ylabel('depth [m]', fontsize=FS)
ax.set_title('b. three definitions of the pycnocline depth (%s)'
             % ('hourly, Godin-filtered' if hourly
                else 'daily Godin-lowpassed output'), fontsize=FS, loc='left')
ax.legend(fontsize=FS - 5, ncol=2, frameon=False, loc='lower left')
ax.grid(**GRID)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

# c. stratification, the context for b
ax = fig.add_subplot(gs[2, 0])
ax.plot(tt, df.dsigma, color=COL, lw=1.2)
ax.axhline(args.dsig, color=CB['red'], lw=1.0, ls='--')
ax.set_ylabel('$\\Delta\\sigma_0$ [kg m$^{-3}$]', fontsize=FS - 1)
ax.set_title('c. stratification', fontsize=FS, loc='left')
ax.grid(**GRID)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))

# d. seasonal cycle
ax = fig.add_subplot(gs[2, 1])
ax.plot(MO.index, MO.z_50, color=COL, lw=2.0, marker='o', ms=4)
ax.invert_yaxis()
ax2 = ax.twinx()
ax2.plot(MO.index, MO.dsigma, color=CB['grey'], lw=1.3, ls=':')
ax2.set_ylabel('$\\Delta\\sigma_0$ (dotted)', fontsize=FS - 4,
               color=CB['grey'])
ax2.tick_params(labelsize=FS - 5, colors=CB['grey'])
ax.set_xlabel('month', fontsize=FS - 1)
ax.set_ylabel('$z_{50}$ [m]', fontsize=FS - 1)
ax.set_title('d. seasonal cycle', fontsize=FS, loc='left')
ax.set_xticks(range(1, 13, 2))
ax.grid(**GRID)

# e. distribution
ax = fig.add_subplot(gs[2, 2])
zmean = np.nanmean(N2, axis=0)
ax.plot(zmean, np.nanmean(d_int, axis=0), color=COL, lw=2.0)
ax.axhline(df.z_50.mean(), color=COL, lw=1.2,
           label='$z_{50}$ %.1f m' % df.z_50.mean())
ax.axhline(df.z_pyc.mean(), color=CB['purple'], lw=1.2, ls='--',
           label='$z_{pyc}$ %.1f m' % df.z_pyc.mean())
ax.axhline(df.z_max.mean(), color=CB['grey'], lw=1.2, ls=':',
           label='$z_{max}$ %.1f m' % df.z_max.mean())
ax.invert_yaxis()
ax.set_xlabel('mean $N^2$ [s$^{-2}$]', fontsize=FS - 1)
ax.set_ylabel('depth [m]', fontsize=FS - 1)
ax.set_title('e. one broad skewed peak', fontsize=FS, loc='left')
ax.legend(fontsize=FS - 4, frameon=False)
ax.grid(**GRID)

fig.suptitle('Pycnocline at %s (Penn Cove, h = %.1f m), %s  %s'
             % (args.sn, h, args.gtagex, span_lbl), fontsize=FS + 1)
fn_out = out_dir / ('pycnocline_%s_%s.png' % (args.sn, tag))
fig.savefig(fn_out, dpi=200, bbox_inches='tight', transparent=True)
print('\nwrote %s' % fn_out)
plt.close(fig)
