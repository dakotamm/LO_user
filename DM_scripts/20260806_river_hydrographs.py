"""
River hydrographs for wb1_t0_xn11abbur00 -- every source, and the total.

Runs at home, off the pickle that 20260806_river_reduce.py makes on apogee
from the ~730 daily rivers.nc files. So this is the forcing as the model
actually ran, not a reconstruction of it, without needing the forcing tree
locally. Run the reducer on apogee first and copy the file across; it prints
the scp line.

wb1 has 32 sources and they are NOT equivalent:

  LO rivers (3)    snohomish, stillaguamish, skagit. Filled from USGS gauges,
                   scaled by the drainage-area ratio in river_info.p. These
                   carry real events and real interannual difference.
  tiny rivers (6)  Mohamedali et al. (2020) climatology, by day of year.
  WWTPs (23)       Moh20 / Wasielewski et al. (2024) climatology, by day of
                   year. The was24 "raw" branch in the forcing code only fires
                   for dates before 2021, so nothing here uses it.

So 29 of the 32 sources are a fixed annual cycle that repeats exactly in 2024
and 2025, and every bit of event and interannual variability in the run lives
in 3 rivers. That is worth stating out loud because it bounds what a
"seasonal signal" in this run can possibly mean: the seasonality of the small
sources is prescribed, not simulated.

That claim is checked two independent ways, which should agree:

  LOG     each day's forcing writes Info/screen_output.txt, which names the
          fill method per source in the model's own words.
  REPEAT  matching 2024 and 2025 on DAY OF YEAR: a climatological source has
          a max difference of exactly 0. Day of year, not month-day -- 2024 is
          a leap year, and the traps climatologies are indexed by dayofyear.

river_transport is signed by the source's isign -- the sign is which way the
flux points on the grid, not whether it is a source or a sink. Magnitude is
what a hydrograph wants, so everything below is abs().

run 20260806_river_hydrographs.py
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.path import Path as MplPath
from scipy.signal import find_peaks

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
gtx = 'wb1_t0_xn11abbur00'
FRC = 'trapsN00'
DATE0, DATE1 = '2024.01.01', '2025.12.31'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_river_hydrographs'
Lfun.make_dir(out_dir)

# written by 20260806_river_reduce.py on apogee, into this same folder
CACHE = out_dir / ('river_daily_%s_%s_%s.p' % (FRC, DATE0, DATE1))

CB = dict(blue='#0072B2', orange='#D55E00', green='#009E73', red='#CC0000',
          purple='#7B3294', dgreen='#1B7837', yellow='#E69F00', pink='#CC79A7')
GRP_C = {'LO river': CB['blue'], 'tiny river': CB['green'], 'WWTP': CB['orange']}

# ------------------------------------------------------------------- load ---
if not CACHE.is_file():
    raise SystemExit(
        'no reduced river file at\n  %s\n'
        'Run 20260806_river_reduce.py on apogee and copy it here:\n'
        '  scp apogee:LO_output/DM_outs/20260806_river_hydrographs/%s \\\n'
        '      ~/LO_output/DM_outs/20260806_river_hydrographs/'
        % (CACHE, CACHE.name))

C = pd.read_pickle(CACHE)
Q, meta, prov, missing = C['Q'], C['meta'], C['prov'], C['missing']
T = C['V']['temp']
print('reduced river forcing from %s' % CACHE.name)
print('  made %s from %s on the %s grid'
      % (C['info']['made'], C['info']['frc'], Ldir['gridname']))
print('%s: %d sources, %s to %s (%d days)'
      % (gtx, Q.shape[1], Q.index[0].date(), Q.index[-1].date(), len(Q)))
if missing:
    print('  ** %d model days had no %s forcing file, e.g. %s'
          % (len(missing), FRC, ', '.join(d.strftime('%Y.%m.%d')
                                          for d in missing[:5])))

# ------------------------------------------------------------ classify ---
# direction 2 is a vertical point source, which is how the traps code writes a
# WWTP; 0/1 are u/v cell-face rivers. Within the rivers, the pre-existing LO
# rivers are the ones named in the grid's river_info.csv. Classifying on
# direction rather than on name avoids the tiny-river names that get an ' R'
# appended when they collide with an LO river.
lo_names = list(pd.read_csv(Ldir['grid'] / 'river_info.csv',
                            index_col='rname').index)
meta['group'] = np.where(meta.direction == 2, 'WWTP',
                         np.where(meta.index.isin(lo_names),
                                  'LO river', 'tiny river'))
meta = meta.loc[Q.columns]

# provenance in the model's own words, carried over from the forcing log by
# the reducer.
# The tiny-river code appends ' R' to a name that would otherwise collide with
# a WWTP (e.g. the river 'Port Townsend R' vs 'PORT TOWNSEND STP'), but the log
# prints the un-suffixed name, so fall back to that before giving up
meta['fill'] = [prov.get(n, prov.get(n[:-2] if n.endswith(' R') else n,
                                     'unknown')) for n in meta.index]

# independent check: does the source repeat exactly between the two years?
# Match on DAY OF YEAR, not month-day. The traps climatologies are indexed by
# dayofyear (yd_ind in the forcing code), so in a leap year every date after
# Feb 29 gets a day-of-year one higher than the same calendar date in a
# non-leap year. Matching on month-day would compare DOY 61 against DOY 60 for
# all of Mar-Dec and report a small nonzero difference for sources that are in
# fact pure climatology.
doy = Q.index.dayofyear
y = Q.index.year
yrs = sorted(set(y))
if len(yrs) >= 2:
    a = Q[y == yrs[0]].set_index(doy[y == yrs[0]])
    b = Q[y == yrs[-1]].set_index(doy[y == yrs[-1]])
    common = a.index.intersection(b.index)      # drops DOY 366 automatically
    meta['repeat_maxdiff'] = (a.loc[common] - b.loc[common]).abs().max()
else:
    meta['repeat_maxdiff'] = np.nan
meta['climatological'] = meta.repeat_maxdiff < 1e-9

n_clim = int(meta.climatological.sum())
log_clim = meta.fill.str.contains('climatology|dataset', regex=True)
print('  groups: ' + ', '.join('%s %d' % (g, n)
                               for g, n in meta.group.value_counts().items()))
if len(yrs) >= 2:
    print('  %d of %d sources repeat exactly between %d and %d on day of year '
          '(climatological); largest between-year difference %.3g m3/s'
          % (n_clim, len(meta), yrs[0], yrs[-1], meta.repeat_maxdiff.max()))
    agree = (log_clim == meta.climatological).all()
    print('  forcing log and repeat test %s'
          % ('AGREE' if agree else '** DISAGREE **'))
    if not agree:
        print(meta.loc[log_clim != meta.climatological,
                       ['group', 'fill', 'repeat_maxdiff']])
else:
    # the repeat test needs two years; fall back to the forcing log alone
    # rather than reporting a disagreement it cannot actually test
    meta['climatological'] = log_clim
    print('  only %d year in the record, so the repeat test cannot run; '
          '%d of %d sources are climatological per the forcing log'
          % (len(yrs), int(log_clim.sum()), len(meta)))

# ------------------------------------------------------------- totals ---
riv_cols = list(meta.index[meta.group != 'WWTP'])
lo_cols = list(meta.index[meta.group == 'LO river'])
tiny_cols = list(meta.index[meta.group == 'tiny river'])
wwtp_cols = list(meta.index[meta.group == 'WWTP'])

tot = pd.DataFrame(index=Q.index)
tot['rivers'] = Q[riv_cols].sum(axis=1)
tot['LO rivers'] = Q[lo_cols].sum(axis=1)
tot['tiny rivers'] = Q[tiny_cols].sum(axis=1)
tot['WWTPs'] = Q[wwtp_cols].sum(axis=1)
tot['all'] = Q.sum(axis=1)
tot['rivers_7d'] = tot.rivers.rolling(7, center=True, min_periods=1).mean()
tot['rivers_30d'] = tot.rivers.rolling(30, center=True, min_periods=1).mean()
# flow-weighted river temperature: what the freshwater actually arrives at
tot['riv_temp'] = ((T[riv_cols] * Q[riv_cols]).sum(axis=1)
                   / Q[riv_cols].sum(axis=1))

SEC_PER_DAY = 86400.0

# --------------------------------------------------------- per source ---
S = pd.DataFrame(index=Q.columns)
S['group'] = meta.group
S['fill'] = meta.fill
S['climatological'] = meta.climatological
S['mean_m3s'] = Q.mean()
S['min_m3s'] = Q.min()
S['max_m3s'] = Q.max()
S['pct_of_total'] = 100 * Q.mean() / Q.mean().sum()
S['volume_km3_per_yr'] = Q.mean() * SEC_PER_DAY * 365.25 / 1e9
S['peak_date'] = [Q[c].idxmax().date() for c in Q.columns]
S['peak_month'] = [Q[c].idxmax().month for c in Q.columns]
# a climatological source has no meaningful flashiness, but the ratio is still
# the honest way to say how peaked its prescribed cycle is
S['peak_over_mean'] = S.max_m3s / S.mean_m3s
seas = dict(DJF=[12, 1, 2], MAM=[3, 4, 5], JJA=[6, 7, 8], SON=[9, 10, 11])
for s, mo in seas.items():
    S[s + '_m3s'] = Q[Q.index.month.isin(mo)].mean()
S['summer_fraction'] = S.JJA_m3s / S.mean_m3s
S = S.sort_values('mean_m3s', ascending=False)

# ------------------------------------------------- seasonal characterisation ---
def center_of_volume(q):
    """Date by which half the period's water volume has been delivered."""
    c = q.cumsum() / q.sum()
    return c.index[np.searchsorted(c.values, 0.5)]


print('\n--- totals ---')
print('all sources      mean %8.1f  min %8.1f  max %8.1f m3/s'
      % (tot['all'].mean(), tot['all'].min(), tot['all'].max()))
for k in ['rivers', 'LO rivers', 'tiny rivers', 'WWTPs']:
    print('%-16s mean %8.2f m3/s  = %5.2f%% of all  (%.2f km3/yr)'
          % (k, tot[k].mean(), 100 * tot[k].mean() / tot['all'].mean(),
             tot[k].mean() * SEC_PER_DAY * 365.25 / 1e9))

print('\n--- seasonal shape of the total river flow ---')
for yr in yrs:
    q = tot.rivers[tot.index.year == yr]
    if len(q) < 300:
        print('%d: %d days only, skipped' % (yr, len(q)))
        continue
    qs = tot.rivers_30d[tot.index.year == yr]
    print('%d  mean %6.1f   max %6.1f on %s   30-d low %5.1f on %s   '
          'CoV %s   Nov-Mar %4.1f%% of volume'
          % (yr, q.mean(), q.max(), q.idxmax().strftime('%b %d'),
             qs.min(), qs.idxmin().strftime('%b %d'),
             center_of_volume(q).strftime('%b %d'),
             100 * q[q.index.month.isin([11, 12, 1, 2, 3])].sum() / q.sum()))
# water year 2025 is the only one the run covers end to end
wy = tot.rivers['2024-10-01':'2025-09-30']
if len(wy) > 360:
    print('WY2025 (Oct 2024 - Sep 2025)  mean %6.1f   CoV %s'
          % (wy.mean(), center_of_volume(wy).strftime('%b %d')))

print('\n--- month of peak flow, by river (only the 3 gauged ones can move) ---')
mn = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for c in lo_cols + tiny_cols:
    m = Q[c].groupby(Q.index.month).mean()
    print('  %-16s %-11s monthly-mean max in %s, min in %s   '
          '(peak day %s, %.0f m3/s)'
          % (c, '[repeats]' if meta.loc[c, 'climatological'] else '[gauged]',
             mn[m.idxmax() - 1], mn[m.idxmin() - 1],
             S.loc[c, 'peak_date'], S.loc[c, 'max_m3s']))

print('\n--- biggest river events (peaks in daily total river flow) ---')
pk, _ = find_peaks(tot.rivers.values,
                   height=np.percentile(tot.rivers.values, 90), distance=7)
ev = tot.rivers.iloc[pk].sort_values(ascending=False)
print('%-12s %10s %10s' % ('date', 'Q m3/s', 'x median'))
for d, v in ev.head(12).items():
    print('%-12s %10.1f %10.2f' % (d.date(), v, v / tot.rivers.median()))
print('%d peaks above the 90th percentile in %d days' % (len(pk), len(tot)))

print('\n--- top 12 sources by mean flow ---')
print('%-45s %-11s %9s %7s %8s' % ('source', 'group', 'mean m3/s', '% tot',
                                   'JJA/mean'))
for c, r in S.head(12).iterrows():
    print('%-45s %-11s %9.2f %7.2f %8.2f'
          % (c[:45], r.group, r.mean_m3s, r.pct_of_total, r.summer_fraction))

zero = S.index[S.max_m3s == 0]
if len(zero):
    print('\n%d source(s) carry no water at all for the whole run: %s'
          % (len(zero), ', '.join(zero)))

print('\n--- flow-weighted river temperature ---')
tm = tot.riv_temp.groupby(tot.index.month).mean()
print('  coldest %s %.1f C, warmest %s %.1f C, annual range %.1f C'
      % (mn[tm.idxmin() - 1], tm.min(), mn[tm.idxmax() - 1], tm.max(),
         tm.max() - tm.min()))

# ------------------------------------------------ Penn Cove local sources ---
# X/Epos are u/v-face indices for rivers and rho indices for point sources;
# the two differ by less than a cell, which is well inside the tolerance for
# asking "is this source in Penn Cove".
try:
    dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
    lon, lat = dsg.lon_rho.values, dsg.lat_rho.values
    dsg.close()
    ii = meta.Epos.astype(int).clip(0, lon.shape[0] - 1)
    jj = meta.Xpos.astype(int).clip(0, lon.shape[1] - 1)
    meta['lon'] = lon[ii, jj]
    meta['lat'] = lat[ii, jj]
    pc = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc.p')
    inpc = MplPath(np.column_stack([pc.x.values, pc.y.values])).contains_points(
        np.column_stack([meta.lon.values, meta.lat.values]))
    meta['in_pc'] = inpc
    S['in_pc'] = meta.in_pc
    if inpc.any():
        print('\n--- sources inside the Penn Cove polygon ---')
        for c in meta.index[inpc]:
            print('  %-45s %-11s %7.3f m3/s' % (c[:45], meta.loc[c, 'group'],
                                                S.loc[c, 'mean_m3s']))
        print('  total %.3f m3/s = %.3f%% of all sources'
              % (Q[list(meta.index[inpc])].sum(axis=1).mean(),
                 100 * Q[list(meta.index[inpc])].sum(axis=1).mean()
                 / tot['all'].mean()))
    else:
        print('\nno sources fall inside the Penn Cove polygon')
except Exception as e:
    print('\nskipped the Penn Cove flag: %s' % e)

# ------------------------------------------------------------------ save ---
Q.round(4).to_csv(out_dir / 'daily_flow.csv', index_label='date')
tot.round(4).to_csv(out_dir / 'totals.csv', index_label='date')
S.round(4).to_csv(out_dir / 'source_summary.csv',
                  index_label='source')

# ---------------------------------------------------------- figure 1: run ---
plt.close('all')
fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True,
                        layout='constrained',
                        gridspec_kw=dict(height_ratios=[2, 2, 1.2, 1.2]))

A = axs[0]
A.fill_between(tot.index, 0, tot.rivers, color='0.85')
A.plot(tot.index, tot.rivers, color='0.55', lw=0.8, label='daily')
A.plot(tot.index, tot.rivers_7d, color='k', lw=1.6, label='7-day mean')
A.plot(ev.head(8).index, ev.head(8).values, 'v', color=CB['red'], ms=7,
       label='8 largest events')
A.set_ylabel('total river flow (m$^3$ s$^{-1}$)')
A.set_title('%s -- river forcing as the model ran it (%s), %s to %s'
            % (gtx, FRC, Q.index[0].date(), Q.index[-1].date()))
A.grid(color='lightgray', ls='--', alpha=0.5)
A.legend(fontsize=8, ncol=3, loc='upper right')

B = axs[1]
for c, col in zip(lo_cols, [CB['blue'], CB['green'], CB['orange']]):
    B.plot(Q.index, Q[c], lw=1.1, color=col, label='%s (%.0f m$^3$ s$^{-1}$)'
           % (c, S.loc[c, 'mean_m3s']))
B.set_ylabel('gauged rivers (m$^3$ s$^{-1}$)')
B.grid(color='lightgray', ls='--', alpha=0.5)
B.legend(fontsize=8, ncol=3, loc='upper right')
B.text(0.005, 0.9, 'USGS gauge data -- the only sources with real variability',
       transform=B.transAxes, fontsize=8, color='0.3')

C = axs[2]
for c in tiny_cols:
    C.plot(Q.index, Q[c], lw=1.0, label=c)
C.set_yscale('log')
C.set_ylabel('tiny rivers\n(m$^3$ s$^{-1}$)')
C.grid(color='lightgray', ls='--', alpha=0.5)
C.legend(fontsize=7, ncol=3, loc='lower right')
C.text(0.005, 0.06, 'climatology -- 2025 repeats 2024 exactly',
       transform=C.transAxes, fontsize=8, color='0.3')

D = axs[3]
D.plot(tot.index, tot.WWTPs, color=CB['purple'], lw=1.4, label='all 23 WWTPs')
for c in S[S.group == 'WWTP'].head(3).index:
    D.plot(Q.index, Q[c], lw=0.9, alpha=0.8, label=c[:32])
D.set_ylabel('WWTPs\n(m$^3$ s$^{-1}$)')
D.set_xlabel('date')
D.grid(color='lightgray', ls='--', alpha=0.5)
D.legend(fontsize=7, ncol=2, loc='lower right')
D.text(0.005, 0.06, 'climatology -- 2025 repeats 2024 exactly',
       transform=D.transAxes, fontsize=8, color='0.3')
D.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

fn1 = out_dir / 'hydrograph.png'
fig.savefig(fn1, dpi=200, bbox_inches='tight')

# ----------------------------------------------------- figure 2: seasonal ---
fig, axs = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')

A = axs[0, 0]
# Share of the total rather than absolute flow: absolute is already figure 1
# panel A, and stacking 32 daily series just makes a solid block. Share is the
# question this panel can answer that the other one cannot -- the small
# prescribed sources are a rounding error in winter but not in late summer,
# because only the gauged rivers drop away. Smoothed, or the daily spikes in
# the denominator make the bands flicker.
sm = Q.rolling(7, center=True, min_periods=1).mean()
bands = ([(c, sm[c]) for c in S[S.group == 'LO river'].index]
         + [('tiny rivers (%d)' % len(tiny_cols), sm[tiny_cols].sum(axis=1)),
            ('WWTPs (%d)' % len(wwtp_cols), sm[wwtp_cols].sum(axis=1))])
den = sm.sum(axis=1)
A.stackplot(Q.index, [100 * v / den for _, v in bands],
            labels=[k for k, _ in bands],
            colors=[CB['blue'], CB['green'], CB['orange'], CB['purple'],
                    CB['red']])
A.set_ylim(0, 100)
A.set_ylabel('% of total freshwater input')
A.set_title('composition of the total input, 7-day smoothed')
A.legend(fontsize=7, ncol=3, loc='lower center', framealpha=0.9)
A.grid(color='lightgray', ls='--', alpha=0.5)
A.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

# the small sources only matter as a share, so say by how much and when
shr = 100 * (sm[tiny_cols].sum(axis=1) + sm[wwtp_cols].sum(axis=1)) / den
print('\n--- share of total input from the 29 prescribed small sources ---')
print('  min %.2f%% (%s), max %.2f%% (%s), mean %.2f%%'
      % (shr.min(), shr.idxmin().strftime('%b %d %Y'),
         shr.max(), shr.idxmax().strftime('%b %d %Y'), shr.mean()))

B = axs[0, 1]
w = 0.27
for k, (g, col) in enumerate(GRP_C.items()):
    cols = list(meta.index[meta.group == g])
    m = Q[cols].sum(axis=1).groupby(Q.index.month).mean()
    B.bar(m.index + (k - 1) * w, m.values, width=w, color=col, label=g)
B.set_yscale('log')
B.set_xticks(range(1, 13))
B.set_xticklabels(mn, fontsize=8)
B.set_ylabel('m$^3$ s$^{-1}$ (log)')
B.set_title('monthly mean by group, both years pooled')
B.legend(fontsize=8)
B.grid(color='lightgray', ls='--', alpha=0.5, axis='y')

C = axs[1, 0]
for yr in yrs:
    q = tot.rivers[tot.index.year == yr]
    if len(q) < 300:
        continue
    c = 100 * q.cumsum() / q.sum()
    C.plot(q.index.dayofyear, c.values, lw=1.6, label='%d total' % yr)
    cv = center_of_volume(q)
    C.plot(cv.dayofyear, 50, 'o', ms=7, mfc='none', mew=1.6,
           color=C.lines[-1].get_color())
for c_, col in zip(lo_cols, [CB['blue'], CB['green'], CB['orange']]):
    q = Q[c_][Q.index.year == yrs[-1]]
    C.plot(q.index.dayofyear, 100 * q.cumsum() / q.sum(), lw=1.0, ls='--',
           color=col, label='%s %d' % (c_, yrs[-1]))
C.axhline(50, color='0.6', lw=0.8)
C.set_xlabel('day of year')
C.set_ylabel('cumulative % of annual volume')
C.set_title('when the water arrives (circle = half the annual volume)')
C.legend(fontsize=7, loc='upper left')
C.grid(color='lightgray', ls='--', alpha=0.5)

D = axs[1, 1]
D.plot(tot.index, tot.riv_temp, color=CB['red'], lw=1.0)
D.plot(tot.index, tot.riv_temp.rolling(15, center=True, min_periods=1).mean(),
       color='k', lw=1.5)
D.set_ylabel('flow-weighted river temperature ($^\\circ$C)')
D.set_title('temperature of the freshwater entering the domain')
D.grid(color='lightgray', ls='--', alpha=0.5)
D.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

fn2 = out_dir / 'seasonality.png'
fig.savefig(fn2, dpi=200, bbox_inches='tight')

# -------------------------------------------- figure 3: every source alone ---
n = len(S)
ncol = 4
nrow = int(np.ceil(n / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(16, 2.0 * nrow), sharex=True,
                        layout='constrained')
for ax, c in zip(axs.ravel(), S.index):
    g = S.loc[c, 'group']
    ax.fill_between(Q.index, 0, Q[c], color=GRP_C[g], alpha=0.35)
    ax.plot(Q.index, Q[c], color=GRP_C[g], lw=0.7)
    ax.set_title('%s\n%s, mean %.3g m$^3$ s$^{-1}$%s'
                 % (c[:38], g, S.loc[c, 'mean_m3s'],
                    '' if S.loc[c, 'climatological'] else '  [gauged]'),
                 fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(color='lightgray', ls='--', alpha=0.5)
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%y'))
for ax in axs.ravel()[n:]:
    ax.axis('off')
fig.suptitle('%s -- all %d sources, each on its own scale (%s to %s)'
             % (gtx, n, Q.index[0].date(), Q.index[-1].date()), fontsize=12)

fn3 = out_dir / 'all_sources.png'
fig.savefig(fn3, dpi=160, bbox_inches='tight')

for f in [fn1, fn2, fn3, out_dir / 'daily_flow.csv',
          out_dir / 'totals.csv',
          out_dir / 'source_summary.csv']:
    print('saved %s' % f)
