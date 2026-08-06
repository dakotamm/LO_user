"""
Hypoxic area and volume in Penn Cove, wb1_t0_xn11abbur00.

Runs at home off the pickle 20260806_hypoxia_reduce.py makes on apogee, so the
series is the model's own 3D oxygen field rather than anything reconstructed
from a map.

The reducer keeps three measures of the same low-oxygen water because they say
different things, and the figures here are built to keep them apart:

  VOLUME    m3 below the threshold. Scales with the deficit; the only one that
            needs the 3D field.
  A_bot     area with the BOTTOM cell below the threshold -- what the benthos
            sees, and what a bottom mooring or a grab survey would report.
  A_col     area with ANY level below it -- the footprint, i.e. the
            water-column minimum criterion of 20260609_hypoxic_days_map.py.

  V / A_col is the mean thickness of the low-oxygen layer where it exists, and
  A_col - A_bot is the area where that layer has lifted off the bed. A cove can
  lose hypoxic volume either by ventilating it or by thinning it over the same
  footprint, and only the pair of series distinguishes those.

Everything is shown against the REGION's own volume and area as well as in
absolute units. Penn Cove holds ~0.2 km3 in 12.4 km2, four thousandths of the
domain's volume, so an absolute-only axis makes the cove look like nothing is
happening in it no matter what is happening in it.

Thresholds are nested (0.5 < 2 < 3 < 5 mg/L) and drawn nested, darkest lowest.
2 mg/L is the conventional hypoxia line; 5 mg/L is the low-DO line already used
in wb1_penncove_region.py. Whether the cove ever crosses 2 is a result, not an
assumption, so all four are carried and the script says which ones are live.

Maps follow the standing wb1 region-plot rule: rectangular window around the
whole polygon plus a margin, but only cells inside wb.p are drawn.

run 20260806_hypoxia_timeseries.py
run 20260806_hypoxia_timeseries.py -reg skagit_delta
run 20260806_hypoxia_timeseries.py -gtx wb1_t1_xn11abbur00 -src lowpass
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.dates import DateFormatter
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-src', '--source', default='lowpass', type=str)
parser.add_argument('-reg', '--region', default='pc', type=str)
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gtx, SOURCE, REG = args.gtagex, args.source, args.region
TZ = 'America/Los_Angeles'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_hypoxia'
Lfun.make_dir(out_dir)
CACHE = out_dir / ('hypoxia_%s_%s_%s_%s.p' % (SOURCE, gtx, args.ds0, args.ds1))

CB = dict(blue='#0072B2', orange='#D55E00', green='#009E73', red='#CC0000',
          purple='#7B3294', dgreen='#1B7837', yellow='#E69F00', pink='#CC79A7')
# nested thresholds, darkest = lowest oxygen
THC = {'5': '#fdd0a2', '3': '#fc9272', '2': '#de2d26', '0.5': '#67000d'}
RC = dict(pc='tab:red', skagit_delta='tab:blue', wb_north='tab:purple',
          wb='tab:green', domain='0.35')
MN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
PAD_CELLS = 10
CLIP_POLY = 'wb'

# ------------------------------------------------------------------- load ---
if not CACHE.is_file():
    raise SystemExit(
        'no reduced hypoxia file at\n  %s\n'
        'Run 20260806_hypoxia_reduce.py on apogee and copy it here:\n'
        '  scp apogee:LO_output/DM_outs/20260806_hypoxia/%s \\\n'
        '      ~/LO_output/DM_outs/20260806_hypoxia/' % (CACHE, CACHE.name))

C = pd.read_pickle(CACHE)
T, MF, YF, RUN = C['T'], C['MF'], C['YF'], C['RUN']
lon, lat, water, h = C['lon'], C['lat'], C['water'], C['h']
regions, THRESH = C['regions'], C['thresh']
fs = C['fstride']
TKEY = ['%g' % t for t in THRESH]
FKEY = ['%g' % t for t in C['field_thresh']]

for nm in regions:
    T[nm].index = pd.DatetimeIndex(T[nm].index).tz_localize('UTC').tz_convert(TZ)
D = T[REG]
t = D.index

print('reduced hypoxia from %s' % CACHE.name)
print('  made %s from %s output of %s'
      % (C['info']['made'], C['info']['source'], C['info']['gtx']))
print('%s: %d steps, %s to %s (%.1f days), %d regions'
      % (gtx, len(D), t[0], t[-1], (t[-1] - t[0]).total_seconds() / 86400,
         len(regions)))
if C['bad']:
    print('  ** %d files failed in the reduction, e.g. %s'
          % (len(C['bad']), C['bad'][:2]))
print('%s: %d cells, %.1f km2, %.3f km3, mean depth %.1f m'
      % (REG, C['ncell'][REG], D.A_tot.mean() / 1e6, D.V_tot.mean() / 1e9,
         D.V_tot.mean() / D.A_tot.mean()))

# ---------------------------------------------------------- which are live ---
# a threshold with 0% or 100% occupancy carries no series; say so once here
# rather than plotting four flat lines and letting the reader work it out
print('\n--- which thresholds are live, by region ---')
print('%-14s %8s %8s   %s' % ('region', 'min DO', 'mean bot',
                              ' '.join('%12s' % ('<%s mg/L' % k)
                                       for k in TKEY)))
for nm in regions:
    d = T[nm]
    occ = ' '.join('%11.1f%%' % (100 * (d['V_' + k] > 0).mean()) for k in TKEY)
    print('%-14s %8.2f %8.2f   %s'
          % (nm, d.do_min.min(), d.do_bot_mean.mean(), occ))
print('  (%% of time steps with any water below the threshold anywhere in the')
print('   region -- 0%% or 100%% means that threshold has no series to plot)')

LIVE = [k for k in TKEY if 0 < (D['V_' + k] > 0).mean() < 1]
print('  live in %s: %s' % (REG, ', '.join(LIVE) if LIVE else 'none'))
# what the figures lead with: the lowest live threshold, and 5 mg/L for context
MAIN = LIVE[0] if LIVE else TKEY[1]

# ----------------------------------------------------- the series, in words ---
def spells(b):
    """Lengths of runs of True, in time steps."""
    x = np.asarray(b, dtype=int)
    if x.sum() == 0:
        return np.array([], dtype=int)
    dd = np.diff(np.concatenate([[0], x, [0]]))
    return np.flatnonzero(dd == -1) - np.flatnonzero(dd == 1)


dt_days = float(np.median(np.diff(t.values)) / np.timedelta64(1, 'D'))
print('\n--- %s: how much, how long, and when ---' % REG)
print('%-6s %10s %9s %8s %9s %10s %10s'
      % ('thresh', 'peak 1e6m3', '% of vol', 'days on', 'longest', 'peak day',
         'mean thick'))
for k in TKEY:
    v, ac = D['V_' + k], D['A_col_' + k]
    on = v > 0
    if not on.any():
        print('%-6s %10s' % ('<' + k, 'never'))
        continue
    sp = spells(on.values)
    thick = (v[on] / ac[on]).mean()
    print('%-6s %10.2f %8.1f%% %8.0f %7.0f d %10s %8.1f m'
          % ('<' + k, v.max() / 1e6, 100 * (v / D.V_tot).max(),
             on.sum() * dt_days, sp.max() * dt_days,
             v.idxmax().strftime('%Y-%m-%d'), thick))
print('  "days on" counts time steps with any water below the threshold;')
print('  "mean thick" is V/A_col over those steps -- how deep the layer is')
print('  where it exists, which is what reconciles the area and volume series')

print('\n--- bottom water vs the whole column ---')
for k in FKEY:
    ab, ac = D['A_bot_' + k], D['A_col_' + k]
    if ac.max() == 0:
        continue
    on = ac > 0
    print('  <%s mg/L: footprint peaks at %.1f km2 (%.0f%% of %s), bottom-hypoxic'
          ' area peaks at %.1f km2; the layer is off the bed over %.0f%% of the'
          ' footprint on average'
          % (k, ac.max() / 1e6, 100 * (ac / D.A_tot).max(), REG,
             ab.max() / 1e6, 100 * (1 - (ab[on] / ac[on]).mean())))

# seasonal timing: the day by which half the year's hypoxic volume-days are in
yrs = [y for y in sorted(set(t.year)) if (t.year == y).sum() > 30]
if yrs:
    print('\n--- when in the year the low oxygen sits ---')
    for k in [MAIN, '5']:
        if D['V_' + k].max() <= 0:
            continue
        for yr in yrs:
            q = D['V_' + k][t.year == yr]
            if q.sum() <= 0:
                continue
            c = q.cumsum() / q.sum()
            half = c.index[np.searchsorted(c.values, 0.5)]
            on = q > 0
            print('  %d  <%s mg/L: first %s, last %s, half the volume-days in '
                  'by %s, %.0f%% of it in Jul-Oct'
                  % (yr, k, q[on].index[0].strftime('%b %d'),
                     q[on].index[-1].strftime('%b %d'),
                     half.strftime('%b %d'),
                     100 * q[q.index.month.isin([7, 8, 9, 10])].sum() / q.sum()))

# --------------------------------------------------- stratification & wind ---
if 'dsalt' in D.columns:
    print('\n--- does it track stratification ---')
    v = D['V_' + MAIN]
    for lag in [0, 1, 2, 3, 5, 7]:
        r = v.corr(D.dsalt.shift(lag))
        print('  V(<%s) vs bottom-minus-surface salinity lagged %d step%s: '
              'r = %+.3f' % (MAIN, lag, ' ' if lag == 1 else 's', r))
    print('  positive r = more stratified, more low-oxygen water. A peak at a')
    print('  lag of a few days is the cove integrating, not responding.')

# the wind pickle is the other forcing reduction from the same day; if it is
# here, ask the obvious question rather than making anyone open both files
wfn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
       / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
if wfn.is_file() and 'dsalt' in D.columns:
    try:
        Wc = pd.read_pickle(wfn)
        W = Wc['W']
        W.index = W.index.tz_localize('UTC').tz_convert(TZ)
        u3 = W['ustar3_' + (REG if 'ustar3_' + REG in W.columns
                            else 'domain')].resample('1D').mean()
        v = D['V_' + MAIN].resample('1D').mean()
        j = pd.concat([v.rename('V'), u3.rename('u3')], axis=1).dropna()
        dv = j.V.diff()
        print('\n--- does the wind knock it down ---')
        print('  daily u*^3 vs same-day CHANGE in hypoxic volume: r = %+.3f'
              % dv.corr(j.u3))
        print('  daily u*^3 vs hypoxic volume itself:              r = %+.3f'
              % j.V.corr(j.u3))
        hi = j.u3 > j.u3.quantile(0.9)
        print('  on the windiest 10%% of days the volume changes by %+.2f x1e6 '
              'm3, against %+.2f on the rest'
              % (dv[hi].mean() / 1e6, dv[~hi].mean() / 1e6))
    except Exception as ex:
        print('\n  wind cross-check skipped: %s' % ex)

# ------------------------------------------------------------------- save ---
D.round(4).to_csv(out_dir / ('hypoxia_series_%s_%s.csv' % (gtx, REG)),
                  index_label='time_local')
allr = pd.concat({nm: T[nm] for nm in regions}, axis=1)
allr.round(4).to_csv(out_dir / ('hypoxia_series_%s_all_regions.csv' % gtx),
                     index_label='time_local')
mo = D.resample('MS').mean()
mo.round(4).to_csv(out_dir / ('hypoxia_monthly_%s_%s.csv' % (gtx, REG)),
                   index_label='month')

# --------------------------------------------------------------- geometry ---
sect_dir = Ldir['LOo'] / 'section_lines'
XY = np.column_stack([lon.ravel(), lat.ravel()])
POLY = {nm: pd.read_pickle(sect_dir / (nm + '.p'))
        for nm in ['pc', 'skagit_delta', 'wb_north', 'wb']}
in_wb = MplPath(np.column_stack([POLY[CLIP_POLY].x.values,
                                 POLY[CLIP_POLY].y.values])
                ).contains_points(XY).reshape(lon.shape)
dx = float(np.diff(lon[0, :]).mean())
dy = float(np.diff(lat[:, 0]).mean())


def region_aa(nm, pad=PAD_CELLS):
    """Rectangular window: the whole polygon plus a margin of grid cells."""
    p = POLY[nm]
    return [p.x.min() - pad * dx, p.x.max() + pad * dx,
            p.y.min() - pad * dy, p.y.max() + pad * dy]


def show(field, sub=1, clip=True):
    """Field ready to draw: land and everything outside wb.p masked out."""
    m = (water & in_wb) if clip else water
    return np.where(m[::sub, ::sub], field, np.nan)


# ------------------------------------------------ figure 1: the series ---
plt.close('all')
fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True,
                        layout='constrained',
                        gridspec_kw=dict(height_ratios=[2, 1.6, 1.6, 1.4]))

A = axs[0]
for k in sorted(TKEY, key=float, reverse=True):     # nested: widest first
    if D['V_' + k].max() <= 0:
        continue
    A.fill_between(t, 0, D['V_' + k] / 1e6, color=THC[k], lw=0,
                   label='< %s mg L$^{-1}$' % k)
A.set_ylabel('volume below threshold\n(10$^6$ m$^3$)')
A2 = A.twinx()
A2.set_ylim(np.array(A.get_ylim()) * 1e6 / D.V_tot.mean() * 100)
A2.set_ylabel('% of %s volume' % REG)
A.set_title('%s -- low-oxygen volume in %s (%s, %s to %s)'
            % (gtx, REG, SOURCE, args.ds0, args.ds1))
A.grid(color='lightgray', ls='--', alpha=0.5)
A.legend(fontsize=8, ncol=4, loc='upper left')

B = axs[1]
for k in FKEY:
    if D['A_col_' + k].max() <= 0:
        continue
    B.fill_between(t, 0, D['A_col_' + k] / 1e6, color=THC[k], alpha=0.55, lw=0,
                   label='footprint, any level < %s' % k)
    B.plot(t, D['A_bot_' + k] / 1e6, color=THC[k], lw=1.3,
           label='bottom cell < %s' % k)
B.set_ylabel('area below threshold\n(km$^2$)')
B2 = B.twinx()
B2.set_ylim(np.array(B.get_ylim()) * 1e6 / D.A_tot.mean() * 100)
B2.set_ylabel('% of %s area' % REG)
B.grid(color='lightgray', ls='--', alpha=0.5)
B.legend(fontsize=8, ncol=2, loc='upper left')
B.text(0.005, 0.05, 'gap between line and fill = low-oxygen layer off the bed',
       transform=B.transAxes, fontsize=8, color='0.3')

Cx = axs[2]
Cx.plot(t, D.do_surf_mean, color=CB['blue'], lw=1.0, label='surface mean')
Cx.plot(t, D.do_vol_mean, color='k', lw=1.4, label='volume mean')
Cx.plot(t, D.do_bot_mean, color=CB['orange'], lw=1.4, label='bottom mean')
Cx.plot(t, D.do_min, color=CB['red'], lw=0.9, ls='--',
        label='minimum anywhere')
for k in ['5', '2']:
    Cx.axhline(float(k), color=THC[k], lw=1.2, ls=':')
    Cx.text(t[0], float(k), ' %s mg/L' % k, fontsize=7, va='bottom',
            color=THC[k])
Cx.set_ylabel('dissolved oxygen\n(mg L$^{-1}$)')
Cx.grid(color='lightgray', ls='--', alpha=0.5)
Cx.legend(fontsize=8, ncol=4, loc='upper right')

Dx = axs[3]
for k in FKEY:
    ac = D['A_col_' + k]
    th = (D['V_' + k] / ac.where(ac > 0)).astype(float)
    if th.notna().sum() == 0:
        continue
    Dx.plot(t, th, color=THC[k], lw=1.3, label='mean thickness < %s mg/L' % k)
Dx.axhline(D.V_tot.mean() / D.A_tot.mean(), color='0.5', lw=1, ls='--',
           label='mean depth of %s' % REG)
Dx.set_ylabel('low-oxygen layer\nthickness (m)')
Dx.set_xlabel('date (%s)' % TZ)
Dx.grid(color='lightgray', ls='--', alpha=0.5)
Dx.legend(fontsize=8, ncol=3, loc='upper left')
Dx.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
if 'dsalt' in D.columns:
    D2 = Dx.twinx()
    D2.plot(t, D.dsalt, color=CB['purple'], lw=0.9, alpha=0.8)
    D2.set_ylabel('bottom - surface salinity (g kg$^{-1}$)', color=CB['purple'])
    D2.tick_params(axis='y', labelcolor=CB['purple'], labelsize=8)

fn1 = out_dir / ('hypoxia_series_%s_%s.png' % (gtx, REG))
fig.savefig(fn1, dpi=200, bbox_inches='tight')

# --------------------------------------------- figure 2: region comparison ---
fig, axs = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')

for ax, k in [(axs[0, 0], '5'), (axs[0, 1], MAIN)]:
    for nm in regions:
        d = T[nm]
        if d['V_' + k].max() <= 0:
            continue
        ax.plot(d.index, 100 * d['V_' + k] / d.V_tot, color=RC.get(nm, '0.5'),
                lw=1.2 if nm == REG else 0.9,
                alpha=1.0 if nm == REG else 0.75, label=nm)
    ax.set_ylabel('% of region volume < %s mg L$^{-1}$' % k)
    ax.set_title('how much of each region is below %s mg/L' % k, fontsize=10)
    ax.grid(color='lightgray', ls='--', alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

Cx = axs[1, 0]
# monthly mean, one bar group per year, so a two-year run shows whether the
# season repeats -- the same question the wind figures ask of the forcing
w = 0.8 / max(len(yrs), 1)
for i, yr in enumerate(yrs):
    q = D[t.year == yr]
    m = (100 * q['V_' + MAIN] / q.V_tot).groupby(q.index.month).mean()
    Cx.bar(m.index + (i - (len(yrs) - 1) / 2) * w, m.values, width=w,
           color=[CB['blue'], CB['orange']][i % 2], label=str(yr))
Cx.set_xticks(range(1, 13))
Cx.set_xticklabels(MN, fontsize=8)
Cx.set_ylabel('%% of %s volume < %s mg L$^{-1}$' % (REG, MAIN))
Cx.set_title('seasonal cycle in %s' % REG, fontsize=10)
Cx.grid(color='lightgray', ls='--', alpha=0.5, axis='y')
Cx.legend(fontsize=8)

Dx = axs[1, 1]
for nm in regions:
    d = T[nm]
    v = d['V_' + MAIN]
    if v.sum() <= 0:
        continue
    c = 100 * v.cumsum() / v.sum()
    Dx.plot(d.index.dayofyear + (d.index.year - min(yrs)) * 365.25 if len(yrs) > 1
            else d.index.dayofyear, c.values, color=RC.get(nm, '0.5'), lw=1.4,
            label=nm)
Dx.axhline(50, color='0.6', lw=0.8)
Dx.set_xlabel('day of the run')
Dx.set_ylabel('cumulative %% of low-oxygen volume-days (< %s mg/L)' % MAIN)
Dx.set_title('when the low-oxygen water is delivered', fontsize=10)
Dx.grid(color='lightgray', ls='--', alpha=0.5)
Dx.legend(fontsize=8)

fn2 = out_dir / ('hypoxia_regions_%s.png' % gtx)
fig.savefig(fn2, dpi=200, bbox_inches='tight')

# --------------------------------------------------------- figure 3: maps ---
aa_reg, aa_wb = region_aa(REG), region_aa('wb', pad=4)
kf = MAIN if MAIN in FKEY else FKEY[0]


def draw(ax, field, aa, cmap, vmin, vmax, label, sub=1, under=None):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('lightgray')
    if under is not None:
        cm.set_under(under)
    p = ax.pcolormesh(lon[::sub, ::sub], lat[::sub, ::sub], field, cmap=cm,
                      shading='nearest', vmin=vmin, vmax=vmax)
    fig.colorbar(p, ax=ax, orientation='horizontal', pad=0.04, label=label)
    pfun.add_coast(ax, color='gray', linewidth=0.5)
    ax.axis(aa)
    pfun.dar(ax)
    for nm, col in [('pc', CB['red']), ('skagit_delta', CB['yellow'])]:
        ax.plot(POLY[nm].x, POLY[nm].y, color=col, lw=1.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=7, labelrotation=45)


fig, axs = plt.subplots(1, 4, figsize=(18, 7), layout='constrained')

draw(axs[0], show(RUN['bot_do']), aa_reg, 'viridis',
     np.nanmin(show(RUN['bot_do'])[np.isfinite(show(RUN['bot_do']))]) if
     np.isfinite(show(RUN['bot_do'])).any() else 0, 10,
     'run-mean bottom DO (mg L$^{-1}$)')
axs[0].set_title('mean bottom oxygen', fontsize=10)

draw(axs[1], show(RUN['col_do_min']), aa_reg, 'magma', 0, 8,
     'run-minimum column DO (mg L$^{-1}$)')
axs[1].set_title('lowest oxygen ever reached\nanywhere in the column',
                 fontsize=10)

draw(axs[2], show(100 * RUN['fhyp_col_' + kf]), aa_reg, 'YlOrRd', 0.5, None,
     '%% of time any level < %s mg/L' % kf, under='white')
axs[2].set_title('how often the column\ngoes below %s mg/L' % kf, fontsize=10)

draw(axs[3], show(100 * RUN['fhyp_col_' + kf]), aa_wb, 'YlOrRd', 0.5, None,
     '%% of time any level < %s mg/L' % kf, under='white')
axs[3].set_title('the same, whole Whidbey Basin', fontsize=10)

fig.suptitle('%s -- where the low oxygen is (%d %s fields, %s to %s). '
             'Rectangular window, wb-clipped content; white = never below '
             'threshold.' % (gtx, RUN['n'], SOURCE, args.ds0, args.ds1),
             fontsize=11)
fn3 = out_dir / ('hypoxia_maps_%s_%s.png' % (gtx, REG))
fig.savefig(fn3, dpi=180, bbox_inches='tight')

# ---------------------------------------------- figure 4: monthly bottom DO ---
# the series says when, this says where within the cove, month by month
keys = sorted(MF)
ncol = 6
nrow = int(np.ceil(len(keys) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.6 * nrow),
                        layout='constrained', squeeze=False)
vals = np.concatenate([show(MF[k]['bot_do'], sub=fs).ravel() for k in keys])
vmin, vmax = np.nanpercentile(vals, [1, 99])
p = None
for i, key in enumerate(keys):
    ax = axs[i // ncol, i % ncol]
    cm = plt.get_cmap('viridis').copy()
    cm.set_bad('lightgray')
    p = ax.pcolormesh(lon[::fs, ::fs], lat[::fs, ::fs],
                      show(MF[key]['bot_do'], sub=fs), cmap=cm,
                      shading='nearest', vmin=vmin, vmax=vmax)
    ax.contour(lon[::fs, ::fs], lat[::fs, ::fs],
               np.nan_to_num(show(MF[key]['fhyp_bot_' + kf], sub=fs)),
               [0.5], colors='w', linewidths=1.2)
    ax.plot(POLY['pc'].x, POLY['pc'].y, color=CB['red'], lw=1.0)
    ax.axis(aa_reg)
    pfun.dar(ax)
    ax.set_title('%s %d  (n = %d)' % (MN[key[1] - 1], key[0], MF[key]['n']),
                 fontsize=9)
    ax.tick_params(labelsize=6, labelrotation=45)
for i in range(len(keys), nrow * ncol):
    axs[i // ncol, i % ncol].axis('off')
if p is not None:
    fig.colorbar(p, ax=axs, shrink=0.6, label='monthly mean bottom DO (mg L$^{-1}$)')
fig.suptitle('%s -- monthly mean bottom oxygen around %s. White contour: the '
             'bed is below %s mg/L more than half the month.' % (gtx, REG, kf),
             fontsize=11)
fn4 = out_dir / ('hypoxia_monthly_maps_%s_%s.png' % (gtx, REG))
fig.savefig(fn4, dpi=150, bbox_inches='tight')

for f in [fn1, fn2, fn3, fn4,
          out_dir / ('hypoxia_series_%s_%s.csv' % (gtx, REG)),
          out_dir / ('hypoxia_series_%s_all_regions.csv' % gtx),
          out_dir / ('hypoxia_monthly_%s_%s.csv' % (gtx, REG))]:
    print('saved %s' % f)
