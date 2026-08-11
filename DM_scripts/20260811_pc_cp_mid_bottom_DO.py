"""
Bottom dissolved oxygen along the Penn Cove axis, wb1_t0_xn11abbur00.

Three moorings, job 'pc4' in LO_user/extract/moor/job_lists.py, laid out from
the head of the cove to the passage outside it:

  cp_mid  midpoint of the Coupeville line (the pc_cp TEF section of the wb1_pc1
          collection). pc_cp is the u-face at i=52/53, wet over j=214..223, so
          the middle rho cell is (j=219, i=53), h = 20.6 m -- within 1 m of the
          deepest cell on the line, so the series is not on a shoal edge.

  lp_mid  midpoint of the Long Point line (pc_lp), i.e. the cove mouth. pc_lp is
          the u-face at i=67/68, wet over j=217..228, so the middle rho cell is
          (j=222, i=68), h = 27.0 m, on the flat channel floor. Same grid cell
          as pc3's M_entrance.

  M5      Saratoga Passage, OUTSIDE the cove in the gyre east of the mouth.
          Same point as pc0's M5, h = 37.4 m, so this series is comparable to
          anything already extracted at that station.

The transect is the point: the cove's bottom water either ventilates from
Saratoga Passage or it doesn't, and an interior series alone cannot tell a
drawdown inside the cove from a low-oxygen source arriving at the mouth. lp_mid
sits between the two and says which end the signal comes from. Read the
differences, not just the levels.

The bottom value is s_rho index 0, the deepest rho level, NOT a fixed depth. Its
height above the bed moves with zeta and with the terrain-following grid, so the
script carries that height and reports its range for each station. That matters
here because the three stations are 21, 27 and 37 m deep: the same 30 sigma
layers give a thicker bottom cell at M5 than at cp_mid, so the three "bottom"
values are not sampled at the same height off the bed, and the printed summary
says what each one is.

SOURCE. The extraction is run with -lt lowpass, so each sample is one
Godin-filtered field per day, stamped 12:00 UTC -- the tide is already removed
and there is no tidal spread to plot. If you re-run it with -lt hourly the
script notices the 1 h spacing on its own and adds the raw series behind its own
Godin average; nothing below needs changing.

Thresholds are nested (2 < 3 < 5 mg/L) and shaded darkest lowest, matching
20260806_hypoxia_timeseries.py. 2 mg/L is the conventional hypoxia line.
Whether any station crosses any of them is a result, so all three are drawn and
the printed summary says which ones are live. Note that a daily lowpassed series
cannot show a brief tidal excursion below a threshold, so threshold counts here
are a floor, not a tally.

The extraction covers the whole run (2024.01.01-2025.12.31); -year windows the
figure and the printed stats without re-extracting. The extraction command is in
the "no mooring file" message below.

run 20260811_pc_cp_mid_bottom_DO.py                 # 2025, all three stations
run 20260811_pc_cp_mid_bottom_DO.py -year all       # both years
run 20260811_pc_cp_mid_bottom_DO.py -sn cp_mid,M5   # a subset
run 20260811_pc_cp_mid_bottom_DO.py -gtx wb1_t1_xn11abbur00
"""
import argparse
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-job', default='pc4', type=str)
p.add_argument('-sn', default='cp_mid,lp_mid,M5', type=str,
               help='comma-separated station names, head of cove outward')
p.add_argument('-0', '--ds0', default='2024.01.01', type=str,
               help='start of the EXTRACTION (part of the filename)')
p.add_argument('-1', '--ds1', default='2025.12.31', type=str,
               help='end of the EXTRACTION (part of the filename)')
p.add_argument('-lt', '--list_type', default='lowpass', type=str,
               help='only used to print the right extraction command')
p.add_argument('-year', default='2025', type=str,
               help="calendar year to plot, or 'all' for the whole record")
p.add_argument('--tz', default='America/Los_Angeles',
               help='timezone for the plot axis; the CSVs stay UTC')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gtagex.split('_')[0])
moor_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'moor'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_cp_mid_bottom_DO'
Lfun.make_dir(out_dir)

SN_LIST = [s.strip() for s in args.sn.split(',') if s.strip()]

O2_MMOL_TO_MGL = 32.0 / 1000.0
CB = dict(blue='#0072B2', orange='#D55E00', green='#009E73', red='#CC0000',
          purple='#7B3294', grey='#7f7f7f')
# section colors match 20260807_pc_alongchannel_wind.py: pc_cp orange, pc_lp blue
SC = {'cp_mid': CB['orange'], 'lp_mid': CB['blue'], 'M5': CB['purple']}
LBL = {'cp_mid': 'cp_mid (Coupeville line, inner cove)',
       'lp_mid': 'lp_mid (Long Point line, cove mouth)',
       'M5': 'M5 (Saratoga Passage, outside)'}
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
# nested thresholds, mg/L -> shading color, darkest lowest
THRESH = [(2.0, '#7f2704'), (3.0, '#d94801'), (5.0, '#fdae6b')]

EXTRACT_MSG = (
    'Extract on apogee, where the run lives, using the LO driver:\n'
    '  cd ~/LO/extract/moor\n'
    '  python multi_mooring_driver.py -gtx %s -ro 2 \\\n'
    '      -0 %s -1 %s -lt %s -job %s -get_all True -Nproc 100 \\\n'
    '      > %s.log &\n'
    % (args.gtagex, args.ds0, args.ds1, args.list_type, args.job, args.job))


def read_station(sn):
    """Bottom/surface DO and height-above-bed for one station, indexed in UTC."""
    # multi_mooring_driver.py files results under moor/<job>/; a bare
    # extract_moor.py call leaves them in moor/ -- accept either.
    stem = '%s_%s_%s.nc' % (sn, args.ds0, args.ds1)
    cands = [moor_dir / args.job / stem, moor_dir / stem]
    fn = next((c for c in cands if c.is_file()), None)
    if fn is None:
        print('No mooring file for %s. Looked for:' % sn)
        for c in cands:
            print('  ' + str(c))
        print('\n' + EXTRACT_MSG)
        sys.exit(1)
    print('reading ' + str(fn))

    ds = xr.open_dataset(fn, decode_times=True)
    if 'oxygen' not in ds.data_vars:
        print('*** no oxygen in %s -- the extraction needs -get_bio True '
              '(or -get_all True)' % fn.name)
        sys.exit(1)

    t_utc = pd.to_datetime(ds.ocean_time.values).tz_localize('UTC')
    df = pd.DataFrame(
        {'do_bot_mgL': ds.oxygen.values[:, 0] * O2_MMOL_TO_MGL,  # 0 = bottom
         'do_surf_mgL': ds.oxygen.values[:, -1] * O2_MMOL_TO_MGL,
         # height of the bottom rho level above the bed
         'hab_m': ds.z_rho.values[:, 0] - ds.z_w.values[:, 0]},
        index=t_utc)
    df.index.name = 'time_utc'

    # Only hourly input needs (or tolerates) a Godin filter here; a lowpass
    # extraction is already Godin-averaged, one field per day. Filtering is done
    # on the FULL record so windowing to a year does not blank its first and
    # last ~35 h.
    dt_h = np.median(np.diff(df.index.values)) / np.timedelta64(1, 'h')
    hourly = bool(np.isclose(dt_h, 1.0, atol=0.01))
    if hourly:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df['do_bot_lp'] = zfun.lowpass(df.do_bot_mgL.to_numpy(dtype=float),
                                           f='godin')
    else:
        df['do_bot_lp'] = df.do_bot_mgL

    meta = dict(lon=float(ds.lon_rho), lat=float(ds.lat_rho), h=float(ds.h),
                hourly=hourly, dt_h=dt_h)
    ds.close()
    return df, meta


# ---------------------------------------------------------------------------
# read, window, save
# ---------------------------------------------------------------------------
D, M = {}, {}
for sn in SN_LIST:
    D[sn], M[sn] = read_station(sn)
    D[sn].to_csv(out_dir / ('bottom_DO_%s_%s_%s_%s.csv'
                            % (sn, args.gtagex, args.ds0, args.ds1)))

ANY_HOURLY = any(M[sn]['hourly'] for sn in SN_LIST)
print('\nsampling interval: ' + ', '.join(
    '%s %.1f h' % (sn, M[sn]['dt_h']) for sn in SN_LIST))

if args.year.lower() == 'all':
    tag, span_lbl = 'all', '2024-2025'
else:
    yr = int(args.year)
    tag, span_lbl = str(yr), str(yr)
    for sn in SN_LIST:
        D[sn] = D[sn][D[sn].index.tz_convert(args.tz).year == yr]
        if len(D[sn]) == 0:
            print('*** %s has no samples in %d -- the extraction covers %s to %s'
                  % (sn, yr, args.ds0, args.ds1))
            sys.exit(1)

# ---------------------------------------------------------------------------
# what the series actually say
# ---------------------------------------------------------------------------
for sn in SN_LIST:
    df, meta = D[sn], M[sn]
    lo = df.do_bot_mgL
    print('\n--- %s at (%.6f, %.6f), h = %.1f m'
          % (sn, meta['lon'], meta['lat'], meta['h']))
    print('bottom rho level sits %.2f-%.2f m above the bed (mean %.2f)'
          % (df.hab_m.min(), df.hab_m.max(), df.hab_m.mean()))
    print('%d samples, %s to %s' % (len(df), df.index[0].date(),
                                    df.index[-1].date()))
    print('bottom DO: min %.2f, mean %.2f, max %.2f mg/L'
          % (lo.min(), lo.mean(), lo.max()))
    print('minimum on %s' % lo.idxmin().tz_convert(args.tz))
    for thr, _ in THRESH:
        n = int((lo < thr).sum())
        if n:
            first, last = lo[lo < thr].index[[0, -1]]
            print('  < %.0f mg/L: %d samples (%.1f%%), %s to %s'
                  % (thr, n, 100 * n / len(lo),
                     first.tz_convert(args.tz).date(),
                     last.tz_convert(args.tz).date()))
        else:
            print('  < %.0f mg/L: never' % thr)

# every station against the outermost one, which is the ventilation end member
if len(SN_LIST) > 1:
    ref = SN_LIST[-1]
    print('\nbottom DO relative to %s (lowpassed):' % ref)
    for sn in SN_LIST[:-1]:
        j = pd.concat([D[sn].do_bot_lp.rename('a'),
                       D[ref].do_bot_lp.rename('b')], axis=1, join='inner').dropna()
        if len(j):
            d = j.a - j.b
            print('  %-7s - %-7s: mean %+.2f, min %+.2f, max %+.2f mg/L'
                  % (sn, ref, d.mean(), d.min(), d.max()))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                         gridspec_kw=dict(height_ratios=[2, 1]))

ax = axes[0]
all_bot = pd.concat([D[sn].do_bot_mgL for sn in SN_LIST])
ymin = min(1.5, float(all_bot.min()) - 0.3)
for thr, col in THRESH:
    ax.axhspan(ymin, thr, color=col, alpha=0.16, lw=0, zorder=0)
    ax.axhline(thr, color=col, lw=0.8, ls='--', zorder=1)
    ax.text(0.997, thr, ' %.0f mg/L' % thr, transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', fontsize=8, color=col)
for sn in SN_LIST:
    df, c = D[sn], SC.get(sn, CB['grey'])
    tl = df.index.tz_convert(args.tz)
    if M[sn]['hourly']:
        # raw behind its own Godin average
        ax.plot(tl, df.do_bot_mgL, color=c, lw=0.4, alpha=0.35, zorder=2)
    ax.plot(tl, df.do_bot_lp, color=c, lw=1.8, zorder=3,
            label='%s, h = %.1f m' % (LBL.get(sn, sn), M[sn]['h']))
    lo = df.do_bot_mgL
    ax.plot(lo.idxmin().tz_convert(args.tz), lo.min(), 'v', color=c, ms=7,
            mec='k', mew=0.5, zorder=4)
ax.set_ylim(ymin, float(all_bot.max()) + 0.5)
ax.set_ylabel('bottom DO [mg L$^{-1}$]')
ax.set_title('Bottom dissolved oxygen along the Penn Cove axis, %s\n'
             '%s (%s; triangle = minimum)'
             % (span_lbl, args.gtagex,
                'thin = hourly, thick = Godin' if ANY_HOURLY
                else 'daily Godin-lowpassed output'))
ax.legend(loc='lower left', frameon=False, fontsize=9)
ax.grid(**GRID)

# Context: surface-minus-bottom says whether a low bottom value is
# stratification or a whole-column drawdown.
ax = axes[1]
for sn in SN_LIST:
    df = D[sn]
    ax.plot(df.index.tz_convert(args.tz), df.do_surf_mgL - df.do_bot_mgL,
            color=SC.get(sn, CB['grey']), lw=0.9, label=sn)
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('surface - bottom\nDO [mg L$^{-1}$]')
ax.set_xlabel('date (%s)' % args.tz)
ax.grid(**GRID)

tl = D[SN_LIST[0]].index.tz_convert(args.tz)
span_yr = (tl[-1] - tl[0]).days / 365.25
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1 if span_yr <= 1.2 else 2))
ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%b' if span_yr <= 1.2 else '%b\n%Y'))
ax.set_xlim(tl[0], tl[-1])

fig.tight_layout()
fn_out = out_dir / ('bottom_DO_%s_%s.png' % (args.gtagex, tag))
fig.savefig(fn_out, dpi=170, bbox_inches='tight')
print('\nwrote ' + str(fn_out))
