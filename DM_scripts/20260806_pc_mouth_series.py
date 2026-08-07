"""
A plain stacked time series at the two Penn Cove mouth points, one week long.

Everything else in this set is a composite -- averaged over the tidal cycle,
over spring/neap, over the season. Composites answer "what does it do on
average", and they hide the thing you actually want to see first, which is
what the raw signal LOOKS like. This is that: hourly, unfiltered, one panel
per quantity, all on the same time axis. A week is short enough that the
individual ebbs and floods are still resolvable by eye.

  1  sea surface height at pc_lp, with the spring/neap phase shaded behind it
     and the mean daily range printed on each span, so every feature in the
     panels below can be read against where in the fortnightly cycle it sits
  2  surface salinity   north vs south
  3  bottom salinity    north vs south
  4  surface velocity   north vs south
  5  bottom velocity    north vs south

THE TWO POINTS are faces of pc_lp, not arbitrary lon/lat: p=2 (north,
lat 48.2438) and p=11 (south, lat 48.2275), h = 19.8 and 20.0 m. They are
DEPTH-MATCHED on purpose, so anything that differs between them is lateral
and not a depth artifact.

HOURLY AND SUBTIDAL ARE BOTH DRAWN. The thin line is the raw hourly value;
the thick line is its Godin lowpass. At this mouth the tidal excursion in
salinity is small (~0.05 g/kg surface, ~0.16 g/kg bed) and rides on a much
larger subtidal drift, so plotting only the hourly line makes the tide look
like noise on a ramp. Both together show which of the two is moving. The
lowpass is computed on a window padded 5 days either side and then trimmed,
so the 71-hour filter never eats into the week being shown.

SIGN IS VERIFIED, NOT ASSUMED. q at pc_lp is positive minus-side -> plus-side;
which of those is flood is settled by correlating section-summed q against
d(ssh)/dt over the full two-year record. Velocity is then plotted as u_in,
positive INTO the cove, and the panels are shaded blue above zero (in) and
orange below (out). Note u is the section-NORMAL component only -- the tef2
extraction never carries the along-section component -- so |u| here is a lower
bound on the true speed.

Runs on the mac from the local extractions_avg.

run 20260806_pc_mouth_series.py
run 20260806_pc_mouth_series.py --t0 2025.09.01 --t1 2025.09.30
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('--ds0', default='2024.01.01', help='start of the extraction')
p.add_argument('--ds1', default='2025.12.31', help='end of the extraction')
p.add_argument('--t0', default='2025.09.01', help='first day to plot')
p.add_argument('--t1', default='2025.09.07', help='last day to plot')
p.add_argument('--sect', default='pc_lp')
p.add_argument('--faces', default='2,11')
p.add_argument('--names', default='north,south')
p.add_argument('--tz', default='America/Los_Angeles')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
in_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_pc_mouth_series'
Lfun.make_dir(out_dir)

FACES = [int(v) for v in args.faces.split(',')]
NAMES = [s.strip() for s in args.names.split(',')]
PCOLOR = {'north': '#0072B2', 'south': '#D55E00'}
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
PHCOLOR = {'spring': '#4565e8', 'neap': '#7f7f7f', 'transition': '#9c9c9c'}
PHALPHA = {'spring': 0.10, 'neap': 0.10, 'transition': 0.0}
PAD_D = 5                      # days of padding for the Godin filter

t0 = pd.Timestamp(args.t0.replace('.', '-')).tz_localize(args.tz)
t1 = (pd.Timestamp(args.t1.replace('.', '-')) + pd.Timedelta(days=1)
      ).tz_localize(args.tz)


def godin(a):
    """Godin lowpass of a 1-d array, NaN at the ends."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


# ------------------------------------------------------------------- load ---
# hourly_flux carries ssh and qnet for the whole record; the flood sign is
# settled on the whole record, not on the one month being drawn
fn_flux = tef2 / ('hourly_flux_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag))
d = xr.open_dataset(fn_flux)
ssh_all = d.ssh.sel(sect=args.sect).to_pandas()
qnet_all = d.qnet.sel(sect=args.sect).to_pandas()
d.close()
r_check = np.corrcoef(qnet_all.values, np.gradient(ssh_all.values))[0, 1]
flood_sign = -1.0 if r_check < 0 else 1.0
print('flood check at %s: corr(qnet, d(ssh)/dt) = %+.2f  ->  flood is q %s 0, '
      'u_in = %+.0f * u' % (args.sect, r_check, '<' if flood_sign < 0 else '>',
                            flood_sign))

# face latitudes live in structure_*.nc; the extraction has no lon/lat
fn_str = tef2 / ('structure_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag))
LATF = None
if fn_str.is_file():
    dstr = xr.open_dataset(fn_str)
    LATF = dstr['%s_lat' % args.sect].values
    dstr.close()

# the extraction is 450 MB, so slice the padded window before pulling values
ds = xr.open_dataset(in_dir / (args.sect + '.nc'))
tt = pd.to_datetime(ds.time.values).tz_localize('UTC').tz_convert(args.tz)
keep = (tt >= t0 - pd.Timedelta(days=PAD_D)) & (tt < t1 + pd.Timedelta(days=PAD_D))
if keep.sum() == 0:
    raise SystemExit('no data in %s to %s' % (t0, t1))
sub = ds.isel(time=np.where(keep)[0], p=FACES)
salt = sub.salt.values                                  # (time, z, npt)
q = sub.q.values
DZ = sub.DZ.values
dd = sub.dd.values
h = sub.h.values
zeta = sub.zeta.values
ds.close()

area = DZ * dd[np.newaxis, np.newaxis, :]
with np.errstate(divide='ignore', invalid='ignore'):
    u_in = flood_sign * np.where(area > 0, q / area, np.nan)   # + is INTO cove

idx = tt[keep].tz_localize(None)                        # naive local, for mpl
show = (idx >= t0.tz_localize(None)) & (idx < t1.tz_localize(None))

print('\n%s faces used (z index 0 = bed, -1 = surface):' % args.sect)
for j, (nm, k) in enumerate(zip(NAMES, FACES)):
    print('  %-6s p=%2d  lat %s  h %5.1f m  dd %6.1f m'
          % (nm, k, '%.4f' % LATF[k] if LATF is not None else 'n/a',
             h[j], dd[j]))
print('window %s to %s local, %d hourly steps drawn (%d loaded with padding)'
      % (idx[show][0], idx[show][-1], show.sum(), len(idx)))

# ---- per point: hourly and Godin-lowpassed, top and bottom, salt and u_in ---
S = {}
for j, nm in enumerate(NAMES):
    S[nm] = {'s_top': salt[:, -1, j], 's_bot': salt[:, 0, j],
             'u_top': u_in[:, -1, j], 'u_bot': u_in[:, 0, j]}
    for vn in list(S[nm]):
        S[nm][vn + '_lp'] = godin(S[nm][vn])

ssh = pd.Series(np.nanmean(zeta, axis=1), index=idx)    # mean of the two faces

# -------------------------------------------------------- spring vs neap ---
fn_phase = Ldir['LOo'] / 'DM_outs' / '20260806_tidal_phase' / 'phase_daily.csv'
if fn_phase.is_file():
    P = pd.read_csv(fn_phase, index_col='date_local', parse_dates=True)
    print('spring/neap from %s' % fn_phase.name)
else:
    # fall back to terciles of the daily range of the ssh actually loaded
    rng = ssh.resample('D').max() - ssh.resample('D').min()
    qq = np.nanpercentile(rng, [33, 67])
    P = pd.DataFrame({'range_m': rng,
                      'model_phase': np.where(rng >= qq[1], 'spring',
                                              np.where(rng <= qq[0], 'neap',
                                                       'transition'))})
    print('no phase_daily.csv -- spring/neap from terciles of the daily range')
Pw = P.loc[(P.index >= t0.tz_localize(None).normalize())
           & (P.index <= t1.tz_localize(None))]

# the moon events that DEFINE the fortnightly cycle, so spring/neap is not
# just an assertion in a shaded box
EVENTS = []
fn_ev = Ldir['LOo'] / 'DM_outs' / '20260806_tidal_phase' / 'phase_events.csv'
if fn_ev.is_file():
    E = pd.read_csv(fn_ev, parse_dates=['time_local'])
    E = E[(E.cycle == 'synodic') | (E.event.str.contains('perigee|apogee'))]
    E = E[(E.time_local >= idx[show][0]) & (E.time_local <= idx[show][-1])]
    EVENTS = list(zip(E.time_local, E.event))
    print('events in window: %s' % (', '.join('%s %s' % (t.strftime('%b %d %H:%M'), e)
                                              for t, e in EVENTS) or 'none'))

# consecutive days of the same phase, collapsed into spans to shade
spans = []
if len(Pw):
    ph = Pw.model_phase.values
    cut = np.where(ph[1:] != ph[:-1])[0] + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(ph)]):
        spans.append((Pw.index[a], Pw.index[b - 1] + pd.Timedelta(days=1),
                      ph[a], Pw.range_m.values[a:b].mean()))

# ------------------------------------------------------------------- plot ---
PANELS = [('range', 'daily tidal\nrange', 'm'),
          ('ssh', 'sea surface height', 'm'),
          ('s_top', 'surface salinity', 'g kg$^{-1}$'),
          ('s_bot', 'bottom salinity', 'g kg$^{-1}$'),
          ('u_top', 'surface velocity', 'm s$^{-1}$'),
          ('u_bot', 'bottom velocity', 'm s$^{-1}$')]

fig, axs = plt.subplots(len(PANELS), 1, figsize=(14, 13.6), sharex=True,
                        layout='constrained',
                        height_ratios=[0.42] + [1] * (len(PANELS) - 1))

for ax, (vn, lab, unit) in zip(axs, PANELS):
    # spring/neap behind everything
    for a, b, phase, rng in spans:
        if PHALPHA.get(phase, 0) > 0:
            ax.axvspan(a, b, color=PHCOLOR[phase], alpha=PHALPHA[phase], lw=0,
                       zorder=0)
    for te, elab in EVENTS:
        ax.axvline(te, color='0.35', lw=0.9, ls=':', zorder=1)

    if vn == 'range':
        # The spring/neap CATEGORY is cut on thresholds set over the whole
        # record, so a window can sit entirely inside one category while its
        # fortnightly envelope is still moving. September 2025 is exactly
        # that -- labelled 'transition' from the 4th on, while the range
        # swings 2.5 to 3.6 m. The range itself never goes flat, so it, and
        # not the label, is what to read the fortnight off.
        rr = np.r_[Pw.range_m.values, Pw.range_m.values[-1]]
        te = list(Pw.index) + [Pw.index[-1] + pd.Timedelta(days=1)]
        ax.step(te, rr, where='post', color='#404040', lw=1.8)
        ax.set_ylim(0, np.nanmax(rr) * 1.55)
        y0, y1 = ax.get_ylim()
        for a, b, phase, rng in spans:
            # bottom-left of the span: the event names live along the top and
            # a centred phase label collides with them on a long window
            aa = max(a, idx[show][0])
            ax.text(aa, y0, ' %s' % phase, ha='left', va='bottom',
                    fontsize=8.5, color=PHCOLOR.get(phase, '0.4'),
                    fontweight='bold')
        for te_, elab in EVENTS:
            ax.text(te_, y1, ' ' + elab, rotation=90, ha='left', va='top',
                    fontsize=7.5, color='0.35')

    elif vn == 'ssh':
        ax.plot(idx[show], ssh.values[show], '-', lw=1.0, color='k')
        ax.axhline(0, color='0.5', lw=0.8)
    else:
        for nm in NAMES:
            ax.plot(idx[show], S[nm][vn][show], '-', lw=0.6, alpha=0.45,
                    color=PCOLOR[nm])
            ax.plot(idx[show], S[nm][vn + '_lp'][show], '-', lw=2.2,
                    color=PCOLOR[nm], label='%s (thick = Godin lowpass)' % nm)
        if vn.startswith('u_'):
            ax.axhline(0, color='0.4', lw=0.9)
            y0, y1 = ax.get_ylim()
            ax.fill_between(idx[show], 0, y1, color='#4565e8', alpha=0.045,
                            lw=0, zorder=0)
            ax.fill_between(idx[show], y0, 0, color='#D55E00', alpha=0.045,
                            lw=0, zorder=0)
            ax.set_ylim(y0, y1)
            ax.text(0.004, 0.94, 'into the cove', transform=ax.transAxes,
                    fontsize=8, color='#4565e8', va='top')
            ax.text(0.004, 0.06, 'out of the cove', transform=ax.transAxes,
                    fontsize=8, color='#D55E00', va='bottom')
        ax.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.85)

    ax.grid(**GRID)
    ax.set_ylabel('%s\n(%s)' % (lab, unit))
    ax.set_xlim(idx[show][0], idx[show][-1])

ndays = (idx[show][-1] - idx[show][0]).total_seconds() / 86400
axs[-1].xaxis.set_major_locator(DayLocator(interval=max(1, int(ndays // 15))))
axs[-1].xaxis.set_major_formatter(DateFormatter('%b %d'))
if ndays <= 14:
    # a week is short enough to mark the quarter-days; the semidiurnal tide
    # walks about 50 min per day and the minor ticks make that visible
    axs[-1].xaxis.set_minor_locator(HourLocator(byhour=[6, 12, 18]))
    for ax in axs:
        ax.grid(which='minor', axis='x', color='lightgray', linestyle=':',
                alpha=0.5)
axs[-1].set_xlabel('local time (%s)' % args.tz)
plt.setp(axs[-1].get_xticklabels(), rotation=45, ha='right')

lat_txt = ('north p=%d lat %.4f, south p=%d lat %.4f'
           % (FACES[0], LATF[FACES[0]], FACES[1], LATF[FACES[1]])
           if LATF is not None else 'faces %s' % FACES)
fig.suptitle('%s -- Penn Cove mouth (%s), %s to %s\n'
             '%s;  h = %.1f and %.1f m (depth-matched).  '
             'thin = hourly, thick = Godin lowpass.  '
             'velocity is the section-normal component, positive into the cove.'
             % (args.gtx, args.sect, args.t0, args.t1, lat_txt, h[0], h[1]),
             fontsize=12)

fn = out_dir / ('mouth_series_%s_%s.png' % (args.t0, args.t1))
fig.savefig(fn, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn)

# -------------------------------------------------------------- the numbers ---
out = pd.DataFrame(index=idx[show])
out.index.name = 'time_local'
out['ssh'] = ssh.values[show]
for nm in NAMES:
    for vn in ['s_top', 's_bot', 'u_top', 'u_bot']:
        out['%s_%s' % (nm, vn)] = S[nm][vn][show]
        out['%s_%s_lp' % (nm, vn)] = S[nm][vn + '_lp'][show]
out['model_phase'] = P.model_phase.reindex(out.index.normalize()).values \
    if 'model_phase' in P else ''
fn_csv = out_dir / ('mouth_series_%s_%s.csv' % (args.t0, args.t1))
out.to_csv(fn_csv, float_format='%.5f')
print('saved %s' % fn_csv)

rows = []
for nm in NAMES:
    for vn in ['s_top', 's_bot', 'u_top', 'u_bot']:
        v = S[nm][vn][show]
        td = v - S[nm][vn + '_lp'][show]        # tidal band
        rows.append(dict(point=nm, var=vn, mean=np.nanmean(v),
                         min=np.nanmin(v), max=np.nanmax(v),
                         range=np.nanmax(v) - np.nanmin(v),
                         tidal_rms=np.sqrt(np.nanmean(td ** 2)),
                         subtidal_range=(np.nanmax(S[nm][vn + '_lp'][show])
                                         - np.nanmin(S[nm][vn + '_lp'][show]))))
R = pd.DataFrame(rows)
print('\n%s to %s, hourly:' % (args.t0, args.t1))
print(R.round(4).to_string(index=False))
R.to_csv(out_dir / ('mouth_series_stats_%s_%s.csv' % (args.t0, args.t1)),
         index=False, float_format='%.5f')
