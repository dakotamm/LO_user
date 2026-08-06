"""
Tidal-phase calendar for wb1_t0_xn11abbur00.

Two independent views of the same thing, so they can check each other:

  MODEL   hourly ssh from the wb1_pc1 tef2 extraction. Daily range
          (max - min) is the day's full tidal excursion; its ~14.8 d modulation
          is the spring-neap cycle as the model actually ran it.

  MOON    ephem gives the astronomical drivers:
            syzygy (new/full)      -> spring, 14.77 d synodic beat  (M2-S2)
            quadrature (quarters)  -> neap
            perigee / apogee       -> 27.55 d anomalistic beat      (M2-N2)
            max |declination|      -> tropic tides, 13.66 d         (K1-O1)
            zero declination       -> equatorial tides

The declination cycle is NOT decoration here. Puget Sound is mixed and
strongly diurnal (K1+O1 rival M2), so the biggest ranges track lunar
declination at least as much as they track syzygy. A "spring tide" that
lands on an equatorial moon can be weaker than a tropic neap. The daily
table carries all three phases so any day can be labelled honestly.

Perigean springs (syzygy within ~2 d of perigee) are flagged -- those are
the strongest forcing of the run.

Everything is binned and reported in Pacific local time (TZ below), which is
what NOAA quotes tides in and what field days are planned in. The raw model
ssh is UTC and is converted on load; a time_utc column is kept alongside.

Two consequences of local binning, both handled:
  - a local day is 23 or 25 h at the DST switches, so a day is defined as
    the fixed 24 h from local midnight rather than by calendar resampling.
  - the first and last local days of the record are partial and are dropped.
Set TZ = 'Etc/GMT+8' for fixed PST if you would rather keep 24 h days
year-round.

run 20260806_tidal_phase_calendar.py
"""
import numpy as np
import pandas as pd
import ephem
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pathlib import Path
from scipy.signal import find_peaks

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
gtx = 'wb1_t0_xn11abbur00'
coll = 'wb1_pc1'
SECT = 'pc_lp'          # Penn Cove mouth; sections differ by < 1 cm
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

TZ = 'America/Los_Angeles'   # 'Etc/GMT+8' for fixed PST, no DST
PERIGEAN_WINDOW = 2.0   # days between syzygy and perigee to call it perigean
KM_PER_AU = 149597870.7

# ------------------------------------------------------------------ model ---
fn = sorted((Ldir['LOo'] / 'extract' / gtx / 'tef2').glob(
    'hourly_flux_*_%s.nc' % coll))[0]
d = xr.open_dataset(fn)
ssh = d.ssh.sel(sect=SECT).to_pandas()
d.close()
ssh.index = ssh.index.tz_localize('UTC').tz_convert(TZ)
print('model ssh from %s' % fn.name)
print('  section %s, %s to %s (%s)'
      % (SECT, ssh.index[0], ssh.index[-1], TZ))

t0, t1 = ssh.index[0], ssh.index[-1]

# Full tidal excursion, one value per LOCAL day (not a NOAA datum).
# A day is the fixed 24 h starting at local midnight, NOT resample('D').
# Resampling on a local calendar makes the DST days 23 and 25 h long, and a
# 23 h window can clip one of the day's extremes -- which would put a fake
# notch in the spring-neap envelope twice a year. A fixed 24 h window keeps
# every value comparable and leaves no gaps for find_peaks to trip on.
midn = pd.date_range(t0.ceil('D'), t1.floor('D'), freq='D', tz=TZ)
v = ssh.values
pos = ssh.index.searchsorted(midn)
ok = (pos + 24) <= len(v)
midn, pos = midn[ok], pos[ok]
w = np.lib.stride_tricks.sliding_window_view(v, 24)[pos]
rng = pd.Series(w.max(axis=1) - w.min(axis=1), index=midn, name='range_m')
print('  %d local days, %s to %s' % (len(rng), rng.index[0].date(),
                                     rng.index[-1].date()))
# 3-day smooth only to make the 14.8 d cycle's extrema unambiguous
rng_s = rng.rolling(3, center=True, min_periods=1).mean()

# spring = local max of the envelope, neap = local min. distance=10 d keeps
# find_peaks from splitting one cycle in two when the two beats interfere.
i_sp, _ = find_peaks(rng_s.values, distance=10, prominence=0.15)
i_np, _ = find_peaks(-rng_s.values, distance=10, prominence=0.15)
model_spring = rng.index[i_sp]
model_neap = rng.index[i_np]
print('  model: %d spring maxima, %d neap minima' % (len(i_sp), len(i_np)))

# ------------------------------------------------------------------- moon ---
# ephem knows only UTC, so cross the boundary in exactly two places: _eph()
# going in, _loc() coming out. Never hand ephem a local timestamp.
def _eph(t):
    """Local (or naive-UTC) timestamp -> ephem.Date."""
    t = pd.Timestamp(t)
    return ephem.Date(t.tz_convert('UTC').tz_localize(None)
                      if t.tzinfo is not None else t)


def _loc(t):
    """Naive UTC datetime -> tz-aware local Timestamp."""
    return pd.Timestamp(t).tz_localize('UTC').tz_convert(TZ)


def _events(fn_next, start, stop):
    """All times of an ephem next_* event in [start, stop), local."""
    out, t, stop_e = [], _eph(start), _eph(stop)
    while True:
        t = fn_next(t)
        if t > stop_e:
            break
        out.append(_loc(t.datetime()))
        t = ephem.Date(t + 0.5)   # step past the one just found
    return out


def _extrema(f, start, stop, step_h=1.0, want='min'):
    """Times of local extrema of f(t), refined by parabola on the sample grid."""
    tt = pd.date_range(start, stop, freq='%dmin' % int(step_h * 60))
    y = np.array([f(_eph(t)) for t in tt])
    s = 1.0 if want == 'min' else -1.0
    k, _ = find_peaks(-s * y)
    out = []
    for i in k:
        # vertex of the parabola through (i-1, i, i+1), in sample units
        a, b, c = y[i - 1], y[i], y[i + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c)
        out.append((tt[i] + pd.Timedelta(hours=step_h * shift),
                    b - 0.125 * (a - c) * shift))
    return out


def _moon_dist(t):
    m = ephem.Moon(t)
    return m.earth_distance


def _moon_dec(t):
    m = ephem.Moon(t)
    return float(m.dec)


pad0, pad1 = t0 - pd.Timedelta('35D'), t1 + pd.Timedelta('35D')

ev = []
for name, f in (('new moon', ephem.next_new_moon),
                ('full moon', ephem.next_full_moon),
                ('first quarter', ephem.next_first_quarter_moon),
                ('last quarter', ephem.next_last_quarter_moon)):
    for t in _events(f, pad0, pad1):
        ev.append(dict(time=t, event=name,
                       cycle='synodic',
                       phase='spring' if 'moon' in name else 'neap'))

for t, dist in _extrema(_moon_dist, pad0, pad1, want='min'):
    ev.append(dict(time=t, event='perigee', cycle='anomalistic', phase='',
                   value=dist * KM_PER_AU))
for t, dist in _extrema(_moon_dist, pad0, pad1, want='max'):
    ev.append(dict(time=t, event='apogee', cycle='anomalistic', phase='',
                   value=dist * KM_PER_AU))

# declination: extrema are the tropic tides, zero crossings the equatorial
for t, dec in _extrema(_moon_dec, pad0, pad1, step_h=1.0, want='max'):
    ev.append(dict(time=t, event='max N declination', cycle='tropical',
                   phase='tropic', value=np.degrees(dec)))
for t, dec in _extrema(_moon_dec, pad0, pad1, step_h=1.0, want='min'):
    ev.append(dict(time=t, event='max S declination', cycle='tropical',
                   phase='tropic', value=np.degrees(dec)))

tt = pd.date_range(pad0, pad1, freq='h')
dec = np.array([_moon_dec(_eph(t)) for t in tt])
zc = np.where(np.diff(np.sign(dec)) != 0)[0]
for i in zc:
    frac = -dec[i] / (dec[i + 1] - dec[i])
    ev.append(dict(time=tt[i] + frac * pd.Timedelta('1h'),
                   event='equatorial (dec=0)', cycle='tropical',
                   phase='equatorial', value=0.0))

for t in model_spring:
    ev.append(dict(time=t + pd.Timedelta('12h'), event='model range maximum',
                   cycle='model', phase='spring', value=rng[t]))
for t in model_neap:
    ev.append(dict(time=t + pd.Timedelta('12h'), event='model range minimum',
                   cycle='model', phase='neap', value=rng[t]))

E = pd.DataFrame(ev).sort_values('time').reset_index(drop=True)

# ------------------------------------------------- perigean / apogean flag ---
peri = E.loc[E.event == 'perigee', 'time'].values
apo = E.loc[E.event == 'apogee', 'time'].values
syz = E.event.isin(['new moon', 'full moon'])

def _utc64(t):
    """tz-aware Timestamp -> naive-UTC datetime64, so it can meet .values."""
    return pd.Timestamp(t).tz_convert('UTC').tz_localize(None).to_datetime64()


def _nearest_days(t, arr):
    return np.min(np.abs((arr - _utc64(t)) / np.timedelta64(1, 'D')))

E['days_to_perigee'] = [_nearest_days(t, peri) for t in E.time]
E['days_to_apogee'] = [_nearest_days(t, apo) for t in E.time]
E['modifier'] = ''
E.loc[syz & (E.days_to_perigee <= PERIGEAN_WINDOW), 'modifier'] = 'perigean'
E.loc[syz & (E.days_to_apogee <= PERIGEAN_WINDOW), 'modifier'] = 'apogean'

# trim to the run, add UTC alongside and the observed range on that local day
E = E[(E.time >= t0) & (E.time <= t1)].reset_index(drop=True)
E['date_local'] = E.time.dt.strftime('%Y-%m-%d')
E['time_local'] = E.time.dt.tz_localize(None)
E['time_utc'] = E.time.dt.tz_convert('UTC').dt.tz_localize(None)
E['range_m'] = [rng.get(pd.Timestamp(t).normalize(), np.nan) for t in E.time]
E_out = E[['date_local', 'time_local', 'time_utc', 'event', 'cycle', 'phase',
           'modifier', 'value', 'days_to_perigee', 'days_to_apogee', 'range_m']]

# ------------------------------------------------------- daily phase table ---
D = pd.DataFrame(index=rng.index)
D['range_m'] = rng
D['range_smooth_m'] = rng_s

def _signed_days(idx, arr):
    """Days from each day (noon) to the nearest event in arr, signed."""
    x = (idx + pd.Timedelta('12h')).values[:, None] - arr[None, :]
    x = x / np.timedelta64(1, 'D')
    k = np.argmin(np.abs(x), axis=1)
    return x[np.arange(len(idx)), k]

allE = pd.DataFrame(ev)
sy = allE.loc[allE.event.isin(['new moon', 'full moon']), 'time'].sort_values().values
qu = allE.loc[allE.event.isin(['first quarter', 'last quarter']), 'time'].sort_values().values
tr = allE.loc[allE.event.isin(['max N declination', 'max S declination']), 'time'].sort_values().values

D['days_from_syzygy'] = _signed_days(D.index, sy)
D['days_from_quadrature'] = _signed_days(D.index, qu)
D['days_from_perigee'] = _signed_days(D.index, np.sort(peri))
D['days_from_tropic'] = _signed_days(D.index, tr)

noon = D.index + pd.Timedelta('12h')
D['moon_dist_km'] = [_moon_dist(_eph(t)) * KM_PER_AU for t in noon]
D['moon_dec_deg'] = [np.degrees(_moon_dec(_eph(t))) for t in noon]
# 0 at syzygy/spring, 1 at quadrature/neap; 0 at perigee, 1 at apogee
D['synodic_phase'] = np.abs(D.days_from_syzygy) / (29.5306 / 4)
D['anomalistic_phase'] = np.abs(D.days_from_perigee) / (27.5546 / 2)
D['tropic_phase'] = np.abs(D.days_from_tropic) / (27.3216 / 4)

# plain-language label from the model envelope: terciles of the smoothed range
q = D.range_smooth_m.quantile([1 / 3, 2 / 3]).values
D['model_phase'] = np.where(D.range_smooth_m >= q[1], 'spring',
                            np.where(D.range_smooth_m <= q[0], 'neap', 'transition'))

E_out.to_csv(out_dir / '20260806_tidal_phase_events.csv', index=False)
D_out = D.copy()
D_out.index = D_out.index.strftime('%Y-%m-%d')
D_out.to_csv(out_dir / '20260806_tidal_phase_daily.csv',
             index_label='date_local', float_format='%.4f')

# ------------------------------------------------------------------ report ---
print('\nspring tides, strongest first (syzygy events, by model daily range):')
sp = E[E.event.isin(['new moon', 'full moon'])].sort_values('range_m', ascending=False)
print('%-19s %-11s %-9s %8s' % ('date (local)', 'event', 'modifier', 'range m'))
for _, r in sp.head(12).iterrows():
    print('%-19s %-11s %-9s %8.2f' % (r.time.strftime('%Y-%m-%d %H:%M'),
                                      r.event, r.modifier, r.range_m))
print('\nweakest neaps (quadrature events):')
nq = E[E.event.isin(['first quarter', 'last quarter'])].sort_values('range_m')
for _, r in nq.head(6).iterrows():
    print('%-19s %-11s %-9s %8.2f' % (r.time.strftime('%Y-%m-%d %H:%M'),
                                      r.event, r.modifier, r.range_m))
n_per = (E.modifier == 'perigean').sum()
print('\n%d perigean springs (syzygy within %.0f d of perigee) in the run'
      % (n_per, PERIGEAN_WINDOW))
print('model daily range: min %.2f m, max %.2f m, mean %.2f m'
      % (rng.min(), rng.max(), rng.mean()))
print('correlation of daily range with:  |syzygy| %+.2f   |perigee| %+.2f   '
      '|tropic| %+.2f'
      % (D.range_m.corr(D.synodic_phase), D.range_m.corr(D.anomalistic_phase),
         D.range_m.corr(D.tropic_phase)))

# ------------------------------------------------------------------ figure ---
plt.close('all')
fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                        layout='constrained',
                        gridspec_kw=dict(height_ratios=[2, 1, 1]))

A = axs[0]
A.plot(rng.index, rng.values, color='0.75', lw=1, label='daily range')
A.plot(rng_s.index, rng_s.values, color='k', lw=1.6, label='3-day smooth')
for _, r in E[E.event.isin(['new moon', 'full moon'])].iterrows():
    c = '#CC0000' if r.modifier == 'perigean' else '#0072B2'
    A.axvline(r.time, color=c, lw=1.2 if r.modifier == 'perigean' else 0.6,
              alpha=0.8 if r.modifier == 'perigean' else 0.35, zorder=0)
A.plot([], [], color='#0072B2', lw=1, label='syzygy (spring)')
A.plot([], [], color='#CC0000', lw=1.4, label='perigean spring')
A.plot(model_spring, rng[model_spring], 'v', color='#009E73', ms=6,
       label='model range maximum')
A.plot(model_neap, rng[model_neap], '^', color='#D55E00', ms=6,
       label='model range minimum')
A.set_ylabel('daily tidal range, max - min (m)')
A.set_title('%s -- tidal phase at %s, hourly model ssh (Pacific local, %s)'
            % (gtx, SECT, TZ))
A.grid(color='lightgray', ls='--', alpha=0.5)
A.legend(fontsize=8, ncol=3, loc='upper left')

B = axs[1]
B.plot(D.index, D.moon_dist_km / 1000, color='#7B3294', lw=1.2)
for _, r in E[E.event == 'perigee'].iterrows():
    B.plot(r.time, r.value / 1000, 'o', color='#CC0000', ms=4)
for _, r in E[E.event == 'apogee'].iterrows():
    B.plot(r.time, r.value / 1000, 'o', color='#0072B2', ms=4)
B.set_ylabel('moon distance\n(1000 km)')
B.grid(color='lightgray', ls='--', alpha=0.5)
B.text(0.005, 0.06, 'red = perigee, blue = apogee (27.55 d)',
       transform=B.transAxes, fontsize=8, color='0.3')

C = axs[2]
C.plot(D.index, D.moon_dec_deg, color='#1B7837', lw=1.2)
C.axhline(0, color='0.6', lw=0.8)
C.set_ylabel('moon declination\n(deg)')
C.set_xlabel('date (Pacific local)')
C.grid(color='lightgray', ls='--', alpha=0.5)
C.text(0.005, 0.06, 'extrema = tropic tides, zero = equatorial (13.66 d)',
       transform=C.transAxes, fontsize=8, color='0.3')
C.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

fn_out = out_dir / '20260806_tidal_phase_calendar.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn_out)
print('saved %s' % (out_dir / '20260806_tidal_phase_events.csv'))
print('saved %s' % (out_dir / '20260806_tidal_phase_daily.csv'))
