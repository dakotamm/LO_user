"""
How big is the velocity at the Penn Cove sections, and how does it vary from
tidal to seasonal timescales?

Velocity is not stored anywhere in the tef2 chain, but extractions_avg holds
the volume flux q(time, z, p) together with the face width dd(p) and the cell
thickness DZ(time, z, p), so the section-normal velocity is just

    u(time, z, p) = q / (dd * DZ)                               [m s-1]

for every face, every sigma level, every hour of the two-year record. No new
extraction is needed. NOTE u is the NORMAL component only -- the along-section
component never enters the tef2 extraction, so |u| here is a lower bound on
the true current speed.

THE CENTRAL POINT: TWO DIFFERENT VELOCITIES
There are two ways to say "the velocity at a section" and at Penn Cove they
differ by a factor of three:

    ubar = qnet / area        the section average. This is the tidal-prism
                              velocity, what filling and emptying the cove
                              requires. rms about 0.03 m/s.
    urms = area-weighted rms  the actual local velocity, cell by cell.
           of u               rms about 0.09 m/s, peaks above 1 m/s.

The gap is not noise. The residual flux changes sign ACROSS every one of these
sections (the lateral gyre), so opposing flows inside the same section cancel
in the average. Any argument built on the prism velocity alone understates the
water actually moving past a point by roughly three times, which matters for
anything shear- or stress-dependent.

Both are carried through every figure, and their ratio is reported per section.

Five figures:
  1. magnitude   how big, section by section, and where in the section it lives
  2. ebb/flood   composite over the tidal cycle, split spring vs neap
  3. spring/neap the fortnightly envelope, on 30-day anomalies
  4. seasonal    monthly climatology + annual/semiannual harmonic fit
  5. variance    which band carries the velocity signal, with utide

The two mouth points from the salinity work -- pc_lp p=2 (north) and p=11
(south), depth-matched at h = 19.8 and 20.0 m -- are carried as named points
so the velocity and salinity stories can be read against each other.

Sign is VERIFIED, not assumed: qnet is checked against d(ssh)/dt per section
and u_in is reported positive INTO the cove.

Runs on the mac from the local extractions_avg.
run 20260806_pc_velocity_tides.py
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d
from scipy.stats import t as tdist

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('--ds0', default='2024.01.01')
p.add_argument('--ds1', default='2025.12.31')
p.add_argument('--ref', default='pc_lp', help='section defining tidal phase')
p.add_argument('--tz', default='America/Los_Angeles')
p.add_argument('--no-utide', dest='utide', action='store_false')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
in_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_pc_velocity_tides'
Lfun.make_dir(out_dir)

SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
COLOR = {'pc_cp': '#009E73', 'pc_lj': '#0072B2', 'pc_lp': '#CC79A7'}
# the two depth-matched mouth points, as (section, face, label)
POINTS = [('pc_lp', 2, 'north'), ('pc_lp', 11, 'south')]
PCOLOR = {'north': '#0072B2', 'south': '#D55E00'}
SEASONS = {'Winter (Dec-Mar)': [12, 1, 2, 3],
           'Spring (Apr-Jul)': [4, 5, 6, 7],
           'Low-DO (Aug-Nov)': [8, 9, 10, 11]}
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)

GODIN = zfun.godin_shape()
NPAD = len(GODIN) // 2


def godin_nd(a):
    """Godin lowpass along axis 0, NaN at the ends.

    zfun.lowpass flattens the array before convolving, which lets one column's
    last hours bleed into the next column's first hours. Filtering strictly
    along the time axis keeps every cell independent.
    """
    out = convolve1d(a, GODIN, axis=0, mode='nearest')
    out[:NPAD] = np.nan
    out[-NPAD:] = np.nan
    return out


def savefig(fig, name):
    fn = out_dir / name
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    print('saved %s' % fn)


# ------------------------------------------------------------------- load ---
fn_flux = tef2 / ('hourly_flux_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag))
d = xr.open_dataset(fn_flux)
ssh_ref = d.ssh.sel(sect=args.ref).to_pandas()
QNET = {sn: d.qnet.sel(sect=sn).to_pandas() for sn in SECTS}
d.close()

idx = ssh_ref.index.tz_localize('UTC').tz_convert(args.tz)
ssh_ref.index = idx
print('tidal phase from %s ssh; %d hourly steps, %s to %s'
      % (args.ref, len(idx), idx[0], idx[-1]))

# structure_*.nc carries the face latitudes; extractions_avg does not
fn_str = tef2 / ('structure_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag))
LAT = {}
if fn_str.is_file():
    dstr = xr.open_dataset(fn_str)
    for sn in SECTS:
        LAT[sn] = dstr['%s_lat' % sn].values
    dstr.close()

SEC = {}          # per section: hourly series and per-cell statistics
print('\ncomputing u = q/(dd*DZ) ...')
for sn in SECTS:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    q = ds.q.values
    DZ = ds.DZ.values
    dd = ds.dd.values
    h = ds.h.values
    area = DZ * dd[np.newaxis, np.newaxis, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(area > 0, q / area, np.nan)
    ds.close()

    # flood sign, verified per section rather than assumed
    qn = QNET[sn].values
    r_check = np.corrcoef(qn, np.gradient(ssh_ref.values))[0, 1]
    fs = -1.0 if r_check < 0 else 1.0
    u = fs * u                                   # now positive INTO the cove

    A_tot = np.nansum(area, axis=(1, 2))
    ubar = fs * np.nansum(q, axis=(1, 2)) / A_tot
    urms = np.sqrt(np.nansum(area * u ** 2, axis=(1, 2)) / A_tot)
    umax = np.nanmax(np.abs(u), axis=(1, 2))

    ulp = godin_nd(u)
    utd = u - ulp
    SEC[sn] = dict(
        u=u, utd=utd, area=area, dd=dd, h=h, flood_sign=fs, r_check=r_check,
        A_tot=A_tot,
        ubar=pd.Series(ubar, index=idx),
        urms=pd.Series(urms, index=idx),
        umax=pd.Series(umax, index=idx),
        ubar_td=pd.Series(ubar - zfun.lowpass(ubar, f='godin'), index=idx),
        umean_cell=np.nanmean(u, axis=0),
        utid_cell=np.sqrt(np.nanmean(utd ** 2, axis=0)),
        urms_cell=np.sqrt(np.nanmean(u ** 2, axis=0)))
    print('  %-6s NP=%2d  flood is qnet %s 0 (r=%+.2f)  section-avg rms %.4f  '
          'local rms %.4f  ratio %.2f  max|u| %.3f m/s'
          % (sn, u.shape[2], '<' if fs < 0 else '>', r_check,
             np.sqrt(np.nanmean(ubar ** 2)), np.sqrt(np.nanmean(urms ** 2)),
             np.sqrt(np.nanmean(ubar ** 2)) / np.sqrt(np.nanmean(urms ** 2)),
             np.nanmax(np.abs(u))))

# the two named mouth points, top and bottom
PT = {}
for sn, k, nm in POINTS:
    u = SEC[sn]['u']
    PT[nm] = dict(sect=sn, face=k, h=SEC[sn]['h'][k],
                  lat=LAT.get(sn, np.full(u.shape[2], np.nan))[k],
                  u_top=pd.Series(u[:, -1, k], index=idx),
                  u_bot=pd.Series(u[:, 0, k], index=idx),
                  u_col=pd.Series(np.nanmean(u[:, :, k], axis=1), index=idx))
    for vn in ['u_top', 'u_bot', 'u_col']:
        PT[nm][vn + '_td'] = pd.Series(
            PT[nm][vn].values - zfun.lowpass(PT[nm][vn].values.astype(float),
                                             f='godin'), index=idx)

# --------------------------------------------------------- tidal phase ---
i_hw, _ = find_peaks(ssh_ref.values, distance=8)
t_hw = ssh_ref.index[i_hw]
hw64 = t_hw.tz_convert('UTC').tz_localize(None).values.astype('datetime64[ns]')
all64 = idx.tz_convert('UTC').tz_localize(None).values.astype('datetime64[ns]')
k = np.clip(np.searchsorted(hw64, all64), 1, len(hw64) - 1)
d_prev = (all64 - hw64[k - 1]) / np.timedelta64(1, 'h')
d_next = (all64 - hw64[k]) / np.timedelta64(1, 'h')
hrs_hw = np.where(np.abs(d_prev) <= np.abs(d_next), d_prev, d_next)
print('\n%d high waters (%.2f per day)'
      % (len(t_hw), len(t_hw) / ((idx[-1] - idx[0]).total_seconds() / 86400)))

T = pd.DataFrame(index=idx)
T['hrs_hw'] = hrs_hw
T['month'] = idx.month
for sn in SECTS:
    T['%s_ubar' % sn] = SEC[sn]['ubar']
    T['%s_urms' % sn] = SEC[sn]['urms']
for nm in PT:
    for vn in ['u_top', 'u_bot', 'u_col']:
        T['%s_%s' % (nm, vn)] = PT[nm][vn]

# ------------------------------------------------------------ daily / phase ---
midn = pd.date_range(idx[0].ceil('D'), idx[-1].floor('D'), freq='D', tz=args.tz)
pos = idx.searchsorted(midn)
ok = (pos + 24) <= len(idx)
midn, pos = midn[ok], pos[ok]


def daily(v, how='mean'):
    w = np.lib.stride_tricks.sliding_window_view(np.asarray(v, float), 24)[pos]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if how == 'mean':
            return pd.Series(np.nanmean(w, axis=1), index=midn)
        if how == 'range':
            return pd.Series(np.nanmax(w, axis=1) - np.nanmin(w, axis=1), index=midn)
        if how == 'std':
            return pd.Series(np.nanstd(w, axis=1), index=midn)
        if how == 'absmean':
            return pd.Series(np.nanmean(np.abs(w), axis=1), index=midn)


D = pd.DataFrame(index=midn)
D['tide_range'] = daily(ssh_ref.values, 'range')
for sn in SECTS:
    # the within-day sd of the section-average velocity IS the tidal velocity
    # amplitude that day -- the subtidal part barely moves within 24 h
    D['%s_amp' % sn] = daily(SEC[sn]['ubar'].values, 'std')
    D['%s_urms' % sn] = daily(SEC[sn]['urms'].values, 'mean')
    D['%s_res' % sn] = daily(SEC[sn]['ubar'].values, 'mean')
for nm in PT:
    for vn in ['u_top', 'u_bot']:
        D['%s_%s_amp' % (nm, vn)] = daily(PT[nm][vn].values, 'std')
        D['%s_%s_res' % (nm, vn)] = daily(PT[nm][vn].values, 'mean')

fn_phase = Ldir['LOo'] / 'DM_outs' / '20260806_tidal_phase' / 'phase_daily.csv'
if fn_phase.is_file():
    P = pd.read_csv(fn_phase, parse_dates=['date_local']).set_index('date_local')
    P.index = P.index.tz_localize(args.tz)
    D = D.join(P[['range_smooth_m', 'days_from_syzygy', 'synodic_phase',
                  'model_phase']])
    print('joined spring/neap labels from %s' % fn_phase.name)
else:
    print('no phase_daily.csv -- labelling spring/neap from the daily range')
    rs = D.tide_range.rolling(3, center=True, min_periods=1).mean()
    qq = rs.quantile([1 / 3, 2 / 3]).values
    D['range_smooth_m'] = rs
    D['model_phase'] = np.where(rs >= qq[1], 'spring',
                                np.where(rs <= qq[0], 'neap', 'transition'))
    D['days_from_syzygy'] = np.nan
D['month'] = D.index.month
T['model_phase'] = D.model_phase.reindex(idx.floor('D')).values


def eff_corr(x, y):
    """Pearson r with the p-value corrected for serial correlation.

    Daily values are strongly autocorrelated, so the nominal n would make
    almost anything significant. n_eff uses the lag-1 autocorrelation of both
    series (Bretherton et al. 1999).
    """
    g = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(g) < 10:
        return np.nan, np.nan, np.nan, 0
    r = g.x.corr(g.y)
    slope = np.polyfit(g.x, g.y, 1)[0]
    r1x, r1y = g.x.autocorr(1), g.y.autocorr(1)
    n_eff = float(np.clip(len(g) * (1 - r1x * r1y) / (1 + r1x * r1y), 3, len(g)))
    tstat = r * np.sqrt((n_eff - 2) / max(1e-12, 1 - r ** 2))
    return r, slope, 2 * (1 - tdist.cdf(abs(tstat), n_eff - 2)), n_eff


def seasonal_fit(y):
    """Mean + annual + semiannual least squares."""
    g = y.dropna()
    if len(g) < 60:
        return None
    t = (g.index - g.index[0]).total_seconds().values / 86400.0
    w = 2 * np.pi / 365.25
    X = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t),
                         np.cos(2 * w * t), np.sin(2 * w * t)])
    b, *_ = np.linalg.lstsq(X, g.values, rcond=None)
    fit = pd.Series(X @ b, index=g.index)
    return dict(fit=fit, R2=1 - np.var(g.values - fit.values) / np.var(g.values),
                amp_annual=np.hypot(b[1], b[2]),
                amp_semiannual=np.hypot(b[3], b[4]), mean=b[0])


# =========================================================== 1. MAGNITUDE ===
mag_rows = []
for sn in SECTS:
    S = SEC[sn]
    mag_rows.append(dict(
        section=sn, n_faces=S['u'].shape[2], area_m2=np.nanmean(S['A_tot']),
        sect_avg_rms=np.sqrt(np.nanmean(S['ubar'].values ** 2)),
        local_rms=np.sqrt(np.nanmean(S['urms'].values ** 2)),
        ratio=np.sqrt(np.nanmean(S['ubar'].values ** 2))
        / np.sqrt(np.nanmean(S['urms'].values ** 2)),
        p95_abs=np.nanpercentile(np.abs(S['u']), 95),
        max_abs=np.nanmax(np.abs(S['u'])),
        residual_mean_abs=np.nanmean(np.abs(S['umean_cell'])),
        residual_max_abs=np.nanmax(np.abs(S['umean_cell'])),
        tidal_rms_cell=np.nanmean(S['utid_cell'])))
M = pd.DataFrame(mag_rows)
M.to_csv(out_dir / 'velocity_magnitude.csv', index=False, float_format='%.5f')
print('\nvelocity magnitude by section (m/s):')
print(M.round(4).to_string(index=False))

fig, axs = plt.subplots(3, 3, figsize=(15, 11), layout='constrained')

A = axs[0][0]
x = np.arange(len(SECTS))
A.bar(x - 0.2, M.sect_avg_rms, 0.38, color='#888888',
      label='section average, qnet/area')
A.bar(x + 0.2, M.local_rms, 0.38, color=[COLOR[s] for s in SECTS],
      label='local, area-weighted rms')
for i, r in M.iterrows():
    A.text(i + 0.2, r.local_rms, ' x%.1f' % (1 / r.ratio), ha='center',
           va='bottom', fontsize=9)
A.set_xticks(x)
A.set_xticklabels(SECTS)
A.set_ylabel('velocity (m s$^{-1}$)')
A.set_title('the section average sees a third of the real velocity\n'
            'labels = local rms / section-average rms')
A.grid(**GRID)
A.legend(fontsize=8)

B = axs[0][1]
for sn in SECTS:
    v = np.abs(SEC[sn]['u']).ravel()
    v = v[np.isfinite(v)]
    B.hist(v, bins=np.arange(0, 0.65, 0.01), histtype='step', density=True,
           color=COLOR[sn], lw=1.8, label='%s  p95=%.2f' % (sn, np.percentile(v, 95)))
B.set_xlabel('|u| (m s$^{-1}$)')
B.set_ylabel('density')
B.set_title('distribution of local velocity\nevery cell, every hour')
B.grid(**GRID)
B.legend(fontsize=8)

C = axs[0][2]
for sn in SECTS:
    C.plot(D.index, D['%s_amp' % sn], '-', lw=1.0, color=COLOR[sn], label=sn)
C.set_ylabel('daily tidal amplitude of ubar (m s$^{-1}$)')
C.set_title('section-average tidal velocity over the record')
C.grid(**GRID)
C.legend(fontsize=8)
C.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

# section views: where in the section the velocity actually lives
for j, sn in enumerate(SECTS):
    S = SEC[sn]
    lat = LAT.get(sn)
    dist = ((lat - lat.min()) * 111.0 if lat is not None
            else np.arange(S['u'].shape[2]))
    for row, (fld, lab, cmap) in enumerate(
            [('utid_cell', 'tidal rms |u|', 'viridis'),
             ('umean_cell', 'residual (mean) u, + into cove', 'RdBu_r')]):
        ax = axs[row + 1][j]
        Z = S[fld]
        if fld == 'umean_cell':
            vm = np.nanmax(np.abs(Z))
            im = ax.pcolormesh(dist, np.arange(Z.shape[0]), Z, cmap=cmap,
                               vmin=-vm, vmax=vm, shading='nearest')
        else:
            im = ax.pcolormesh(dist, np.arange(Z.shape[0]), Z, cmap=cmap,
                               shading='nearest')
        plt.colorbar(im, ax=ax, label='m s$^{-1}$')
        ax.set_title('%s -- %s' % (sn, lab), color=COLOR[sn], fontsize=10)
        ax.set_xlabel('distance across section (km)')
        if j == 0:
            ax.set_ylabel('sigma level (0 = bed)')
        # mark the two named mouth points
        for pnm, pd_ in PT.items():
            if pd_['sect'] == sn:
                ax.axvline(dist[pd_['face']], color=PCOLOR[pnm], lw=1.6,
                           ls='--')
                ax.text(dist[pd_['face']], Z.shape[0] * 0.95, pnm,
                        color=PCOLOR[pnm], fontsize=8, ha='center')

fig.suptitle('%s -- Penn Cove section velocity, u = q/(dd*DZ), normal '
             'component only\nresidual reverses ACROSS each section: that is '
             'why the section average cancels' % args.gtx, fontsize=12)
savefig(fig, 'fig1_magnitude.png')
plt.close(fig)

# =========================================================== 2. EBB / FLOOD ===
BINS = np.arange(-6.5, 6.6, 1.0)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
T['hw_bin'] = pd.cut(T.hrs_hw, BINS, labels=CTR)

comp_rows = []
fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), layout='constrained')

for j, sn in enumerate(SECTS):
    ax = axs[0][j]
    for ph, ls in [('spring', '-'), ('neap', '--')]:
        sel = T[T.model_phase == ph]
        if len(sel) == 0:
            continue
        g = sel.groupby('hw_bin', observed=False)['%s_ubar' % sn]
        m, se = g.mean(), g.std() / np.sqrt(g.count())
        ax.plot(CTR, m.values, ls, color=COLOR[sn], lw=2, label=ph)
        ax.fill_between(CTR, m - 2 * se, m + 2 * se, color=COLOR[sn],
                        alpha=0.20, lw=0)
        for c, mm, ss, nn in zip(CTR, m.values, g.std().values, g.count().values):
            comp_rows.append(dict(location=sn, var='ubar', phase=ph,
                                  hrs_from_HW=c, mean=mm, std=ss, n=nn))
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.grid(**GRID)
    ax.set_xlabel('hours from high water')
    pk = ax.get_ylim()
    ax.set_title('%s -- section-average velocity\npeak flood %.3f, peak ebb '
                 '%.3f m s$^{-1}$'
                 % (sn, T.groupby('hw_bin', observed=False)['%s_ubar' % sn]
                    .mean().max(),
                    T.groupby('hw_bin', observed=False)['%s_ubar' % sn]
                    .mean().min()), color=COLOR[sn], fontsize=10)
    if j == 0:
        ax.set_ylabel('u into the cove (m s$^{-1}$)')
        ax.legend(fontsize=8)

for j, (vn, lab) in enumerate([('u_top', 'surface'), ('u_bot', 'bed'),
                               ('u_col', 'column mean')]):
    ax = axs[1][j]
    for nm in PT:
        for ph, ls in [('spring', '-'), ('neap', '--')]:
            sel = T[T.model_phase == ph]
            if len(sel) == 0:
                continue
            g = sel.groupby('hw_bin', observed=False)['%s_%s' % (nm, vn)]
            m, se = g.mean(), g.std() / np.sqrt(g.count())
            ax.plot(CTR, m.values, ls, color=PCOLOR[nm], lw=2,
                    label='%s %s' % (nm, ph))
            ax.fill_between(CTR, m - 2 * se, m + 2 * se, color=PCOLOR[nm],
                            alpha=0.18, lw=0)
            for c, mm, ss, nn in zip(CTR, m.values, g.std().values,
                                     g.count().values):
                comp_rows.append(dict(location=nm, var=vn, phase=ph,
                                      hrs_from_HW=c, mean=mm, std=ss, n=nn))
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.grid(**GRID)
    ax.set_xlabel('hours from high water')
    ax.set_title('mouth points -- %s' % lab, fontsize=10)
    if j == 0:
        ax.set_ylabel('u into the cove (m s$^{-1}$)')
        ax.legend(fontsize=7, ncol=2)

fig.suptitle('%s -- velocity through the tidal cycle. Positive = INTO the '
             'cove (sign verified against d(ssh)/dt).\nband = +/- 2 standard '
             'errors' % args.gtx, fontsize=12)
savefig(fig, 'fig2_tidal_cycle.png')
plt.close(fig)
pd.DataFrame(comp_rows).to_csv(out_dir / 'tidal_composite_velocity.csv',
                               index=False, float_format='%.5f')

# ========================================================== 3. SPRING/NEAP ===
# On 30-day anomalies: the fortnightly signal is a few tenths of the seasonal
# range, so a raw scatter against tidal range is mostly season. 30 days is
# about two spring-neap cycles, so the rolling mean cannot absorb what is
# being tested.
ROLL = 30


def anom(s):
    return s - s.rolling(ROLL, center=True, min_periods=ROLL // 2).mean()


D['tide_range_anom'] = anom(D.tide_range)
sn_rows = []
fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')

A = axs[0][0]
for sn in SECTS:
    D['%s_amp_anom' % sn] = anom(D['%s_amp' % sn])
    x, y = D.tide_range_anom, D['%s_amp_anom' % sn]
    r, slope, pv, ne = eff_corr(x, y)
    A.plot(x, y, '.', ms=3, alpha=0.35, color=COLOR[sn],
           label='%s: r=%+.2f, %.3f per m (p=%.1e, n_eff=%.0f)'
                 % (sn, r, slope, pv, ne))
    g = pd.DataFrame({'x': x, 'y': y}).dropna()
    xx = np.linspace(g.x.min(), g.x.max(), 10)
    A.plot(xx, np.polyval(np.polyfit(g.x, g.y, 1), xx), '-', color=COLOR[sn], lw=2)
    sn_rows.append(dict(location=sn, var='tidal amplitude of ubar',
                        detrended='30 d anomaly', r=r, slope=slope, p=pv,
                        n_eff=ne))
    r0, s0, p0, n0 = eff_corr(D.tide_range, D['%s_amp' % sn])
    sn_rows.append(dict(location=sn, var='tidal amplitude of ubar',
                        detrended='none', r=r0, slope=s0, p=p0, n_eff=n0))
A.axhline(0, color='0.5', lw=0.8)
A.axvline(0, color='0.5', lw=0.8)
A.set_xlabel('daily tidal range anomaly (m)')
A.set_ylabel('tidal velocity amplitude anomaly (m s$^{-1}$)')
A.set_title('does a bigger tide mean faster water?\n(both as 30-day anomalies)')
A.grid(**GRID)
A.legend(fontsize=7)

B = axs[0][1]
if D.days_from_syzygy.notna().any():
    sb = np.arange(-7.5, 7.6, 1.0)
    sc = 0.5 * (sb[:-1] + sb[1:])
    cut = pd.cut(D.days_from_syzygy, sb, labels=sc)
    for sn in SECTS:
        g = D.groupby(cut, observed=False)['%s_amp_anom' % sn]
        se = g.std() / np.sqrt(g.count())
        B.errorbar(sc, g.mean().values, yerr=2 * se.values, fmt='o-',
                   color=COLOR[sn], ms=4, lw=1.6, capsize=2, label=sn)
    B.axvline(0, color='0.5', lw=0.8)
    B.axhline(0, color='0.5', lw=0.8)
    B.set_xlabel('days from syzygy (0 = new/full moon = spring)')
    B.set_ylabel('tidal amplitude anomaly (m s$^{-1}$)')
    B.set_title('composite over the 14.8 d synodic cycle')
    B.grid(**GRID)
    B.legend(fontsize=8)
else:
    B.text(0.5, 0.5, 'no phase_daily.csv', ha='center', transform=B.transAxes)

C = axs[1][0]
for nm in PT:
    for vn, ls in [('u_top', '-'), ('u_bot', '--')]:
        col = '%s_%s_amp' % (nm, vn)
        D[col + '_anom'] = anom(D[col])
        r, slope, pv, ne = eff_corr(D.tide_range_anom, D[col + '_anom'])
        C.plot(D.tide_range_anom, D[col + '_anom'], '.', ms=2.5, alpha=0.3,
               color=PCOLOR[nm])
        g = pd.DataFrame({'x': D.tide_range_anom, 'y': D[col + '_anom']}).dropna()
        xx = np.linspace(g.x.min(), g.x.max(), 10)
        C.plot(xx, np.polyval(np.polyfit(g.x, g.y, 1), xx), ls,
               color=PCOLOR[nm], lw=2,
               label='%s %s: r=%+.2f (p=%.1e)' % (nm, vn, r, pv))
        sn_rows.append(dict(location=nm, var='%s amplitude' % vn,
                            detrended='30 d anomaly', r=r, slope=slope, p=pv,
                            n_eff=ne))
C.axhline(0, color='0.5', lw=0.8)
C.axvline(0, color='0.5', lw=0.8)
C.set_xlabel('daily tidal range anomaly (m)')
C.set_ylabel('tidal amplitude anomaly (m s$^{-1}$)')
C.set_title('the two mouth points, surface and bed')
C.grid(**GRID)
C.legend(fontsize=7)

E = axs[1][1]
E2 = E.twinx()
E2.plot(D.index, D.tide_range, '-', color='0.75', lw=0.8, zorder=0)
E2.set_ylabel('daily tidal range (m)', color='0.5')
E2.tick_params(axis='y', colors='0.5')
for sn in SECTS:
    E.plot(D.index, D['%s_amp' % sn], '-', lw=1.1, color=COLOR[sn], label=sn)
E.set_zorder(E2.get_zorder() + 1)
E.patch.set_visible(False)
E.set_ylabel('tidal velocity amplitude (m s$^{-1}$)')
E.set_title('tidal velocity tracks the fortnightly envelope (grey = range)')
E.grid(**GRID)
E.legend(fontsize=8, loc='upper left')
E.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

fig.suptitle('%s -- Penn Cove velocity: spring/neap response' % args.gtx,
             fontsize=12)
savefig(fig, 'fig3_spring_neap.png')
plt.close(fig)
pd.DataFrame(sn_rows).to_csv(out_dir / 'spring_neap_velocity.csv', index=False,
                             float_format='%.5f')

# ============================================================= 4. SEASONAL ===
fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
mon = np.arange(1, 13)

A = axs[0][0]
for sn in SECTS:
    g = D.groupby('month')['%s_amp' % sn]
    A.plot(mon, g.mean().reindex(mon).values, 'o-', lw=2, ms=4, color=COLOR[sn],
           label=sn)
    A.fill_between(mon, g.quantile(0.25).reindex(mon).values,
                   g.quantile(0.75).reindex(mon).values, color=COLOR[sn],
                   alpha=0.12, lw=0)
A.set_xticks(mon)
A.set_xlabel('month')
A.set_ylabel('tidal velocity amplitude (m s$^{-1}$)')
A.set_title('seasonal cycle of the TIDAL velocity\nband = interquartile range')
A.grid(**GRID)
A.legend(fontsize=8)

B = axs[0][1]
for sn in SECTS:
    g = D.groupby('month')['%s_res' % sn]
    B.plot(mon, g.mean().reindex(mon).values, 'o-', lw=2, ms=4, color=COLOR[sn],
           label=sn)
B.axhline(0, color='0.5', lw=0.8)
B.set_xticks(mon)
B.set_xlabel('month')
B.set_ylabel('residual section-average u (m s$^{-1}$)')
B.set_title('seasonal cycle of the RESIDUAL (net) velocity')
B.grid(**GRID)
B.legend(fontsize=8)

C = axs[1][0]
for nm in PT:
    for vn, ls in [('u_top', '-'), ('u_bot', '--')]:
        g = D.groupby('month')['%s_%s_res' % (nm, vn)]
        C.plot(mon, g.mean().reindex(mon).values, ls, lw=2, color=PCOLOR[nm],
               label='%s %s' % (nm, vn))
C.axhline(0, color='0.5', lw=0.8)
C.set_xticks(mon)
C.set_xlabel('month')
C.set_ylabel('residual u into the cove (m s$^{-1}$)')
C.set_title('mouth points: seasonal cycle of the residual flow\n'
            'surface (solid) and bed (dashed)')
C.grid(**GRID)
C.legend(fontsize=8, ncol=2)

E = axs[1][1]
sfit_rows = []
for sn in SECTS:
    y = D['%s_amp' % sn]
    E.plot(D.index, y, '-', lw=0.7, alpha=0.30, color=COLOR[sn])
    f = seasonal_fit(y)
    if f is None:
        continue
    E.plot(f['fit'].index, f['fit'].values, '-', lw=2.2, color=COLOR[sn],
           label='%s: R2=%.2f, annual amp %.4f' % (sn, f['R2'], f['amp_annual']))
    sfit_rows.append(dict(location=sn, var='tidal amplitude', mean=f['mean'],
                          amp_annual=f['amp_annual'],
                          amp_semiannual=f['amp_semiannual'], R2=f['R2']))
    f2 = seasonal_fit(D['%s_res' % sn])
    if f2 is not None:
        sfit_rows.append(dict(location=sn, var='residual', mean=f2['mean'],
                              amp_annual=f2['amp_annual'],
                              amp_semiannual=f2['amp_semiannual'], R2=f2['R2']))
E.set_ylabel('tidal velocity amplitude (m s$^{-1}$)')
E.set_title('annual + semiannual harmonic fit')
E.grid(**GRID)
E.legend(fontsize=8)
E.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

fig.suptitle('%s -- Penn Cove velocity: seasonal cycle. Two years = two '
             'samples per calendar month:\nenough for a large phase-locked '
             'cycle, NOT enough to call one month anomalous.' % args.gtx,
             fontsize=12)
savefig(fig, 'fig4_seasonal.png')
plt.close(fig)
pd.DataFrame(sfit_rows).to_csv(out_dir / 'seasonal_fit_velocity.csv',
                               index=False, float_format='%.5f')

clim = D.groupby('month').mean(numeric_only=True)
clim.to_csv(out_dir / 'monthly_climatology_velocity.csv', float_format='%.5f')

# ============================================================= 5. VARIANCE ===
var_rows = []
for sn in SECTS:
    S = SEC[sn]
    for lab, raw in [('ubar (section avg)', S['ubar'].values),
                     ('urms (local)', S['urms'].values)]:
        lp = zfun.lowpass(raw.astype(float), f='godin')
        okm = np.isfinite(lp)
        v_tot = float(np.var(raw[okm]))
        var_rows.append(dict(location=sn, var=lab, sd_total=np.sqrt(v_tot),
                             frac_tidal=float(np.var((raw - lp)[okm])) / v_tot,
                             frac_subtidal=float(np.var(lp[okm])) / v_tot))
for nm in PT:
    for vn in ['u_top', 'u_bot']:
        raw = PT[nm][vn].values.astype(float)
        lp = zfun.lowpass(raw, f='godin')
        okm = np.isfinite(lp)
        v_tot = float(np.var(raw[okm]))
        var_rows.append(dict(location=nm, var=vn, sd_total=np.sqrt(v_tot),
                             frac_tidal=float(np.var((raw - lp)[okm])) / v_tot,
                             frac_subtidal=float(np.var(lp[okm])) / v_tot))
V = pd.DataFrame(var_rows)
V.to_csv(out_dir / 'variance_partition_velocity.csv', index=False,
         float_format='%.4f')
print('\nvariance partition of velocity:')
print(V.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), layout='constrained')
A = axs[0]
lbl = ['%s\n%s' % (r.location, r['var']) for _, r in V.iterrows()]
x = np.arange(len(V))
A.bar(x, V.frac_tidal, color='#4565e8', label='tidal band (hourly - Godin)')
A.bar(x, V.frac_subtidal, bottom=V.frac_tidal, color='#e04256',
      label='subtidal (Godin)')
A.set_xticks(x)
A.set_xticklabels(lbl, fontsize=7, rotation=30, ha='right')
A.set_ylabel('fraction of hourly variance')
A.set_title('where the velocity variance lives')
A.grid(**GRID)
A.legend(fontsize=8)

B = axs[1]
if args.utide:
    try:
        import utide
        con_rows = []
        tnum = pd.to_datetime(idx.tz_convert('UTC').tz_localize(None)
                              .to_pydatetime())
        series = [(sn, 'ubar', SEC[sn]['ubar'].values) for sn in SECTS]
        series += [(nm, vn, PT[nm][vn].values) for nm in PT
                   for vn in ['u_top', 'u_bot']]
        for loc, vn, y in series:
            sol = utide.solve(tnum, y.astype(float), lat=48.23, method='ols',
                              conf_int='none', nodal=True, trend=False,
                              verbose=False)
            for cn, amp in zip(sol.name, sol.A):
                con_rows.append(dict(location=loc, var=vn, constituent=cn,
                                     amplitude=amp))
        CN = pd.DataFrame(con_rows)
        CN.to_csv(out_dir / 'utide_velocity.csv', index=False,
                  float_format='%.5f')
        MAIN = ['M2', 'S2', 'N2', 'K1', 'O1', 'P1']
        wide = (CN[CN.constituent.isin(MAIN)]
                .pivot_table(index='constituent', columns=['location', 'var'],
                             values='amplitude').reindex(MAIN))
        cols = list(wide.columns)
        w = 0.8 / max(1, len(cols))
        xx = np.arange(len(MAIN))
        for kk, c in enumerate(cols):
            B.bar(xx + kk * w, wide[c].values, w,
                  color=COLOR.get(c[0], PCOLOR.get(c[0], 'k')),
                  alpha=1.0 if c[1] in ('ubar', 'u_top') else 0.55,
                  label='%s %s' % c)
        B.set_xticks(xx + 0.4 - w / 2)
        B.set_xticklabels(MAIN)
        B.set_ylabel('velocity amplitude (m s$^{-1}$)')
        B.set_title('tidal constituents of VELOCITY (utide)')
        B.grid(**GRID)
        B.legend(fontsize=6, ncol=2)
        print('\nM2 velocity amplitude (m/s):')
        print(CN[CN.constituent == 'M2'].round(4).to_string(index=False))
    except Exception as e:
        B.text(0.5, 0.5, 'utide failed:\n%s' % e, ha='center', va='center',
               transform=B.transAxes, fontsize=8)
        print('utide failed: %s' % e)
else:
    B.axis('off')

fig.suptitle('%s -- Penn Cove velocity: variance and constituents' % args.gtx,
             fontsize=12)
savefig(fig, 'fig5_variance.png')
plt.close(fig)

D.to_csv(out_dir / 'daily_velocity_series.csv', float_format='%.5f')
print('\nsaved figures and tables to\n  %s' % out_dir)
