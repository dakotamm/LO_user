"""Combined time series at M1/Penn Cove around the 2017 hypoxia events.

Panels (top to bottom):
  1. Bottom-top density contrast (kg/m^3) from M1 mooring (daily lowpass).
  2. Salinity contrast (psu) and temperature contrast (degC) components.
  3. Top and bottom dissolved oxygen at M1 (mg/L).
  4. Sea surface height at Penn Cove (m), hourly, with spring/neap shading
     and flood/ebb tick marks; daily lowpass SSH overplotted.
  5. Daily max |zeta_vort| (1/s) for surface, depth-avg, and bottom layers,
     pooled across event runs (concatenated from event_01/02/03 dirs).

Event windows are shaded on every panel (tan=lead-up, salmon=event).
"""
import os, glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.path.expanduser('~/LO_output/swirl/wb1_r0_xn11b')
HE = f'{ROOT}/hypoxia_events'
MOOR = os.path.expanduser(
    '~/LO_output/extract/wb1_r0_xn11b/moor/pc0/M1_2017.01.02_2017.12.30.nc')
PHASE = os.path.expanduser(
    '~/LO_output/tide_phase/wb1_r0_xn11b/'
    'tide_phases_2017.01.01_2017.12.31/penn_cove.nc')
OUT = f'{HE}/_timeseries_combined.png'

TMIN = pd.Timestamp('2017-01-01')
TMAX = pd.Timestamp('2017-12-31')

# --- Events (merged 1+2) ---
raw = pd.read_csv(f'{ROOT}/hypoxia_events_M1_2017.csv',
    parse_dates=['event_start','event_end','lead_start','window_end'])
def _orig_dir(ev):
    return f"{HE}/event_{int(ev.event_id):02d}_{ev.lead_start:%Y.%m.%d}_{ev.window_end:%Y.%m.%d}"
e1 = raw[raw.event_id==1].iloc[0]; e2 = raw[raw.event_id==2].iloc[0]
events = [dict(event_id=12,
    event_start=min(e1.event_start,e2.event_start),
    event_end=max(e1.event_end,e2.event_end),
    lead_start=min(e1.lead_start,e2.lead_start),
    window_end=max(e1.window_end,e2.window_end),
    src_dirs=[_orig_dir(e1), _orig_dir(e2)])]
for _, ev in raw[raw.event_id==3].iterrows():
    events.append(dict(event_id=int(ev.event_id),
        event_start=ev.event_start, event_end=ev.event_end,
        lead_start=ev.lead_start, window_end=ev.window_end,
        src_dirs=[_orig_dir(ev)]))

# --- M1 mooring (daily lowpass) ---
ds = xr.open_dataset(MOOR)
mt = pd.to_datetime(ds.ocean_time.values)
salt = ds.salt.values     # (time, s_rho); 0=bot, -1=top
temp = ds.temp.values
do   = ds.oxygen.values * 32.0/1000.0
z    = ds.z_rho.values
zeta_ssh_daily = ds.zeta.values
ds.close()

salt_top, salt_bot = salt[:, -1], salt[:, 0]
temp_top, temp_bot = temp[:, -1], temp[:, 0]
do_top,   do_bot   = do[:, -1],   do[:, 0]
z_top,    z_bot    = z[:, -1],    z[:, 0]

rho0, alpha, beta = 1025.0, 2e-4, 8e-4
rho_top = rho0*(1 - alpha*temp_top + beta*salt_top)
rho_bot = rho0*(1 - alpha*temp_bot + beta*salt_bot)
drho = rho_bot - rho_top
dsalt = salt_bot - salt_top
dtemp = temp_top - temp_bot

m = (mt >= TMIN) & (mt <= TMAX)
mt = mt[m]; drho = drho[m]; dsalt = dsalt[m]; dtemp = dtemp[m]
do_top = do_top[m]; do_bot = do_bot[m]; zeta_ssh_daily = zeta_ssh_daily[m]

# --- Tide phase NC (hourly) ---
dsp = xr.open_dataset(PHASE)
pt = pd.to_datetime(dsp.time.values)
ssh = dsp.zeta.values
is_spring = dsp.is_spring.values.astype(bool)
is_neap   = dsp.is_neap.values.astype(bool)
is_flood  = dsp.is_flood.values.astype(bool)
is_ebb    = dsp.is_ebb.values.astype(bool)
dsp.close()

pm = (pt >= TMIN) & (pt <= TMAX)
pt = pt[pm]; ssh = ssh[pm]
is_spring = is_spring[pm]; is_neap = is_neap[pm]
is_flood = is_flood[pm]; is_ebb = is_ebb[pm]

# Contiguous spring/neap segments for shading
def runs(flag, t):
    edges = np.diff(flag.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends   = np.where(edges == -1)[0] + 1
    if flag[0]: starts = np.r_[0, starts]
    if flag[-1]: ends = np.r_[ends, len(flag)]
    return [(t[s], t[min(e, len(t)-1)]) for s, e in zip(starts, ends)]

spring_runs = runs(is_spring, pt)
neap_runs   = runs(is_neap,   pt)

# --- Per-layer daily max |zeta_vort| pooled across events ---
def daily_max_vort_pooled(layer):
    parts = []
    for ev in events:
        for d in ev['src_dirs']:
            f = glob.glob(f'{d}/{layer}/ow_vortices_*.csv')
            if f:
                parts.append(pd.read_csv(f[0], parse_dates=['time']))
    if not parts:
        return pd.Series(dtype=float)
    v = pd.concat(parts, ignore_index=True)
    v['day'] = v.time.dt.normalize()
    s = v.groupby('day')['max_vorticity'].apply(lambda x: np.abs(x).max())
    return s.sort_index()

vort_layers = {L: daily_max_vort_pooled(L)
               for L in ['surface','depth_avg','bottom']}

# --- Plot ---
fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True,
                         gridspec_kw=dict(hspace=0.12))

# Panel 1: drho
ax = axes[0]
ax.plot(mt, drho, 'k-', lw=1.2)
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel(r'$\rho_{bot}-\rho_{top}$  [kg m$^{-3}$]')
ax.set_title('M1 stratification, DO, SSH, and vortex strength '
             '(2017 hypoxia events at Penn Cove)')

# Panel 2: density contributions from salinity and temperature (kg/m^3),
# on a shared axis so they're directly comparable and sum to Delta rho.
ax = axes[1]
drho_S = beta  * rho0 * dsalt   # bot-top salinity contribution
drho_T = alpha * rho0 * dtemp   # top-bot temperature contribution (warm top = stable)
ax.plot(mt, drho_S, 'b-', lw=1.2, label=r'$\beta\rho_0\Delta S$ (salinity)')
ax.plot(mt, drho_T, 'r-', lw=1.2, label=r'$\alpha\rho_0\Delta T$ (temperature)')
ax.plot(mt, drho_S + drho_T, 'k--', lw=0.8, alpha=0.7,
        label='sum (= Δρ, panel 1)')
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel(r'contribution to $\Delta\rho$  [kg m$^{-3}$]')
ax.legend(loc='upper right', fontsize=9, ncol=3)

# Panel 3: DO
ax = axes[2]
ax.plot(mt, do_top, color='tab:orange', lw=1.2, label='top')
ax.plot(mt, do_bot, color='tab:blue',   lw=1.2, label='bottom')
ax.axhline(2.0, color='k', ls=':', lw=1, label='2 mg/L')
ax.set_ylabel('DO  [mg L$^{-1}$]')
ax.legend(loc='upper right', fontsize=9, ncol=3)

# Panel 4: SSH with tide-phase markers
ax = axes[3]
for t0, t1 in spring_runs:
    ax.axvspan(t0, t1, color='crimson', alpha=0.18, lw=0)
for t0, t1 in neap_runs:
    ax.axvspan(t0, t1, color='royalblue', alpha=0.18, lw=0)
ax.plot(pt, ssh, color='k', lw=0.5, alpha=0.6, label='hourly')
ax.plot(mt, zeta_ssh_daily, color='magenta', lw=1.2, label='daily lowpass')
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel('SSH  [m]')
# legend including phase patches
from matplotlib.patches import Patch
handles = [
    plt.Line2D([],[], color='k', lw=0.5, label='SSH hourly'),
    plt.Line2D([],[], color='magenta', lw=1.2, label='SSH daily lowpass'),
    Patch(facecolor='crimson', alpha=0.18, label='spring'),
    Patch(facecolor='royalblue', alpha=0.18, label='neap'),
    Patch(facecolor='tan', alpha=0.9, label='hypoxia lead-up (ribbon)'),
    Patch(facecolor='firebrick', alpha=0.9, label='hypoxia event (ribbon)'),
]
ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=3)

# Panel 5: per-layer max |zeta_vort|
ax = axes[4]
colors = dict(surface='tab:orange', depth_avg='tab:green', bottom='tab:blue')
for L, s in vort_layers.items():
    s = s[(s.index >= TMIN) & (s.index <= TMAX)]
    ax.plot(s.index, s.values * 1e5, '-o', ms=3, lw=1,
            color=colors[L], label=L)
ax.set_ylabel(r'daily max $|\zeta_{vort}|$ [$10^{-5}$ s$^{-1}$]')
ax.set_xlabel('2017')
ax.legend(loc='upper right', fontsize=9, ncol=3)

# Mark events on every panel: thin ribbon at TOP edge of each axes
# (tan = lead-up, dark-red = event), plus vertical dotted lines at
# event_start and event_end so they're easy to align across panels.
# Background event fill is omitted so it doesn't collide with spring/neap
# shading on the SSH panel.
from matplotlib.transforms import blended_transform_factory
for ax in axes:
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for ev in events:
        ax.axvline(ev['event_start'], color='k', ls=':', lw=0.8, alpha=0.6)
        ax.axvline(ev['event_end'],   color='k', ls=':', lw=0.8, alpha=0.6)
        # ribbon: y in [0.96, 1.0] of axes height
        ax.fill_between([ev['lead_start'], ev['event_start']],
                        0.96, 1.00, transform=tr,
                        color='tan', alpha=0.9, lw=0)
        ax.fill_between([ev['event_start'], ev['event_end']],
                        0.96, 1.00, transform=tr,
                        color='firebrick', alpha=0.9, lw=0)

axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[-1].set_xlim(TMIN, TMAX)

fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Wrote {OUT}')
