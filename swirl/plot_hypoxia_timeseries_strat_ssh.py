"""Two-panel time series at M1/Penn Cove around the 2017 hypoxia events.

A trimmed version of plot_hypoxia_timeseries.py keeping only:
  1. Bottom-top density contrast (kg/m^3) from M1 mooring.
  2. Sea surface height at Penn Cove (m), hourly, with spring/neap shading
     and daily lowpass SSH overplotted.

Event windows are marked on both panels (tan=lead-up ribbon, firebrick=event
ribbon, dotted verticals at event_start/end). No panel letters.

Styling follows Mascarenas et al. 2026 R1 (transparent, lightgray dashed
grid, frameless legends, dpi=500).
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

# --- paper styling (Mascarenas et al. 2026 R1) ---
RED = '#e04256'
BLUE = '#4565e8'
GRID_KW = dict(color='lightgray', linestyle='--', alpha=0.5)

ROOT = os.path.expanduser('~/LO_output/swirl/wb1_r0_xn11b')
HE = f'{ROOT}/hypoxia_events'
MOOR = os.path.expanduser(
    '~/LO_output/extract/wb1_r0_xn11b/moor/pc0/M1_2017.01.02_2017.12.30.nc')
PHASE = os.path.expanduser(
    '~/LO_output/tide_phase/wb1_r0_xn11b/'
    'tide_phases_2017.01.01_2017.12.31/penn_cove.nc')
OUT = f'{HE}/_timeseries_strat_ssh.png'

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
    window_end=max(e1.window_end,e2.window_end))]
for _, ev in raw[raw.event_id==3].iterrows():
    events.append(dict(event_id=int(ev.event_id),
        event_start=ev.event_start, event_end=ev.event_end,
        lead_start=ev.lead_start, window_end=ev.window_end))

# --- M1 mooring (daily) ---
ds = xr.open_dataset(MOOR)
mt = pd.to_datetime(ds.ocean_time.values)
salt = ds.salt.values     # (time, s_rho); 0=bot, -1=top
temp = ds.temp.values
zeta_ssh_daily = ds.zeta.values
ds.close()

salt_top, salt_bot = salt[:, -1], salt[:, 0]
temp_top, temp_bot = temp[:, -1], temp[:, 0]

rho0, alpha, beta = 1025.0, 2e-4, 8e-4
rho_top = rho0*(1 - alpha*temp_top + beta*salt_top)
rho_bot = rho0*(1 - alpha*temp_bot + beta*salt_bot)
drho = rho_bot - rho_top

m = (mt >= TMIN) & (mt <= TMAX)
mt = mt[m]; drho = drho[m]; zeta_ssh_daily = zeta_ssh_daily[m]

# --- Tide phase NC (hourly) ---
dsp = xr.open_dataset(PHASE)
pt = pd.to_datetime(dsp.time.values)
ssh = dsp.zeta.values
is_spring = dsp.is_spring.values.astype(bool)
is_neap   = dsp.is_neap.values.astype(bool)
dsp.close()

pm = (pt >= TMIN) & (pt <= TMAX)
pt = pt[pm]; ssh = ssh[pm]
is_spring = is_spring[pm]; is_neap = is_neap[pm]

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

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                         gridspec_kw=dict(hspace=0.12))

# Panel 1: drho
ax = axes[0]
ax.plot(mt, drho, 'k-', lw=1.2)
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel(r'$\rho_{bot}-\rho_{top}$  [kg m$^{-3}$]')
ax.set_title('M1 stratification and SSH '
             '(2017 hypoxia events at Penn Cove)')

# Panel 2: SSH with tide-phase markers
ax = axes[1]
for t0, t1 in spring_runs:
    ax.axvspan(t0, t1, color='crimson', alpha=0.18, lw=0)
for t0, t1 in neap_runs:
    ax.axvspan(t0, t1, color='royalblue', alpha=0.18, lw=0)
ax.plot(pt, ssh, color='k', lw=0.5, alpha=0.6)
ax.plot(mt, zeta_ssh_daily, color='magenta', lw=1.2)
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel('SSH  [m]')
ax.set_xlabel('2017')
handles = [
    plt.Line2D([],[], color='k', lw=0.5, label='SSH hourly'),
    plt.Line2D([],[], color='magenta', lw=1.2, label='SSH daily lowpass'),
    Patch(facecolor='crimson', alpha=0.18, label='spring'),
    Patch(facecolor='royalblue', alpha=0.18, label='neap'),
    Patch(facecolor='tan', alpha=0.9, label='hypoxia lead-up (ribbon)'),
    Patch(facecolor='firebrick', alpha=0.9, label='hypoxia event (ribbon)'),
]
ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=3,
          frameon=False)

# Mark events on both panels: thin ribbon at TOP edge + dotted verticals.
for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(**GRID_KW)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for ev in events:
        ax.axvline(ev['event_start'], color='k', ls=':', lw=0.8, alpha=0.6)
        ax.axvline(ev['event_end'],   color='k', ls=':', lw=0.8, alpha=0.6)
        ax.fill_between([ev['lead_start'], ev['event_start']],
                        0.96, 1.00, transform=tr,
                        color='tan', alpha=0.9, lw=0)
        ax.fill_between([ev['event_start'], ev['event_end']],
                        0.96, 1.00, transform=tr,
                        color='firebrick', alpha=0.9, lw=0)

axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[-1].set_xlim(TMIN, TMAX)

fig.savefig(OUT, dpi=500, bbox_inches='tight', transparent=True)
print(f'Wrote {OUT}')
