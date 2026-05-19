"""Stacked time series of bottom DO and per-layer max |zeta_vort| about
event_start for each (merged) hypoxia event at M1.

Panels:
  1. M1 bottom DO  [mg/L]
  2. Daily max |zeta_vort| at SURFACE       [1e-5 s^-1]
  3. Daily max |zeta_vort| at DEPTH-AVG     [1e-5 s^-1]
  4. Daily max |zeta_vort| at BOTTOM        [1e-5 s^-1]

x-axis = days from event_start (day 0).  One curve per event, color-coded.
Events 1 and 2 are merged.
"""
import os, glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = os.path.expanduser('~/LO_output/swirl/wb1_r0_xn11b')
HE = f'{ROOT}/hypoxia_events'
MOOR = os.path.expanduser(
    '~/LO_output/extract/wb1_r0_xn11b/moor/pc0/M1_2017.01.02_2017.12.30.nc')
OUT = f'{HE}/_analysis_stacked_DO_vort_layers.png'

# --- Events (merge 1+2) ---
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

# --- M1 bottom DO ---
ds = xr.open_dataset(MOOR)
do = pd.Series(ds.oxygen.values[:,0] * 32.0/1000.0,
               index=pd.to_datetime(ds.ocean_time.values).normalize(),
               name='do')
ds.close()

# --- Per-layer daily max |zeta_vort| for an event ---
def daily_max_vort(ev, layer):
    parts = []
    for d in ev['src_dirs']:
        f = glob.glob(f'{d}/{layer}/ow_vortices_*.csv')
        if f:
            parts.append(pd.read_csv(f[0], parse_dates=['time']))
    if not parts:
        return pd.Series(dtype=float)
    v = pd.concat(parts, ignore_index=True)
    v['day'] = v.time.dt.normalize()
    return v.groupby('day')['max_vorticity'].apply(
        lambda s: np.abs(s).max()).sort_index()

# --- Build stacked dataframe ---
LAYERS = ['surface','depth_avg','bottom']
rows = []
for ev in events:
    idx = pd.date_range(ev['lead_start'], ev['window_end'], freq='D')
    per_layer = {L: daily_max_vort(ev, L).reindex(idx).fillna(0.0)
                 for L in LAYERS}
    for d in idx:
        r = (d - ev['event_start']).days
        rec = dict(event=int(ev['event_id']), day=d, rel_day=r,
                   do=do.get(d, np.nan))
        for L in LAYERS:
            rec[L] = per_layer[L].loc[d]
        rows.append(rec)
stk = pd.DataFrame(rows)

# --- Plot ---
fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                         gridspec_kw=dict(hspace=0.18))
colors = {12: 'tab:red', 3: 'tab:green'}

# relative-day position of each event's end
event_end_rel = {int(ev['event_id']):
                 (ev['event_end'] - ev['event_start']).days
                 for ev in events}

# Panel 1: bottom DO
ax = axes[0]
for eid, sub in stk.groupby('event'):
    sub = sub.sort_values('rel_day')
    ax.plot(sub.rel_day, sub.do, '-o', ms=4, color=colors[eid],
            label=f'event {eid}')
ax.axhline(2.0, color='k', ls=':', lw=1, label='2 mg/L')
ax.axvline(0, color='gray', ls='--', lw=1)
for eid, rd in event_end_rel.items():
    ax.axvline(rd, color=colors[eid], ls=':', lw=1.2, alpha=0.8)
ax.set_ylabel('M1 bottom DO  [mg L$^{-1}$]')
ax.legend(loc='upper right', fontsize=9, ncol=3)
ax.set_title('Stacked: bottom DO and per-layer max |$\\zeta_{vort}$| about event_start')

# Panels 2-4: per-layer max |zeta_vort|
panel_labels = {
    'surface':   'SURFACE\nmax |$\\zeta_{vort}$|\n[10$^{-5}$ s$^{-1}$]',
    'depth_avg': 'DEPTH-AVG\nmax |$\\zeta_{vort}$|\n[10$^{-5}$ s$^{-1}$]',
    'bottom':    'BOTTOM\nmax |$\\zeta_{vort}$|\n[10$^{-5}$ s$^{-1}$]',
}
for ax, L in zip(axes[1:], LAYERS):
    for eid, sub in stk.groupby('event'):
        sub = sub.sort_values('rel_day')
        ax.plot(sub.rel_day, sub[L].values * 1e5, '-o', ms=4,
                color=colors[eid], label=f'event {eid}')
    ax.axvline(0, color='gray', ls='--', lw=1)
    for eid, rd in event_end_rel.items():
        ax.axvline(rd, color=colors[eid], ls=':', lw=1.2, alpha=0.8)
    ax.set_ylabel(panel_labels[L])

axes[-1].set_xlabel('days from event_start (dashed=start, dotted=end per event)')
fig.align_ylabels(axes)

# prune top/bottom tick labels on shared-x stack so adjacent panels' y-tick
# labels don't crowd at the panel boundaries.
from matplotlib.ticker import MaxNLocator
for ax in axes:
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Wrote {OUT}')
