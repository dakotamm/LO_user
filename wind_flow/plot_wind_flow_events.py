"""
Phase 2: per-event 5-panel stacked plots of wind / DO / along-channel flow
/ across-channel flow / top-bottom density difference at a mooring,
spanning each hypoxia event window from find_hypoxia_events.py.

Usage:
    python plot_wind_flow_events.py \
        -gtx wb1_r0_xn11b -mooring M_inner -year 2017 \
        -wind_mooring_extra M_entrance

    # Just one event:
    python plot_wind_flow_events.py -event_id 3 \
        -wind_mooring_extra M_entrance

    # Full-year overview with all events shaded:
    python plot_wind_flow_events.py -overview True \
        -wind_mooring_extra M_entrance
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from lo_tools import Lfun


HYPOXIA_THRESHOLD_MGL = 2.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-gtx', '--gtagex', type=str, default='wb1_r0_xn11b')
    p.add_argument('-mooring', type=str, default='M_inner')
    p.add_argument('-year', type=int, default=2017)
    p.add_argument('-ds0', type=str, default='2017.01.01')
    p.add_argument('-ds1', type=str, default='2017.12.31')
    p.add_argument('-wf_fn', type=str, default=None,
                   help='Override path to the *_wind_flow_*.nc from Phase 1.')
    p.add_argument('-wind_mooring_extra', type=str, default=None,
                   help='Name of a second mooring whose wind to overlay '
                        '(e.g. M_entrance). Resolved against the same gtx '
                        'and ds0/ds1 unless -wf_fn_extra is given.')
    p.add_argument('-wf_fn_extra', type=str, default=None,
                   help='Override path to the second mooring wind_flow NC.')
    p.add_argument('-events_csv', type=str, default=None,
                   help='Override hypoxia events CSV path.')
    p.add_argument('-out_dir', type=str, default=None)
    p.add_argument('-event_id', type=int, default=None,
                   help='Plot only this single event id.')
    p.add_argument('-overview', type=str, default='False',
                   help='If True, also produce a full-year overview figure.')
    return p.parse_args()


def _bool(s):
    return str(s).lower() in ('true', '1', 'yes', 't')


def load_wind_flow(wf_fn):
    ds = xr.open_dataset(wf_fn)
    df = ds.to_dataframe()
    df.index = pd.to_datetime(df.index)
    return df, ds.attrs


def shade_event(ax, row):
    ax.axvspan(row.lead_start,  row.event_start, color='goldenrod', alpha=0.15)
    ax.axvspan(row.event_start, row.event_end,   color='firebrick', alpha=0.30)
    ax.axvspan(row.event_end,   row.window_end,  color='steelblue', alpha=0.15)


def panel_wind(ax, df, t0, t1, df_extra=None, label='M_inner',
               label_extra=None):
    sub = df.loc[t0:t1]
    ax.plot(sub.index, sub['wind_speed'], color='0.6', lw=0.6,
            label=f'{label} hourly')
    ax.plot(sub.index, sub['wind_speed_lp'], color='k', lw=1.2,
            label=f'{label} Godin')
    if df_extra is not None:
        sub2 = df_extra.loc[t0:t1]
        ax.plot(sub2.index, sub2['wind_speed'], color='lightseagreen',
                lw=0.5, alpha=0.5, label=f'{label_extra} hourly')
        ax.plot(sub2.index, sub2['wind_speed_lp'], color='teal', lw=1.2,
                label=f'{label_extra} Godin')
    # Sparse wind-direction sticks along the top (primary mooring)
    n = max(1, len(sub) // 60)
    s = sub.iloc[::n]
    y0 = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10
    ax.quiver(s.index, np.full(len(s), 0.95 * y0),
              s['Uwind'].values, s['Vwind'].values,
              scale=80, width=0.0025, color='0.3', alpha=0.7,
              pivot='middle')
    if df_extra is not None:
        s2 = df_extra.loc[t0:t1].iloc[::n]
        ax.quiver(s2.index, np.full(len(s2), 0.80 * y0),
                  s2['Uwind'].values, s2['Vwind'].values,
                  scale=80, width=0.0025, color='teal', alpha=0.7,
                  pivot='middle')
    ax.set_ylabel('wind [m/s]')
    ax.legend(loc='upper right', fontsize=7, ncol=2)


def panel_do(ax, df, t0, t1):
    sub = df.loc[t0:t1]
    ax.plot(sub.index, sub['bot_DO_mgL'], color='0.6', lw=0.6)
    ax.plot(sub.index, sub['bot_DO_mgL_lp'], color='k', lw=1.2)
    ax.axhline(HYPOXIA_THRESHOLD_MGL, color='red', lw=1, ls='--',
               label=f'{HYPOXIA_THRESHOLD_MGL:.1f} mg/L')
    ax.set_ylabel('bot DO [mg/L]')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=7)


def panel_flow(ax, df, t0, t1, component):
    """component = 'u' (east, +E) or 'v' (north, +N)."""
    sub = df.loc[t0:t1]
    layers = [('surface',  'tab:orange'),
              ('depthavg', 'k'),
              ('bottom',   'tab:blue')]
    for layer, color in layers:
        col = f'{component}_{layer}_lp'
        ax.plot(sub.index, sub[col], color=color, lw=1.0,
                label=layer)
    ax.axhline(0, color='0.4', lw=0.5)
    label = 'east (+E)' if component == 'u' else 'north (+N)'
    ax.set_ylabel(f'{label} [m/s]\n(Godin)')
    ax.legend(loc='upper right', fontsize=7, ncol=3)


def panel_strat(ax, df, t0, t1):
    sub = df.loc[t0:t1]
    ax.plot(sub.index, sub['d_rho'], color='0.6', lw=0.6)
    ax.plot(sub.index, sub['d_rho_lp'], color='purple', lw=1.2)
    ax.axhline(0, color='0.4', lw=0.5)
    ax.set_ylabel(r'$\Delta\rho$ bot-top'
                  '\n[kg/m$^3$]')


def plot_event(df, row, attrs, out_path, mooring, gtx,
               df_extra=None, mooring_extra=None):
    t0 = pd.Timestamp(row.lead_start)
    t1 = pd.Timestamp(row.window_end)

    fig, axs = plt.subplots(5, 1, figsize=(11, 12), sharex=True,
                            gridspec_kw=dict(hspace=0.08))

    panel_wind(axs[0], df, t0, t1, df_extra=df_extra,
               label=mooring, label_extra=mooring_extra)
    panel_do(axs[1], df, t0, t1)
    panel_flow(axs[2], df, t0, t1, 'u')
    panel_flow(axs[3], df, t0, t1, 'v')
    panel_strat(axs[4], df, t0, t1)

    for ax in axs:
        shade_event(ax, row)
        ax.grid(color='lightgray', ls='--', alpha=0.4)

    axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    th_d = attrs.get('theta_depthavg_deg', float('nan'))
    fig.suptitle(
        f'{gtx} {mooring} hypoxia event #{int(row.event_id)} '
        f'({row.event_start.date()} – {row.event_end.date()})  '
        f'[principal axis = {th_d:+.0f}° CCW from east]',
        fontsize=11,
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_overview(df, events_df, attrs, out_path, mooring, gtx,
                  df_extra=None, mooring_extra=None):
    fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True,
                            gridspec_kw=dict(hspace=0.08))
    t0 = df.index.min()
    t1 = df.index.max()
    panel_wind(axs[0], df, t0, t1, df_extra=df_extra,
               label=mooring, label_extra=mooring_extra)
    panel_do(axs[1], df, t0, t1)
    panel_flow(axs[2], df, t0, t1, 'u')
    panel_flow(axs[3], df, t0, t1, 'v')
    panel_strat(axs[4], df, t0, t1)
    for ax in axs:
        for _, row in events_df.iterrows():
            shade_event(ax, row)
        ax.grid(color='lightgray', ls='--', alpha=0.4)
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.suptitle(f'{gtx} {mooring} wind / flow / strat / bot-DO '
                 f'with hypoxia events shaded', fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

    if args.wf_fn is not None:
        wf_fn = Path(args.wf_fn)
    else:
        wf_fn = (Ldir['LOo'] / 'wind_flow' / args.gtagex
                 / f'{args.mooring}_wind_flow_{args.ds0}_{args.ds1}.nc')
    if not wf_fn.exists():
        raise FileNotFoundError(
            f'Wind-flow NC not found: {wf_fn}\n'
            f'Run compute_wind_flow.py first.')

    # Optional second mooring for wind overlay
    df_extra = None
    mooring_extra = None
    if args.wf_fn_extra is not None or args.wind_mooring_extra is not None:
        if args.wf_fn_extra is not None:
            wf_fn_extra = Path(args.wf_fn_extra)
            mooring_extra = (args.wind_mooring_extra
                             or wf_fn_extra.stem.split('_wind_flow')[0])
        else:
            mooring_extra = args.wind_mooring_extra
            wf_fn_extra = (Ldir['LOo'] / 'wind_flow' / args.gtagex
                           / f'{mooring_extra}_wind_flow_'
                             f'{args.ds0}_{args.ds1}.nc')
        if not wf_fn_extra.exists():
            raise FileNotFoundError(
                f'Extra wind-flow NC not found: {wf_fn_extra}')
        df_extra, _ = load_wind_flow(wf_fn_extra)
        print(f'Overlaying wind from {mooring_extra} ({wf_fn_extra})')

    if args.events_csv is not None:
        events_csv = Path(args.events_csv)
    else:
        events_csv = (Ldir['LOo'] / 'wind_flow' / args.gtagex
                      / f'hypoxia_events_{args.mooring}_{args.year}.csv')
    if not events_csv.exists():
        raise FileNotFoundError(f'Events CSV not found: {events_csv}')

    df, attrs = load_wind_flow(wf_fn)
    events = pd.read_csv(events_csv, parse_dates=[
        'event_start', 'event_end', 'lead_start', 'window_end'])

    if args.event_id is not None:
        events = events[events.event_id == args.event_id]
        if events.empty:
            raise ValueError(f'event_id {args.event_id} not in {events_csv}')

    out_dir = (Path(args.out_dir) if args.out_dir else
               (Ldir['LOo'] / 'wind_flow' / args.gtagex / 'plots'))

    for _, row in events.iterrows():
        out_path = out_dir / (
            f'event_{int(row.event_id):02d}_'
            f'{pd.Timestamp(row.lead_start).strftime("%Y%m%d")}_'
            f'{pd.Timestamp(row.window_end).strftime("%Y%m%d")}_'
            f'{args.mooring}_wind_flow.png')
        plot_event(df, row, attrs, out_path, args.mooring, args.gtagex,
                   df_extra=df_extra, mooring_extra=mooring_extra)
        print(f'Wrote {out_path}')

    if _bool(args.overview):
        out_path = out_dir / f'overview_{args.mooring}_{args.year}.png'
        plot_overview(df, events, attrs, out_path, args.mooring, args.gtagex,
                      df_extra=df_extra, mooring_extra=mooring_extra)
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
