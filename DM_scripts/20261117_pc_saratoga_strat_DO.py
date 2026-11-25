"""
Daily stratification and bottom DO at two points, one inside Penn Cove and one in Saratoga Passage just outside it,
for the cas7 hindcasts. One pass per run reads, at both cells, the full salt / temp / oxygen column and zeta, so any
other column metric can be computed later from the pickle without another apogee run.
(Was 20261117_sp_point_strat_DO.py, which did one point at a time.)

-st picks the stations (comma list, default both). M5 was already extracted for both runs by the old
20261117_sp_point_strat_DO.py (DM_outs/20261117_sp_point_strat_DO/); the comparison plot reads those pickles too, so
only Penn Cove needs running. Run on apogee (both lines can run at the same time):
  cas7_t1_x11ab (50% burial; daily ocean_avg_0001.nc in Parker's /dat1 = roms_out1):
    nohup python 20261117_pc_saratoga_strat_DO.py -gtx cas7_t1_x11ab -ro 1 -0 2012.10.07 -1 2025.06.12 -lt average -st pc_west > pc_west_x11ab.log 2>&1 &
  cas7_t0_x4b (0% burial; lowpassed.nc in Parker's /dat2 = roms_out3):
    nohup python 20261117_pc_saratoga_strat_DO.py -gtx cas7_t0_x4b -ro 3 -0 2012.10.07 -1 2019.12.31 -lt lowpass -st pc_west > pc_west_x4b.log 2>&1 &
  wb1_t0_xn11abbur00 (0% burial nest; lowpassed.nc in our /dat2 = roms_out2; both stations, M5 not done before):
    nohup python 20261117_pc_saratoga_strat_DO.py -gtx wb1_t0_xn11abbur00 -ro 2 -0 2024.01.01 -1 2025.12.31 -lt lowpass -st pc_west,M5 > pc_saratoga_wb1.log 2>&1 &
  Check on them with: tail pc_west_x11ab.log pc_west_x4b.log pc_saratoga_wb1.log
RESUME  each day is appended to <out_name>_partial.csv (flushed every 100 files). If a run dies, rerun the same
  command: days already in the partial file are skipped (a truncated last line is dropped). The final pickle is
  written, and the partial removed, at the end.
Re-plot (e.g. on the Mac after copying the pickles over); also redraws the run comparison:
    python 20261117_pc_saratoga_strat_DO.py -gtx cas7_t1_x11ab -0 2012.10.07 -1 2025.06.12 -st pc_west -plot_only True

POINTS  each snapped to the nearest wet rho cell (printed). Grid from the first ocean file, else that day's
        ocean_his_0002.nc, else grid.nc.
        pc_west  (-122.70878, 48.22710) = the 20261116_pc_cas7_bottom_DO point and the cove script's seed:
                 cas7 j=882, i=557, h 17.0 m (map: 20261116_pc_cas7_point_map.py); wb1 j=217, i=47, h 17.2 m.
        M5       (-122.582462, 48.245981), the pc0/pc4 Saratoga Passage station: cas7 j=886, i=576, h 39.2 m;
                 wb1 j=227, i=94, h 37.4 m.
READ    -lt average: f<date>/ocean_avg_0001.nc (UTC-day average); -lt lowpass: f<date>/lowpassed.nc (one Godin-
        filtered record per day). zeta, and salt, temp, oxygen at every s_rho level, at each cell.
STRAT   drho = sigma0(bottom) - sigma0(top) [kg/m3], TEOS-10 as pc_forcing_states.py / 20261110_pc_column_metrics.py:
        SA = SA_from_SP(salt, -z_rho, lon, lat), CT = CT_from_pt(SA, temp). z_rho from zrfun.get_z with each day's
        zeta and the file's s-coordinate. Also dS and dT (bottom - top).
BOTTOM  DO at s_rho index 0; oxygen mmol m-3 * 32/1000 = mg/L.
OUT     LO_output/DM_outs/20261117_pc_saratoga_strat_DO/
        out_name = <gtagex>_<stations joined by ->_<ds0>_<ds1>
        <out_name>.p        dict: run info + 'st' = {station: time, zeta, z_rho / salt / temp / oxygen / sigma0
                            profiles (Nt x N), cell}
        <out_name>_<sn>.csv daily bottom DO, drho, dS, dT, top/bottom S and T, one per station
        <out_name>.png      bottom DO + drho, all stations in the run;  compare.png  every run and station found
                            here and in DM_outs/20261117_sp_point_strat_DO/, one column per station
"""

import argparse
import sys
import pickle
import numpy as np
import pandas as pd
import netCDF4 as nc
import gsw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun, zrfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str, default='cas7_t1_x11ab')
parser.add_argument('-ro', '--roms_out_num', type=int, default=1)  # 1 = Ldir['roms_out1']
parser.add_argument('-0', '--ds0', type=str, default='2012.10.07')
parser.add_argument('-1', '--ds1', type=str, default='2025.06.12')
parser.add_argument('-lt', type=str, default='average')   # average (ocean_avg_0001) or lowpass (lowpassed.nc)
parser.add_argument('-st', type=str, default='pc_west,M5')   # stations to extract, from STATIONS_ALL
parser.add_argument('-plot_only', type=zfun.boolean_string, default=False)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
if args.roms_out_num > 0:
    Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]

STATIONS_ALL = {'pc_west': (-122.70878, 48.22710), 'M5': (-122.582462, 48.245981)}
STATIONS = {sn: STATIONS_ALL[sn] for sn in args.st.split(',')}
DO_MMOL_TO_MGL = 32.0 / 1000.0
HYP = 2.0                                  # mg/L
RUN_COLORS = {'cas7_t1_x11ab': 'tab:blue', 'cas7_t0_x4b': 'tab:red', 'wb1_t0_xn11abbur00': 'tab:green'}   # as cove script
ST_COLORS = {'pc_west': 'tab:orange', 'M5': 'k'}

out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_saratoga_strat_DO'
Lfun.make_dir(out_dir)
out_name = args.gtagex + '_' + '-'.join(STATIONS) + '_' + args.ds0 + '_' + args.ds1
p_fn = out_dir / (out_name + '.p')

if not args.plot_only:
    fn_list = Lfun.get_fn_list(args.lt, Ldir, args.ds0, args.ds1)
    fn_have = [fn for fn in fn_list if fn.exists()]
    print(f'files  : {len(fn_have)} of {len(fn_list)} days have {fn_list[0].name}')
    if len(fn_have) == 0:
        print(f'*** no files under {Ldir["roms_out"] / args.gtagex}')
        sys.exit()

    # ---- nearest wet rho cell for each station ----
    vns = ['lon_rho', 'lat_rho', 'mask_rho', 'h']
    for gfn in [fn_have[0], fn_have[0].parent / 'ocean_his_0002.nc', Ldir['grid'] / 'grid.nc']:
        if gfn.exists():
            with nc.Dataset(gfn) as ds:
                if all(vn in ds.variables for vn in vns):
                    G = {vn: ds[vn][:].data for vn in vns}
                    break
    else:
        print('*** no grid variables in the ocean file, ocean_his_0002.nc or grid.nc')
        sys.exit()
    print(f'grid   : {gfn}')
    C = {}
    for sn, (lon0, lat0) in STATIONS.items():
        dist2 = ((G['lon_rho'] - lon0) * np.cos(np.deg2rad(lat0)))**2 + (G['lat_rho'] - lat0)**2
        dist2[G['mask_rho'] != 1] = np.inf
        j, i = np.unravel_index(np.argmin(dist2), dist2.shape)
        C[sn] = dict(j=int(j), i=int(i), lon=float(G['lon_rho'][j, i]), lat=float(G['lat_rho'][j, i]),
                     h=float(G['h'][j, i]))
        print(f'{sn:7s}: j={j} i={i}  lon {C[sn]["lon"]:.5f}  lat {C[sn]["lat"]:.5f}  h {C[sn]["h"]:.1f} m  '
              f'({np.sqrt(dist2[j, i]) * 111:.2f} km from the requested point)')
    S = zrfun.get_basic_info(fn_have[0], only_S=True)
    N = int(S['N'])

    # ---- one column per station per file, appended to a partial csv so a killed run can resume ----
    part_fn = out_dir / (out_name + '_partial.csv')
    cell_line = (f'# v1 {args.gtagex} {args.lt} N={N} '
                 + ' '.join(f'{sn}:j={c["j"]},i={c["i"]}' for sn, c in C.items()) + '\n')
    cols = ['fdir', 'time_utc']
    for sn in C:
        cols += [f'{sn}_zeta'] + [f'{sn}_{v}{k}' for v in ['s', 't', 'o'] for k in range(N)]
    cols += ['ok']
    done = set()
    if part_fn.exists():
        with open(part_fn) as f:
            if f.readline() != cell_line:
                print(f'*** {part_fn.name} is for different cells or file type: delete it')
                sys.exit()
        # keep only complete rows (a kill mid-write leaves a short last line, so no ok=1), rewrite, then append
        dfp = pd.read_csv(part_fn, comment='#', on_bad_lines='skip')
        dfp = dfp[pd.to_numeric(dfp['ok'], errors='coerce') == 1]
        with open(part_fn, 'w') as f:
            f.write(cell_line)
            dfp.to_csv(f, index=False)
        done = set(dfp['fdir'])
        print(f'resume : {len(done)} days already in {part_fn.name}')
    else:
        with open(part_fn, 'w') as f:
            f.write(cell_line + ','.join(cols) + '\n')
    fn_todo = [fn for fn in fn_have if fn.parent.name not in done]

    with open(part_fn, 'a') as f:
        for nn, fn in enumerate(fn_todo):
            vals = []
            with nc.Dataset(fn) as ds:
                t0 = nc.num2date(ds['ocean_time'][0], ds['ocean_time'].units,
                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                for c in C.values():
                    vals += [float(ds['zeta'][0, c['j'], c['i']])]
                    for vn in ['salt', 'temp', 'oxygen']:
                        vals += list(np.ma.filled(ds[vn][0, :, c['j'], c['i']].astype(float), np.nan))
            f.write(f'{fn.parent.name},{t0:%Y-%m-%d %H:%M:%S},' + ','.join(f'{v:.4f}' for v in vals) + ',1\n')
            if nn % 100 == 0:
                f.flush()
                print(f'  {nn} / {len(fn_todo)}  {fn.parent.name}', flush=True)

    df = pd.read_csv(part_fn, comment='#').drop_duplicates('fdir').sort_values('time_utc').reset_index(drop=True)
    P = {'gtagex': args.gtagex, 'lt': args.lt, 'time': pd.to_datetime(df['time_utc']).values, 'st': {}}
    for sn, c in C.items():
        zeta = df[f'{sn}_zeta'].values
        z_rho, _ = zrfun.get_z(c['h'] * np.ones(len(zeta)), zeta, S)     # (N, Nt), bottom first
        z_rho = z_rho.T
        salt = df[[f'{sn}_s{k}' for k in range(N)]].values
        temp = df[[f'{sn}_t{k}' for k in range(N)]].values
        SA = gsw.SA_from_SP(salt, -z_rho, c['lon'], c['lat'])
        P['st'][sn] = dict(c, zeta=zeta, z_rho=z_rho, salt=salt, temp=temp,
                           oxygen=df[[f'{sn}_o{k}' for k in range(N)]].values * DO_MMOL_TO_MGL,
                           sigma0=gsw.sigma0(SA, gsw.CT_from_pt(SA, temp)))
    pickle.dump(P, open(p_fn, 'wb'))
    part_fn.unlink()
    print(f'saved  : {p_fn}')


# ---- daily series ----
def series(P, sn):
    s = P['st'][sn]
    return pd.DataFrame({'time_utc': pd.to_datetime(P['time']), 'DO_bot_mgL': s['oxygen'][:, 0],
                         'drho': s['sigma0'][:, 0] - s['sigma0'][:, -1],
                         'dS': s['salt'][:, 0] - s['salt'][:, -1], 'dT': s['temp'][:, 0] - s['temp'][:, -1],
                         'S_top': s['salt'][:, -1], 'S_bot': s['salt'][:, 0],
                         'T_top': s['temp'][:, -1], 'T_bot': s['temp'][:, 0]})


def st_label(s, sn):
    return f'{sn} (j={s["j"]} i={s["i"]}, h {s["h"]:.1f} m)'


P = pickle.load(open(p_fn, 'rb'))
fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
for sn, s in P['st'].items():
    D = series(P, sn)
    D.to_csv(out_dir / (out_name + '_' + sn + '.csv'), index=False)
    t = D['time_utc']
    print(f'{P["gtagex"]} {st_label(s, sn)}: bottom DO min {D.DO_bot_mgL.min():.2f} mg/L on '
          f'{t[D.DO_bot_mgL.idxmin()]:%Y-%m-%d}, days < {HYP:g} mg/L: {int((D.DO_bot_mgL < HYP).sum())} of {len(D)};  '
          f'drho mean {D.drho.mean():.2f}, max {D.drho.max():.2f} kg/m3')
    c = ST_COLORS.get(sn)
    axs[0].plot(t, D.DO_bot_mgL, '-', color=c, lw=0.7, label=st_label(s, sn))
    axs[1].plot(t, D.drho, '-', color=c, lw=0.7, label=st_label(s, sn))
axs[0].axhline(HYP, color='gray', ls=':', lw=1)
axs[0].set_ylabel('Bottom DO [mg/L]')
axs[0].set_ylim(bottom=0)
axs[0].legend(loc='upper right', fontsize=8)
axs[0].set_title(f'{P["gtagex"]} ({P["lt"]})', fontsize=10)
axs[1].set_ylabel(r'$\Delta\sigma_0$ bottom - top [kg m$^{-3}$]')
axs[1].set_ylim(bottom=0)
axs[1].set_xlim(t.iloc[0], t.iloc[-1])
for ax in axs:
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / (out_name + '.png'), dpi=200, transparent=True)
plt.close(fig)

# ---- every run and station found, by station: columns = stations, rows = bottom DO, drho ----
def load_run(fn):
    """A pickle from this script, or a one-station pickle from the old 20261117_sp_point_strat_DO.py."""
    Pr = pickle.load(open(fn, 'rb'))
    if 'st' not in Pr:
        keys = ['zeta', 'z_rho', 'salt', 'temp', 'oxygen', 'sigma0', 'j', 'i', 'lon', 'lat', 'h']
        Pr = {'gtagex': Pr['gtagex'], 'lt': Pr['lt'], 'time': Pr['time'], 'st': {Pr['sn']: {k: Pr[k] for k in keys}}}
    return Pr


old_dir = Ldir['LOo'] / 'DM_outs' / '20261117_sp_point_strat_DO'
runs = sorted([load_run(fn) for fn in sorted(out_dir.glob('*.p')) + sorted(old_dir.glob('*.p'))],
              key=lambda Pr: list(RUN_COLORS).index(Pr['gtagex']) if Pr['gtagex'] in RUN_COLORS else len(RUN_COLORS))   # drawing order: long x11ab first, shorter runs on top
sns = [sn for sn in STATIONS_ALL if any(sn in Pr['st'] for Pr in runs)]
if len(runs) > 1:
    fig, axs = plt.subplots(2, len(sns), figsize=(16, 6), sharex=True, squeeze=False)
    for Pr, c in zip(runs, ['tab:green', 'tab:purple', 'tab:orange', 'tab:brown'] * 3):
        c = RUN_COLORS.get(Pr['gtagex'], c)
        for col, sn in enumerate(sns):
            if sn not in Pr['st']:
                continue
            Dr = series(Pr, sn)
            lab = f'{Pr["gtagex"]} ({Pr["lt"]}, h {Pr["st"][sn]["h"]:.1f} m)'
            axs[0, col].plot(Dr.time_utc, Dr.DO_bot_mgL, '-', color=c, lw=0.6, label=lab)
            axs[1, col].plot(Dr.time_utc, Dr.drho, '-', color=c, lw=0.6, label=lab)
    for col, sn in enumerate(sns):
        axs[0, col].axhline(HYP, color='gray', ls=':', lw=1)
        axs[0, col].set_ylim(bottom=0)
        axs[0, col].set_title(f'{sn} ({STATIONS_ALL[sn][0]:.4f}, {STATIONS_ALL[sn][1]:.4f})', fontsize=10)
        axs[0, col].legend(loc='upper right', fontsize=7)
        axs[1, col].set_ylim(bottom=0)
    axs[0, 0].set_ylabel('Bottom DO [mg/L]')
    axs[1, 0].set_ylabel(r'$\Delta\sigma_0$ bottom - top [kg m$^{-3}$]')
    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'compare.png', dpi=200, transparent=True)
    plt.close(fig)
print(f'saved  : {out_dir}')
