"""
How much temperature and salinity each contribute to stratification in Penn Cove, 2024,
wb1_t0_xn11abbur00, hourly pc_cove box. Built step by step.

STEP 1  hourly, every wet cove cell (mask_rho == 1, xi_rho <= 30, i.e. the cove without the
        Saratoga cells east of the mouth), top (s_rho -1) and bottom (s_rho 0):
            SA = SA_from_SP(salt, 0, lon, lat)   CT = CT_from_pt(SA, temp)   (as pc_strat_counterflow)
            dsig   = sigma0(bot) - sigma0(top)                       total, + = stable
            dsig_S =  rho beta  (SA_bot - SA_top)                    salt part
            dsig_T = -rho alpha (CT_bot - CT_top)                    temperature part
            resid  = dsig - dsig_S - dsig_T                          nonlinear remainder (cabbeling etc.)
        rho, alpha, beta from gsw.rho_alpha_beta at the column mid state ((top + bot)/2, p = 0).
        Cove mean = area-weighted (dA = 1/(pm pn)) over the wet cove cells. Cached to series.npz.

STEP 2  Godin filter the hourly cove means, one value per day at 12:00 UTC, kept for YEAR.
        S share = dsig_S / dsig  (> 1 when temperature is destabilizing, i.e. cold surface).

STEP 3  time series: dsig with its S and T parts; S share; top/bottom SA and CT.

Units: sigma kg/m3; SA g/kg; CT degC.
OUTPUT LO_output/DM_outs/20261117_pc_strat_TS/

run 20261117_pc_strat_TS.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gsw

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
YEAR = 2024
XI_MAX = 30                     # rho columns <= 30 are the cove (pc_cove_box_job)
LON, LAT = -122.69, 48.23
BLK = 744                       # hours per read block (the box is chunked one hour x whole box)
PAD = 72                        # hours read past the end of YEAR so the Godin filter has its tail

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_strat_TS'
out_dir.mkdir(parents=True, exist_ok=True)
cache_fn = out_dir / 'series.npz'
VARS = ['dsig', 'dsig_S', 'dsig_T', 'resid', 'SA_top', 'SA_bot', 'CT_top', 'CT_bot']


# ------------------------------------------------------------------ STEP 1 hourly cove means (cached)
if cache_fn.exists():
    z = np.load(cache_fn)
    T = pd.DatetimeIndex(z['T'])
    out = {v: z[v] for v in VARS}
    NCELL = int(z['ncell'])
    print('loaded ' + str(cache_fn))
else:
    ds = xr.open_dataset(in_fn)
    T_all = pd.DatetimeIndex(ds.ocean_time.values)
    it = np.flatnonzero((T_all.year == YEAR) | (T_all < pd.Timestamp(YEAR + 1, 1, 1) + pd.Timedelta(hours=PAD)))
    T = T_all[it]
    NT = len(T)
    cove = (ds.mask_rho.values == 1)
    cove[:, XI_MAX + 1:] = False
    dA = (1 / (ds.pm.values * ds.pn.values))[cove]
    wA = dA / dA.sum()
    NCELL = int(cove.sum())
    print('%d wet cove cells, %.1f km2, %d hours' % (cove.sum(), dA.sum() / 1e6, NT))

    out = {v: np.zeros(NT) for v in VARS}
    for t0 in range(0, NT, BLK):
        tt = slice(it[t0], it[min(t0 + BLK, NT) - 1] + 1)
        oo = slice(t0, min(t0 + BLK, NT))
        S = ds.salt.isel(ocean_time=tt, s_rho=[0, -1]).to_numpy().astype(float)[:, :, cove]   # (t, bot/top, cell)
        th = ds.temp.isel(ocean_time=tt, s_rho=[0, -1]).to_numpy().astype(float)[:, :, cove]
        SA = gsw.SA_from_SP(S, 0, LON, LAT)
        CT = gsw.CT_from_pt(SA, th)
        sig = gsw.sigma0(SA, CT)
        rho, alpha, beta = gsw.rho_alpha_beta(SA.mean(1), CT.mean(1), 0)       # column mid state
        dSA, dCT = SA[:, 0] - SA[:, 1], CT[:, 0] - CT[:, 1]                    # bottom - top
        cell = dict(dsig=sig[:, 0] - sig[:, 1], dsig_S=rho * beta * dSA, dsig_T=-rho * alpha * dCT,
                    SA_top=SA[:, 1], SA_bot=SA[:, 0], CT_top=CT[:, 1], CT_bot=CT[:, 0])
        cell['resid'] = cell['dsig'] - cell['dsig_S'] - cell['dsig_T']
        for v in VARS:
            out[v][oo] = cell[v] @ wA
        print('  hours %d-%d of %d' % (oo.start, oo.stop, NT), flush=True)
    ds.close()
    np.savez(cache_fn, T=T.values, ncell=NCELL, **out)
    print('saved ' + str(cache_fn))


# ------------------------------------------------------------------ STEP 2 Godin, daily at 12:00, YEAR only
lp = pd.DataFrame({v: zfun.lowpass(out[v], f='godin') for v in VARS}, index=T)
lp = lp[(lp.index.hour == 12) & (lp.index.year == YEAR)]
lp['S_share'] = lp.dsig_S / lp.dsig

print('\n%s Penn Cove, %d, Godin daily (n = %d, %d NaN days)' % (GTAGEX, YEAR, len(lp), lp.dsig.isna().sum()))
print('hourly |resid| max %.3f kg/m3 (rms %.3f), so the linear split holds'
      % (np.nanmax(np.abs(out['resid'])), np.sqrt(np.nanmean(out['resid'] ** 2))))
print('year mean  dsig %.2f = S %.2f + T %.2f + resid %.3f kg/m3   (S share %.2f)'
      % (lp.dsig.mean(), lp.dsig_S.mean(), lp.dsig_T.mean(), lp.resid.mean(), lp.dsig_S.mean() / lp.dsig.mean()))
mon = lp.groupby(lp.index.month)[['dsig', 'dsig_S', 'dsig_T']].mean()
mon['S_share'] = mon.dsig_S / mon.dsig
print('\nmonth   dsig  dsig_S  dsig_T  S share')
for m, r in mon.iterrows():
    print('%5d %6.2f %7.2f %7.2f %8.2f' % (m, r.dsig, r.dsig_S, r.dsig_T, r.S_share))
lp.to_csv(out_dir / ('pc_strat_TS_daily_%d.csv' % YEAR))


# ------------------------------------------------------------------ STEP 3 time series
C_S, C_T, C_TOT = 'tab:blue', 'tab:red', 'k'
t = lp.index

fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True, layout='constrained')

ax = axs[0]
ax.plot(t, lp.dsig, color=C_TOT, lw=1.2, label=r'$\Delta\sigma$ total')
ax.plot(t, lp.dsig_S, color=C_S, lw=1, label=r'$\Delta\sigma_S = \rho\beta\Delta S_A$')
ax.plot(t, lp.dsig_T, color=C_T, lw=1, label=r'$\Delta\sigma_T = -\rho\alpha\Delta\Theta$')
ax.axhline(0, color='0.4', lw=0.5)
ax.set_ylabel(r'bottom - top [kg m$^{-3}$]')
ax.set_title('Stratification and its salt / temperature parts')

ax = axs[1]
ax.plot(t, lp.S_share, color=C_S, lw=1, label=r'$\Delta\sigma_S / \Delta\sigma$')
ax.axhline(1, color='0.4', lw=0.5)
ax.axhline(0.5, color='0.4', lw=0.5, ls=':')
ax.set_ylabel('salt share')
ax.set_title('> 1: temperature destabilizing (cold surface);  < 1: warm surface adds to it')

ax = axs[2]
ax.plot(t, lp.SA_top, color=C_S, lw=1, label='top')
ax.plot(t, lp.SA_bot, color=C_S, lw=1, ls='--', label='bottom')
ax.set_ylabel(r'$S_A$ [g kg$^{-1}$]')

ax = axs[3]
ax.plot(t, lp.CT_top, color=C_T, lw=1, label='top')
ax.plot(t, lp.CT_bot, color=C_T, lw=1, ls='--', label='bottom')
ax.set_ylabel(r'$\Theta$ [$^\circ$C]')

for ax in axs:
    ax.grid(True, color='0.85', lw=0.5)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
loc = mdates.AutoDateLocator()
axs[-1].xaxis.set_major_locator(loc)
axs[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
axs[-1].set_xlim(pd.Timestamp(YEAR, 1, 1), pd.Timestamp(YEAR, 12, 31))
fig.suptitle('Penn Cove (area mean, %d wet cells), %s, %d, Godin daily' % (NCELL, GTAGEX, YEAR))

fn = out_dir / ('pc_strat_TS_%d.png' % YEAR)
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved', fn)
