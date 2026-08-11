"""
Average vertical shear across the two Penn Cove sections, pc_lp (the mouth) and
pc_cp (off Coupeville), as hourly time series over 2024-2025.

WHAT "AVERAGE SHEAR" MEANS HERE
The width-averaged normal-velocity profile at each hour is

    U(t,z) = sum_p q / sum_p (dd * DZ)                              [m s-1]

and the shear lives on the 29 interfaces between the 30 sigma levels

    Sh(t,k) = (U_{k+1} - U_k) / (0.5 * (dz_k + dz_{k+1}))           [s-1]

with dz_k(t) = Az(t,k) / W the width-mean thickness of layer k. Four scalars
per hour, because "average shear" is not one number:

    S_rms   thickness-weighted rms of Sh          magnitude, sign-blind
    S_abs   thickness-weighted mean |Sh|          same, less peak-sensitive
    S_bulk  (u_top - u_bot) / dz_centroid         SIGNED, top third vs bottom
                                                  third by column thickness
    S_max   largest |Sh| in the column, and the depth it sits at

and each is computed twice: on the raw hourly profile (total shear, tide
included) and on the Godin-filtered profile (SUBTIDAL shear -- the shear of
the exchange flow itself, after the tide is averaged out). The two are very
different quantities and the ratio subtidal/total says how much of the shear
is the estuarine circulation versus the tide swinging past.

SIGN. Positive q at both sections runs minus-side -> plus-side = eastward =
OUT of Penn Cove, so q is negated here: positive means INTO the cove. A
classic estuarine profile (surface out, bottom in) therefore gives S_bulk < 0.
Same convention as 20260811_pc_lp_baroclinic_barotropic.py.

WHAT IT SAYS (2024-2025, wb1_t0_xn11abbur00)
pc_cp is the sheared one: S_rms 0.0216 vs 0.0171 s-1, a factor 1.26, in a
column 4.5 m shallower. Both are stratification-controlled, not tidal --
corr(S_rms, dstrat) = +0.80 / +0.82 daily, corr with |U0| only +0.12 / +0.10,
and the spring-neap anomaly correlation is nil (-0.07 / +0.02). Same result
the baroclinic/barotropic script found at pc_lp, now confirmed inside the cove.
Seasonal minimum in September (0.011 / 0.015), maximum in December (0.024 /
0.031), tracking dstrat month for month. Only ~39 % of the shear magnitude
survives the Godin filter at either section, so most of it is tidal-band
motion; the SIGNED subtidal shear averages just -0.0005 / -0.0003 s-1, i.e. a
weakly estuarine mean profile that the fluctuating shear dwarfs. The largest
shear sits near the surface -- median depth of max |dU/dz| 1.9 m at pc_lp,
2.8 m at pc_cp -- not at the bed, so this is pycnocline/wind-layer shear, not
a bottom boundary layer. The two sections co-vary tightly (daily corr +0.95).

SPRING-NEAP. Compared as anomalies from a 30-day running mean, not raw --
the seasonal cycle otherwise buries the fortnightly signal entirely.

CAVEATS. u is the section-NORMAL component only, so all of this is a lower
bound on the true shear. Faces have different depths, so a sigma level is not
a fixed z across the section; the width-average mixes levels of unequal
thickness and "depth" in the profile panel is the record-mean depth of a
level. Shear is resolved at the model's 30 sigma levels only -- roughly 0.5 m
apart mid-column at these depths -- so it is a layer-scale shear, not a
turbulent one.

Runs on the mac from the local extractions_avg.
run 20260811_pc_shear.py
"""
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
ex_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
st_fn = tef2 / ('strat_%s_%s_%s.nc' % (args.ds0, args.ds1, args.gctag))
fx_fn = tef2 / ('hourly_flux_%s_%s_%s.nc' % (args.ds0, args.ds1, args.gctag))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_shear'
Lfun.make_dir(out_dir)

CB = dict(blue='#0072B2', red='#CC0000', green='#009E73', orange='#D55E00',
          purple='#CC79A7', grey='#7f7f7f')
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 13
SECTS = ['pc_lp', 'pc_cp']
SCOL = {'pc_lp': CB['blue'], 'pc_cp': CB['orange']}
SLAB = {'pc_lp': 'pc_lp (mouth)', 'pc_cp': 'pc_cp (Coupeville)'}


def godin(a):
    return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def godin2d(A):
    """Godin filter down the time axis of a (time, z) array."""
    return np.column_stack([godin(A[:, k]) for k in range(A.shape[1])])


def shear_metrics(Uz, dz, k_bot, k_top, area_bot, area_top, dz_cent):
    """All four scalars from a (time, z) profile and its layer thicknesses."""
    w = 0.5 * (dz[:, :-1] + dz[:, 1:])                 # interface thickness
    sh = np.diff(Uz, axis=1) / w                       # (time, nz-1)
    wsum = w.sum(axis=1)
    out = dict(
        S_rms=np.sqrt((w * sh ** 2).sum(axis=1) / wsum),
        S_abs=(w * np.abs(sh)).sum(axis=1) / wsum,
        S_max=np.abs(sh).max(axis=1),
        kmax=np.abs(sh).argmax(axis=1))
    # signed bulk shear from thickness-weighted top-third and bottom-third means
    ub = (Uz[:, k_bot] * area_bot).sum(axis=1) / area_bot.sum(axis=1)
    ut = (Uz[:, k_top] * area_top).sum(axis=1) / area_top.sum(axis=1)
    out['u_top'] = ut
    out['u_bot'] = ub
    out['S_bulk'] = (ut - ub) / dz_cent
    return out, sh


# ---------------------------------------------------------------------------
# load, decompose, and measure both sections
# ---------------------------------------------------------------------------
SER, SHPROF, META = {}, {}, {}
for sect in SECTS:
    ds = xr.open_dataset(ex_dir / (sect + '.nc'))
    tt = pd.to_datetime(ds.time.values)
    dd, hh = ds.dd.values, ds.h.values
    q = -ds.q.values                                   # + = INTO the cove
    A = ds.DZ.values * dd[None, None, :]               # (time, z, p)
    ds.close()

    W = dd.sum()
    Az = A.sum(axis=2)                                 # (time, z) layer area
    Atot = Az.sum(axis=1)
    Uz = q.sum(axis=2) / Az                            # width-mean u per level
    dz = Az / W                                        # width-mean thickness
    U0 = q.sum(axis=(1, 2)) / Atot                     # barotropic

    # fixed top/bottom thirds, defined once from the record-mean thicknesses
    lay = dz.mean(axis=0)
    frac = (np.cumsum(lay) - 0.5 * lay) / lay.sum()    # 0 at bed, 1 at surface
    k_bot = np.where(frac <= 1 / 3)[0]
    k_top = np.where(frac >= 2 / 3)[0]
    Hm = lay.sum()
    zc = Hm * frac                                     # level centre above bed
    dz_cent = (np.average(zc[k_top], weights=lay[k_top])
               - np.average(zc[k_bot], weights=lay[k_bot]))

    Azg = godin2d(Az)                                  # filtered weights too
    tot, sh_tot = shear_metrics(Uz, dz, k_bot, k_top,
                                Az[:, k_bot], Az[:, k_top], dz_cent)
    sub, sh_sub = shear_metrics(godin2d(Uz), godin2d(dz), k_bot, k_top,
                                Azg[:, k_bot], Azg[:, k_top], dz_cent)

    S = pd.DataFrame(index=tt)
    for k, v in tot.items():
        S[k] = v
    for k, v in sub.items():
        S[k + '_sub'] = v
    S['U0'] = U0
    S['zmax'] = Hm - 0.5 * (zc[tot['kmax']] + zc[tot['kmax'] + 1])  # below surf
    for c in ['S_rms', 'S_abs', 'S_bulk', 'S_max']:
        S[c + '_lp'] = godin(S[c].values)              # Godin of the magnitude

    SER[sect] = S
    SHPROF[sect] = dict(z=Hm - 0.5 * (zc[:-1] + zc[1:]),   # interface depth
                        abs_tot=np.abs(sh_tot).mean(axis=0),
                        abs_sub=np.nanmean(np.abs(sh_sub), axis=0),
                        mean_sub=np.nanmean(sh_sub, axis=0),
                        U=Uz.mean(axis=0), zU=Hm - zc)
    META[sect] = dict(NP=len(dd), Hm=Hm, hmin=hh.min(), hmax=hh.max(),
                      W=W, dz_cent=dz_cent, nbot=len(k_bot), ntop=len(k_top))
    print('%s: %d faces, h %.1f-%.1f m, mean depth %.1f m, width %.0f m, '
          'bulk shear over %.1f m (%d bottom / %d top levels)'
          % (sect, META[sect]['NP'], hh.min(), hh.max(), Hm, W, dz_cent,
             len(k_bot), len(k_top)))

# context: stratification and the tide
dsrt = xr.open_dataset(st_fn)
dfx = xr.open_dataset(fx_fn)
tt = pd.to_datetime(dsrt.time.values)
for sect in SECTS:
    SER[sect]['dstrat'] = godin(dsrt.dstrat.sel(sect=sect).values)
    ssh = pd.Series(dfx.ssh.sel(sect=sect).values, index=tt)
    # tidal range: daily peak-to-trough of ssh, as a spring-neap index
    SER[sect]['trange'] = (ssh.resample('D').max() - ssh.resample('D').min()
                           ).reindex(tt, method='ffill').values
dsrt.close()
dfx.close()

# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
rows = []
for sect in SECTS:
    S = SER[sect]
    r = dict(sect=sect, faces=META[sect]['NP'], H_m=META[sect]['Hm'])
    for c in ['S_rms', 'S_abs', 'S_max']:
        r[c + '_mean'] = S[c].mean()
        r[c + '_p90'] = S[c].quantile(0.9)
    r['S_bulk_mean'] = S.S_bulk.mean()
    r['S_bulk_absmean'] = S.S_bulk.abs().mean()
    r['S_rms_sub_mean'] = S.S_rms_sub.mean()
    r['S_bulk_sub_mean'] = S.S_bulk_sub.mean()
    r['subtidal_frac'] = S.S_rms_sub.mean() / S.S_rms.mean()
    r['zmax_median_m'] = S.zmax.median()
    r['du_top_bot_mean'] = (S.u_top - S.u_bot).mean()
    rows.append(r)
SUM = pd.DataFrame(rows).set_index('sect')
print('\n--- record summary (shear in s-1, depths in m) ---')
print(SUM.round(4).to_string())

print('\n--- what the shear tracks (Godin-filtered, daily) ---')
for sect in SECTS:
    D = SER[sect].resample('D').mean().dropna()
    print('  %-6s corr(S_rms, dstrat) %+.2f | corr(S_rms, |U0|) %+.2f | '
          'corr(S_rms, tidal range) %+.2f'
          % (sect, D.S_rms_lp.corr(D.dstrat), D.S_rms_lp.corr(D.U0.abs()),
             D.S_rms_lp.corr(D.trange)))
    a = D.S_rms_lp - D.S_rms_lp.rolling(30, center=True).mean()
    b = D.trange - D.trange.rolling(30, center=True).mean()
    print('         spring-neap, as 30-day anomalies: corr %+.2f (n=%d)'
          % (a.corr(b), a.notna().sum()))

L, C = SER['pc_lp'].resample('D').mean(), SER['pc_cp'].resample('D').mean()
print('\n--- section vs section (daily) ---')
print('  corr(S_rms) %+.2f | corr(S_bulk_sub) %+.2f | ratio of mean S_rms '
      'pc_cp/pc_lp %.2f'
      % (L.S_rms_lp.corr(C.S_rms_lp), L.S_bulk_sub.corr(C.S_bulk_sub),
         SUM.loc['pc_cp', 'S_rms_mean'] / SUM.loc['pc_lp', 'S_rms_mean']))

print('\n--- monthly mean S_rms (s-1), Godin-filtered ---')
MO = pd.DataFrame({s: SER[s].S_rms_lp.groupby(SER[s].index.month).mean()
                   for s in SECTS})
MO['dstrat_lp'] = SER['pc_lp'].dstrat.groupby(SER['pc_lp'].index.month).mean()
MO.index.name = 'month'
print(MO.round(4).to_string())

for sect in SECTS:
    SER[sect].to_csv(out_dir / ('shear_hourly_%s.csv' % sect),
                     float_format='%.6f')
    SER[sect].resample('D').mean().to_csv(
        out_dir / ('shear_daily_%s.csv' % sect), float_format='%.6f')
SUM.to_csv(out_dir / 'shear_summary.csv', float_format='%.6f')
MO.to_csv(out_dir / 'shear_monthly.csv', float_format='%.6f')

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 9.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05], hspace=0.42, wspace=0.45)

# a. total shear magnitude, hourly behind the Godin filter
ax = fig.add_subplot(gs[0, :])
for sect in SECTS[::-1]:
    S = SER[sect]
    ax.plot(S.index, S.S_rms, color=SCOL[sect], lw=0.25, alpha=0.10)
for sect in SECTS[::-1]:
    S = SER[sect]
    ax.plot(S.index, S.S_rms_lp, color=SCOL[sect], lw=1.6, label=SLAB[sect])
ax.set_ylabel('$S_{rms}$ [s$^{-1}$]', fontsize=FS)
ax.set_title('a. total vertical shear magnitude (hourly, with Godin filter)',
             fontsize=FS, loc='left')
ax.legend(fontsize=FS - 3, ncol=2, frameon=False)
ax.grid(**GRID)

# b. subtidal shear of the exchange flow, signed
ax = fig.add_subplot(gs[1, :])
for sect, lw in zip(SECTS[::-1], [1.6, 1.0]):
    S = SER[sect]
    ax.plot(S.index, S.S_bulk_sub, color=SCOL[sect], lw=lw, alpha=0.85,
            label=SLAB[sect])
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('$S_{bulk}$, subtidal [s$^{-1}$]', fontsize=FS)
ax.set_title('b. signed shear of the subtidal exchange flow  '
             '(negative = surface out / bottom in, i.e. estuarine)',
             fontsize=FS, loc='left')
ax.legend(fontsize=FS - 3, ncol=2, frameon=False)
ax.grid(**GRID)

# c. where in the column the shear sits
ax = fig.add_subplot(gs[2, 0])
for sect in SECTS:
    P = SHPROF[sect]
    ax.plot(P['abs_tot'], P['z'], color=SCOL[sect], lw=2.0, label=SLAB[sect])
    ax.plot(P['abs_sub'], P['z'], color=SCOL[sect], lw=1.4, ls='--')
ax.invert_yaxis()
ax.set_xlabel('mean |$\\partial U/\\partial z$| [s$^{-1}$]', fontsize=FS)
ax.set_ylabel('depth below surface [m]', fontsize=FS)
ax.set_title('c. shear profile\n(solid total, dashed subtidal)',
             fontsize=FS, loc='left')
ax.legend(fontsize=FS - 4, frameon=False)
ax.grid(**GRID)

# d. seasonal cycle
ax = fig.add_subplot(gs[2, 1])
for sect in SECTS:
    ax.plot(MO.index, MO[sect], color=SCOL[sect], lw=2.0, marker='o', ms=4,
            label=SLAB[sect])
ax2 = ax.twinx()
ax2.plot(MO.index, MO.dstrat_lp, color=CB['grey'], lw=1.4, ls=':')
ax2.tick_params(labelsize=FS - 4, colors=CB['grey'])
ax2.annotate('$\\Delta s$ pc_lp', xy=(0.05, 0.88), xycoords='axes fraction',
             color=CB['grey'], fontsize=FS - 3)
ax.set_xlabel('month', fontsize=FS)
ax.set_ylabel('$S_{rms}$ [s$^{-1}$]', fontsize=FS)
ax.set_title('d. seasonal cycle', fontsize=FS, loc='left')
ax.set_xticks(range(1, 13, 2))
ax.grid(**GRID)

# e. spring-neap, as 30-day anomalies
ax = fig.add_subplot(gs[2, 2])
for sect in SECTS:
    D = SER[sect].resample('D').mean().dropna()
    a = (D.S_rms_lp - D.S_rms_lp.rolling(30, center=True).mean())
    b = (D.trange - D.trange.rolling(30, center=True).mean())
    m = a.notna() & b.notna()
    ax.scatter(b[m], a[m], s=5, alpha=0.35, color=SCOL[sect],
               label='%s  r=%+.2f' % (sect, a[m].corr(b[m])))
    c = np.polyfit(b[m], a[m], 1)
    xg = np.linspace(b[m].min(), b[m].max(), 10)
    ax.plot(xg, np.polyval(c, xg), color=SCOL[sect], lw=1.6)
ax.axhline(0, color='k', lw=0.6)
ax.axvline(0, color='k', lw=0.6)
ax.set_xlabel('tidal range anomaly [m]', fontsize=FS)
ax.set_ylabel('$S_{rms}$ anomaly [s$^{-1}$]', fontsize=FS)
ax.set_title('e. spring-neap\n(30-day anomalies)', fontsize=FS, loc='left')
ax.legend(fontsize=FS - 4, frameon=False)
ax.grid(**GRID)

fig.suptitle('Vertical shear at the Penn Cove sections, %s  %s to %s'
             % (args.gtagex, args.ds0, args.ds1), fontsize=FS + 1)
fig.savefig(out_dir / 'pc_shear.png', dpi=200, bbox_inches='tight',
            transparent=True)
print('\nwrote %s' % (out_dir / 'pc_shear.png'))
plt.close(fig)
