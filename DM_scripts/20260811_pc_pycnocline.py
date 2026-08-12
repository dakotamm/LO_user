"""
Average pycnocline depth at the two Penn Cove sections, pc_lp (the mouth) and
pc_cp (off Coupeville), as hourly time series over 2024-2025.

WHAT "AVERAGE PYCNOCLINE DEPTH" MEANS HERE
Per FACE, not on a width-averaged profile. The faces of a section have different
depths (pc_lp 15.9-26.8 m, pc_cp 14.9-21.5 m), so a sigma level is not a fixed z
across the section and averaging salt/temp across faces first would smear the
pycnocline before it is ever located. So each face gets its own profile, its own
pycnocline depth, and the section value is the width-weighted mean of those.

On each face and hour, potential density sigma0 comes from gsw (SP -> SA with the
section's mean lon/lat, pt -> CT), and the buoyancy frequency lives on the 29
interfaces between the 30 sigma levels

    N2_k = -(g/rho0) * (sigma_{k+1} - sigma_k) / dz_k        [s-2]

with dz_k = 0.5*(DZ_k + DZ_{k+1}) the interface thickness and d_k the depth of
that interface below the instantaneous free surface. The weight is the buoyancy
JUMP across the interface, w_k = max(N2_k, 0) * dz_k, i.e. only stable
interfaces vote. Three depths, because "the pycnocline depth" is not one number:

    z_pyc   w-weighted centroid of d          the primary series -- continuous,
                                              does not jump between sigma levels
    z_max   d at the single largest N2        the classic definition, quantized
                                              to the 30 levels
    h_pyc   w-weighted std of d about z_pyc   pycnocline THICKNESS, i.e. whether
                                              this is a sharp interface or a
                                              broad gradient

plus N2_max, the record of how sharp it is, and dsigma = sigma_bot - sigma_surf
per face. All are reported raw (hourly, tide included) and Godin-filtered.

WHEN THERE IS NO PYCNOCLINE. A depth computed from a nearly uniform column is
meaningless -- the centroid of noise sits mid-column by construction. Hours with
dsigma < 0.5 kg m-3 on a face are therefore flagged; the summary says how often
that happens and the series carry a `frac_strat` column (width fraction of the
section that is stratified) so a weakly stratified stretch can be greyed out
rather than read as a real mid-column pycnocline.

DEPTHS ARE BELOW THE INSTANTANEOUS SURFACE, positive down, so the series is
comparable to a CTD cast, not to a fixed z. The normalized depth z_pyc/H is
also carried, because pc_lp is ~5 m deeper than pc_cp and the two are only
comparable in absolute metres if you remember that.

WHAT IT SAYS (2024-2025, wb1_t0_xn11abbur00)
The pycnocline sits at 8.0 m at pc_lp and 7.8 m at pc_cp -- essentially the
same absolute depth at both sections, which means it is NOT a fixed fraction of
the column: 0.35 H at the 23 m mouth against 0.42 H in the 18.5 m cove. It is
set from above, by the surface layer, not scaled to the local depth. The two
sections move together almost perfectly (daily corr +0.98, pc_cp shallower by
0.22 m on average), so one series describes the whole cove.

The variability is subtidal, not tidal: sd of the Godin series is 2.0 / 1.7 m
against 1.0 / 1.3 m for the residual the filter removes. Seasonally it is
shallowest in July (6.4 / 6.5 m) and deepest in March (10.1 / 9.4 m) -- winter
mixing pushes it down, summer heating and the freshet pin it near the surface.
Note that March is deepest even though March is not the least stratified month;
depth and strength are only weakly related (corr(z_pyc, dsigma) -0.17 / -0.15,
corr with N2_max -0.44 / -0.34). A deeper pycnocline is a BROADER one
(corr(z_pyc, h_pyc) +0.56 at pc_lp), not a weaker one.

The column is essentially always stratified -- dsigma clears 0.5 kg m-3 over
99.8 % / 99.4 % of the section width -- so the weak-stratification flag never
actually bites here and panel a can be read straight through. The one event
that dominates panel b is mid-December 2025, when surface salinity at pc_lp
falls to ~8.6 and dsigma hits 16.6 kg m-3. That is a real freshet in the model,
not a numerical artifact, and it is the sharpest pycnocline of the record
(Godin peak 14 Dec: dsigma 16.5 kg m-3, N2_max 0.039 s-2, z_pyc 6.3 m).

CAVEATS. Resolution is the model's 30 sigma levels, ~0.5-0.9 m apart mid-column
at these depths, so z_pyc is a layer-scale estimate, not a fine-structure one.
sigma0 ignores the pressure dependence of density, which is the right call for
a 27 m column. The width average weights faces by dd only (all faces are the
same 200.5 m wide here), so a deep-channel face and a shoal face count equally;
weight by area instead if you want the volume-relevant answer.

Runs on the mac from the local extractions_avg.
run 20260811_pc_pycnocline.py
run 20260811_pc_pycnocline.py -year 2025
"""
import argparse
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gsw

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-year', default='all', type=str,
               help="calendar year to plot, or 'all' for the whole record")
p.add_argument('-dsig', default=0.5, type=float,
               help='dsigma [kg m-3] below which a face counts as unstratified')
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
ex_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
sect_fn = Ldir['LOo'] / 'extract' / 'tef2' / ('sect_df_%s.p' % args.gctag)
grid_fn = Ldir['data'] / 'grids' / args.gctag.split('_')[0] / 'grid.nc'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_pycnocline'
Lfun.make_dir(out_dir)

CB = dict(blue='#0072B2', red='#CC0000', green='#009E73', orange='#D55E00',
          purple='#CC79A7', grey='#7f7f7f')
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 13
SECTS = ['pc_lp', 'pc_cp']
SCOL = {'pc_lp': CB['blue'], 'pc_cp': CB['orange']}
SLAB = {'pc_lp': 'pc_lp (mouth)', 'pc_cp': 'pc_cp (Coupeville)'}
G, RHO0 = 9.81, 1025.0


def godin(a):
    return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def section_lonlat(sn):
    """Mean lon/lat of a section's faces, for the SP -> SA conversion."""
    sdf = pd.read_pickle(sect_fn)
    d = sdf[sdf.sn == sn]
    g = xr.open_dataset(grid_fn)
    lo = float(g.lon_rho.values[d.jrp, d.irp].mean())
    la = float(g.lat_rho.values[d.jrp, d.irp].mean())
    g.close()
    return lo, la


def pycnocline(sn):
    """Per-face pycnocline metrics and their width-weighted section means."""
    ds = xr.open_dataset(ex_dir / (sn + '.nc'))
    tt = pd.to_datetime(ds.time.values)
    dd, h = ds.dd.values, ds.h.values                  # (p,)
    zeta = ds.zeta.values                              # (time, p)
    DZ = ds.DZ.values                                  # (time, z, p)
    SP = ds.salt.values.astype(float)
    pt = ds.temp.values.astype(float)
    ds.close()

    # geometry: z of rho levels and of the interfaces between them
    z_w = -h[None, None, :] + np.cumsum(DZ, axis=1)     # top of each layer
    z_rho = z_w - 0.5 * DZ
    lon, lat = section_lonlat(sn)

    pres = gsw.p_from_z(z_rho, lat)
    SA = gsw.SA_from_SP(SP, pres, lon, lat)
    CT = gsw.CT_from_pt(SA, pt)
    sig = gsw.sigma0(SA, CT)                           # (time, z, p)

    # buoyancy frequency on the 29 interfaces
    dzi = 0.5 * (DZ[:, :-1, :] + DZ[:, 1:, :])
    N2 = -(G / RHO0) * np.diff(sig, axis=1) / dzi
    d_int = zeta[:, None, :] - z_w[:, :-1, :]          # depth below surface, + down

    w = np.clip(N2, 0.0, None) * dzi                   # buoyancy jump, stable only
    wsum = w.sum(axis=1)                               # (time, p)
    ok = wsum > 0
    zc = np.where(ok, np.divide((w * d_int).sum(axis=1), wsum,
                                out=np.zeros_like(wsum), where=ok), np.nan)
    var = np.where(ok, np.divide((w * (d_int - zc[:, None, :]) ** 2).sum(axis=1),
                                 wsum, out=np.zeros_like(wsum), where=ok), np.nan)
    kmax = N2.argmax(axis=1)                           # (time, p)
    ti, pi = np.indices(kmax.shape)
    zmx = np.where(ok, d_int[ti, kmax, pi], np.nan)
    n2mx = np.where(ok, N2[ti, kmax, pi], np.nan)

    H = h[None, :] + zeta                              # column depth (time, p)
    dsig = sig[:, 0, :] - sig[:, -1, :]                # bottom minus surface
    strat = dsig >= args.dsig

    # width-weighted section means; a depth is only averaged where it exists
    W = dd[None, :] * np.isfinite(zc)
    def wmean(A):
        m = np.isfinite(A)
        num = np.nansum(np.where(m, A, 0.0) * dd[None, :], axis=1)
        den = (m * dd[None, :]).sum(axis=1)
        return np.where(den > 0, num / den, np.nan)

    S = pd.DataFrame(index=tt)
    S['z_pyc'] = wmean(zc)
    S['h_pyc'] = wmean(np.sqrt(var))
    S['z_max'] = wmean(zmx)
    S['N2_max'] = wmean(n2mx)
    S['dsigma'] = wmean(np.where(np.isfinite(zc), dsig, np.nan))
    S['H'] = wmean(H)
    S['z_pyc_norm'] = S.z_pyc / S.H
    S['frac_strat'] = (strat * dd[None, :]).sum(axis=1) / dd.sum()
    # stratified faces only -- the depth you would quote if you refuse to
    # locate a pycnocline in a column that does not have one
    S['z_pyc_strat'] = wmean(np.where(strat, zc, np.nan))
    for c in ['z_pyc', 'h_pyc', 'z_max', 'N2_max', 'dsigma', 'z_pyc_norm',
              'z_pyc_strat']:
        S[c + '_lp'] = godin(S[c].values)

    # record-mean N2 profile, on the mean interface depth, for the profile panel
    prof = dict(z=np.nanmean(d_int, axis=(0, 2)),
                N2=np.nanmean(N2, axis=(0, 2)))
    meta = dict(NP=len(dd), W=dd.sum(), hmin=h.min(), hmax=h.max(),
                Hm=float(np.nanmean(H)), lon=lon, lat=lat)
    return S, prof, meta, (zc, dd, tt)


# ---------------------------------------------------------------------------
# both sections, plus the two of them together
# ---------------------------------------------------------------------------
SER, PROF, META, RAW = {}, {}, {}, {}
for sn in SECTS:
    SER[sn], PROF[sn], META[sn], RAW[sn] = pycnocline(sn)
    m = META[sn]
    print('%s: %d faces, h %.1f-%.1f m, mean column %.1f m, width %.0f m, '
          'at (%.4f, %.4f)'
          % (sn, m['NP'], m['hmin'], m['hmax'], m['Hm'], m['W'],
             m['lon'], m['lat']))

# the combined series: one width-weighted average over ALL faces of both
zc_all = np.concatenate([RAW[s][0] for s in SECTS], axis=1)
dd_all = np.concatenate([RAW[s][1] for s in SECTS])
msk = np.isfinite(zc_all)
BOTH = pd.DataFrame(index=RAW[SECTS[0]][2])
BOTH['z_pyc'] = (np.nansum(np.where(msk, zc_all, 0.0) * dd_all[None, :], axis=1)
                 / (msk * dd_all[None, :]).sum(axis=1))
BOTH['z_pyc_lp'] = godin(BOTH.z_pyc.values)

if args.year.lower() != 'all':
    yr = int(args.year)
    for sn in SECTS:
        SER[sn] = SER[sn][SER[sn].index.year == yr]
    BOTH = BOTH[BOTH.index.year == yr]
span_lbl = '2024-2025' if args.year.lower() == 'all' else args.year

# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
rows = []
for sn in SECTS:
    S = SER[sn]
    rows.append(dict(
        sect=sn, faces=META[sn]['NP'], H_m=META[sn]['Hm'],
        z_pyc_mean=S.z_pyc.mean(), z_pyc_med=S.z_pyc.median(),
        z_pyc_p10=S.z_pyc.quantile(0.1), z_pyc_p90=S.z_pyc.quantile(0.9),
        z_pyc_norm=S.z_pyc_norm.mean(), h_pyc_mean=S.h_pyc.mean(),
        z_max_mean=S.z_max.mean(), N2_max_mean=S.N2_max.mean(),
        dsigma_mean=S.dsigma.mean(),
        strat_frac=S.frac_strat.mean(),
        z_pyc_strat=S.z_pyc_strat.mean(),
        tidal_sd=(S.z_pyc - S.z_pyc_lp).std(), subtidal_sd=S.z_pyc_lp.std()))
SUM = pd.DataFrame(rows).set_index('sect')
print('\n--- record summary, %s (depths in m below the surface) ---' % span_lbl)
print(SUM.round(3).to_string())

print('\ncombined pc_lp + pc_cp: mean z_pyc %.2f m, median %.2f, '
      '10th-90th %.2f-%.2f m'
      % (BOTH.z_pyc.mean(), BOTH.z_pyc.median(),
         BOTH.z_pyc.quantile(0.1), BOTH.z_pyc.quantile(0.9)))

print('\n--- what the depth tracks (Godin-filtered, daily) ---')
for sn in SECTS:
    D = SER[sn].resample('D').mean().dropna()
    print('  %-6s corr(z_pyc, dsigma) %+.2f | corr(z_pyc, N2_max) %+.2f | '
          'corr(z_pyc, h_pyc) %+.2f'
          % (sn, D.z_pyc_lp.corr(D.dsigma_lp), D.z_pyc_lp.corr(D.N2_max_lp),
             D.z_pyc_lp.corr(D.h_pyc_lp)))
L, C = (SER[s].resample('D').mean() for s in SECTS)
print('  section vs section: corr(z_pyc) %+.2f | pc_cp minus pc_lp %+.2f m'
      % (L.z_pyc_lp.corr(C.z_pyc_lp), (C.z_pyc_lp - L.z_pyc_lp).mean()))

# the strongest-stratification event, because it sets the scale of panel b and
# is worth knowing about before reading a shallow pycnocline off panel a
for sn in SECTS:
    S = SER[sn]
    i = S.dsigma_lp.idxmax()
    print('  %-6s peak stratification %s: dsigma %.1f kg m-3, z_pyc %.1f m, '
          'N2_max %.3f s-2' % (sn, i.date(), S.dsigma_lp[i], S.z_pyc_lp[i],
                              S.N2_max_lp[i]))
    w = S.frac_strat < 1.0
    print('         width fully stratified %.1f%% of hours; mean stratified '
          'width fraction %.3f' % (100 * (~w).mean(), S.frac_strat.mean()))

print('\n--- monthly mean pycnocline depth [m], Godin-filtered ---')
MO = pd.DataFrame({s: SER[s].z_pyc_lp.groupby(SER[s].index.month).mean()
                   for s in SECTS})
for s in SECTS:
    MO[s + '_dsig'] = SER[s].dsigma_lp.groupby(SER[s].index.month).mean()
    MO[s + '_strat'] = SER[s].frac_strat.groupby(SER[s].index.month).mean()
MO.index.name = 'month'
print(MO.round(3).to_string())

for sn in SECTS:
    SER[sn].to_csv(out_dir / ('pycnocline_hourly_%s.csv' % sn),
                   float_format='%.5f')
    SER[sn].resample('D').mean().to_csv(
        out_dir / ('pycnocline_daily_%s.csv' % sn), float_format='%.5f')
BOTH.to_csv(out_dir / 'pycnocline_hourly_both.csv', float_format='%.5f')
SUM.to_csv(out_dir / 'pycnocline_summary.csv', float_format='%.5f')
MO.to_csv(out_dir / 'pycnocline_monthly.csv', float_format='%.5f')

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1.05], hspace=0.45,
                      wspace=0.42)

# a. the series itself, hourly behind its Godin filter, depth increasing down
ax = fig.add_subplot(gs[0, :])
for sn in SECTS[::-1]:
    ax.plot(SER[sn].index, SER[sn].z_pyc, color=SCOL[sn], lw=0.25, alpha=0.12)
for sn in SECTS[::-1]:
    ax.plot(SER[sn].index, SER[sn].z_pyc_lp, color=SCOL[sn], lw=1.7,
            label=SLAB[sn])
ax.plot(BOTH.index, BOTH.z_pyc_lp, color='k', lw=1.2, ls='--',
        label='both sections, width-weighted')
for sn in SECTS:
    ax.axhline(META[sn]['Hm'], color=SCOL[sn], lw=0.8, ls=':', alpha=0.7)
    ax.text(0.998, META[sn]['Hm'], 'mean bed, %s ' % sn, ha='right', va='bottom',
            fontsize=FS - 5, color=SCOL[sn],
            transform=ax.get_yaxis_transform())
ax.invert_yaxis()
ax.set_ylabel('pycnocline depth [m]', fontsize=FS)
ax.set_title('a. average pycnocline depth (hourly, with Godin filter); '
             'N$^2$-jump-weighted centroid, per face then width-averaged',
             fontsize=FS, loc='left')
ax.legend(fontsize=FS - 4, ncol=3, frameon=False, loc='lower left')
ax.grid(**GRID)

# b. is there a pycnocline at all? read panel a only where this is high
ax = fig.add_subplot(gs[1, :])
for sn in SECTS:
    ax.plot(SER[sn].index, SER[sn].dsigma_lp, color=SCOL[sn], lw=1.5,
            label=SLAB[sn])
ax.axhline(args.dsig, color=CB['grey'], lw=1.0, ls='--')
ax.text(0.002, args.dsig, ' weak-stratification cutoff, %.1f' % args.dsig,
        transform=ax.get_yaxis_transform(), va='bottom', fontsize=FS - 4,
        color=CB['grey'])
ax.set_ylabel('$\\Delta\\sigma_0$ bottom - surface\n[kg m$^{-3}$]', fontsize=FS)
ax.set_title('b. stratification, the context for panel a '
             '(a depth located in an unstratified column means nothing)',
             fontsize=FS, loc='left')
ax.legend(fontsize=FS - 4, ncol=2, frameon=False)
ax.grid(**GRID)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

# c. where the stratification sits in the mean
ax = fig.add_subplot(gs[2, 0])
for sn in SECTS:
    ax.plot(PROF[sn]['N2'], PROF[sn]['z'], color=SCOL[sn], lw=2.0,
            label=SLAB[sn])
    ax.axhline(SER[sn].z_pyc.mean(), color=SCOL[sn], lw=1.0, ls='--')
ax.invert_yaxis()
ax.set_xlabel('mean $N^2$ [s$^{-2}$]', fontsize=FS)
ax.set_ylabel('depth below surface [m]', fontsize=FS)
ax.set_title('c. mean $N^2$ profile\n(dashed = mean $z_{pyc}$)', fontsize=FS,
             loc='left')
ax.legend(fontsize=FS - 5, frameon=False)
ax.grid(**GRID)

# d. seasonal cycle
ax = fig.add_subplot(gs[2, 1])
for sn in SECTS:
    ax.plot(MO.index, MO[sn], color=SCOL[sn], lw=2.0, marker='o', ms=4,
            label=SLAB[sn])
ax.invert_yaxis()
ax2 = ax.twinx()
for sn in SECTS:
    ax2.plot(MO.index, MO[sn + '_dsig'], color=SCOL[sn], lw=1.2, ls=':')
ax2.tick_params(labelsize=FS - 4, colors=CB['grey'])
ax2.set_ylabel('$\\Delta\\sigma_0$ (dotted)', fontsize=FS - 3, color=CB['grey'])
ax.set_xlabel('month', fontsize=FS)
ax.set_ylabel('$z_{pyc}$ [m]', fontsize=FS)
ax.set_title('d. seasonal cycle', fontsize=FS, loc='left')
ax.set_xticks(range(1, 13, 2))
ax.grid(**GRID)

# e. how much of the spread is tidal
ax = fig.add_subplot(gs[2, 2])
for sn in SECTS:
    S = SER[sn]
    ax.hist(S.z_pyc.dropna(), bins=40, orientation='horizontal', density=True,
            color=SCOL[sn], alpha=0.35,
            label='%s\nhourly sd %.2f m' % (sn, S.z_pyc.std()))
    ax.hist(S.z_pyc_lp.dropna(), bins=40, orientation='horizontal',
            density=True, histtype='step', color=SCOL[sn], lw=1.6)
ax.invert_yaxis()
ax.set_xlabel('density', fontsize=FS)
ax.set_ylabel('$z_{pyc}$ [m]', fontsize=FS)
ax.set_title('e. distribution\n(filled hourly, line Godin)', fontsize=FS,
             loc='left')
ax.legend(fontsize=FS - 5, frameon=False)
ax.grid(**GRID)

fig.suptitle('Pycnocline depth at the Penn Cove sections, %s  %s'
             % (args.gtagex, span_lbl), fontsize=FS + 1)
fn_out = out_dir / ('pc_pycnocline_%s.png' % (args.year.lower()))
fig.savefig(fn_out, dpi=200, bbox_inches='tight', transparent=True)
print('\nwrote %s' % fn_out)
plt.close(fig)
