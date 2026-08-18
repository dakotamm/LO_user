"""
The LATERAL OVERTURNING CELL in Penn Cove, seen directly, on the NATIVE C-GRID.

WHY THIS EXISTS
20260818_pc_lateral_baroclinic.py showed the mouth carries a depth-varying
lateral exchange (~390 m3/s, 35 % of the lateral exchange) controlled by
stratification and not at all by the tide -- a baroclinic lateral exchange. But
tef2 stores only the SECTION-NORMAL velocity, so that result is the lateral
structure of the ALONG-channel flow. The cross-channel (v) and vertical (w)
velocities that would close the cell were in no extraction. This script uses a
box extraction that carries them. On apogee:

  python extract_box.py -gtx wb1_t0_xn11abbur00 -ro 2 -0 2024.01.01 -1 2025.12.31 -lt hourly -job pc_cove

-ro 2 because wb1_t0_xn11abbur00 sits on /dat2/dakotamm/LO_roms (roms_out2),
the same path the tef2 files were ncrcat'd from. -ro 0 is the mac test bed.

DO NOT PASS -uv_to_rho. This script wants the native staggered grids, and the
interpolation actively breaks this analysis three ways:
  1. It leaves a NaN ring around the OUTERMOST row and column of the box
     (`uuu[:,:,1:-1,1:-1] = UU`). The pc_cove east edge IS the pc_lp column, so
     u and v at the mouth would come back entirely NaN.
  2. It zeros masked velocities BEFORE averaging, so a rho cell against the
     shoreline gets half the true face value -- exactly where the lateral
     cell's return flow lives.
  3. It breaks discrete continuity. On the C-grid dv/dy and dw/dz are the
     natural finite-difference pair; after interpolating v to rho they are not,
     so the continuity check below would partly measure regridding error.

WHAT IT COMPUTES
pc_lp, pc_lj and pc_cp are lines of constant longitude, so the lateral plane at
each is (y, z) with velocities (v, w) -- no rotation needed. At a rho column i:

  v lives on eta_v, one face BETWEEN each pair of rho rows: face k spans rho
  rows k and k+1. w lives on rho rows at s_w interfaces. That staggering is
  what makes the two diagnostics below exact rather than approximate.

  v'(k,z) = v(k,z) - (1/h_k) int v dz        depth mean removed at each face
  psi(k,z) = int_{-h}^{z} v' dz'             [m2 s-1], at (v-face, w-level)

psi is zero at bed and surface by construction, so its contours are CLOSED
streamlines and max|psi| is the lateral overturning strength. It is formed on
the model's own sigma layers from DZ_v = diff(z_w averaged onto the v-face), so
no vertical interpolation enters it.

THE HONESTY CHECK
psi is a 2-D streamfunction, meaningful only if the lateral plane is nearly
non-divergent (along-channel divergence small). At a cove mouth that is not
guaranteed. On the C-grid the exact discrete test at rho row j is

  div_y(j,z) = ( v(face j) - v(face j-1) ) / dy(j)
  w_2D(j,z)  = - int_{-h}^{z} div_y dz'

compared against the model's own w at the same rho row and w-levels. High r
means a genuine 2-D lateral overturning; low r means along-channel divergence
does the work and psi is a diagnostic, not a material circulation. Reported,
not assumed -- on the September 2017 wb1_r0_xn11b test bed this was weak, so
expect it to be the load-bearing caveat. The full (not depth-demeaned) v is
used here on purpose: the depth-mean part carries real divergence too.

MASKING. Masked v-faces are NaN in ROMS output and are set to 0 -- that is the
exact no-normal-flow wall condition, not a patch. mask_rho is taken from the
EXTRACTION, never from grid.nc: they disagree at 16 cells around Penn Cove,
see [[wb1-grid-vs-run-mask]].

TIDAL vs RESIDUAL. The cell is a residual feature, so v and DZ are Godin
filtered BEFORE the depth mean is removed and before integration, never after
-- see [[lowpassed-transport-stokes]] for what the wrong order costs.

SIGN. v is northward, w is up. Positive psi is a cell with northward flow near
the surface over southward flow near the bed.

run 20260818_pc_lateral_circulation.py
run 20260818_pc_lateral_circulation.py -gtx wb1_r0_xn11b -0 2017.09.05 -1 2017.09.17
"""
import argparse
import warnings

import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-job', default='pc_cove', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname='wb1')
box_fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'box' /
          ('%s_%s_%s.nc' % (args.job, args.ds0, args.ds1)))
tef_csv = (Ldir['LOo'] / 'DM_outs' / '20260818_pc_lateral_baroclinic' / 'daily_series.csv')
out_dir = Ldir['LOo'] / 'DM_outs' / '20260818_pc_lateral_circulation'
Lfun.make_dir(out_dir)

CB = dict(blue='#0072B2', red='#CC0000', green='#009E73', orange='#D55E00',
          purple='#CC79A7', grey='#7f7f7f')
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 13
STATIONS = [('pc_lp (mouth)', -122.6534), ('pc_lj (mid)', -122.6936),
            ('pc_cp (Coupeville)', -122.7218)]


def godin_t(A):
    """Godin filter down axis 0 of an array of any shape."""
    sh = A.shape
    B = A.reshape(sh[0], -1)
    G = np.full(B.shape, np.nan)
    for m in range(B.shape[1]):
        col = B[:, m]
        if np.isfinite(col).all():
            G[:, m] = zfun.lowpass(col.astype(float), f='godin')
    return G.reshape(sh)


def neff_r(x, y):
    r = np.corrcoef(x, y)[0, 1]
    ac = lambda a: np.corrcoef(a[:-1], a[1:])[0, 1]
    ne = max(len(x) * (1 - ac(x) * ac(y)) / (1 + ac(x) * ac(y)), 3.0)
    t = r * np.sqrt((ne - 2) / max(1 - r ** 2, 1e-12))
    return r, ne, 2 * stats.t.sf(abs(t), ne - 2)


# ------------------------------------------------------------------ load
ds = xr.open_dataset(box_fn)
if 'eta_v' not in ds.sizes:
    raise SystemExit('This box was made with -uv_to_rho; re-extract without it '
                     '(see the docstring for why).')
lon = ds.lon_rho.values[0, :]
lat_r = ds.lat_rho.values[:, 0]
lat_v = ds.lat_v.values[:, 0]
mask = ds.mask_rho.values.astype(bool)          # RUN mask, from the extraction
h = ds.h.values
pn = ds.pn.values
tt = pd.to_datetime(ds.ocean_time.values)
print('box %s : %d times, eta_rho %d, xi_rho %d, eta_v %d'
      % (box_fn.name, len(tt), *mask.shape, ds.sizes['eta_v']))

results, panels, series = [], [], {}
for name, xlon in STATIONS:
    i = int(np.argmin(np.abs(lon - xlon)))
    wetr = np.flatnonzero(mask[:, i])
    if len(wetr) < 4:
        print('  %s: only %d wet rho rows, skipped' % (name, len(wetr)))
        continue
    # v-face k spans rho rows k and k+1 -> keep faces inside the wet run
    k0, k1 = wetr.min(), wetr.max() - 1
    kk = np.arange(k0, k1 + 1)
    # isel keeps the read lazy; .values on the whole DataArray would load the box
    v = ds.v.isel(xi_v=i, eta_v=kk).values                      # (t, z, face)
    zwr = ds.z_w.isel(xi_rho=i).values                          # (t, s_w, eta_rho)
    zw_v = 0.5 * (zwr[:, :, kk] + zwr[:, :, kk + 1])            # z_w on the v-face
    DZv = np.diff(zw_v, axis=1)
    nwall = int((~np.isfinite(v)).any(axis=(0, 1)).sum())
    v = np.where(np.isfinite(v), v, 0.0)                        # wall: no normal flow
    y = (lat_v[kk] - lat_v[kk].mean()) * 111.32

    vG, DZG = godin_t(v), godin_t(DZv)
    ok = np.isfinite(vG).all(axis=(1, 2)) & np.isfinite(DZG).all(axis=(1, 2))
    print('  %s: %d wet rho rows, %d v-faces (%d wall), %d/%d times after Godin'
          % (name, len(wetr), len(kk), nwall, int(ok.sum()), len(tt)))
    vG, DZG = vG[ok], DZG[ok]

    vbar = (vG * DZG).sum(1) / DZG.sum(1)
    vp = vG - vbar[:, None, :]
    psi = np.cumsum(vp * DZG, axis=1)
    psi_m = psi.mean(0)
    kmax = np.unravel_index(np.abs(psi_m).argmax(), psi_m.shape)

    # ---- exact C-grid continuity check at interior rho rows
    jr = np.arange(k0 + 1, k1 + 1)                              # rho rows with a face each side
    dyr = 1.0 / pn[jr, i]
    div = (vG[:, :, jr - k0] - vG[:, :, jr - k0 - 1]) / dyr[None, None, :]
    DZr = godin_t(np.diff(zwr, axis=1)[:, :, jr])[ok]
    w2d = -np.cumsum(div * DZr, axis=1)                         # at w-levels 1..N
    wmod = godin_t(ds.w.isel(xi_rho=i, eta_rho=jr).values)[ok][:, 1:, :]
    a, b = w2d.ravel(), wmod.ravel()
    gd = np.isfinite(a) & np.isfinite(b)
    r_cont = np.corrcoef(a[gd], b[gd])[0, 1]
    amp = np.sqrt((a[gd] ** 2).mean()) / np.sqrt((b[gd] ** 2).mean())

    zm = (0.5 * (zw_v[ok][:, :-1, :] + zw_v[ok][:, 1:, :])).mean(0)
    dx_cell = 1.0 / ds.pm.isel(xi_rho=i, eta_rho=kk).values.mean()

    PSI_t = psi[:, kmax[0], kmax[1]]
    salt = ds.salt.isel(xi_rho=i, eta_rho=wetr).values[ok]
    zrr = ds.z_rho.isel(xi_rho=i, eta_rho=wetr).values[ok]
    SA = gsw.SA_from_SP(salt, -zrr, lon[i], lat_r[wetr].mean())
    temp = ds.temp.isel(xi_rho=i, eta_rho=wetr).values[ok]
    rho = gsw.rho(SA, gsw.CT_from_pt(SA, temp), 0.0)
    STRAT_t = np.nanmean(rho[:, 0, :] - rho[:, -1, :], axis=1)
    series[name] = pd.DataFrame(dict(psi_core=PSI_t, psi_absmax=np.abs(psi).max((1, 2)),
                                     strat=STRAT_t), index=tt[ok])

    results.append(dict(station=name, lon=lon[i], n_rho=len(wetr), n_vface=len(kk),
                        n_wall=nwall, width_km=float(y.max() - y.min()),
                        hmax=float(h[wetr, i].max()), psi_mean=float(psi_m[kmax]),
                        z_core=float(zm[kmax]), y_core=float(y[kmax[1]]),
                        Q_equiv=float(psi_m[kmax] * dx_cell),
                        r_continuity=float(r_cont), amp_ratio=float(amp),
                        vprime_rms=float(np.sqrt(np.nanmean(vp ** 2)))))
    panels.append((name, y, zm, psi_m, vp.mean(0)))

R = pd.DataFrame(results)
txt = ['PENN COVE LATERAL OVERTURNING (native C-grid) -- %s, %s to %s'
       % (args.gtagex, args.ds0, args.ds1), '',
       'psi_mean: time-mean subtidal lateral overturning streamfunction at its core (m2/s).',
       'Q_equiv: psi scaled by one along-channel cell (m3/s).',
       'r_continuity: model w vs w from lateral divergence alone (exact C-grid test) --',
       '  if low, the lateral plane is NOT non-divergent and psi is a diagnostic,',
       '  not a material circulation.', '',
       R.to_string(index=False, float_format=lambda v: '%.3f' % v)]

DD = {}
for name, sf in series.items():
    d = sf.resample('D').mean().dropna()
    DD[name] = d
    if len(d) < 120:
        continue
    txt += ['', '-- %s : what controls the cell (daily, Bretherton n_eff) --' % name]
    for tgt in ['psi_core', 'psi_absmax']:
        r, ne, pv = neff_r(d[tgt].values, d['strat'].values)
        txt.append('   corr(%s, stratification) = %+.3f  (n_eff %.0f, p = %.2g)'
                   % (tgt, r, ne, pv))
    mon = d.groupby(d.index.month)['psi_absmax'].mean()
    txt.append('   monthly mean |psi| max month %d (%.4f), min month %d (%.4f) m2/s'
               % (mon.idxmax(), mon.max(), mon.idxmin(), mon.min()))
    if tef_csv.is_file() and name.startswith('pc_lp'):
        tf = pd.read_csv(tef_csv, index_col=0, parse_dates=True)
        j = d.join(tf[['QLAT_BC']], how='inner').dropna()
        if len(j) > 120:
            r, ne, pv = neff_r(j['psi_absmax'].values, j['QLAT_BC'].values)
            txt.append('   corr(|psi|max, tef2 QLAT_BC) = %+.3f  (n_eff %.0f, p = %.2g)'
                       ' -- do the two views agree?' % (r, ne, pv))
report = '\n'.join(txt)
print('\n' + report)
(out_dir / 'report.txt').write_text(report + '\n')
R.to_csv(out_dir / 'lateral_cells.csv', index=False)
for name, d in DD.items():
    d.to_csv(out_dir / ('daily_%s.csv' % name.split(' ')[0]))

# ------------------------------------------------------------------ figure
n = len(panels)
long_rec = any(len(d) >= 120 for d in DD.values())
nrow = 3 if long_rec else 2
fig, axs = plt.subplots(nrow, n, figsize=(5.4 * n, 4.2 * nrow), squeeze=False)
for c, (name, y, zm, psi_m, vpm) in enumerate(panels):
    Y = np.tile(y, (zm.shape[0], 1))
    ax = axs[0, c]
    lim = np.abs(psi_m).max()
    pc = ax.pcolormesh(Y, zm, psi_m, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='gouraud')
    cs = ax.contour(Y, zm, psi_m, 9, colors='k', linewidths=0.7)
    ax.clabel(cs, fontsize=7, fmt='%.2f')
    plt.colorbar(pc, ax=ax, label=r'$\psi$ (m$^2$ s$^{-1}$)')
    ax.set_title('%s\nlateral overturning streamfunction' % name, fontsize=FS)
    ax.set_xlabel('distance north of section centre (km)')
    ax.set_ylabel('z (m)')

    ax = axs[1, c]
    lim = np.abs(100 * vpm).max()
    pc = ax.pcolormesh(Y, zm, 100 * vpm, cmap='PuOr_r', vmin=-lim, vmax=lim, shading='gouraud')
    ax.contour(Y, zm, vpm, [0], colors='k', linewidths=1.2)
    plt.colorbar(pc, ax=ax, label='cm s$^{-1}$')
    ax.set_title(r"cross-channel $v'$   (r$_{cont}$ = %+.2f)" % R.iloc[c].r_continuity,
                 fontsize=FS)
    ax.set_xlabel('distance north of section centre (km)')
    ax.set_ylabel('z (m)')

    if long_rec:
        ax = axs[2, c]
        d = DD[name]
        ax.plot(d['strat'], d['psi_absmax'], '.', ms=3, color=CB['purple'], alpha=0.5)
        bb = np.polyfit(d['strat'], d['psi_absmax'], 1)
        xs = np.linspace(d['strat'].min(), d['strat'].max(), 20)
        ax.plot(xs, np.polyval(bb, xs), color=CB['red'], lw=2)
        r, ne, pv = neff_r(d['psi_absmax'].values, d['strat'].values)
        ax.text(0.03, 0.96, 'r = %+.2f\nn$_{eff}$ = %.0f' % (r, ne),
                transform=ax.transAxes, va='top', fontsize=11)
        ax.grid(**GRID)
        ax.set_xlabel(r'stratification $\Delta\rho$ bottom-top (kg m$^{-3}$)')
        ax.set_ylabel(r'max$|\psi|$ (m$^2$ s$^{-1}$)')
        ax.set_title('is the cell density-driven?', fontsize=FS)

fig.suptitle('Penn Cove lateral circulation (native C-grid) -- %s  %s to %s'
             % (args.gtagex, args.ds0, args.ds1), fontsize=FS + 2)
fig.tight_layout()
fig.savefig(out_dir / 'pc_lateral_circulation.png', dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % out_dir)
