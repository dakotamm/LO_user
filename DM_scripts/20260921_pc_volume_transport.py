"""
Penn Cove volume transport, subtidal and tidal, two geometries: in the section
plane (from the tef2 section extractions) and in plan view depth-integrated
(from the pc_cove box). Annual mean plus the four seasons.

WHAT IS PLOTTED, and in what units

  sections   transport per unit area, sum_c of it * dA_c = the section volume
             transport. Dimensionally a velocity, so it is drawn in cm s-1, but
             it is a transport density: what you integrate, not what a current
             meter at a point reads. Each panel is annotated with the
             section-integrated volume transport in m3 s-1 so the actual
             transport is on the figure, not just its shape.

  maps       depth-integrated transport per unit width, U = int u dz [m2 s-1].
             Multiply by a cell width and you have m3 s-1 through that face.
             This is "depth-averaged transport" written so that it integrates
             correctly; the depth-mean VELOCITY U/D is in the report instead,
             because dividing by D flatters the shallow flanks.

SUBTIDAL vs TIDAL
    subtidal   <q>, the Godin average
    tidal      q - <q>, summarised as its RMS over the averaging period, since
               its mean is zero by construction. For a single clean constituent
               amplitude = sqrt(2) * RMS; the Penn Cove tide is mixed, so RMS is
               the honest summary and the sqrt(2) is not applied.
    On the maps the tidal part also gets a principal axis: the major axis of the
    covariance of (U', V') over the period, drawn headless and to scale, which
    is the direction the tidal transport actually oscillates along.

THE ONE RULE (same as 20260916_exchange_fun.py, and [[lowpassed-transport-stokes]])
Tidal averaging is applied to the TRANSPORT, never to a velocity and a
thickness separately. For the sections q is already stored hourly as Huon/Hvom,
so <q> carries the <u' dz'> Stokes correlation. For the box, u and z_w are
stored hourly and U = sum_k u dz is formed HOURLY and filtered after; forming
<u> and <dz> first and multiplying throws that term away. The report prints the
size of the difference so the choice is auditable, along with a check of U
against ROMS' own ubar * D.

SIGN
  sections   positive is INTO Penn Cove, via xfun.INFLOW_SIGN. Because all three
             pc sections are u-face lines whose own positive direction points
             east, i.e. out of the cove, that is a flip of the stored q.
  maps       geographic. u eastward, v northward, untouched.
  section x  km north of the section centre, so north is to the RIGHT and the
             panels are a view looking landward (west, up-cove), matching
             20260819_pc_section_tidal_movie.py.

WALLS. Masked velocity faces are set to zero, not NaN, before anything is
averaged or interpolated to rho points: zero normal flow is the correct value
at a wall, and a NaN there would eat the whole coastal cell.

FIGURE SIZING. Everything here is drawn to be read at 800 px wide -- that is
roughly half the saved pixel width at dpi 200, so fonts are set large and no
figure has more than three columns.

run 20260921_pc_volume_transport.py
run 20260921_pc_volume_transport.py -tchunk 1000
"""
import argparse
import importlib.util
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-job', default='pc_cove', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-sect', default='pc_cp,pc_lj,pc_lp', type=str,
               help='comma separated, ordered landward -> seaward')
p.add_argument('-tchunk', default=1500, type=int,
               help='hours of box data to read at a time')
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20260921_pc_volume_transport'
Lfun.make_dir(out_dir)

_fn = Path(__file__).parent / '20260916_exchange_fun.py'
_spec = importlib.util.spec_from_file_location('exchange_fun', _fn)
xfun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xfun)

SECTS = [s.strip() for s in args.sect.split(',') if s.strip()]
SLAB = {'pc_cp': 'pc_cp  (head, Coupeville)', 'pc_lj': 'pc_lj  (mid cove)',
        'pc_lp': 'pc_lp  (mouth)'}

# ---------------------------------------------------------------- house style
CB = dict(blue='#0072B2', red='#CC0000', green='#009E73', orange='#D55E00',
          purple='#CC79A7', grey='#7f7f7f')
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
LAND = '#d9d9d9'
CM_MAG = 'YlOrRd'      # magnitudes: white -> red, black arrows stay legible
CM_DIV = 'RdBu_r'      # signed transport, blue out / red in
LAT0 = 48.23

# Readable at 800 px in a markdown document. Saved at dpi 200, so a figure
# 8.4 in wide is 1680 px and displays at ~48%; these sizes are chosen for how
# they land AFTER that shrink, not for how they look full size.
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'legend.fontsize': 10.5,
                     'figure.titlesize': 15})
FIGW = 8.4

# Calendar seasons. The annual mean comes first and is given the big panel;
# earlier Penn Cove work (20260819_pc_planview_circulation.py) used data-derived
# Nov-Dec / Aug-Oct extremes instead, which is the sharper contrast but is not
# what "seasonal averages" means, so the standard four are used here.
SEASONS = [('annual', None),
           ('winter (DJF)', [12, 1, 2]), ('spring (MAM)', [3, 4, 5]),
           ('summer (JJA)', [6, 7, 8]), ('autumn (SON)', [9, 10, 11])]
SNAMES = [s for s, _ in SEASONS]


def season_mask(tt, months):
    return np.ones(len(tt), bool) if months is None else np.isin(tt.month, months)


# =============================================================== SECTIONS ====
print('=== sections ===')
SEC = dict()
sec_rows = []
for sn in SECTS:
    S = xfun.load_section(sn, args.gtagex, args.ds0, args.ds1, Ldir=Ldir)
    tt = pd.to_datetime(S['time'])
    q = S['q']                      # (NT, NZ, NP), + into the cove
    dA = S['dA']
    NT, NZ, NP = q.shape

    # THE ONE RULE: filter the transport itself.
    q_lp = zfun.lowpass(q, f='godin')
    dA_lp = zfun.lowpass(dA, f='godin')
    q_td = q - q_lp                 # NaN in the 35 h Godin pad at each end

    # what the wrong order would have given, for the report
    u_lp = zfun.lowpass(q / dA, f='godin')
    q_wrong = np.nansum(u_lp * dA_lp, axis=(1, 2))

    qsec = np.nansum(q, axis=(1, 2))            # hourly section transport
    qsec_lp = np.nansum(q_lp, axis=(1, 2))
    qsec_td = qsec - zfun.lowpass(qsec, f='godin', nanpad=False)

    # section geometry: p = 0 is the NORTH end (sect_df runs j downward), and
    # dd is uniform, so x increases southward with p. Flip it so north is right.
    dd = S['dd']
    xs = np.cumsum(dd) - dd / 2                 # metres from the north end
    x_km = -(xs - xs.mean()) / 1000.            # km north of section centre
    dz = np.nanmean(dA_lp, axis=0) / dd[None, :]
    zw = np.vstack([np.zeros(NP), np.cumsum(dz, axis=0)]) - S['h'][None, :]
    zr = 0.5 * (zw[:-1] + zw[1:])
    Xs = np.tile(x_km, (NZ, 1))

    d = dict(x=x_km, X=Xs, z=zr, h=S['h'], dd=dd, NP=NP, NZ=NZ)
    for name, months in SEASONS:
        m = season_mask(tt, months)
        A = np.nanmean(dA_lp[m], axis=0)
        sub = np.nanmean(q_lp[m], axis=0) / A           # m s-1, + into cove
        tid = np.sqrt(np.nanmean(q_td[m] ** 2, axis=0)) / A
        qc = np.nanmean(q_lp[m], axis=0)                # m3 s-1 per cell
        d[name] = dict(sub=sub, tid=tid,
                       Qin=np.nansum(qc[qc > 0]), Qout=np.nansum(qc[qc < 0]),
                       Qnet=np.nanmean(qsec_lp[m]),
                       Qprism=np.nanmean(np.abs(qsec_td[m])) / 2)
        sec_rows.append(dict(section=sn, period=name,
                             Qin=d[name]['Qin'], Qout=d[name]['Qout'],
                             Qnet=d[name]['Qnet'], Qprism=d[name]['Qprism'],
                             area_m2=np.nansum(A),
                             sub_max_cms=100 * np.nanmax(np.abs(sub)),
                             tid_rms_cms=100 * np.sqrt(np.nanmean(tid ** 2))))
    SEC[sn] = d
    print('  %-6s %d x %d, area %.0f m2, Qnet %.3f, Qprism %.0f m3/s'
          % (sn, NZ, NP, np.nansum(np.nanmean(dA_lp, axis=0)),
             d['annual']['Qnet'], d['annual']['Qprism']))
    print('       filter-then-sum %.4f vs sum-of-<u><dA> %.4f m3/s  ->  the '
          'Stokes term is %.4f' % (d['annual']['Qnet'], np.nanmean(q_wrong),
                                   d['annual']['Qnet'] - np.nanmean(q_wrong)))

SR = pd.DataFrame(sec_rows)
SR.to_csv(out_dir / 'section_transport.csv', index=False)

# ==================================================================== BOX ====
print('\n=== box (depth-integrated) ===')
box_fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'box' /
          ('%s_%s_%s.nc' % (args.job, args.ds0, args.ds1)))
ds = xr.open_dataset(box_fn)
lon, lat = ds.lon_rho.values, ds.lat_rho.values
mask = ds.mask_rho.values.astype(bool)
h = ds.h.values
ttb = pd.to_datetime(ds.ocean_time.values)
NR, NC = mask.shape
NTB = len(ttb)
print('  %s : %d h, %d x %d, %d wet' % (box_fn.name, NTB, NR, NC, mask.sum()))


def to_rho(Fu, Fv):
    """u,v on faces -> rho points; zero outside, one-sided at the box edge."""
    ur = np.zeros((Fu.shape[0], NR, NC))
    ur[:, :, 1:-1] = 0.5 * (Fu[:, :, :-1] + Fu[:, :, 1:])
    ur[:, :, 0], ur[:, :, -1] = Fu[:, :, 0], Fu[:, :, -1]
    vr = np.zeros((Fv.shape[0], NR, NC))
    vr[:, 1:-1, :] = 0.5 * (Fv[:, :-1, :] + Fv[:, 1:, :])
    vr[:, 0, :], vr[:, -1, :] = Fv[:, 0, :], Fv[:, -1, :]
    return ur, vr


# Pass 1: hourly depth-INTEGRATED transport per unit width, at rho points.
# Held as (NT, NR, NC) float64, ~95 MB each, so the Godin filter afterwards is
# a single in-memory operation. Only the vertical sum is done chunk by chunk.
U = np.zeros((NTB, NR, NC))
V = np.zeros((NTB, NR, NC))
Dh = np.zeros((NTB, NR, NC))            # water column depth at rho, h + zeta
Ubar = np.zeros((NTB, NR, NC))          # ROMS ubar * D, for the cross-check
Vbar = np.zeros((NTB, NR, NC))
for a in range(0, NTB, args.tchunk):
    b = min(a + args.tchunk, NTB)
    sl = dict(ocean_time=slice(a, b))
    zw = ds.z_w.isel(**sl).values
    DZ = np.diff(zw, axis=1)
    DZu = 0.5 * (DZ[:, :, :, :-1] + DZ[:, :, :, 1:])
    DZv = 0.5 * (DZ[:, :, :-1, :] + DZ[:, :, 1:, :])
    u = np.nan_to_num(ds.u.isel(**sl).values)          # wall -> 0
    v = np.nan_to_num(ds.v.isel(**sl).values)
    Uf = (u * DZu).sum(1)                              # m2 s-1
    Vf = (v * DZv).sum(1)
    U[a:b], V[a:b] = to_rho(Uf, Vf)
    Du = DZu.sum(1)
    Dv = DZv.sum(1)
    ub = np.nan_to_num(ds.ubar.isel(**sl).values) * Du
    vb = np.nan_to_num(ds.vbar.isel(**sl).values) * Dv
    Ubar[a:b], Vbar[a:b] = to_rho(ub, vb)
    Dh[a:b] = h[None] + ds.zeta.isel(**sl).values
    print('    %5d - %5d h' % (a, b), end='\r')
ds.close()
print('    read %d h                    ' % NTB)

ok = mask[None]
print('  check vs ROMS ubar*D : rms|U - ubar*D| = %.4g m2/s, on rms|U| = %.4g'
      % (np.sqrt(np.nanmean(((U - Ubar) ** 2)[np.broadcast_to(ok, U.shape)])),
         np.sqrt(np.nanmean((U ** 2)[np.broadcast_to(ok, U.shape)]))))

U_lp, V_lp = zfun.lowpass(U, f='godin'), zfun.lowpass(V, f='godin')
U_td, V_td = U - U_lp, V - V_lp
D_lp = zfun.lowpass(Dh, f='godin')

BOX = dict()
box_rows = []
for name, months in SEASONS:
    m = season_mask(ttb, months)
    Um, Vm = np.nanmean(U_lp[m], axis=0), np.nanmean(V_lp[m], axis=0)
    Dm = np.nanmean(D_lp[m], axis=0)
    # tidal covariance -> RMS magnitude and principal (major) axis
    uu = np.nanmean(U_td[m] ** 2, axis=0)
    vv = np.nanmean(V_td[m] ** 2, axis=0)
    uv = np.nanmean(U_td[m] * V_td[m], axis=0)
    tr, det = uu + vv, uu * vv - uv ** 2
    rt = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    lmaj, lmin = tr / 2 + rt, np.maximum(tr / 2 - rt, 0)
    th = 0.5 * np.arctan2(2 * uv, uu - vv)              # major-axis angle
    BOX[name] = dict(U=Um, V=Vm, D=Dm, mag=np.hypot(Um, Vm),
                     trms=np.sqrt(tr), maj=np.sqrt(lmaj), th=th,
                     ell=np.sqrt(lmin / np.maximum(lmaj, 1e-12)))
    box_rows.append(dict(period=name,
                         sub_mean_m2s=np.nanmean(BOX[name]['mag'][mask]),
                         sub_max_m2s=np.nanmax(BOX[name]['mag'][mask]),
                         sub_mean_cms=100 * np.nanmean((np.hypot(Um, Vm) / Dm)[mask]),
                         tid_mean_m2s=np.nanmean(BOX[name]['trms'][mask]),
                         tid_max_m2s=np.nanmax(BOX[name]['trms'][mask]),
                         tid_mean_cms=100 * np.nanmean((BOX[name]['trms'] / Dm)[mask]),
                         tid_over_sub=(np.nanmean(BOX[name]['trms'][mask])
                                       / np.nanmean(BOX[name]['mag'][mask]))))
BR = pd.DataFrame(box_rows)
BR.to_csv(out_dir / 'box_transport.csv', index=False)

# ================================================================= report ====
txt = ['PENN COVE VOLUME TRANSPORT -- %s, %s to %s' % (args.gtagex, args.ds0, args.ds1),
       '', 'SECTIONS, volume transport [m3 s-1], positive = INTO the cove.',
       'Qin/Qout are a sign split of the subtidal cell transports (a GROSS',
       'measure, see 20260916_exchange_fun.py); Qprism is the tidal transport',
       'amplitude, mean|q - <q>| / 2, same definition as bulk_calc_avg.py.', '',
       SR.to_string(index=False, float_format=lambda v: '%.3f' % v), '',
       'BOX, depth-integrated transport per unit width [m2 s-1] and the',
       'depth-mean velocity U/D [cm s-1] it corresponds to. Cove means are over',
       'wet rho cells only.', '',
       BR.to_string(index=False, float_format=lambda v: '%.3f' % v)]
print('\n' + '\n'.join(txt))
(out_dir / 'report.txt').write_text('\n'.join(txt) + '\n')

# ================================================================ figures ====
plt.close('all')
hv = np.where(mask, h, np.nan)
landv = np.where(mask, np.nan, 1.0)
ASP = 1 / np.cos(np.deg2rad(LAT0))


def basemap(ax):
    ax.pcolormesh(lon, lat, landv, cmap=plt.matplotlib.colors.ListedColormap([LAND]),
                  shading='nearest', zorder=0)
    ax.contour(lon, lat, hv, levels=[10, 20], colors='0.45', linewidths=0.5,
               alpha=0.7, zorder=3)
    ax.set_aspect(ASP)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('0.6')


def map_mosaic():
    """Annual across the top, the four seasons 2x2 under it. Panel heights are
    set by set_aspect, so the figure height only has to be generous enough not
    to squeeze the annual panel."""
    mos = [[SNAMES[0], SNAMES[0]], [SNAMES[1], SNAMES[2]], [SNAMES[3], SNAMES[4]]]
    fig, ax = plt.subplot_mosaic(mos, figsize=(FIGW, 9.6), layout='constrained',
                                 height_ratios=[1.9, 1, 1])
    return fig, ax


def map_panel(ax, k, name, stat):
    """Shared furniture: short bold title, stats inside, panel letter."""
    ax.set_title(name, fontsize=13 if k == 0 else 12, fontweight='bold', pad=4)
    ax.text(0.5, 0.965, stat, transform=ax.transAxes, ha='center', va='top',
            fontsize=10.5 if k == 0 else 9.5, zorder=6,
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.8))
    ax.text(0.015, 0.03, 'abcde'[k], transform=ax.transAxes, fontweight='bold',
            fontsize=13, zorder=6)


# ---- figure 1: subtidal depth-integrated transport ---------------------------
vmax = np.nanpercentile(np.concatenate([BOX[s]['mag'][mask] for s in SNAMES]), 99)
# arrow scale: a vector of size vmax spans ~1.2 grid cells, so neighbouring
# arrows do not run into each other. NC cells across the panel.
qscale = 1.2 * NC * vmax
fig, AX = map_mosaic()
for k, name in enumerate(SNAMES):
    ax, B = AX[name], BOX[name]
    basemap(ax)
    pc = ax.pcolormesh(lon, lat, np.where(mask, B['mag'], np.nan), cmap=CM_MAG,
                       vmin=0, vmax=vmax, shading='nearest', zorder=1)
    qv = ax.quiver(lon, lat, np.where(mask, B['U'], np.nan),
                   np.where(mask, B['V'], np.nan), color='k', scale=qscale,
                   width=0.004 if k == 0 else 0.006, zorder=4)
    map_panel(ax, k, name, 'cove mean %.2f, max %.2f m$^2$ s$^{-1}$'
              % (np.nanmean(B['mag'][mask]), np.nanmax(B['mag'][mask])))
    if k == 0:
        ax.quiverkey(qv, 0.135, 0.13, 0.5, r'0.5 m$^2$ s$^{-1}$',
                     labelpos='E', coordinates='axes', fontproperties={'size': 11})
fig.colorbar(pc, ax=[AX[s] for s in SNAMES], location='bottom', shrink=0.55,
             aspect=34, pad=0.015,
             label='|subtidal depth-integrated transport|  (m$^2$ s$^{-1}$)')
fig.suptitle('Penn Cove SUBTIDAL depth-integrated volume transport\n'
             r'$\mathbf{U}=\int \mathbf{u}\,dz$, Godin averaged; '
             'one colour and arrow scale for all five panels')
fig.savefig(out_dir / 'pc_map_subtidal.png', dpi=200, transparent=True,
            bbox_inches='tight')
plt.close(fig)
print('\nwrote pc_map_subtidal.png')

# ---- figure 2: tidal depth-integrated transport ------------------------------
tmax = np.nanpercentile(np.concatenate([BOX[s]['trms'][mask] for s in SNAMES]), 99)
# the bars are drawn out from the cell centre in BOTH directions, so a bar of
# size tmax has to be half as long as an arrow to take up the same room
ascale = 2.4 * NC * tmax
fig, AX = map_mosaic()
for k, name in enumerate(SNAMES):
    ax, B = AX[name], BOX[name]
    basemap(ax)
    pc = ax.pcolormesh(lon, lat, np.where(mask, B['trms'], np.nan), cmap=CM_MAG,
                       vmin=0, vmax=tmax, shading='nearest', zorder=1)
    # principal axis, drawn headless in both directions because the tidal
    # transport oscillates along it rather than pointing down it
    ex = np.where(mask, B['maj'] * np.cos(B['th']), np.nan)
    ey = np.where(mask, B['maj'] * np.sin(B['th']), np.nan)
    w = 0.0035 if k == 0 else 0.005
    for sg in (1, -1):
        qv = ax.quiver(lon, lat, sg * ex, sg * ey, color='k', scale=ascale,
                       width=w, headwidth=0, headlength=0, headaxislength=0,
                       zorder=4)
    map_panel(ax, k, name, 'cove mean %.2f, max %.2f m$^2$ s$^{-1}$'
              % (np.nanmean(B['trms'][mask]), np.nanmax(B['trms'][mask])))
    if k == 0:
        ax.quiverkey(qv, 0.135, 0.13, 0.5, r'0.5 m$^2$ s$^{-1}$ major axis',
                     labelpos='E', coordinates='axes', fontproperties={'size': 11})
fig.colorbar(pc, ax=[AX[s] for s in SNAMES], location='bottom', shrink=0.55,
             aspect=34, pad=0.015,
             label='RMS tidal depth-integrated transport  (m$^2$ s$^{-1}$)')
fig.suptitle('Penn Cove TIDAL depth-integrated volume transport\n'
             r"RMS of $\mathbf{U}-\langle\mathbf{U}\rangle$; bars are the "
             'principal axis, half-length drawn each way')
fig.savefig(out_dir / 'pc_map_tidal.png', dpi=200, transparent=True,
            bbox_inches='tight')
plt.close(fig)
print('wrote pc_map_tidal.png')


# ---- figures 3 and 4: the section plane --------------------------------------
def section_figure(key, cmap, sym, cbl, sup, fn, annot):
    vals = np.concatenate([SEC[sn][s][key].ravel() for sn in SECTS for s in SNAMES])
    lim = np.nanpercentile(np.abs(vals), 99.5) * 100
    fig, axs = plt.subplots(len(SNAMES), len(SECTS), figsize=(FIGW, 9.2),
                            layout='constrained', sharex='col', sharey=True)
    for r, name in enumerate(SNAMES):
        for c, sn in enumerate(SECTS):
            ax, d = axs[r, c], SEC[sn]
            F = 100 * d[name][key]
            pc = ax.pcolormesh(d['X'], d['z'], F, cmap=cmap,
                               vmin=-lim if sym else 0, vmax=lim,
                               shading='nearest')
            if sym:
                # the dividing line between inflow and outflow, which is what
                # the eye is looking for in these panels
                ax.contour(d['X'], d['z'], F, levels=[0], colors='k',
                           linewidths=0.8, alpha=0.65)
            ax.plot(d['x'], -d['h'], color='0.25', lw=1.2)
            ax.fill_between(d['x'], -d['h'], -32, color=LAND, zorder=2)
            ax.set_ylim(-28, 0.5)
            ax.text(0.035, 0.07, annot(d[name]), transform=ax.transAxes,
                    fontsize=10, zorder=3,
                    bbox=dict(fc='white', ec='none', alpha=0.78, pad=1.8))
            if r == 0:
                ax.set_title(SLAB.get(sn, sn), fontsize=11.5)
            if c == 0:
                ax.set_ylabel('%s\nz (m)' % name, fontsize=11.5,
                              fontweight='bold')
    fig.colorbar(pc, ax=axs, location='bottom', shrink=0.55, aspect=34,
                 pad=0.012, label=cbl)
    fig.supxlabel('km north of section centre  (north to the right; the view '
                  'looks up-cove)', fontsize=11.5)
    fig.suptitle(sup)
    fig.savefig(out_dir / fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % fn)


section_figure(
    'sub', CM_DIV, True,
    'subtidal transport per unit area  (cm s$^{-1}$, + into the cove)',
    'Penn Cove SUBTIDAL volume transport in the section plane\n'
    'red = into the cove, blue = out, black line = the divide; labels are the '
    'section-integrated transport in m$^3$ s$^{-1}$',
    'pc_sections_subtidal.png',
    lambda a: '$Q_{in}$ %.0f    $Q_{net}$ %+.2f' % (a['Qin'], a['Qnet']))

section_figure(
    'tid', CM_MAG, False,
    'RMS tidal transport per unit area  (cm s$^{-1}$)',
    'Penn Cove TIDAL volume transport in the section plane\n'
    'RMS of $q-\\langle q\\rangle$ per unit area; labels are the tidal prism '
    'transport $Q_{prism}$ in m$^3$ s$^{-1}$',
    'pc_sections_tidal.png',
    lambda a: '$Q_{prism}$ %.0f' % a['Qprism'])

# ---- figure 5: the integrated numbers ----------------------------------------
# Qin and Qprism share an axis on purpose: the point of the pair is that the
# tidal transport is the same size as the subtidal one at the head and only
# comparable to it at the mouth. Qnet gets its own axis because it is four
# orders of magnitude smaller -- Penn Cove has essentially no river.
fig, axs = plt.subplots(1, 3, figsize=(FIGW, 3.6), layout='constrained')
xb = np.arange(len(SNAMES))
w = 0.8 / len(SECTS)
COLS = [CB['blue'], CB['orange'], CB['green']]
for ax, (key, ttl, ylab) in zip(
        axs, [('Qin', 'subtidal inflow $Q_{in}$', 'm$^3$ s$^{-1}$'),
              ('Qprism', 'tidal prism $Q_{prism}$', 'm$^3$ s$^{-1}$'),
              ('Qnet', 'net $Q_{net}$', 'm$^3$ s$^{-1}$')]):
    for j, sn in enumerate(SECTS):
        ax.bar(xb + (j - 1) * w, [SEC[sn][s][key] for s in SNAMES], w,
               color=COLS[j], label=sn)
    ax.set_title(ttl, fontsize=12)
    ax.set_ylabel(ylab)
    ax.set_xticks(xb)
    ax.set_xticklabels(['ann', 'DJF', 'MAM', 'JJA', 'SON'], fontsize=10.5)
    ax.grid(**GRID, axis='y')
    ax.set_axisbelow(True)
    ax.axhline(0, color='k', lw=0.8)
ylim = 1.06 * max(SEC[sn][s][k] for sn in SECTS for s in SNAMES
                  for k in ['Qin', 'Qprism'])
axs[0].set_ylim(0, ylim)
axs[1].set_ylim(0, ylim)
axs[0].legend(fontsize=9.5, ncol=1, loc='upper left')
fig.suptitle('Section-integrated volume transport by season\n'
             'left two panels share a scale; $Q_{net}$ does not')
fig.savefig(out_dir / 'pc_section_integrated.png', dpi=200, transparent=True,
            bbox_inches='tight')
plt.close(fig)
print('wrote pc_section_integrated.png')
print('\nall output in %s' % out_dir)
