"""
Place modeled Penn Cove (wb1_t0_xn11abbur00) in the Geyer & MacCready (2014,
Annu. Rev. Fluid Mech. 46:175-197) estuarine parameter space, their Figure 6.

The two axes are

    Fr_f = U_R / c_0                         freshwater Froude number
    M    = sqrt( C_D U_T^2 / (om N_0 H^2) )  tidal mixing parameter

with c_0 = sqrt(beta g s_ocn H) the maximum internal wave speed, N_0 = c_0/H,
U_R = Q_R/A the river velocity through the mouth cross-section, U_T the
amplitude of the section-averaged tidal velocity, om the M2 frequency and
C_D = 2.5e-3.

The point of the exercise is that Penn Cove has no river. Its local freshwater
input in this run is two WWTPs totaling ~0.009 m3/s, so the literal Fr_f is
~1e-7 and the GM14 x-axis is undefined in any useful sense. So we compute
Fr_f two ways:

  'local'     Q_R = the actual freshwater discharged inside the cove.
  'knudsen'   Q_R = Q_in (s_in - s_out) / s_out at the mouth, i.e. the
              freshwater flux the exchange flow is actually carrying,
              whichever end of the cove it came from. For Penn Cove that is
              Skagit water imported through Saratoga Passage, not runoff.

M needs no river and is computed straight from the model.

Sections come from the wb1_pc1 tef2 collection. pc_lp is the cove mouth;
pc_cp (off Coupeville) is carried along as an inner-cove comparison. The
wb1_pc1 sign convention is positive = from the minus-side cell to the
plus-side cell, and upper Saratoga Passage (pc_lp_p) is the plus side of
pc_lp, so positive q at pc_lp is flow OUT of Penn Cove.

Run:
    python 20260818_gm2014_parameter_space.py
    python 20260818_gm2014_parameter_space.py --show-lit    # literature points
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lo_tools import Lfun, zfun

# ---- arguments ------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('--dates', default='2024.01.01_2025.12.31')
p.add_argument('--sects', default='pc_lp,pc_cp')
p.add_argument('--show-lit', action='store_true',
               help='overlay approximate GM14 Fig 6 estuary positions (eyeballed, see LIT)')
args = p.parse_args()

Ldir = Lfun.Lstart()
in_dir = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260818_gm2014_parameter_space'
Lfun.make_dir(out_dir)

sect_list = args.sects.split(',')

# ---- physical constants ---------------------------------------------------
BETA = 7.7e-4        # haline contraction coefficient, 1/(g/kg)
GRAV = 9.81
CD = 2.5e-3          # drag coefficient, GM14 value
OM_M2 = 1.4052e-4    # M2 radian frequency, 1/s

# Local freshwater sources inside Penn Cove for the trapsN00 forcing used by
# this run: COUPEVILLE STP + PENN COVE WWTP, both in segment pc_cp_p.
Q_LOCAL_FRESH = 0.008 + 0.001   # m3/s

# Approximate positions of the estuaries plotted in GM14 Fig 6. These are
# EYEBALLED from the published figure, not digitized -- they are here only to
# give the eye a scale and must be replaced with real values before any of
# this goes in a paper. Off by default.
LIT = {
    'Mississippi': (0.30, 0.18),
    'Fraser':      (0.15, 0.35),
    'Merrimack':   (0.04, 0.55),
    'Columbia':    (0.02, 0.90),
    'Hudson':      (5e-3, 0.90),
    'Chesapeake':  (1e-3, 1.10),
    'Delaware':    (3e-4, 1.60),
}


# ---- helpers --------------------------------------------------------------
def section_geometry(st, sect):
    """Cross-sectional area, width and mean depth of a tef2 section."""
    h = st[sect + '_h'].values
    dd = st[sect + '_dd'].values
    A = float(np.sum(h * dd))
    W = float(np.sum(dd))
    return A, W, A / W


def tef_two_layer(fn):
    """Collapse a multi-layer bulk file to Q_in/s_in/Q_out/s_out for the COVE.

    Positive q at pc_lp is out of the cove (see module docstring), so the
    q < 0 layers are the inflow.
    """
    ds = xr.open_dataset(fn)
    q = ds.q.values
    s = ds.salt.values
    time = pd.to_datetime(ds.time.values)

    qneg = np.where(q < 0, q, 0.0)
    qpos = np.where(q > 0, q, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        Qin = -np.nansum(qneg, axis=1)
        Qout = np.nansum(qpos, axis=1)
        sin_ = np.nansum(np.where(q < 0, q * s, 0.0), axis=1) / np.nansum(qneg, axis=1)
        sout = np.nansum(np.where(q > 0, q * s, 0.0), axis=1) / Qout

    df = pd.DataFrame({'Qin': Qin, 'Qout': Qout, 's_in': sin_, 's_out': sout,
                       'qprism': ds.qprism.values, 'qnet': ds.qnet.values},
                      index=time)
    # Knudsen freshwater flux implied by the exchange.
    df['Qr_knudsen'] = df.Qin * (df.s_in - df.s_out) / df.s_out
    ds.close()
    return df


def tidal_velocity(qnet, A):
    """Amplitude of the section-averaged tidal velocity from hourly transport.

    The subtidal part is removed with a Godin filter first (raw std would fold
    subtidal and spring/neap variability into what should be a tidal scale),
    then amplitude = sqrt(2) * rms of the residual.
    """
    u = qnet / A
    u_sub = zfun.lowpass(u, f='godin')
    u_tide = u - u_sub
    return u_tide


def gm14(Qr, U_T, H, s_ocn, A):
    """Geyer & MacCready (2014) Fr_f and M."""
    c0 = np.sqrt(BETA * GRAV * s_ocn * H)
    N0 = c0 / H
    U_R = Qr / A
    Fr_f = U_R / c0
    M = np.sqrt(CD * U_T ** 2 / (OM_M2 * N0 * H ** 2))
    return Fr_f, M, c0, N0


# ---- load -----------------------------------------------------------------
st = xr.open_dataset(in_dir / ('structure_%s_%s.nc' % (args.dates, args.coll)))
hf = xr.open_dataset(in_dir / ('hourly_flux_%s_%s.nc' % (args.dates, args.coll)))
sr = xr.open_dataset(in_dir / ('strat_%s_%s.nc' % (args.dates, args.coll)))

rows = []
monthly = {}

for sect in sect_list:
    A, W, H = section_geometry(st, sect)

    qnet_h = hf.qnet.sel(sect=sect).values
    t_h = pd.to_datetime(hf.time.values)
    u_tide = tidal_velocity(qnet_h, A)

    bulk = tef_two_layer(in_dir / ('bulk_avg_' + args.dates) / (sect + '.nc'))

    dstrat = pd.Series(sr.dstrat.sel(sect=sect).values, index=pd.to_datetime(sr.time.values))
    s_bot = pd.Series(sr.s_bot.sel(sect=sect).values, index=dstrat.index)
    s_top = pd.Series(sr.s_top.sel(sect=sect).values, index=dstrat.index)

    ut_ser = pd.Series(u_tide, index=t_h)

    # --- full-record values ---
    s_ocn = float(bulk.s_in.mean())
    U_T = np.sqrt(2) * float(ut_ser.std())
    Qr_kn = float(bulk.Qr_knudsen.mean())

    for tag, Qr in [('knudsen', Qr_kn), ('local', Q_LOCAL_FRESH)]:
        Fr_f, M, c0, N0 = gm14(Qr, U_T, H, s_ocn, A)
        rows.append(dict(sect=sect, Qr_kind=tag, period='full', A=A, W=W, H=H,
                         s_ocn=s_ocn, c0=c0, N0=N0, U_T=U_T, Qr=Qr,
                         U_R=Qr / A, Fr_f=Fr_f, M=M,
                         Qin=float(bulk.Qin.mean()),
                         ds_inout=float((bulk.s_in - bulk.s_out).mean()),
                         dstrat=float(dstrat.mean()),
                         ds_over_s=float(dstrat.mean() / ((s_top + s_bot).mean() / 2))))

    # --- monthly climatology ---
    mrec = []
    for m in range(1, 13):
        U_Tm = np.sqrt(2) * float(ut_ser[ut_ser.index.month == m].std())
        bm = bulk[bulk.index.month == m]
        s_ocnm = float(bm.s_in.mean())
        Qr_m = float(bm.Qr_knudsen.mean())
        Fr_f, M, c0, N0 = gm14(Qr_m, U_Tm, H, s_ocnm, A)
        dsm = dstrat[dstrat.index.month == m]
        mrec.append(dict(month=m, U_T=U_Tm, Qr=Qr_m, s_ocn=s_ocnm,
                         Fr_f=Fr_f, M=M, c0=c0,
                         Qin=float(bm.Qin.mean()),
                         ds_inout=float((bm.s_in - bm.s_out).mean()),
                         dstrat=float(dsm.mean())))
    monthly[sect] = pd.DataFrame(mrec).set_index('month')

DF = pd.DataFrame(rows)
DF.to_csv(out_dir / 'gm2014_parameters.csv', index=False)
for sect, mdf in monthly.items():
    mdf.to_csv(out_dir / ('gm2014_monthly_%s.csv' % sect))

# ---- report ---------------------------------------------------------------
pd.set_option('display.width', 200)
print('\n=== Geyer & MacCready 2014 parameters, %s ===' % args.gtx)
print(DF[['sect', 'Qr_kind', 'H', 'A', 's_ocn', 'c0', 'U_T', 'Qr', 'U_R',
          'Fr_f', 'M', 'ds_over_s']].to_string(index=False,
          float_format=lambda x: '%.4g' % x))
print('\nGM14 Fig 6 is drawn over roughly Fr_f in [1e-4, 1] and M in [0.1, 3].')
for _, r in DF.iterrows():
    inx = 1e-4 <= r.Fr_f <= 1
    iny = 0.1 <= r.M <= 3
    print('  %-6s %-8s Fr_f=%9.3g (%s)  M=%6.3g (%s)'
          % (r.sect, r.Qr_kind, r.Fr_f, 'in ' if inx else 'OFF',
             r.M, 'in ' if iny else 'OFF'))

print('\nMonthly (pc_lp, Knudsen Qr):')
print(monthly['pc_lp'][['U_T', 'Qr', 'ds_inout', 'Fr_f', 'M', 'dstrat']].to_string(
    float_format=lambda x: '%.4g' % x))

# ---- figure ---------------------------------------------------------------
fig = plt.figure(figsize=(13, 5.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.35, wspace=0.25)
ax = fig.add_subplot(gs[:, 0])
axb = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[1, 1])

# GM14 Fig 6 domain
ax.add_patch(Rectangle((1e-4, 0.1), 1 - 1e-4, 3 - 0.1, facecolor='0.92',
                       edgecolor='0.55', lw=1.2, zorder=0))
ax.text(3e-4, 2.55, 'domain of GM14 Fig. 6', fontsize=9, color='0.35',
        style='italic', zorder=1)

# Qualitative regime labels (relative arrangement only -- the actual
# stratification and length contours must be traced from the paper).
for x, y, lab in [(0.20, 0.16, 'salt wedge'),
                  (0.02, 0.55, 'strongly\nstratified'),
                  (2e-3, 1.10, 'partially\nmixed'),
                  (2e-4, 2.10, 'well\nmixed')]:
    ax.text(x, y, lab, fontsize=9, color='0.45', ha='center', va='center',
            zorder=1)

if args.show_lit:
    for name, (x, y) in LIT.items():
        ax.plot(x, y, marker='o', ms=4, color='0.55', zorder=2)
        ax.text(x * 1.15, y, name, fontsize=7, color='0.5', va='center', zorder=2)
    ax.text(1e-4 * 1.3, 0.115, 'literature points EYEBALLED, not digitized',
            fontsize=7, color='crimson', style='italic', zorder=3)

cmap = plt.get_cmap('twilight_shifted')
mk = {'pc_lp': 'o', 'pc_cp': 's'}
lab_off = {'pc_lp': (12, 14), 'pc_cp': (12, -20)}

# Months with Qr < 0 are inverse-estuarine and have no place on a log Fr_f
# axis. They are plotted at |Fr_f| with a hollow face so they stay visible
# instead of being silently dropped.
n_neg = 0
for sect in sect_list:
    mdf = monthly[sect]
    pos = mdf.Fr_f > 0
    n_neg += int((~pos).sum())
    ax.plot(mdf.Fr_f.abs(), mdf.M, '-', color='0.75', lw=0.8, zorder=3)
    sc = ax.scatter(mdf.Fr_f[pos].abs(), mdf.M[pos], c=mdf.index[pos],
                    cmap=cmap, vmin=1, vmax=12, s=46,
                    marker=mk.get(sect, 'o'), edgecolor='k', linewidth=0.6,
                    zorder=4)
    if (~pos).any():
        ax.scatter(mdf.Fr_f[~pos].abs(), mdf.M[~pos], facecolor='none',
                   edgecolor='crimson', linewidth=1.4, s=70,
                   marker=mk.get(sect, 'o'), zorder=5)
    r = DF[(DF.sect == sect) & (DF.Qr_kind == 'knudsen')].iloc[0]
    ax.plot(r.Fr_f, r.M, marker=mk.get(sect, 'o'), ms=14, mfc='none',
            mec='k', mew=1.8, zorder=6)
    ax.annotate('%s\nKnudsen $Q_R$' % sect, (r.Fr_f, r.M),
                textcoords='offset points', xytext=lab_off.get(sect, (12, 12)),
                fontsize=8.5, fontweight='bold', ha='left')

for sect in sect_list:
    r = DF[(DF.sect == sect) & (DF.Qr_kind == 'local')].iloc[0]
    ax.plot(r.Fr_f, r.M, marker=mk.get(sect, 'o'), ms=9, mfc='crimson',
            mec='k', mew=0.8, zorder=6)
ax.annotate('local runoff only\n(0.009 m$^3$/s)',
            (DF[DF.Qr_kind == 'local'].Fr_f.min(),
             DF[DF.Qr_kind == 'local'].M.min()),
            textcoords='offset points', xytext=(-4, -34), fontsize=8.5,
            color='crimson', ha='left')

if n_neg:
    ax.plot([], [], marker='o', ls='none', mfc='none', mec='crimson', mew=1.4,
            ms=8, label='month with $Q_R<0$ (inverse), plotted at $|Fr_f|$')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

cb = fig.colorbar(sc, ax=ax, pad=0.02, ticks=range(1, 13))
cb.set_label('month')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(2e-8, 3)
ax.set_ylim(6e-3, 4)
ax.set_xlabel(r'$Fr_f = U_R / c_0$')
ax.set_ylabel(r'$M = \sqrt{C_D U_T^2 / (\omega N_0 H^2)}$')
ax.set_title('Penn Cove in the Geyer & MacCready (2014) parameter space\n%s'
             % args.gtx, fontsize=11)
ax.grid(True, which='major', alpha=0.3, lw=0.6)

# panel b: seasonality of the freshwater forcing
mdf = monthly['pc_lp']
axb.bar(mdf.index, mdf.Qr, color='tab:blue', alpha=0.8)
axb.axhline(Q_LOCAL_FRESH, color='crimson', lw=1.4,
            label='local runoff (%.3f m$^3$/s)' % Q_LOCAL_FRESH)
axb.set_ylabel(r'Knudsen $Q_R$ (m$^3$/s)')
axb.set_title('pc_lp: freshwater flux carried by the exchange', fontsize=10)
axb.legend(fontsize=8)
axb.grid(True, alpha=0.25, lw=0.5)
axb.set_xticks(range(1, 13))

# panel c: what actually sets the stratification. Note the two curves need
# separate axes -- that size gap is the whole point.
axc.plot(mdf.index, mdf.dstrat, 'o-', color='tab:purple')
axc.set_xlabel('month')
axc.set_ylabel(r'$\Delta s$ top$-$bot (g/kg)', color='tab:purple')
axc.tick_params(axis='y', labelcolor='tab:purple')
axc.set_ylim(0, None)
axc2 = axc.twinx()
axc2.plot(mdf.index, mdf.ds_inout, 's-', color='tab:orange')
axc2.axhline(0, color='0.6', lw=0.8)
axc2.set_ylabel(r'$s_{in}-s_{out}$ (g/kg)', color='tab:orange')
axc2.tick_params(axis='y', labelcolor='tab:orange')
axc.set_title('pc_lp: imported stratification (left) vs.\nestuarine exchange '
              '$\\Delta s$ (right, note scale)', fontsize=9.5)
axc.grid(True, alpha=0.25, lw=0.5)
axc.set_xticks(range(1, 13))

fig.savefig(out_dir / 'gm2014_parameter_space.png', dpi=200,
            bbox_inches='tight', transparent=True)
plt.close(fig)
print('\nsaved %s' % (out_dir / 'gm2014_parameter_space.png'))
print('saved %s' % (out_dir / 'gm2014_parameters.csv'))
