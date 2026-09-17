"""
Where in the section does the Penn Cove oxygen flux happen?

The oxygen-coordinate TEF output (bulk_avg_DO_*) cannot answer this -- binning
into oxygen classes throws away z and p. So this goes back to the hourly
section extraction and decomposes the subtidal oxygen flux cell by cell,

    <q O>_c   =   <q>_c <O>_c   +   <q' O'>_c
     total         advective         tidal pumping

exactly as 20260916_exchange_fun.flux_decomp does section-integrated, but kept
resolved in (z, p). <> is the Godin average; <O>_c is area weighted, so a cell
that is thin at low tide does not count as much as a thick one.

THE ONE RULE (same as exchange_fun): tidal averaging is applied to q, never to
a velocity and an area separately. q is stored hourly as Huon/Hvom, so <q O>
already carries the tidal correlation. That correlation is not a small
correction here -- at pc_lp the advective term exports ~+308 g s-1 and tidal
pumping imports ~-215, and the ~93 that survives is the difference. A method
that paired <q> with <O> and stopped would overstate the export threefold.

LAYOUT follows 20260805_plot_section_structure.py: section view on top,
collapsed onto each axis below. Columns are the three terms rather than the
three sections, and one figure is written per section.

    row 1   section view, distance across the cove on x, true depth on y
    row 2   vertical profile, sum over width, against mean depth
    row 3   lateral profile, sum over depth, against distance from south

The map is flux DENSITY [g s-1 m-2], not flux per cell. The house version
plots per-cell flux, which is fine for volume transport but biased here: sigma
layers are much thinner near the surface, so a per-cell map makes the surface
look quiet when per unit area it is the most active part of the section. The
profiles below are integrated [g s-1] and do sum to the section totals.

NET VS GROSS the net flux is a small residual of much larger opposing terms --
at pc_lp the north half imports ~4600 g s-1 and the south half exports ~4700.
Read the structure, and quote the gross terms; the net is the difference.

AND DO NOT READ IMPORTANCE OFF THE COLOUR SCALE. At pc_lp:

    term        sum|F_cell|      net    net/sum   max density
    advective          9635   -307.8       3.2%         0.635
    tidal               410   +214.6      52.3%         0.020

where sum|F_cell| adds every cell's time-mean flux ignoring sign, and net lets
them cancel. So net/sum is a measure of SPATIAL COHERENCE: 100% means every
cell has the same sign, 0% a perfectly balanced dipole. Nothing temporal -- the
time mean is already taken per cell before either sum.

It is resolution dependent in principle (collapse the section to one cell and
it is 100% by definition) but not in practice: coarsening the section 15x moves
it from 3.2 to 3.3% for the advective term and 52.3 to 54.5% for tidal, so the
contrast between the two is a property of the fields, not of the grid.

Tidal pumping is ~32x weaker per unit area and yet its net is 70% of the
advective net, because the advective term is a north/south dipole that almost
entirely self-cancels while tidal pumping is coherent in sign across most of
the section. Amplitude on the map does not predict integrated flux; each panel
prints its gross and surviving fraction for exactly this reason.

SIGN positive is INTO the cove everywhere (INFLOW_SIGN from
20260916_exchange_fun.py), so red is import and blue is export.

run 20260917_pc_o2_flux_structure.py
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-sect', default='pc_cp,pc_lj,pc_lp', type=str)
parser.add_argument('-seasons', default='all,Winter,Spring,Low-DO', type=str,
                    help='comma separated; house 4-month bins plus "all"')
args = parser.parse_args()

_fn = Path(__file__).parent / '20260916_exchange_fun.py'
_spec = importlib.util.spec_from_file_location('exchange_fun', _fn)
xfun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xfun)

Ldir = Lfun.Lstart(gridname='wb1')
in_dir = xfun.section_dir(args.gtagex, args.ds0, args.ds1, Ldir=Ldir)
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

SECTS = [s.strip() for s in args.sect.split(',') if s.strip()]
CONV = 31.998 / 1000        # mmol s-1 -> g s-1 of O2
PAD = xfun.GODIN_PAD

# Dakota's 3 season bins, December folding forward into the next year
# (same map as 20260805_tef_flushing_time.py)
SEASON = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Winter',
          4: 'Spring', 5: 'Spring', 6: 'Spring', 7: 'Spring',
          8: 'Low-DO', 9: 'Low-DO', 10: 'Low-DO', 11: 'Low-DO'}
WANT = [w.strip() for w in args.seasons.split(',') if w.strip()]

# the three terms, in plot order. CVD-validated trio from the house palette.
TERMS = [('total', 'total  ' + r'$\langle qO \rangle$', '#000000'),
         ('adv', 'advective  ' + r'$\langle q \rangle \langle O \rangle$', '#0072B2'),
         ('tid', 'tidal pumping  ' + r"$\langle q'O' \rangle$", '#D55E00')]

# section geometry: latitude of each face, to order the section south -> north
sect_df = pd.read_pickle(Ldir['LOo'] / 'extract' / 'tef2' / 'sect_df_wb1_pc1.p')
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')

def make_figure(sn, season, F, Abar, X, Y, dist_c, zlev, ho, tsel):
    plt.close('all')
    fig, axes = plt.subplots(3, 3, figsize=(16, 11),
                             gridspec_kw=dict(height_ratios=[2, 1, 1]))

    dens = {k: F[k] / Abar for k in F}
    # Each term gets its OWN symmetric scale. Sharing one would be tidier, but
    # tidal pumping is ~30x weaker per unit area than the advective term and
    # renders as a blank panel on a shared scale. The limit is printed on every
    # colorbar so the panels are never compared by eye without it.
    vmax = {k: np.nanpercentile(np.abs(dens[k]), 99) for k in dens}

    for c, (key, label, col) in enumerate(TERMS):
        ax = axes[0, c]
        cs = ax.pcolormesh(X, Y, dens[key], cmap='RdBu_r',
                           vmin=-vmax[key], vmax=vmax[key], shading='flat')
        ax.plot(dist_c, -ho, color='0.3', lw=1.5)
        plt.colorbar(cs, ax=ax, label='O$_2$ flux density [g s$^{-1}$ m$^{-2}$]'
                                      '\nred = into the cove'
                                      '\nOWN SCALE, $\\pm$%.2f' % vmax[key])
        gross = np.abs(F[key]).sum()
        coh = 100 * abs(F[key].sum()) / gross
        net = F[key].sum()
        ax.set_title('%s\nnet %+.1f g s$^{-1}$  =  %s'
                     % (label, net, 'IMPORT' if net > 0 else 'EXPORT'),
                     color=col, fontweight='bold')
        # Amplitude on this map is a BAD predictor of integrated flux, and the
        # per-panel colour scale makes that easy to misread. The advective term
        # is a north/south dipole -- large amplitude, ~97% self-cancelling --
        # while tidal pumping is ~30x weaker per unit area but coherent in sign,
        # so half of it survives integration. Printing gross and the surviving
        # fraction on every panel keeps the two facts together.
        ax.text(0.02, 0.03, r'$\Sigma|F_{cell}|$ = %.0f    net is %.0f%% of it'
                % (gross, coh), transform=ax.transAxes, fontsize=9,
                va='bottom', ha='left',
                bbox=dict(fc='w', ec='0.7', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.set_xlabel('distance across cove from south [km]')
        if c == 0:
            ax.set_ylabel('depth [m]')

        ax = axes[1, c]
        ax.plot(F[key].sum(axis=1), zlev, '-o', ms=4, color=col)
        ax.axvline(0, color='k', lw=1)
        ax.set_xlabel('O$_2$ flux [g s$^{-1}$]   + = into the cove')
        if c == 0:
            ax.set_ylabel('mean depth [m]')
        ax.set_title('vertical profile (sum over width)', fontsize=10)
        ax.grid(color='lightgray', ls='--', alpha=0.5)

        ax = axes[2, c]
        ax.plot(dist_c, F[key].sum(axis=0), '-o', ms=4, color=col)
        ax.axhline(0, color='k', lw=1)
        ax.set_xlabel('distance across cove from south [km]')
        if c == 0:
            ax.set_ylabel('O$_2$ flux [g s$^{-1}$]\n+ = into the cove')
        ax.set_title('lateral profile (sum over depth)', fontsize=10)
        ax.grid(color='lightgray', ls='--', alpha=0.5)

    fig.suptitle('%s  %s: where the subtidal oxygen flux sits in the section\n'
                 '%d days drawn from %s-%s\n'
                 'SIGN: + = into the cove, - = out.  net %+.1f g s$^{-1}$ = %s, '
                 'the small residual of north %+.0f and south %+.0f'
                 '\nrow 1 panels each carry their own colour scale'
                 % (sn, season, len(tsel), tsel.year.min(), tsel.year.max(),
                    F['total'].sum(),
                    'EXPORT' if F['total'].sum() < 0 else 'IMPORT',
                    F['total'][:, dist_c > dist_c.mean()].sum(),
                    F['total'][:, dist_c <= dist_c.mean()].sum()),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fn_out = out_dir / ('20260917_pc_o2_flux_structure_' + sn + '_'
                        + season.replace('-', '') + '.png')
    fig.savefig(fn_out, dpi=200, bbox_inches='tight', transparent=True)
    print('   saved %s' % fn_out.name)


print('%-8s %-8s %7s %12s %12s %12s   %s'
      % ('sect', 'season', 'days', 'total', 'advective', 'tidal',
         '[g/s into the cove]'))

for sn in SECTS:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    sgn = xfun.INFLOW_SIGN[sn]
    q = sgn * ds.q.to_numpy()
    o = ds.oxygen.to_numpy().astype(float)
    DZ = ds.DZ.to_numpy()
    dd = ds.dd.to_numpy()
    h = ds.h.to_numpy()
    tt = ds.time.to_numpy()
    ds.close()
    dA = DZ * dd[np.newaxis, np.newaxis, :]

    # per-cell subtidal decomposition [g s-1]
    qo = xfun.godin_daily(q * o, pad=PAD) * CONV
    q_lp = xfun.godin_daily(q, pad=PAD)
    dA_lp = xfun.godin_daily(dA, pad=PAD)
    with np.errstate(invalid='ignore', divide='ignore'):
        o_lp = xfun.godin_daily(o * dA, pad=PAD) / dA_lp
    adv = q_lp * o_lp * CONV
    # daily, still resolved in (z,p) -- season masks are applied below, BEFORE
    # the time mean, so each season gets its own section structure
    td = pd.DatetimeIndex(xfun.daily_time(tt))
    smon = pd.Series(td.month).map(SEASON).to_numpy()
    SEASONS = [(w, np.ones(len(td), bool) if w == 'all' else (smon == w))
               for w in WANT]
    Abar = dA_lp.mean(axis=0)                       # mean cell face area [m2]
    NZ, NP = qo.shape[1:]

    # ---- geometry, ordered south -> north so x reads like the house figure
    si = sect_df[sect_df.sn == sn]
    lat = np.array([g.lat_u.values[r.j, r.i] if r.uv == 'u'
                    else g.lat_v.values[r.j, r.i] for r in si.itertuples()])
    order = np.argsort(lat)
    ddo, ho = dd[order], h[order]
    qo = qo[:, :, order]
    adv = adv[:, :, order]
    Abar = Abar[:, order]

    dist_e = np.concatenate([[0], np.cumsum(ddo)]) / 1000        # km
    dist_c = (dist_e[:-1] + dist_e[1:]) / 2
    # true cell edges: z_w built up from the bed with the mean layer thicknesses
    DZm = DZ.mean(axis=0)[:, order]
    zw = np.vstack([np.zeros((1, NP)), np.cumsum(DZm, axis=0)]) - ho[np.newaxis, :]
    zr = (zw[:-1] + zw[1:]) / 2
    # 2D edge arrays for a section view with real bathymetry
    X = np.tile(dist_e[np.newaxis, :], (NZ + 1, 1))
    Y = np.hstack([zw[:, :1], (zw[:, :-1] + zw[:, 1:]) / 2, zw[:, -1:]])
    # width-weighted mean depth of each sigma level, for the vertical profile
    zlev = (zr * DZm).sum(axis=1) / DZm.sum(axis=1)

    # ------------------------------------------------------ season loop ----
    for season, mask in SEASONS:
        if mask.sum() == 0:
            print('%-8s %-8s  no days, skipped' % (sn, season))
            continue
        F = {'total': qo[mask].mean(axis=0), 'adv': adv[mask].mean(axis=0)}
        F['tid'] = F['total'] - F['adv']
        print('%-8s %-8s %5d d %12.2f %12.2f %12.2f'
              % (sn, season, mask.sum(), F['total'].sum(), F['adv'].sum(),
                 F['tid'].sum()))
        make_figure(sn, season, F, Abar, X, Y, dist_c, zlev, ho, td[mask])
