"""
MacCready (2011, JPO 41, 1116-1124) Section 2 velocity decomposition (after
Lerczak et al. 2006), u = u0 + u1 + u2, at the three Penn Cove sections
pc_lp (mouth), pc_lj (middle) and pc_cp (off Coupeville) of wb1_t0_xn11abbur00.
Velocity only: the same equations as 20261013_maccready2011_sec2.py, no salt.

Equations
    (2.2)  A0  = < int dA >                  section area, tidally averaged
    (2.3)  u0  = < int u dA > / A0           section-mean subtidal velocity (one number per hour)
    (2.2)  dA0 = < dA >                      per element
    (2.4)  u1  = < u dA > / dA0 - u0         per element; subtidal, varies across the section;
                                             int u1 dA0 = 0 every hour
    (2.5)  u2  = u - u0 - u1                 per element, hourly; the tidal part

ELEMENTS dA. The ROMS cells on the section, c = (k, p), 30 s-levels x the
section's u-faces, expanding and contracting with the tide. In each hour t,
    dA_c(t) = dy_p * dz_c(t)
    dy_p    = 0.5 * (1/pn[irm] + 1/pn[irp])
    dz_c(t) = 0.5 * (Hz[irm] + Hz[irp]),  Hz = diff(z_w)  (ROMS u-point thickness)

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       hourly ocean_his SNAPSHOTS (on the hour). The box rho origin is global
       (j, i) = (209, 37), so a u-face at global (j, i) is box (eta_u, xi_u) =
       (j - 209, i - 37), between box xi_rho i - 37 and i - 36. Rows run north
       (p = 0) to south, matching sect_df_wb1_pc1.p and the tef2 extraction.

SIGN  u is positive UP-ESTUARY = into Penn Cove. The sections' own positive
direction is east = out of the cove, so it is flipped.

<> is the Godin filter, kept HOURLY; u0, u1, u2 are NaN for 35 h at each end,
and every statistic is taken over the hours where they are valid (ok).

OUTPUT LO_output/DM_outs/20261103_pc_eulerian_u0u1u2/
    u0.png          u0 vs time, hourly, one panel per section
    u1_mean.png     time-mean u1 on each section, looking into the cove (west; north at right)
    u2_example.png  u, u0 + u1 and u2 at one element of each section, 15 days
    u_terms_max_flood.png, u_terms_max_ebb.png
                    u, u0, u1, u2 on each section, averaged over the peak-|Q| hour
                    of each flood (each ebb); Q = int u dA, unfiltered, hourly

CYCLE (-cycle springneap | tropeq | strat | wind | windhs | windls | cross | stratcalm; see pc_tidal_cycles.py): only the u_terms figures, with each of max
flood / max ebb split into the high and low third of the cycle (pc_tidal_cycles.py),
e.g. u_terms_max_flood_spring.png and u_terms_max_flood_neap.png; each pair shares
one color scale. u0 / u1_mean / u2_example are not redrawn. Default (none): unchanged.

run 20261103_pc_eulerian_u0u1u2.py [-cycle springneap|tropeq|strat|wind|windhs|windls|cross|stratcalm]
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from lo_tools import Lfun, zfun
import argparse
from pc_tidal_cycles import cycle_masks, CYCLES
parser = argparse.ArgumentParser()
parser.add_argument('-cycle', default='none', choices=['none', 'springneap', 'tropeq', 'strat', 'wind', 'windhs', 'windls', 'cross', 'stratcalm'])   # split max flood / ebb by the cycle
CYCLE = parser.parse_args().cycle

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove
# box indices of each section's u-faces (global i - 37, global j - 209), north to south;
# listed west to east (inner cove to mouth): the column order of the section figures
FACES = {
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
KK, PP = -1, 2                            # example element for u2: surface, north side

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261103_pc_eulerian_u0u1u2'
out_dir.mkdir(parents=True, exist_ok=True)


def lp(x):
    """Tidal low-pass <>: Godin filter on axis 0, hourly (NaN at the ends)."""
    return zfun.lowpass(x, f='godin')


def isum(x):
    """Integral over the section, int ... dA, as a sum over elements (k, p)."""
    return x.sum(axis=(-2, -1))


def peak_hours(mask, Q, nmin=3):
    """One hour per contiguous run of mask (one flood or one ebb): the hour of
    largest |Q|. Runs shorter than nmin hours are slack-water flicker, skipped."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    out = np.zeros_like(mask)
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= nmin:
            out[a + np.argmax(np.abs(Q[a:b]))] = True
    return out


ds = xr.open_dataset(in_fn)
time = ds.ocean_time.to_numpy()
CM = cycle_masks(time) if CYCLE != 'none' else None
R = {}
for SN in FACES:
    XI_U, ETA = FACES[SN]['XI_U'], FACES[SN]['ETA']
    IRM, IRP = XI_U, XI_U + 1             # rho cells west (minus) and east (plus) of the face

    # ------------------------------------------------------------------ load snapshots
    u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)   # (t, k, p)
    zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()            # (t, w, p, 2)
    pn = ds.pn.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()             # (p, 2)

    dz = np.diff(zw, axis=1).mean(axis=-1)    # (t, k, p)
    dy = (1 / pn).mean(axis=-1)               # (p,)
    dA = dz * dy[None, None, :]               # (t, k, p)
    h = -zw[0, 0, :, :].mean(axis=-1)         # (p,) depth at the face (bottom z_w = -h)

    # ------------------------------------------------------------------ (2.2) A0
    A0h = lp(isum(dA))                        # < int dA >          (t,)

    # ------------------------------------------------------------------ (2.3) u0
    Q_lph = lp(isum(u * dA))                  # < int u dA >        (t,)
    u0h = Q_lph / A0h

    # ------------------------------------------------------------------ (2.4) u1
    dA0h = lp(dA)                             # < dA >              (t, k, p)
    u1h = lp(u * dA) / dA0h - u0h[:, None, None]

    # ------------------------------------------------------------------ (2.5) u2
    u2 = u - u0h[:, None, None] - u1h         # hourly              (t, k, p)

    # ------------------------------------------------------------------ checks and statistics
    ok = np.isfinite(u0h)                     # hours where u0, u1, u2 are all valid

    # ------------------------------------------------------------------ max flood / max ebb hours
    # flood = section transport into the cove, Q = int u dA > 0; ebb Q < 0.
    # One hour per flood (per ebb): the hour of largest |Q| in that run.
    Qh = isum(u * dA)                         # hourly, unfiltered  (t,)
    PF = peak_hours(Qh > 0, Qh) & ok
    PE = peak_hours(Qh < 0, Qh) & ok
    print('  %d max-flood hours, %d max-ebb hours' % (PF.sum(), PE.sum()))

    # composite (plain mean over those hours) of each term; u = u0 + u1 + u2 holds in the mean too
    comp = {}
    PHM = [('max flood', PF), ('max ebb', PE)]
    if CYCLE != 'none':                      # each also split by the cycle, e.g. 'max flood, spring'
        PHM += [('%s, %s' % (ph, half), m & CM[half]) for ph, m in PHM for half in CYCLES[CYCLE]]
    for ph, m in PHM:
        comp[ph] = {'u': u[m].mean(axis=0),
                    'u0': np.broadcast_to(u0h[m].mean(), u.shape[1:]),
                    'u1': u1h[m].mean(axis=0),
                    'u2': u2[m].mean(axis=0)}
        resid = np.abs(comp[ph]['u'] - comp[ph]['u0'] - comp[ph]['u1'] - comp[ph]['u2']).max()
        print('  %s: max |u - (u0 + u1 + u2)| = %.1e m/s' % (ph, resid))
    print('\n' + SN + ', %d valid hours' % ok.sum())
    print('  max |int u1 dA0|     = %.2e m3/s   (should be 0)' % np.nanmax(np.abs(isum(u1h * dA0h))))
    print('  mean / std u0        = %+.2e / %.2e m/s' % (u0h[ok].mean(), u0h[ok].std()))
    print('  max |time-mean u1|   = %.4f m/s' % np.abs(u1h[ok].mean(axis=0)).max())
    print('  rms u1, rms u2       = %.4f, %.4f m/s' % (np.sqrt(np.mean(u1h[ok]**2)), np.sqrt(np.mean(u2[ok]**2))))

    # time-mean cell-centre depth from the tidally averaged element thickness
    dz0m = (dA0h[ok] / dy[None, None, :]).mean(axis=0)       # (k, p)
    z0 = -h[None, :] + np.cumsum(dz0m, axis=0) - dz0m / 2   # (k, p)

    R[SN] = dict(u=u, u0h=u0h, u1h=u1h, u2=u2, ok=ok, dy=dy, h=h, z0=z0, comp=comp)
ds.close()

WR = [R[SN]['dy'].sum() for SN in FACES]  # column widths = real section widths, west to east
if CYCLE == 'none':                       # these three do not depend on the tidal phase
    # ------------------------------------------------------------------ plot u0 (hourly)
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True, sharey=True)
    for ax, SN in zip(axs, FACES):
        u0h, ok = R[SN]['u0h'], R[SN]['ok']
        ax.plot(time, u0h * 1000, color='tab:blue', lw=0.6,
                label=r'$u_0$ (mean %+.3f, std %.3f mm s$^{-1}$)' % (u0h[ok].mean() * 1000, u0h[ok].std() * 1000))
        ax.axhline(0, color='k', lw=0.6)
        ax.set_ylabel(NAME[SN] + '\n' + r'$u_0$ [mm s$^{-1}$]')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    axs[0].set_title(r'$u_0=\langle\int u\,dA\rangle/A_0$ (Eq. 2.3), hourly low-passed, + into the cove')
    fig.tight_layout()
    fig.savefig(out_dir / 'u0.png', dpi=200, transparent=True)
    plt.close(fig)

    # ------------------------------------------------------------------ plot time-mean u1 on each section
    # looking up-estuary (west, into the cove), so north is at the RIGHT; one color scale for all three
    vm = max(np.abs(R[SN]['u1h'][R[SN]['ok']].mean(axis=0)).max() for SN in FACES) * 100
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, layout='constrained', gridspec_kw={'width_ratios': WR})
    for ax, SN in zip(axs, FACES):
        u1h, ok, dy, h, z0 = R[SN]['u1h'], R[SN]['ok'], R[SN]['dy'], R[SN]['h'], R[SN]['z0']
        x = np.cumsum(dy) / 1000 - dy[0] / 2000            # face centres, km from north end
        X = np.broadcast_to(x[None, :], z0.shape)
        xe = np.concatenate(([0], np.cumsum(dy) / 1000))   # face edges
        fld = u1h[ok].mean(axis=0) * 100
        cs = ax.pcolormesh(X, z0, fld, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='nearest')
        ax.contour(X, z0, fld, levels=[0], colors='k', linewidths=1)
        ax.fill_between(np.repeat(xe, 2)[1:-1], np.repeat(-h, 2), -30, color='0.6')
        ax.set_xlim(xe[-1], xe[0])           # reversed: north at right = looking west, into the cove
        ax.set_ylim(-28, 0)
        ax.set_xlabel('distance from north end [km]')
        ax.grid(True, alpha=0.3)
        ax.set_title(NAME[SN], fontsize=10)
    axs[0].set_ylabel('z [m]')
    fig.colorbar(cs, ax=axs, label=r'[cm s$^{-1}$], + up-estuary (into cove)')
    fig.suptitle(r'time-mean $u_1=\langle u\,dA\rangle/dA_0-u_0$ (Eq. 2.4), ' + DS0 + ' to ' + DS1 + ', sections west to east, each looking into the cove (west), north at right', fontsize=11)
    fig.savefig(out_dir / 'u1_mean.png', dpi=200, transparent=True)
    plt.close(fig)

    # ------------------------------------------------------------------ plot u = u0 + u1 + u2 at one element, over 15 days
    w = (time >= np.datetime64('2024-08-01')) & (time < np.datetime64('2024-08-16'))
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True, sharey=True)
    for ax, SN in zip(axs, FACES):
        u, u0h, u1h, u2 = R[SN]['u'], R[SN]['u0h'], R[SN]['u1h'], R[SN]['u2']
        ax.plot(time[w], u[w, KK, PP], color='k', lw=1, label=r'$u$')
        ax.plot(time[w], (u0h[:, None, None] + u1h)[w, KK, PP], color='tab:blue', lw=1.8, label=r'$u_0+u_1$')
        ax.plot(time[w], u2[w, KK, PP], color='tab:orange', lw=1, label=r'$u_2$')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_ylabel(NAME[SN] + '\n' + r'[m s$^{-1}$]')
        ax.legend(loc='upper left', fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
    axs[0].set_title('Eq. 2.5 at element k=%d (surface), p=%d (north side), hourly, + into the cove'
                     % (R['pc_lp']['u'].shape[1] + KK, PP))
    fig.tight_layout()
    fig.savefig(out_dir / 'u2_example.png', dpi=200, transparent=True)
    plt.close(fig)

# ------------------------------------------------------------------ plot u, u0, u1, u2 on each section at max flood / max ebb
# looking up-estuary (west, into the cove), so north is at the RIGHT. u and u2 share one color
# scale (tidal); u0 and u1 share a second, smaller one (subtidal).
TERMS = [('u', r'$u$'), ('u0', r'$u_0$'), ('u1', r'$u_1$'), ('u2', r'$u_2$')]
PAIRS = [[ph] for ph in ['max flood', 'max ebb']] if CYCLE == 'none' else [['%s, %s' % (ph, half) for half in CYCLES[CYCLE]] for ph in ['max flood', 'max ebb']]
for ph in [p for pair in PAIRS for p in pair]:
    pair = [p for p in PAIRS if ph in p][0]   # one color scale for both halves of a cycle
    vt = max(np.abs(R[SN]['comp'][p][k]).max() for p in pair for SN in FACES for k in ['u', 'u2']) * 100
    vr = max(np.abs(R[SN]['comp'][p][k]).max() for p in pair for SN in FACES for k in ['u0', 'u1']) * 100
    # columns = sections west to east, rows = the terms u, u0, u1, u2
    fig, axs = plt.subplots(4, 3, figsize=(14, 13), sharey=True, layout='constrained', gridspec_kw={'width_ratios': WR})
    for j, SN in enumerate(FACES):
        dy, h, z0 = R[SN]['dy'], R[SN]['h'], R[SN]['z0']
        x = np.cumsum(dy) / 1000 - dy[0] / 2000            # face centres, km from north end
        X = np.broadcast_to(x[None, :], z0.shape)
        xe = np.concatenate(([0], np.cumsum(dy) / 1000))   # face edges
        for i, (k, lab) in enumerate(TERMS):
            ax = axs[i, j]
            fld = R[SN]['comp'][ph][k] * 100
            vm = vt if k in ['u', 'u2'] else vr
            cs = ax.pcolormesh(X, z0, fld, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='nearest')
            if np.ptp(fld) > 0:
                ax.contour(X, z0, fld, levels=[0], colors='k', linewidths=1)
            ax.fill_between(np.repeat(xe, 2)[1:-1], np.repeat(-h, 2), -30, color='0.6')
            ax.set_xlim(xe[-1], xe[0])           # reversed: north at right = looking west, into the cove
            ax.set_ylim(-28, 0)
            ax.grid(True, alpha=0.3)
            ax.set_title(NAME[SN] + ': ' + lab, fontsize=10)
            if i == 3:
                ax.set_xlabel('distance from north end [km]')
            if j == 2:
                fig.colorbar(cs, ax=axs[i, :], location='right', shrink=0.9,
                             label=lab + r' [cm s$^{-1}$], + into cove')
            if j == 0:
                ax.set_ylabel('z [m]')
    fig.suptitle((r'$u=u_0+u_1+u_2$ (Eq. 2.5) at %s: the peak-$|Q|$ hour of each %s, averaged over %s to %s' + '\n' + 'sections west to east, each looking into the cove (west), north at right')
                 % (ph, ph.split(',')[0].split()[1], DS0, DS1), fontsize=11)
    fig.savefig(out_dir / ('u_terms_%s.png' % ph.replace(', ', '_').replace(' ', '_')), dpi=200, transparent=True)
    plt.close(fig)
print('\nsaved to ' + str(out_dir))
