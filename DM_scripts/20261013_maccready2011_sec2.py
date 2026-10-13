"""
MacCready (2011, JPO 41, 1116-1124) Section 2, "Eulerian salt flux
decomposition" (after Lerczak et al. 2006), built one equation at a time at the
Penn Cove mouth section pc_lp of wb1_t0_xn11abbur00.

Built so far
    (2.2)  A0 = < int dA >
    (2.3)  u0 = < int u dA > / A0           (includes Stokes drift)
           Q_R = -u0 A0
    (2.3)  s0 = < int s dA > / A0
    (2.2)  dA0 = < dA >                      per element
    (2.4)  u1 = < u dA > / dA0 - u0          per element; int u1 dA0 = 0
    (2.4)  s1 = < s dA > / dA0 - s0          per element; int s1 dA0 = 0
    (2.5)  u2 = u - u0 - u1                  hourly, per element
           s2 = s - s0 - s1
           the paper uses <u2 dA> = 0 and <s2 dA> = 0; with a real filter these
           hold only approximately, and the size of each is printed
    (2.6)  F_R = u0 s0 A0                    hourly (low-passed), river term
           F_E = int u1 s1 dA0               hourly (low-passed), exchange term
           F_T = < int u2 s2 dA >            computed DIRECTLY; tidal term
    (2.7)  Storage = d/dt < int s dV > = F_R + F_E + F_T
           F_T is a second filter pass over u2, s2 (NaN for 35 h at each end), so it
           is valid only 70 h in from each end of the record

ELEMENTS dA. Following the paper, A is divided into a constant number of
elements that expand and contract with the tide: the ROMS cells on the section,
c = (k, p), 30 s-levels x 12 u-faces. At each hourly snapshot t,
    dA_c(t) = dy_p * dz_c(t)
    dy_p    = 0.5 * (1/pn[irm] + 1/pn[irp])
    dz_c(t) = 0.5 * (Hz[irm] + Hz[irp]),  Hz = diff(z_w)  (ROMS u-point thickness)
    s_c(t)  = 0.5 * (salt[irm] + salt[irp])               (salinity at the face)

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       hourly ocean_his SNAPSHOTS (on the hour). The pc_lp u-faces are box
       xi_u = 30 (global 67) between box xi_rho 30, 31 (global irm 67, irp 68);
       rows eta 19..8 = global j 228..217, so p = 0 is the north end, matching
       sect_df and the tef2 extraction.

SIGN  u is positive UP-ESTUARY = into Penn Cove (the paper's convention).
The section's own positive direction is east = out of the cove, so it is
flipped. Then Q_R = -u0 A0 is positive for river outflow.

<> is the Godin filter, kept HOURLY throughout: u2 and s2 need u0, u1, s0, s1
at every hour, and all saved fields, plots and statistics are hourly too (no
daily subsampling). Single-pass terms are NaN for 35 h at each end, and F_T
(two passes) for 70 h. Every statistic is taken over the same hours, the ones
where all terms are valid (mask ok).

OUTPUT LO_output/DM_outs/20261013_maccready2011_sec2/

run 20261013_maccready2011_sec2.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SN = 'pc_lp'
SIGN = -1        # section-positive (east) is out of the cove
PAD = 36
XI_U = 30        # box index of the pc_lp u-faces
IRM, IRP = 30, 31
ETA = np.arange(19, 7, -1)   # north (p=0) to south (p=11)

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_maccready2011_sec2'
out_dir.mkdir(parents=True, exist_ok=True)


def lp(x):
    """Tidal low-pass <>: Godin filter on axis 0, hourly (NaN at the ends)."""
    return zfun.lowpass(x, f='godin')


def isum(x):
    """Integral over the section, int ... dA, as a sum over elements (k, p)."""
    return x.sum(axis=(-2, -1))


# ------------------------------------------------------------------ load snapshots
ds = xr.open_dataset(in_fn)
time = ds.ocean_time.to_numpy()
u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)   # (t, k, p)
zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()            # (t, w, p, 2)
s = ds.salt.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().astype(float).mean(axis=-1)  # (t, k, p)
pn = ds.pn.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()             # (p, 2)
ds.close()

dz = np.diff(zw, axis=1).mean(axis=-1)    # (t, k, p)
dy = (1 / pn).mean(axis=-1)               # (p,)
dA = dz * dy[None, None, :]               # (t, k, p)
h = -zw[0, 0, :, :].mean(axis=-1)         # (p,) depth at the face (bottom z_w = -h)
print('elements: %d levels x %d faces; NaN in u %d, in s %d, in dA %d'
      % (u.shape[1], u.shape[2], np.isnan(u).sum(), np.isnan(s).sum(),
         np.isnan(dA).sum()))

# ------------------------------------------------------------------ (2.2) A0
A0h = lp(isum(dA))                        # < int dA >          (t,)

# ------------------------------------------------------------------ (2.3) u0
Q_lph = lp(isum(u * dA))                  # < int u dA >        (t,)
u0h = Q_lph / A0h

# ------------------------------------------------------------------ (2.3) s0
s0h = lp(isum(s * dA)) / A0h              # < int s dA > / A0   (t,)

# ------------------------------------------------------------------ (2.4) u1
dA0h = lp(dA)                             # < dA >              (t, k, p)
u1h = lp(u * dA) / dA0h - u0h[:, None, None]
print('max |int u1 dA0| = %.2e m3/s' % np.nanmax(np.abs(isum(u1h * dA0h))))

# ------------------------------------------------------------------ (2.4) s1
s1h = lp(s * dA) / dA0h - s0h[:, None, None]
print('max |int s1 dA0| = %.2e g/kg m2' % np.nanmax(np.abs(isum(s1h * dA0h))))

# ------------------------------------------------------------------ (2.5) u2, s2
u2 = u - u0h[:, None, None] - u1h         # hourly              (t, k, p)
s2 = s - s0h[:, None, None] - s1h

# The paper's <u2 dA> = 0 and <s2 dA> = 0. u2 is NaN for the first and last 35 h,
# so filtering it again is valid only 70 h in from each end.
NT = len(time)
idx = np.arange(NT)
ok = (idx >= 2 * (PAD - 1)) & (idx <= NT - 1 - 2 * (PAD - 1))
u2dA_lp = lp(np.nan_to_num(u2 * dA))[ok]  # < u2 dA >
s2dA_lp = lp(np.nan_to_num(s2 * dA))[ok]  # < s2 dA >
# as a velocity / salinity on dA0, compared with u1 / s1 at the same hours
u2_res = u2dA_lp / dA0h[ok]
s2_res = s2dA_lp / dA0h[ok]
print('rms <u2 dA>/dA0 = %.2e m/s   vs rms u1 = %.2e m/s'
      % (np.sqrt(np.mean(u2_res**2)), np.sqrt(np.mean(u1h[ok]**2))))
print('rms <s2 dA>/dA0 = %.2e g/kg  vs rms s1 = %.2e g/kg'
      % (np.sqrt(np.mean(s2_res**2)), np.sqrt(np.mean(s1h[ok]**2))))

# ------------------------------------------------------------------ (2.6) F_R
F_Rh = u0h * s0h * A0h                    # hourly, NaN for 35 h at each end

# ------------------------------------------------------------------ (2.6) F_E
F_Eh = isum(u1h * s1h * dA0h)             # hourly, NaN for 35 h at each end

# ------------------------------------------------------------------ (2.6) F_T, direct
F_Th = lp(isum(np.nan_to_num(u2 * s2 * dA)))
F_Th[~ok] = np.nan                        # second filter pass: valid 70 h in

# ------------------------------------------------------------------ (2.7) storage
St_h = F_Rh + F_Eh + F_Th                 # d/dt < int s dV >, as the sum

# ------------------------------------------------------------------ record statistics
# all over the hours where every term is valid (ok: 70 h in from each end)
Q_Rh = -u0h * A0h
mean = pd.Series({'A0 [m2]': A0h[ok].mean(),
                  '<int u dA> [m3/s]': Q_lph[ok].mean(),
                  'u0 [m/s]': u0h[ok].mean(),
                  'Q_R = -u0 A0 [m3/s]': Q_Rh[ok].mean(),
                  'std of Q_R [m3/s]': Q_Rh[ok].std(),
                  's0 [g/kg]': s0h[ok].mean(),
                  'std of s0 [g/kg]': s0h[ok].std(),
                  'min / max s0 [g/kg]': '%.3f / %.3f' % (s0h[ok].min(), s0h[ok].max()),
                  'max |time-mean u1| [m/s]': np.abs(u1h[ok].mean(axis=0)).max(),
                  'max |u1| [m/s]': np.abs(u1h[ok]).max(),
                  'max |time-mean s1| [g/kg]': np.abs(s1h[ok].mean(axis=0)).max(),
                  'max |s1| [g/kg]': np.abs(s1h[ok]).max(),
                  'rms u2 [m/s]': np.sqrt(np.mean(u2[ok]**2)),
                  'rms s2 [g/kg]': np.sqrt(np.mean(s2[ok]**2))})
terms = pd.DataFrame({'mean': [x[ok].mean() for x in (F_Rh, F_Eh, F_Th, St_h)],
                      'std': [x[ok].std() for x in (F_Rh, F_Eh, F_Th, St_h)]},
                     index=['F_R', 'F_E', 'F_T (direct)',
                            'Storage = F_R + F_E + F_T'])
print('\n' + SN + ', hourly, over the %d hours where all terms are valid'
      % ok.sum())
print(mean.to_string(float_format=lambda x: '%12.4g' % x))
print('\nsalt flux terms [g/kg m3/s]')
print(terms.to_string(float_format=lambda x: '%10.2f' % x))

# ------------------------------------------------------------------ save
# time-mean cell-centre depth from the tidally averaged element thickness
dz0m = (dA0h[ok] / dy[None, None, :]).mean(axis=0)       # (k, p)
z0 = -h[None, :] + np.cumsum(dz0m, axis=0) - dz0m / 2   # (k, p)

f4 = lambda x: x.astype(np.float32)
xr.Dataset({'A0': ('time', A0h), 'u0': ('time', u0h), 'Q_R': ('time', Q_Rh),
            's0': ('time', s0h), 'F_R': ('time', F_Rh), 'F_E': ('time', F_Eh),
            'F_T': ('time', F_Th), 'Storage': ('time', St_h), 'ok': ('time', ok),
            'dA0': (('time', 'k', 'p'), f4(dA0h)), 'u1': (('time', 'k', 'p'), f4(u1h)),
            's1': (('time', 'k', 'p'), f4(s1h)),
            'z0_mean': (('k', 'p'), z0), 'h': ('p', h), 'dy': ('p', dy)},
           coords={'time': time},
           attrs={'source': str(in_fn),
                  'time': 'hourly; ok marks hours where every term is valid',
                  'sign': 'u positive up-estuary (into Penn Cove)',
                  'units': 'A0, dA0 m2; u0, u1 m/s; Q_R m3/s; s0, s1 g/kg; F_R, F_E, F_T, Storage g/kg m3/s; z m'}
           ).to_netcdf(out_dir / (SN + '_sec2.nc'))

# ------------------------------------------------------------------ plots (hourly)
fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axs[0].plot(time, A0h, color='k', lw=0.8)
axs[0].set_ylabel(r'$A_0$ [m$^2$]')
axs[1].plot(time, Q_Rh, color='tab:blue', lw=0.6)
axs[1].axhline(0, color='k', lw=0.6)
axs[1].set_ylabel(r'$Q_R=-u_0A_0$ [m$^3$ s$^{-1}$]')
axs[2].plot(time, s0h, color='tab:green', lw=0.8)
axs[2].set_ylabel(r'$s_0$ [g kg$^{-1}$]')
for a in axs:
    a.grid(True, alpha=0.3)
fig.suptitle(SN + r': MacCready (2011) Eq. 2.2-2.3, hourly low-passed')
fig.tight_layout()
fig.savefig(out_dir / (SN + '_sec2.png'), dpi=200, transparent=True)
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(time, u0h * 1000, color='tab:blue', lw=0.6,
        label=r'$u_0$ (mean %.2e mm s$^{-1}$)' % (u0h[ok].mean() * 1000))
ax.axhline(0, color='k', lw=0.6)
ax.set_ylabel(r'$u_0$ [mm s$^{-1}$], + up-estuary (into cove)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title(SN + r': $u_0=\langle\int u\,dA\rangle/A_0$, hourly low-passed')
fig.tight_layout()
fig.savefig(out_dir / (SN + '_u0.png'), dpi=200, transparent=True)
plt.close(fig)

# time-mean u1 and s1 on the section, looking up-estuary (into the cove)
x = np.cumsum(dy) / 1000 - dy[0] / 2000   # face centres, km from north end
X = np.broadcast_to(x[None, :], z0.shape)
xe = np.concatenate(([0], np.cumsum(dy) / 1000))   # face edges
for fld, cmap, clab, ttl, fn in [
        (u1h[ok].mean(axis=0) * 100, 'RdBu_r',
         r'[cm s$^{-1}$], + up-estuary (into cove)',
         r'time-mean $u_1=\langle u\,dA\rangle/dA_0-u_0$', '_u1_mean.png'),
        (s1h[ok].mean(axis=0), 'BrBG_r', r'[g kg$^{-1}$]',
         r'time-mean $s_1=\langle s\,dA\rangle/dA_0-s_0$', '_s1_mean.png')]:
    vm = np.abs(fld).max()
    fig, ax = plt.subplots(figsize=(7, 4))
    cs = ax.pcolormesh(X, z0, fld, cmap=cmap, vmin=-vm, vmax=vm, shading='nearest')
    fig.colorbar(cs, ax=ax, label=clab)
    ax.fill_between(np.repeat(xe, 2)[1:-1], np.repeat(-h, 2), -30, color='0.6')
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(-28, 0)
    ax.set_xlabel('distance from north end [km]')
    ax.set_ylabel('z [m]')
    ax.grid(True, alpha=0.3)
    ax.set_title(SN + ': ' + ttl + ', ' + DS0 + ' to ' + DS1, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / (SN + fn), dpi=200, transparent=True)
    plt.close(fig)

# u = u0 + u1 + u2 and s = s0 + s1 + s2 at one element, over 15 days
KK, PP = -1, 2                            # surface, north side (inflow)
w = (time >= np.datetime64('2024-08-01')) & (time < np.datetime64('2024-08-16'))
fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
a = axs[0]
a.plot(time[w], u[w, KK, PP], color='k', lw=1, label=r'$u$')
a.plot(time[w], (u0h[:, None, None] + u1h)[w, KK, PP], color='tab:blue', lw=1.8,
       label=r'$u_0+u_1$')
a.plot(time[w], u2[w, KK, PP], color='tab:orange', lw=1, label=r'$u_2$')
a.axhline(0, color='k', lw=0.6)
a.set_ylabel(r'[m s$^{-1}$]')
a = axs[1]
a.plot(time[w], s[w, KK, PP], color='k', lw=1, label=r'$s$')
a.plot(time[w], (s0h[:, None, None] + s1h)[w, KK, PP], color='tab:blue', lw=1.8,
       label=r'$s_0+s_1$')
a.set_ylabel(r'[g kg$^{-1}$]')
a = axs[2]
a.plot(time[w], s2[w, KK, PP], color='tab:orange', lw=1, label=r'$s_2$')
a.axhline(0, color='k', lw=0.6)
a.set_ylabel(r'[g kg$^{-1}$]')
for a in axs:
    a.legend(loc='upper left', fontsize=8, ncol=3)
    a.grid(True, alpha=0.3)
axs[0].set_title(SN + ': Eq. 2.5 at element k=%d (surface), p=%d (north), hourly'
                 % (u.shape[1] + KK, PP))
fig.tight_layout()
fig.savefig(out_dir / (SN + '_u2_s2_example.png'), dpi=200, transparent=True)
plt.close(fig)

# each flux term on its own, hourly low-passed
for y, c, lab, ttl, fn in [
        (F_Rh, 'tab:gray', r'$F_R=u_0s_0A_0$', r'$F_R$, Eq. 2.6', '_F_R.png'),
        (F_Eh, 'tab:blue', r'$F_E=\int u_1s_1\,dA_0$', r'$F_E$, Eq. 2.6', '_F_E.png'),
        (F_Th, 'tab:orange', r'$F_T=\langle\int u_2s_2\,dA\rangle$',
         r'$F_T$ computed directly, Eq. 2.6', '_F_T.png')]:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(time, y, color=c, lw=0.6, label='hourly (mean %.2f)' % y[ok].mean())
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel(lab + r' [g kg$^{-1}$ m$^3$ s$^{-1}$]' + '\n+ up-estuary (into cove)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(SN + ': ' + ttl)
    fig.tight_layout()
    fig.savefig(out_dir / (SN + fn), dpi=200, transparent=True)
    plt.close(fig)

# Eq. 2.7: storage as the sum of the three flux terms
fig, ax = plt.subplots(figsize=(11, 4))
for y, lab, c, lw in [(F_Rh, r'$F_R$', 'tab:gray', 0.6),
                      (F_Eh, r'$F_E$', 'tab:blue', 0.6),
                      (F_Th, r'$F_T$', 'tab:orange', 0.6),
                      (St_h, r'Storage $=F_R+F_E+F_T$', 'k', 1.0)]:
    ax.plot(time, y, color=c, lw=lw, label=lab + ' (mean %.1f)' % y[ok].mean())
ax.axhline(0, color='k', lw=0.6)
ax.set_ylabel(r'[g kg$^{-1}$ m$^3$ s$^{-1}$]' + '\n+ up-estuary (into cove)')
ax.legend(loc='upper left', fontsize=8, ncol=4)
ax.grid(True, alpha=0.3)
ax.set_title(SN + r': Eq. 2.7, $\frac{d}{dt}\langle\int s\,dV\rangle=F_R+F_E+F_T$, hourly low-passed')
fig.tight_layout()
fig.savefig(out_dir / (SN + '_storage.png'), dpi=200, transparent=True)
plt.close(fig)
print('\nsaved to ' + str(out_dir))
