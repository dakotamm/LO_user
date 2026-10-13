"""
MacCready (2011, JPO 41, 1116-1124) Section 3, "Isohaline salt flux
decomposition", built one equation at a time at the Penn Cove mouth section
pc_lp of wb1_t0_xn11abbur00.

HOURLY-AVERAGE VERSION of 20261013_maccready2011_sec3.py (which uses ocean_his
snapshots); identical method, different input. Companion to
20261013_maccready2011_sec2_avg.py. Companion to 20261013_maccready2011_sec2.py: same
input, same elements dA, same face salinity, same sign, hourly throughout.

Built so far
    (3.1)  Q(s) = < int_{A_s} u dA >        A_s = part of the section with salinity > s
    (3.2)  -dQ/ds, as the tidally averaged transport in finite salinity bins
           ("a histogram of transport versus salinity, which is then integrated
           to find Q")

HOW. In each hour t, every element c carries a transport u_c dA_c and
a salinity s_c. Bin the transports by salinity,
    q(s_i, t) = sum over {c : s_c(t) in bin i} of u_c(t) dA_c(t)       [m3/s]
then tidally average each bin,
    qbar(s_i) = < q(s_i, t) >                                           [m3/s]
which is the transport in salinity class i, i.e. -dQ/ds * ds. Then
    Q(s_edge_i) = sum over j >= i of qbar(s_j)        (cumulate from the salty end)
so Q(s_ocn) = 0 and Q(0) = < int u dA > = -Q_R. Elements change bin from hour to
hour as their salinity changes, which is what makes this isohaline.

BINS  NS = 1000 over 0-36 g/kg (ds = 0.036), the same as process_sections_avg.py,
so the result can be compared with bulk_avg_*.

ELEMENTS dA (as in Section 2). The ROMS cells on the section, c = (k, p),
30 s-levels x 12 u-faces. In each hour t,
    u_c dA_c = q_c(t)               hour-mean Huon, the exact volume flux
    dA_c(t)  = dd_p * DZ_c(t)       DZ from the hour-mean zeta (exact hour-mean
                                    thickness, since z_w is linear in zeta)
    s_c(t)   = hour-mean salinity at the face (mean of the two rho cells)
    u_c(t)   = q_c / dA_c           only for plotting and rms u2
Products like u s dA are q * s: the covariance of Huon and salt WITHIN each hour
is not resolved (ROMS can write <Huon*salt> with Aout(iHUTav), off in this run).

INPUT  LO_output/extract/[gtagex]/tef2/extractions_avg_[ds0]_[ds1]/pc_lp.nc
       hourly ocean_avg AVERAGES, stamped at the hour centre (hh:30). Hour
       averages are used instead of ocean_his snapshots because snapshots
       alias sub-hourly motion into the subtidal band: <int u dA> from
       snapshots has ~6x the true subtidal variance and does not close the
       cove volume budget; the hour-mean Huon closes it to r = 0.999999.
       p = 0 is the north end.

SIGN  u positive UP-ESTUARY = into Penn Cove, so positive -dQ/ds is inflow in
that salinity class (the paper's convention).

<> is the Godin filter, kept HOURLY (NaN for 35 h at each end). Statistics are
over the hours where every term is valid (mask ok).

OUTPUT LO_output/DM_outs/20261013_maccready2011_sec3/

run 20261013_maccready2011_sec3.py
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
NS, S_LO, S_HI = 1000, 0.0, 36.0

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'tef2'
         / ('extractions_avg_' + DS0 + '_' + DS1) / (SN + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_maccready2011_sec3_avg'
out_dir.mkdir(parents=True, exist_ok=True)


def lp(x):
    """Tidal low-pass <>: Godin filter on axis 0, hourly (NaN at the ends)."""
    return zfun.lowpass(x, f='godin')


def isum(x):
    """Integral over the section, int ... dA, as a sum over elements (k, p)."""
    return x.sum(axis=(-2, -1))


# ------------------------------------------------------------------ load hourly averages
ds = xr.open_dataset(in_fn)
time = ds.time.to_numpy()                 # hour centres, hh:30
q = SIGN * ds.q.to_numpy()                # (t, k, p) hour-mean Huon = u dA [m3/s]
s = ds.salt.to_numpy().astype(float)      # (t, k, p) hour-mean face salinity
dz = ds.DZ.to_numpy()                     # (t, k, p) from hour-mean zeta
dy = ds.dd.to_numpy()                     # (p,)
h = ds.h.to_numpy()                       # (p,) depth at the face
ds.close()

dA = dz * dy[None, None, :]               # (t, k, p)
u = q / dA                                # hour-mean, area-weighted velocity
NT = len(time)
print('elements: %d levels x %d faces; salinity range %.3f to %.3f g/kg'
      % (u.shape[1], u.shape[2], s.min(), s.max()))

# ------------------------------------------------------------------ bins
sedges = np.linspace(S_LO, S_HI, NS + 1)
sbins = 0.5 * (sedges[:-1] + sedges[1:])
dsb = sedges[1] - sedges[0]

# ------------------------------------------------------------------ (3.2) -dQ/ds
# hourly transport in each salinity bin, q(s_i, t) [m3/s]
ib = np.clip(np.digitize(s, sedges) - 1, 0, NS - 1)            # (t, k, p) bin index
it = np.broadcast_to(np.arange(NT)[:, None, None], s.shape)    # (t, k, p) hour index
q_ts = np.bincount((it * NS + ib).ravel(), weights=(u * dA).ravel(),
                   minlength=NT * NS).reshape(NT, NS)
print('max |sum_i q(s_i,t) - int u dA| = %.2e m3/s'
      % np.abs(q_ts.sum(axis=1) - isum(u * dA)).max())

qbar = lp(q_ts)                           # < q(s_i, t) > = -dQ/ds * ds   (t, NS)
dQds = -qbar / dsb                        # dQ/ds [m3/s per g/kg]; -dQ/ds > 0 is inflow

# ------------------------------------------------------------------ (3.1) Q(s)
# cumulate from the salty end, at the bin edges: Q(sedges[i]) = sum_{j>=i} qbar_j
Q = np.zeros((NT, NS + 1))
Q[:, :-1] = np.cumsum(qbar[:, ::-1], axis=1)[:, ::-1]
Q[np.isnan(qbar[:, 0]), :] = np.nan

# Q(0) is the whole section: Q(0) = < int u dA > = -Q_R
Q_lph = lp(isum(u * dA))
idx = np.arange(NT)
ok = (idx >= PAD - 1) & (idx <= NT - PAD)
print('max |Q(0) - <int u dA>| = %.2e m3/s' % np.abs(Q[ok, 0] - Q_lph[ok]).max())
print('Q(s_ocn) = Q(%.0f) = %.1f (zero by construction)' % (S_HI, np.abs(Q[ok, -1]).max()))

# ------------------------------------------------------------------ record statistics
qm = qbar[ok].mean(axis=0)                # time-mean transport per bin [m3/s]
Qm = Q[ok].mean(axis=0)                   # time-mean Q(s) at edges
nz = np.nonzero(np.abs(qm) > 0)[0]
print('\n' + SN + ', hourly, over the %d hours where all terms are valid' % ok.sum())
print(pd.Series({'Q(0) = <int u dA> [m3/s]': Qm[0],
                 'max Q(s) [m3/s]': Qm.max(),
                 's at max Q(s) [g/kg]': sedges[np.argmax(Qm)],
                 'salinity classes carrying transport': '%.2f to %.2f g/kg'
                 % (sedges[nz[0]], sedges[nz[-1] + 1]),
                 'sum of inflowing bins [m3/s]': qm[qm > 0].sum(),
                 'sum of outflowing bins [m3/s]': qm[qm < 0].sum()}
                ).to_string(float_format=lambda x: '%12.4g' % x))

# ------------------------------------------------------------------ save
f4 = lambda x: x.astype(np.float32)
xr.Dataset({'qbar': (('time', 'sbin'), f4(qbar)),
            'Q': (('time', 'sedge'), f4(Q)),
            'ok': ('time', ok)},
           coords={'time': time, 'sbin': sbins, 'sedge': sedges},
           attrs={'source': str(in_fn),
                  'qbar': '< q(s_i, t) >, transport in salinity bin i = -dQ/ds * ds',
                  'Q': 'Q(s) = < int_{A_s} u dA >, at bin edges',
                  'time': 'hourly; ok marks hours where every term is valid',
                  'sign': 'u positive up-estuary (into Penn Cove)',
                  'units': 'qbar, Q m3/s; s g/kg'}
           ).to_netcdf(out_dir / (SN + '_sec3.nc'))

# ------------------------------------------------------------------ plots
# time-mean -dQ/ds and Q(s) over the salinity range that carries transport
s_lo = np.floor(sedges[nz[0]]) - 0.5
s_hi = np.ceil(sedges[nz[-1] + 1]) + 0.5
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
a = axs[0]
a.barh(sbins, -dQds[ok].mean(axis=0), height=dsb, color='tab:blue')
a.axvline(0, color='k', lw=0.6)
a.set_xlabel(r'$-\partial Q/\partial s$ [m$^3$ s$^{-1}$ (g kg$^{-1}$)$^{-1}$]'
             + '\n+ inflow (into cove)')
a.set_ylabel(r'$s$ [g kg$^{-1}$]')
a.set_title(r'(3.2) time-mean $-\partial Q/\partial s$')
a = axs[1]
a.plot(Qm, sedges, color='k', lw=1.2)
a.axvline(0, color='k', lw=0.6)
a.set_xlabel(r'$Q(s)$ [m$^3$ s$^{-1}$]')
a.set_title(r'(3.1) time-mean $Q(s)=\langle\int_{A_s}u\,dA\rangle$')
for a in axs:
    a.set_ylim(s_lo, s_hi)
    a.grid(True, alpha=0.3)
fig.suptitle(SN + ', ' + DS0 + ' to ' + DS1 + ', hourly averages, %d bins (ds = %.3f)'
             % (NS, dsb))
fig.tight_layout()
fig.savefig(out_dir / (SN + '_Q_of_s_mean.png'), dpi=200, transparent=True)
plt.close(fig)
print('\nsaved to ' + str(out_dir))
