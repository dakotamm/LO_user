"""
MacCready (2011) Section 3 from (3.10) on: the Eulerian versions of the TEF
quantities, compared with TEF itself, at the Penn Cove mouth section pc_lp of
wb1_t0_xn11abbur00.

HOURLY-AVERAGE VERSION: reads the outputs of the _avg Section 2 and 3 scripts.

INPUTS (run these first; both hourly, same time axis)
    20261013_maccready2011_sec2_avg.py -> pc_lp_sec2.nc   u0, s0, u1, s1, dA0, F_R, F_E, F_T
    20261013_maccready2011_sec3_avg.py -> pc_lp_sec3.nc   qbar = < q(s_i, t) >, 1000 bins

TEF (3.3)-(3.7), at every hour t, from the tidally averaged bin transports qbar_i
(= -dQ/ds * ds, positive = into the cove):
    Q_in  = sum over {qbar_i > 0} of qbar_i          Q_out = sum over {qbar_i < 0}
    F_in  = sum over {qbar_i > 0} of s_i qbar_i      F_out likewise
    s_in  = F_in / Q_in                              s_out = F_out / Q_out
    Ds    = s_in - s_out
    (3.4) Q_in + Q_out = Q(0) = -Q_R
    (3.7) d/dt int s dV = Q_in s_in + Q_out s_out = F_in + F_out
"in" counts a bin only when its sign brings water into the cove, bin by bin, as
in the paper -- not the Lorenz multi-layer divider that bulk_calc_avg.py uses.

EULERIAN (3.10), at every hour t:
    Q^Eu(s) = int over A^Eu_s of (u0 + u1) dA0
    A^Eu_s  = the part of the section where the tidally averaged salinity
              s0 + s1 > s
Each element carries the subtidal transport (u0 + u1) dA0 and the subtidal
salinity s0 + s1; binning these with the same 1000 bins gives
    qEu_i = sum over {c : s0 + s1 in bin i} of (u0 + u1)_c dA0_c
and Q^Eu_in,out, F^Eu_in,out, s^Eu_in,out, Ds^Eu exactly as for TEF. No further
filtering: every field is already tidally averaged.
    Q^Eu_in + Q^Eu_out = int (u0 + u1) dA0 = u0 A0 = -Q_R       (since int u1 dA0 = 0)
    F^Eu_in + F^Eu_out = int (u0 + u1)(s0 + s1) dA0 = F_R + F_E   (no F_T)

(3.8) KNUDSEN. With the true storage S = d/dt int s dV and Q_R = -Q(0),
    Q_in   = (s_out / Ds) Q_R + S / Ds
    -Q_out = (s_in  / Ds) Q_R + S / Ds
exact for TEF; the Eulerian quantities miss F_T and so do not satisfy it. S is
taken as the TEF salt flux F_in + F_out (3.7).

AVERAGING. Every term is hourly; record means are over the hours where every
input is valid. Mean salinities are flux weighted, (4.1):
    [[s_in]] = [[Q_in s_in]] / [[Q_in]]

SIGN  positive = into Penn Cove (up-estuary).

OUTPUT LO_output/DM_outs/20261013_maccready2011_eu_vs_tef_avg/

run 20261013_maccready2011_eu_vs_tef.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

SN = 'pc_lp'

Ldir = Lfun.Lstart(gridname='wb1')
out_root = Ldir['LOo'] / 'DM_outs'
fn2 = out_root / '20261013_maccready2011_sec2_avg' / (SN + '_sec2.nc')
fn3 = out_root / '20261013_maccready2011_sec3_avg' / (SN + '_sec3.nc')
out_dir = out_root / '20261013_maccready2011_eu_vs_tef_avg'
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ load
d2 = xr.open_dataset(fn2)
d3 = xr.open_dataset(fn3)
assert (d2.time.values == d3.time.values).all(), 'sec2 and sec3 time axes differ'
time = d2.time.to_numpy()
u0 = d2.u0.to_numpy()
s0 = d2.s0.to_numpy()
u1 = d2.u1.to_numpy().astype(float)       # (t, k, p)
s1 = d2.s1.to_numpy().astype(float)
dA0 = d2.dA0.to_numpy().astype(float)
F_R, F_E, F_T = d2.F_R.to_numpy(), d2.F_E.to_numpy(), d2.F_T.to_numpy()
qbar = d3.qbar.to_numpy().astype(float)   # (t, NS)
sbins = d3.sbin.to_numpy()
sedges = d3.sedge.to_numpy()
ok = d2.ok.to_numpy() & d3.ok.to_numpy()
d2.close()
d3.close()
NT, NS = qbar.shape
dsb = sedges[1] - sedges[0]
print('hours valid in both: %d of %d' % (ok.sum(), NT))


def tef_terms(qb):
    """(3.3)-(3.6) from bin transports qb (t, NS), bin by bin sign."""
    qi = np.where(qb > 0, qb, 0.0)
    qo = np.where(qb < 0, qb, 0.0)
    Q_in, Q_out = qi.sum(axis=1), qo.sum(axis=1)
    F_in, F_out = (qi * sbins).sum(axis=1), (qo * sbins).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):   # invalid hours: 0/0
        s_in, s_out = F_in / Q_in, F_out / Q_out
    return {'Q_in': Q_in, 'Q_out': Q_out, 'F_in': F_in, 'F_out': F_out,
            's_in': s_in, 's_out': s_out}


# ------------------------------------------------------------------ TEF
qbar[~ok] = np.nan
T = tef_terms(qbar)

# ------------------------------------------------------------------ Eulerian (3.10)
uE = (u0[:, None, None] + u1) * dA0       # subtidal transport per element (t, k, p)
sE = s0[:, None, None] + s1               # subtidal salinity per element
good = np.isfinite(uE) & np.isfinite(sE)
ib = np.clip(np.digitize(np.where(good, sE, 0), sedges) - 1, 0, NS - 1)
it = np.broadcast_to(np.arange(NT)[:, None, None], sE.shape)
qEu = np.bincount((it * NS + ib).ravel(), weights=np.where(good, uE, 0).ravel(),
                  minlength=NT * NS).reshape(NT, NS)
qEu[~ok] = np.nan
E = tef_terms(qEu)

# ------------------------------------------------------------------ identities
QR = -np.nansum(qbar, axis=1)             # Q_R = -Q(0)
QR[~ok] = np.nan
print('max |Q_in + Q_out + Q_R|       TEF %.1e   Eu %.1e m3/s'
      % (np.nanmax(np.abs(T['Q_in'] + T['Q_out'] + QR)),
         np.nanmax(np.abs(E['Q_in'] + E['Q_out'] + QR))))
FR_E = F_R + F_E
print('max |F^Eu_in + F^Eu_out - (F_R + F_E)| = %.2f g/kg m3/s  (bin-centre salinity)'
      % np.nanmax(np.abs(E['F_in'] + E['F_out'] - FR_E)[ok]))

# ------------------------------------------------------------------ (3.8) Knudsen
S = T['F_in'] + T['F_out']                # storage, (3.7)
for D in (T, E):
    D['Ds'] = D['s_in'] - D['s_out']
    D['Q_in_K'] = (D['s_out'] * QR + S) / D['Ds']
    D['Q_out_K'] = -(D['s_in'] * QR + S) / D['Ds']

# ------------------------------------------------------------------ record means
def m(x):
    return np.nanmean(x[ok])


def fw(D, side):
    """(4.1) flux-weighted mean salinity."""
    return m(D['F_' + side]) / m(D['Q_' + side])


rows = {}
for lab, D in [('TEF', T), ('Eulerian', E)]:
    s_in, s_out = fw(D, 'in'), fw(D, 'out')
    Ds = s_in - s_out
    rows[lab] = {'Q_in [m3/s]': m(D['Q_in']), 'Q_out [m3/s]': m(D['Q_out']),
                 's_in [g/kg]': s_in, 's_out [g/kg]': s_out, 'Ds [g/kg]': Ds,
                 'F_in + F_out [g/kg m3/s]': m(D['F_in'] + D['F_out']),
                 'Q_in from (3.8), means [m3/s]': (s_out * m(QR) + m(S)) / Ds}
tab = pd.DataFrame(rows)
tab['Eu / TEF'] = tab['Eulerian'] / tab['TEF']
print('\n' + SN + ', record means over %d hours, positive into the cove' % ok.sum())
print(tab.to_string(float_format=lambda x: '%10.3f' % x))
print('\nfor reference (Section 2): Q_R %.3f m3/s;  F_R + F_E %.2f;  F_T %.2f;'
      '  F_R + F_E + F_T %.2f g/kg m3/s'
      % (m(QR), m(FR_E), m(F_T), m(F_R + F_E + F_T)))
print('hourly (3.8) misfit, rms of Q_in - Q_in_K:  TEF %.2e   Eu %.1f m3/s'
      % (np.sqrt(m((T['Q_in'] - T['Q_in_K'])**2)),
         np.sqrt(m((E['Q_in'] - E['Q_in_K'])**2))))

# ------------------------------------------------------------------ save
xr.Dataset({k + '_tef': ('time', T[k]) for k in
            ['Q_in', 'Q_out', 'F_in', 'F_out', 's_in', 's_out', 'Ds']}
           | {k + '_eu': ('time', E[k]) for k in
              ['Q_in', 'Q_out', 'F_in', 'F_out', 's_in', 's_out', 'Ds', 'Q_in_K']}
           | {'Q_R': ('time', QR), 'storage_tef': ('time', S), 'ok': ('time', ok),
              'qEu': (('time', 'sbin'), qEu.astype(np.float32))},
           coords={'time': time, 'sbin': sbins},
           attrs={'inputs': str(fn2) + ', ' + str(fn3),
                  'sign': 'positive into Penn Cove',
                  'units': 'Q m3/s; F g/kg m3/s; s g/kg'}
           ).to_netcdf(out_dir / (SN + '_eu_vs_tef.nc'))

# ------------------------------------------------------------------ plot: -dQ/ds (cf. MacCready 2011 Fig. 3)
qm_T = np.nanmean(qbar[ok], axis=0) / dsb
qm_E = np.nanmean(qEu[ok], axis=0) / dsb
nz = np.nonzero((np.abs(qm_T) > 0) | (np.abs(qm_E) > 0))[0]
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(qm_T, sbins, color='k', lw=1, label='TEF')
ax.plot(qm_E, sbins, color='0.6', lw=1, label='Eulerian (3.10)')
ax.axvline(0, color='k', lw=0.6)
ax.set_ylim(np.floor(sbins[nz[0]]) - 0.5, np.ceil(sbins[nz[-1]]) + 0.5)
ax.set_xlabel(r'time-mean $-\partial Q/\partial s$ [m$^3$ s$^{-1}$ (g kg$^{-1}$)$^{-1}$]'
              + '\n+ inflow (into cove)')
ax.set_ylabel(r'$s$ [g kg$^{-1}$]')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title(SN + ': TEF vs Eulerian', fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / (SN + '_dQds_tef_vs_eu.png'), dpi=200, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ plot: hourly time series (cf. Fig. 6)
fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
a = axs[0]
for D, c, lab in [(T, 'k', 'TEF'), (E, '0.6', 'Eulerian')]:
    a.plot(time, D['Q_in'], color=c, lw=0.6, label=r'$Q_{in}$ ' + lab
           + ' (mean %.0f)' % m(D['Q_in']))
    a.plot(time, D['Q_out'], color=c, lw=0.6, ls='--', label=r'$Q_{out}$ ' + lab
           + ' (mean %.0f)' % m(D['Q_out']))
a.axhline(0, color='k', lw=0.6)
a.set_ylabel(r'[m$^3$ s$^{-1}$], + into cove')
a = axs[1]
for D, c, lab in [(T, 'k', 'TEF'), (E, '0.6', 'Eulerian')]:
    a.plot(time, D['s_in'], color=c, lw=0.6, label=r'$s_{in}$ ' + lab)
    a.plot(time, D['s_out'], color=c, lw=0.6, ls='--', label=r'$s_{out}$ ' + lab)
a.set_ylabel(r'[g kg$^{-1}$]')
for a in axs:
    a.legend(loc='lower left', fontsize=8, ncol=2)
    a.grid(True, alpha=0.3)
axs[0].set_title(SN + ': TEF vs Eulerian (3.10), hourly low-passed')
fig.tight_layout()
fig.savefig(out_dir / (SN + '_tef_vs_eu_timeseries.png'), dpi=200, transparent=True)
plt.close(fig)
print('\nsaved to ' + str(out_dir))
