"""
Dissolved oxygen through the Penn Cove mouth (pc_lp) in the TEF and Eulerian
frameworks of MacCready (2011), wb1_t0_xn11abbur00: which salinity classes carry
oxygen in and out, the oxygen of the inflow and outflow, and the steady vs tidal
split of the oxygen flux. Record means (2024-2025) and monthly means for 2025.

SOURCE (-src)
    avg   hourly-AVERAGE output (ocean_avg Huon, salt, oxygen at the faces, hh:30),
          extractions_avg/pc_lp.nc. Closes the cove volume and salt budgets, so the
          oxygen budget is closed here too and its residual is reported. (default)
    snap  hourly SNAPSHOT output (ocean_his, on the hour): u, z_w, salt from the
          pc_cove box and oxygen from the pc_cove_o2 box. Sub-hourly motion aliases
          into the subtidal transport (~ +-6-9 m3/s monthly net, ~ +-50-70 g O2/s),
          so the oxygen budget is NOT closed in this version. Class-by-class DO and
          O_in / O_out are fine.

Positive = into Penn Cove. <> is the Godin filter, kept hourly. Elements c are
the 30 s-levels x 12 faces of pc_lp; q_c = u_c dA_c, O_c is oxygen at the face.

TEF (salinity coordinates), every hour: bin q and q*O by the HOURLY salinity
(1000 bins, 0-36 g/kg), filter each bin, and count a bin as "in" when its
filtered volume transport is positive (MacCready 2011, 3.3, 3.5):
    Q_in  = sum_in <q>_i          G_in = sum_in <q O>_i          O_in = G_in / Q_in
    Q_out, G_out, O_out likewise;  s_in, s_out as in (3.6)
    G = G_in + G_out = < int u O dA >                     total oxygen flux
The oxygen flux is binned directly (q*O), not as O_i * <q>_i, because oxygen is
not the sorting coordinate.

EULERIAN (3.10 analogue): the filtered element transport <q>_c = (u0 + u1) dA0
binned by the filtered salinity s0 + s1 = <s dA>/<dA>, carrying the filtered
oxygen Obar_c = <O dA>/<dA> = O0 + O1. Its total is the steady oxygen flux
    G_steady = sum_c <q>_c Obar_c = G_R + G_E
    G_R = u0 O0 A0 (river / net-flow part),  G_E = G_steady - G_R (exchange part)
TIDAL oxygen flux, computed DIRECTLY as in Section 2 (2.5-2.6):
    G_T = < int u2 O2 dA >,  u2 dA = q - ubar_c dA,  O2 = O - Obar_c
(second filter pass: valid 70 h in from each end). G - G_steady - G_T is the
cross-term residual the decomposition drops.

OXYGEN BUDGET (avg only), cove landward of pc_lp (segments pc_lp_m, pc_cp_p,
pc_cp_m; ocean_his snapshots, forward difference onto the hh:30 flux clock):
    dO/dt = G + R
    R = net interior source: photosynthesis - respiration - sediment demand,
        plus air-sea exchange (not separated here)
R < 0 is net consumption. R / V is the cove-mean rate in mmol m-3 per day.

Monthly salinities and oxygen are flux weighted, (4.1): [[O_in]] = [[G_in]]/[[Q_in]].
Units: oxygen in mmol m-3 (x 0.032 = mg/L); oxygen flux in mmol/s (x 0.032 = g/s).

OUTPUT LO_output/DM_outs/20261020_pc_do_tef_avg/  or  _snap/
    pc_lp_do_tef.nc                    hourly series
    pc_lp_do_monthly_2025.csv          monthly table
    pc_lp_do_monthly_2025.png          monthly Q_in, O_in/O_out, oxygen fluxes
    by_month/pc_lp_do_2025-MM.png      -dQ/ds, -dG/ds, and DO of each salinity class

run 20261020_pc_do_tef.py -src avg
run 20261020_pc_do_tef.py -src snap
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-src', choices=['avg', 'snap'], default='avg')
args = parser.parse_args()

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SN = 'pc_lp'
YEAR = 2025
SIGN = -1                 # section-positive (east) is out of the cove
PAD = 36
NS, S_LO, S_HI = 1000, 0.0, 36.0
K = 0.032                 # mmol O2 -> g, and mmol m-3 -> mg/L
DT = 3600.0
SEGS = ['pc_lp_m', 'pc_cp_p', 'pc_cp_m']
SRC_LAB = {'avg': 'hourly-average output', 'snap': 'hourly snapshots'}[args.src]

Ldir = Lfun.Lstart(gridname='wb1')
ext = Ldir['LOo'] / 'extract' / GTAGEX
out_dir = Ldir['LOo'] / 'DM_outs' / ('20261020_pc_do_tef_' + args.src)
month_dir = out_dir / 'by_month'
month_dir.mkdir(parents=True, exist_ok=True)


def lp(x):
    """Tidal low-pass <>: Godin filter on axis 0, hourly (NaN at the ends)."""
    return zfun.lowpass(x, f='godin')


def isum(x):
    """Integral over the section, as a sum over elements (k, p)."""
    return x.sum(axis=(-2, -1))


# ------------------------------------------------------------------ load
if args.src == 'avg':
    ds = xr.open_dataset(ext / 'tef2' / ('extractions_avg_' + DS0 + '_' + DS1) / (SN + '.nc'))
    time = ds.time.to_numpy()
    q = SIGN * ds.q.to_numpy()                            # (t, k, p) hour-mean Huon
    s = ds.salt.to_numpy().astype(float)
    O = ds.oxygen.to_numpy().astype(float)
    dA = ds.DZ.to_numpy() * ds.dd.to_numpy()[None, None, :]
    ds.close()
else:
    XI_U, IRM, IRP = 30, 30, 31                           # pc_lp in the pc_cove box
    ETA = np.arange(19, 7, -1)                            # north (p=0) to south
    bx = xr.open_dataset(ext / 'box' / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
    bo = xr.open_dataset(ext / 'box' / ('pc_cove_o2_' + DS0 + '_' + DS1 + '.nc'))
    assert (bx.ocean_time.values == bo.ocean_time.values).all(), 'box time axes differ'
    time = bx.ocean_time.to_numpy()
    u = SIGN * bx.u.isel(xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)
    zw = bx.z_w.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()
    s = bx.salt.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().astype(float).mean(axis=-1)
    O = bo.oxygen.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy().astype(float).mean(axis=-1)
    pn = bx.pn.isel(eta_rho=ETA, xi_rho=[IRM, IRP]).to_numpy()
    bx.close(), bo.close()
    dA = np.diff(zw, axis=1).mean(axis=-1) * (1 / pn).mean(axis=-1)[None, None, :]
    q = u * dA
NT = len(time)
idx = np.arange(NT)
ok = (idx >= 2 * (PAD - 1)) & (idx <= NT - 1 - 2 * (PAD - 1))   # two filter passes
print('%s: %d hours, oxygen %.0f to %.0f mmol/m3' % (SRC_LAB, NT, np.nanmin(O), np.nanmax(O)))

# ------------------------------------------------------------------ element filtered fields
q_lp = lp(q)                              # <q>_c = (u0 + u1) dA0
dA_lp = lp(dA)                            # dA0
sbar = lp(s * dA) / dA_lp                 # s0 + s1
Obar = lp(O * dA) / dA_lp                 # O0 + O1

# ------------------------------------------------------------------ oxygen flux split
G = lp(isum(q * O))                       # < int u O dA >
G_steady = isum(q_lp * Obar)              # sum (u0 + u1)(O0 + O1) dA0
O0 = isum(lp(O * dA)) / isum(dA_lp)
G_R = isum(q_lp) * O0                     # u0 O0 A0
G_E = G_steady - G_R
u2dA = q - (q_lp / dA_lp) * dA            # u2 dA
G_T = lp(isum(np.nan_to_num(u2dA * (O - Obar))))
G_T[~ok] = np.nan                         # second filter pass

# ------------------------------------------------------------------ binning
sedges = np.linspace(S_LO, S_HI, NS + 1)
sbins = 0.5 * (sedges[:-1] + sedges[1:])
dsb = sedges[1] - sedges[0]
it = np.broadcast_to(idx[:, None, None], s.shape)


def binsum(sal, w):
    """Hourly sum of w over the elements in each salinity bin of sal -> (t, NS)."""
    good = np.isfinite(sal) & np.isfinite(w)
    ib = np.clip(np.digitize(np.where(good, sal, 0), sedges) - 1, 0, NS - 1)
    return np.bincount((it * NS + ib).ravel(), weights=np.where(good, w, 0).ravel(),
                       minlength=NT * NS).reshape(NT, NS)


# TEF: hourly fields binned by the hourly salinity, then filtered
qT = lp(binsum(s, q))
gT = lp(binsum(s, q * O))
# section water: area and oxygen per hourly-salinity class (unfiltered, for DO by class)
aT = binsum(s, dA)
oaT = binsum(s, O * dA)
# Eulerian (3.10): filtered transport binned by the filtered salinity
qE = binsum(sbar, q_lp)
gE = binsum(sbar, q_lp * Obar)
for a in (qT, gT, qE, gE):
    a[~ok] = np.nan
print('max |sum_i <qO>_i - G| = %.2e mmol/s;  max |sum_i gEu_i - G_steady| = %.2e mmol/s'
      % (np.nanmax(np.abs(gT.sum(axis=1) - G)[ok]),
         np.nanmax(np.abs(gE.sum(axis=1) - G_steady)[ok])))


def tef_terms(qb, gb):
    """In/out volume, salt and oxygen transports from filtered bin transports."""
    inn, out = qb > 0, qb < 0
    T = {'Q_in': np.where(inn, qb, 0).sum(1), 'Q_out': np.where(out, qb, 0).sum(1),
         'F_in': np.where(inn, qb * sbins, 0).sum(1),
         'F_out': np.where(out, qb * sbins, 0).sum(1),
         'G_in': np.where(inn, gb, 0).sum(1), 'G_out': np.where(out, gb, 0).sum(1)}
    for k in T:
        T[k][~ok] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        T['s_in'], T['s_out'] = T['F_in'] / T['Q_in'], T['F_out'] / T['Q_out']
        T['O_in'], T['O_out'] = T['G_in'] / T['Q_in'], T['G_out'] / T['Q_out']
    return T


T = tef_terms(qT, gT)
E = tef_terms(qE, gE)

# ------------------------------------------------------------------ oxygen budget (avg only)
if args.src == 'avg':
    sg = xr.open_dataset(ext / 'tef2' / ('segments_' + DS0 + '_' + DS1
                                         + '_wb1_pc1_trapsN00.nc')).sel(seg=SEGS)
    assert (sg.time.values[:-1] + np.timedelta64(30, 'm') == time).all(), \
        'segment snapshots do not bracket the flux hours'
    Vs = sg.volume.sum('seg').to_numpy()
    Os = (sg.oxygen * sg.volume).sum('seg').to_numpy()     # mmol
    sg.close()
    V = lp(0.5 * (Vs[1:] + Vs[:-1]))
    dOdt = lp(np.diff(Os) / DT)                            # mmol/s, hour n
    R = dOdt - G                                           # net interior source
    R_rate = R / V * 86400                                 # mmol m-3 per day
else:
    dOdt = R = R_rate = np.full(NT, np.nan)

# ------------------------------------------------------------------ statistics
tt = pd.DatetimeIndex(time)


def summarise(w):
    """Means over mask w; salinities and oxygen flux weighted (4.1)."""
    m = lambda x: np.nanmean(x[w]) if np.isfinite(x[w]).any() else np.nan
    r = {'hours': int(w.sum())}
    for lab, D in [('tef', T), ('eu', E)]:
        Qi, Qo = m(D['Q_in']), m(D['Q_out'])
        r.update({'Q_in_' + lab: Qi, 'Q_out_' + lab: Qo,
                  's_in_' + lab: m(D['F_in']) / Qi, 's_out_' + lab: m(D['F_out']) / Qo,
                  'O_in_' + lab: m(D['G_in']) / Qi, 'O_out_' + lab: m(D['G_out']) / Qo})
    r.update({'G_total': m(G), 'G_R': m(G_R), 'G_E': m(G_E), 'G_T': m(G_T),
              'G_cross': m(G - G_steady - G_T),
              'dOdt': m(dOdt), 'R': m(R), 'R_rate': m(R_rate)})
    return r


rec = pd.Series(summarise(ok))
rows = {'%d-%02d' % (YEAR, mo): summarise(ok & (tt.year == YEAR) & (tt.month == mo))
        for mo in range(1, 13)}
df = pd.DataFrame(rows).T
df.index.name = 'month'
df.to_csv(out_dir / (SN + '_do_monthly_%d.csv' % YEAR), float_format='%.4f')

show = lambda D: pd.DataFrame({
    'hours': D['hours'],
    'Q_in TEF': D['Q_in_tef'], 'Q_in Eu': D['Q_in_eu'],
    'O_in TEF [mg/L]': D['O_in_tef'] * K, 'O_out TEF [mg/L]': D['O_out_tef'] * K,
    'O_in Eu': D['O_in_eu'] * K, 'O_out Eu': D['O_out_eu'] * K,
    'G total [g/s]': D['G_total'] * K, 'G_R': D['G_R'] * K, 'G_E': D['G_E'] * K,
    'G_T': D['G_T'] * K, 'cross': D['G_cross'] * K,
    'dO/dt [g/s]': D['dOdt'] * K, 'R [g/s]': D['R'] * K,
    'R/V [mmol/m3/d]': D['R_rate']}, index=D.index if hasattr(D, 'index') and
    not isinstance(D, pd.Series) else [0])
print('\n' + SN + ', ' + SRC_LAB + ', positive into the cove')
print('\nrecord means, %s to %s (%d h)' % (DS0, DS1, rec['hours']))
print(show(pd.DataFrame([rec])).T.to_string(header=False, float_format=lambda x: '%10.2f' % x))
print('\nmonthly means, %d' % YEAR)
print(show(df).to_string(float_format=lambda x: '%7.2f' % x))

# ------------------------------------------------------------------ save hourly
xr.Dataset({**{k + '_tef': ('time', T[k]) for k in T},
            **{k + '_eu': ('time', E[k]) for k in E},
            'G': ('time', G), 'G_R': ('time', G_R), 'G_E': ('time', G_E),
            'G_T': ('time', G_T), 'dOdt': ('time', dOdt), 'R': ('time', R),
            'R_rate': ('time', R_rate), 'ok': ('time', ok)},
           coords={'time': time},
           attrs={'source': SRC_LAB, 'sign': 'positive into Penn Cove',
                  'units': 'Q m3/s; s g/kg; O mmol m-3; G, dOdt, R mmol/s; '
                           'R_rate mmol m-3 d-1'}
           ).to_netcdf(out_dir / (SN + '_do_tef.nc'))

# ------------------------------------------------------------------ per-month figures
def s_range(curves, frac=0.01):
    keep = np.zeros(NS, bool)
    for c in curves:
        keep |= np.abs(c) > frac * np.nanmax(np.abs(c))
    i = np.nonzero(keep)[0]
    return np.floor(sbins[i[0]]) - 0.5, np.ceil(sbins[i[-1]]) + 0.5


for mo in range(1, 13):
    lab = '%d-%02d' % (YEAR, mo)
    w = ok & (tt.year == YEAR) & (tt.month == mo)
    dQT, dQE = np.nanmean(qT[w], 0) / dsb, np.nanmean(qE[w], 0) / dsb
    dGT, dGE = np.nanmean(gT[w], 0) / dsb * K, np.nanmean(gE[w], 0) / dsb * K
    with np.errstate(invalid='ignore', divide='ignore'):
        Ocls = oaT[w].sum(0) / aT[w].sum(0) * K           # DO of section water by class
    r = df.loc[lab]
    fig, axs = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    a = axs[0]
    a.plot(dQT, sbins, color='k', lw=0.8, label='TEF ($Q_{in}$ %.0f)' % r['Q_in_tef'])
    a.plot(dQE, sbins, color='0.6', lw=0.8, label='Eulerian ($Q_{in}$ %.0f)' % r['Q_in_eu'])
    a.set_xlabel(r'$-\partial Q/\partial s$ [m$^3$ s$^{-1}$ (g kg$^{-1}$)$^{-1}$]')
    a.set_ylabel(r'$s$ [g kg$^{-1}$]')
    a.set_title('volume transport by salinity class', fontsize=10)
    a = axs[1]
    a.plot(dGT, sbins, color='k', lw=0.8, label='TEF')
    a.plot(dGE, sbins, color='0.6', lw=0.8, label='Eulerian')
    a.set_xlabel(r'$-\partial G/\partial s$ [g O$_2$ s$^{-1}$ (g kg$^{-1}$)$^{-1}$]')
    a.set_title('oxygen transport by salinity class', fontsize=10)
    a = axs[2]
    a.plot(Ocls, sbins, color='tab:blue', lw=0.8, label='section water')
    a.plot(r['O_in_tef'] * K, r['s_in_tef'], '^', color='k', ms=8,
           label=r'TEF in  (%.2f mg/L)' % (r['O_in_tef'] * K))
    a.plot(r['O_out_tef'] * K, r['s_out_tef'], 'v', color='k', ms=8, mfc='w',
           label=r'TEF out (%.2f mg/L)' % (r['O_out_tef'] * K))
    a.plot(r['O_in_eu'] * K, r['s_in_eu'], '^', color='0.6', ms=7, label='Eulerian in')
    a.plot(r['O_out_eu'] * K, r['s_out_eu'], 'v', color='0.6', ms=7, mfc='w',
           label='Eulerian out')
    a.set_xlabel(r'DO [mg L$^{-1}$]')
    a.set_title('DO of each salinity class', fontsize=10)
    for a in axs[:2]:
        a.axvline(0, color='k', lw=0.6)
    for a in axs:
        a.grid(True, alpha=0.3)
        a.legend(loc='lower right', fontsize=7)
    axs[0].set_ylim(*s_range([dQT, dQE]))
    fig.suptitle(SN + ', %s (%d h, %s), + into cove' % (lab, r['hours'], SRC_LAB))
    fig.tight_layout()
    fig.savefig(month_dir / (SN + '_do_%s.png' % lab), dpi=200, transparent=True)
    plt.close(fig)

# ------------------------------------------------------------------ monthly summary figure
x = np.arange(1, 13)
mlab = [pd.Timestamp(YEAR, m, 1).strftime('%b') for m in x]
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
a = axs[0]
a.plot(x, df['Q_in_tef'], 'o-', color='k', label='TEF')
a.plot(x, df['Q_in_eu'], 'o-', color='0.6', label='Eulerian')
a.set_ylabel(r'$Q_{in}$ [m$^3$ s$^{-1}$]')
a = axs[1]
a.plot(x, df['O_in_tef'] * K, '^-', color='k', label=r'TEF $O_{in}$')
a.plot(x, df['O_out_tef'] * K, 'v--', color='k', mfc='w', label=r'TEF $O_{out}$')
a.plot(x, df['O_in_eu'] * K, '^-', color='0.6', label=r'Eulerian $O_{in}$')
a.plot(x, df['O_out_eu'] * K, 'v--', color='0.6', mfc='w', label=r'Eulerian $O_{out}$')
a.set_ylabel(r'DO [mg L$^{-1}$]')
a = axs[2]
for col, c, l in [('G_total', 'k', r'$G$ total (TEF)'),
                  ('G_E', 'tab:blue', r'$G_E$ steady exchange'),
                  ('G_T', 'tab:orange', r'$G_T$ tidal (direct)'),
                  ('G_R', 'tab:gray', r'$G_R$')]:
    a.plot(x, df[col] * K, 'o-', color=c, label=l)
if args.src == 'avg':
    a.plot(x, df['dOdt'] * K, 's-', color='tab:green', label=r'$dO/dt$ storage')
    a.plot(x, df['R'] * K, 's-', color='tab:red', label=r'$R$ net interior source')
a.axhline(0, color='k', lw=0.6)
a.set_ylabel(r'oxygen flux [g O$_2$ s$^{-1}$]')
for a in axs:
    a.grid(True, alpha=0.3)
    a.legend(loc='best', fontsize=8, ncol=2)
axs[-1].set_xticks(x)
axs[-1].set_xticklabels(mlab)
axs[0].set_title(SN + ': oxygen exchange, monthly means %d (%s), + into cove'
                 % (YEAR, SRC_LAB))
fig.tight_layout()
fig.savefig(out_dir / (SN + '_do_monthly_%d.png' % YEAR), dpi=200, transparent=True)
plt.close(fig)
print('\nsaved to ' + str(out_dir))
