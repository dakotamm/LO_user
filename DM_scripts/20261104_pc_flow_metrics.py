"""
Hourly flow metrics at the three Penn Cove sections, pc_cp (off Coupeville),
pc_lj (middle) and pc_lp (mouth), of wb1_t0_xn11abbur00, saved for
20261104_pc_flow_metrics_plot.py, which sets them against the drivers.

Per section and hour (q = u dA per element, + into the cove; elements as in
20261013_maccready2011_sec2.py; u normal to the section, v along it as the mean of
the 4 v points around the face, as in 20261103_pc_section_Ri.py):

TRANSPORT
    Q   = sum q                         net transport [m3/s]
    G   = sum |q|                       gross transport
TWO-LAYER / EXCHANGE (transports; the plot script forms daily ratios, sum X / sum G,
    so slack hours, when Q ~ 0 and any fraction is noise, carry little weight)
    Qx  = min(Q_in, |Q_out|)            counterflow: transport moving against the rest
    LC  = sum_p max(0, -sgn(Q) Q_p)     LATERAL counterflow: through faces whose depth-
                                        integrated transport Q_p opposes the section's net
    VC  = sum_p sum_k max(0, -sgn(Q_p) q) LAYERED counterflow: within each face, against
                                        that face's own net
SPEED
    umax = max |u|; u95 = area-weighted 95th percentile of |u|   [m/s]
SHEAR (on the layers between adjacent cell centres of each face)
    Srms = sqrt( area-weighted mean of (du/dz)^2 + (dv/dz)^2 )   [1/s]
    dU   = width-mean u(top cell) - u(bottom cell); + = surface more into the cove [m/s]
STRATIFICATION AND MIXING
    drho  = width-mean sigma0(bottom cell) - sigma0(top cell)   [kg/m3], TEOS-10
    fcrit = share of the layer area with Ri = N^2 / S^2 < 1/4 (N^2 < 0 counts)

WIND (cove-wide, one value per hour: ~14 WRF cells cover the cove), from the surface
stress ROMS applied (sustr, svstr; pc_cove_mix box), over the cove's water cells
(rho columns <= 30), on the cove axis (ax, ay) = (-0.9789, -0.2043), mouth -> head:
    tau_al = along-cove stress [Pa], + toward the head (the forcing classes' sign)
    tau_cr = cross-cove stress [Pa], + toward the NORTH shore: normal (ay, -ax), the
             corrected sign ([[wcross-sign-bug]]; reduce_wind_cove.py had it backwards)
    tau_mag = cove mean of |tau| per cell (stress at rho points from the two faces each way)
    ustar3 = cove mean of (|tau| / rho_w)^1.5 per cell [m3/s3]: rate of wind energy input
             to surface turbulence - the wind's mixing power, as opposed to its push
    w_al   = along-cove cove-mean WRF wind [m/s] (20260806_wind_reduce.py), for reference

READING. The box is stored in compressed chunks of one hour x the whole box, so it is
read once, a month of hours at a time, and all three sections are cut from each block.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (hourly ocean_his snapshots)
       LO_output/extract/[gtagex]/box/pc_cove_mix_[ds0]_[ds1].nc (sustr, svstr; same times)
       LO_output/DM_outs/20260806_wind/wind_hourly_atm00_[ds0]_[ds1].p (WRF cove-mean wind)
OUTPUT LO_output/DM_outs/20261104_pc_flow_metrics/flow_metrics.nc   (time, sect) + wind (time)

run 20261104_pc_flow_metrics.py [-test True]
"""
import argparse
from time import time as clock

import numpy as np
import xarray as xr
import gsw

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-test', default=False, type=Lfun.boolean_string)   # True: first month only
TEST = parser.parse_args().test

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove; v is not flipped (+ = north)
# box indices of each section's u-faces (global i - 37, global j - 209), north to south; west to east
FACES = {
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
G_, RHO0 = 9.81, 1025.0
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head
WIND = ['tau_al', 'tau_cr', 'tau_mag', 'ustar3', 'w_al']
LON, LAT = -122.69, 48.23
RI_CRIT = 0.25
BLK = 744                                 # hours per read (31 days)
METRICS = ['Q', 'G', 'Qx', 'LC', 'VC', 'umax', 'u95', 'Srms', 'dU', 'drho', 'fcrit']
UNITS = {'Q': 'm3/s', 'G': 'm3/s', 'Qx': 'm3/s', 'LC': 'm3/s', 'VC': 'm3/s', 'umax': 'm/s', 'u95': 'm/s',
         'Srms': '1/s', 'dU': 'm/s', 'drho': 'kg/m3', 'fcrit': '1',
         'tau_al': 'Pa', 'tau_cr': 'Pa', 'tau_mag': 'Pa', 'ustar3': 'm3/s3', 'w_al': 'm/s'}

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
mix_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_mix_' + DS0 + '_' + DS1 + '.nc')
wind_fn = Ldir['LOo'] / 'DM_outs' / '20260806_wind' / ('wind_hourly_atm00_%s_%s.p' % (DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261104_pc_flow_metrics'
out_dir.mkdir(parents=True, exist_ok=True)


def wpercentile(x, w, p):
    """Weighted percentile p (0-1) along axis 1 of x (t, n) with weights w (t, n)."""
    i = np.argsort(x, axis=1)
    xs, ws = np.take_along_axis(x, i, axis=1), np.take_along_axis(w, i, axis=1)
    cw = np.cumsum(ws, axis=1) / ws.sum(axis=1, keepdims=True)
    j = np.argmax(cw >= p, axis=1)
    return xs[np.arange(len(x)), j]


def metrics(u, v, salt, temp, zw, zr, dy):
    """All metrics for one section over a block of hours. u, v, salt, temp, zr (t, k, p);
    zw (t, k+1, p); dy (p,). Returns a dict of (t,) arrays."""
    nt = len(u)
    dz = np.diff(zw, axis=1)                                  # (t, k, p) cell thickness
    dA = dz * dy[None, None, :]
    q = u * dA                                                # (t, k, p) transport per element

    # ------------------------------------------------------------------ transport and counterflow
    Q = q.sum(axis=(1, 2))
    G = np.abs(q).sum(axis=(1, 2))
    Qin, Qout = np.clip(q, 0, None).sum(axis=(1, 2)), np.clip(q, None, 0).sum(axis=(1, 2))
    Qx = np.minimum(Qin, -Qout)
    Qp = q.sum(axis=1)                                        # (t, p) face transport
    LC = np.clip(-np.sign(Q)[:, None] * Qp, 0, None).sum(axis=1)
    VC = np.clip(-np.sign(Qp)[:, None, :] * q, 0, None).sum(axis=(1, 2))

    # ------------------------------------------------------------------ speed
    au = np.abs(u).reshape(nt, -1)
    umax = au.max(axis=1)
    u95 = wpercentile(au, dA.reshape(nt, -1), 0.95)

    # ------------------------------------------------------------------ shear, stratification, Ri on the layers between cell centres
    dzr = np.diff(zr, axis=1)                                 # (t, k-1, p)
    S2 = (np.diff(u, axis=1) / dzr)**2 + (np.diff(v, axis=1) / dzr)**2
    SA = gsw.SA_from_SP(salt, -zr, LON, LAT)
    sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, temp))
    N2 = -(G_ / RHO0) * np.diff(sig, axis=1) / dzr
    wl = dzr * dy[None, None, :]                              # layer area
    Srms = np.sqrt((S2 * wl).sum(axis=(1, 2)) / wl.sum(axis=(1, 2)))
    fcrit = ((N2 < RI_CRIT * S2) * wl).sum(axis=(1, 2)) / wl.sum(axis=(1, 2))
    W = dy / dy.sum()                                         # width weights
    dU = ((u[:, -1, :] - u[:, 0, :]) * W).sum(axis=1)
    drho = ((sig[:, 0, :] - sig[:, -1, :]) * W).sum(axis=1)
    return dict(Q=Q, G=G, Qx=Qx, LC=LC, VC=VC, umax=umax, u95=u95, Srms=Srms, dU=dU, drho=drho, fcrit=fcrit)


def wind(T):
    """Cove-wide hourly wind-stress metrics from the stress ROMS applied, and the WRF along-cove wind."""
    import pandas as pd
    m = xr.open_dataset(mix_fn)
    assert np.array_equal(m.ocean_time.values, T)       # same instants as the velocity box
    wet = m.mask_rho.values == 1
    wet[:, 31:] = False                                 # the cove: rho columns <= 30
    su = np.nan_to_num(m.sustr.values)                  # (t, eta_u, xi_u), land = 0
    sv = np.nan_to_num(m.svstr.values)                  # (t, eta_v, xi_v)
    m.close()
    tx = np.full((len(T),) + wet.shape, np.nan)
    ty = np.full((len(T),) + wet.shape, np.nan)
    tx[:, :, 1:-1] = 0.5 * (su[:, :, 1:] + su[:, :, :-1])   # stress at interior rho points
    ty[:, 1:-1, :] = 0.5 * (sv[:, 1:, :] + sv[:, :-1, :])
    ok = wet.copy()
    ok[:, [0, -1]] = False
    ok[[0, -1], :] = False
    tx, ty = tx[:, ok], ty[:, ok]                       # (t, cells)
    mag = np.hypot(tx, ty)
    W = pd.read_pickle(wind_fn)['W']
    w_al = (W.u_pc * AXIS[0] + W.v_pc * AXIS[1]).reindex(pd.DatetimeIndex(T), method='nearest').to_numpy()
    TX, TY = tx.mean(axis=1), ty.mean(axis=1)
    print('wind: %d cove cells' % ok.sum())
    return dict(tau_al=TX * AXIS[0] + TY * AXIS[1], tau_cr=TX * AXIS[1] - TY * AXIS[0],
                tau_mag=mag.mean(axis=1), ustar3=((mag / RHO0)**1.5).mean(axis=1), w_al=w_al)


# ------------------------------------------------------------------ main: read the box a block of hours at a time
ds = xr.open_dataset(in_fn)
T = ds.ocean_time.values
NT = BLK if TEST else len(T)
OUT = {SN: {k: np.full(NT, np.nan) for k in METRICS} for SN in FACES}
tt0 = clock()
for t0 in range(0, NT, BLK):
    tt = slice(t0, min(t0 + BLK, NT))
    B = {vn: ds[vn].isel(ocean_time=tt).to_numpy().astype(float) for vn in ['u', 'v', 'salt', 'temp', 'z_w', 'z_rho']}
    for SN, F in FACES.items():
        XI_U, ETA = F['XI_U'], F['ETA']
        IRM, IRP = XI_U, XI_U + 1
        face = lambda x: x[:, :, ETA][:, :, :, [IRM, IRP]].mean(axis=-1)               # rho -> face, (t, k, p)
        u = SIGN * B['u'][:, :, ETA, XI_U]
        v = np.nan_to_num(np.concatenate([B['v'][:, :, ETA + dj][:, :, :, [IRM, IRP]] for dj in (-1, 0)], axis=-1)).mean(axis=-1)
        dy = (1 / ds.pn.values[ETA][:, [IRM, IRP]]).mean(axis=-1)
        M = metrics(u, v, face(B['salt']), face(B['temp']), face(B['z_w']), face(B['z_rho']), dy)
        for k in METRICS:
            OUT[SN][k][tt] = M[k]
    print('  hours %d-%d of %d, %.0f s' % (tt.start, tt.stop, NT, clock() - tt0))
ds.close()
WD = {k: x[:NT] for k, x in wind(T).items()}

# ------------------------------------------------------------------ save
dsout = xr.Dataset({k: (('time', 'sect'), np.stack([OUT[SN][k] for SN in FACES], axis=1), {'units': UNITS[k]}) for k in METRICS},
                   coords={'time': T[:NT], 'sect': list(FACES)},
                   attrs={'source': str(in_fn), 'sign': 'u, Q, dU positive into the cove (west); tau_al + toward the head, tau_cr + toward the north shore',
                          'note': 'hourly snapshots; see the docstring of 20261104_pc_flow_metrics.py for definitions'})
for k in WIND:
    dsout[k] = (('time',), WD[k], {'units': UNITS[k]})
fn = out_dir / ('flow_metrics%s.nc' % ('_test' if TEST else ''))
dsout.to_netcdf(fn)
print('saved ' + str(fn))
for SN in FACES:
    o = OUT[SN]
    print('%s: Qx/G %.2f, LC/G %.2f, VC/G %.2f, u95 %.1f cm/s, umax %.1f cm/s, Srms %.3f 1/s, dU %+.1f cm/s, drho %.2f, fcrit %.2f'
          % (SN, o['Qx'].sum() / o['G'].sum(), o['LC'].sum() / o['G'].sum(), o['VC'].sum() / o['G'].sum(), 100 * o['u95'].mean(),
             100 * o['umax'].mean(), o['Srms'].mean(), 100 * o['dU'].mean(), o['drho'].mean(), o['fcrit'].mean()))
print('wind: tau_al mean %+.4f Pa, tau_cr mean %+.4f Pa, |tau| %.4f Pa, corr(tau_al, w_al) %.2f'
      % (WD['tau_al'].mean(), WD['tau_cr'].mean(), WD['tau_mag'].mean(), np.corrcoef(WD['tau_al'], WD['w_al'])[0, 1]))
