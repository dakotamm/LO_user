"""
Where is the DIVIDE in the two-layer flow at the Penn Cove mouth, and how does it
move with tidal stage, the tidal cycles and the season?

pc_cove box, mouth section (last rho column west of pc_lp), hourly TOTAL
velocity, along-cove component (+ = OUT). Each column is interpolated to fixed
depths BELOW THE MOVING SURFACE (0.25 to 26 m, every 0.5 m), then averaged
across the section (columns too shallow for a depth drop out), giving one
profile per hour for the whole section and one each for the north and south
halves.

Layers as in 20261027_pc_mouth_layers.py: runs of one sign with |U| >= 1 cm/s,
at least 2 m thick, same-sign neighbours merged. For TWO-layer hours:
  d_int    depth of the divide (zero crossing between the two runs) [m below
           the surface], and as a fraction of the section-mean water depth
  orient   'out over in' (surface out, deep in) or 'in over out'
  U_up, U_lo, H_up   mean velocity of each layer, upper-layer thickness
Stage = section-mean velocity / that day's peak (as before); bins = band-fit
spring/neap, tropic/equatorial flags; seasons = trimesters.

Output: LO_output/DM_outs/20261028_pc_mouth_interface/
  hourly_interface.csv, report.txt (figures: 20261028_pc_water_column_schematic.py)

  python 20261028_pc_mouth_interface.py
"""

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
box_fn = Ldir['LOo'] / 'extract' / GTX / 'box' / ('%s_%s_%s.nc' % (JOB, DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_mouth_interface'
Lfun.make_dir(out_dir)
OUT = np.array([0.9789, 0.2043])
UMIN, HMIN, DZG = 0.01, 2.0, 0.5
DG = np.arange(0.25, 26.0, DZG) # depth below the surface [m]

ds = xr.open_dataset(box_fn)
tt = pd.DatetimeIndex(ds.ocean_time.values)
lon, lat = ds.lon_rho.values, ds.lat_rho.values
mask = ds.mask_rho.values == 1
LON_LP = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()
i = int(np.flatnonzero(lon[0] < LON_LP)[-1])
jw = np.array([j for j in np.flatnonzero(mask[:, i]) if 0 < j < mask.shape[0] - 1])
u = np.nan_to_num(ds.u[:, :, :, i - 1:i + 1].values).mean(axis=-1)[:, :, jw]
v = np.nan_to_num(ds.v[:, :, :, i].values)
v = 0.5 * (v[:, :, jw - 1] + v[:, :, jw])
zr = ds.z_rho[:, :, :, i].values[:, :, jw]
zw = ds.z_w[:, :, :, i].values[:, :, jw]
ds.close()
uo = u * OUT[0] + v * OUT[1] # (T, NZ, nj)
NT, NZ, NJ = uo.shape
dz = np.diff(zw, axis=1)
Ubar = (uo * dz).sum((1, 2)) / dz.sum((1, 2))
Hsec = (zw[:, -1] - zw[:, 0]).mean(1) # section-mean water depth
latj = lat[jw, i]
north = latj > np.median(latj)
print('mouth column %d, %d cells (%d north), %d hours' % (i, NJ, north.sum(), NT))

# profiles on depth below the surface, per column
dep = zw[:, -1:, :] - zr # (T, NZ, nj), positive down, decreasing with k
UG = np.full((NT, len(DG), NJ), np.nan, np.float32)
for j in range(NJ):
    for t in range(NT):
        d_ = dep[t, ::-1, j]
        uu = uo[t, ::-1, j]
        hcol = zw[t, -1, j] - zw[t, 0, j]
        g = DG <= hcol - 0.25
        UG[t, g, j] = np.interp(DG[g], d_, uu)
print('interpolated')

def runs_of(U):
    """[[sign, top_index, bottom_index, thickness, mean U], ...] after the 1 cm/s / 2 m rules"""
    sg = np.where(np.isfinite(U) & (np.abs(U) >= UMIN), np.sign(U), 0)
    rr = []
    for n, s_ in enumerate(sg):
        if s_ == 0:
            continue
        if rr and rr[-1][0] == s_ and rr[-1][2] == n - 1:
            rr[-1][2] = n
        else:
            rr.append([s_, n, n])
    rr = [r for r in rr if (r[2] - r[1] + 1) * DZG >= HMIN]
    merged = []
    for r in rr:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    return [[s_, a, b, (b - a + 1) * DZG, np.nanmean(U[a:b + 1])] for s_, a, b in merged]

def divide(U, r):
    """zero crossing between the upper run r[0] and the lower run r[1]"""
    a, b = r[0][2], r[1][1]
    seg = U[a:b + 1]
    d = DG[a:b + 1]
    k = np.flatnonzero(np.sign(seg[:-1]) != np.sign(seg[1:]))
    if len(k):
        k = k[0]
        return d[k] + (d[k + 1] - d[k]) * seg[k] / (seg[k] - seg[k + 1])
    return 0.5 * (d[0] + d[-1])

rows = []
for nm, sel in [('all', np.ones(NJ, bool)), ('north', north), ('south', ~north)]:
    P = np.nanmean(UG[:, :, sel], axis=2) # (T, depth)
    for t in range(NT):
        r = runs_of(P[t])
        rec = dict(time=tt[t], part=nm, NL=min(len(r), 3))
        if len(r) == 2:
            rec.update(d_int=divide(P[t], r), orient='out over in' if r[0][0] > 0 else 'in over out',
                       U_up=r[0][4], U_lo=r[1][4], H_up=DG[r[0][2]] + DZG / 2)
        if len(r) >= 3:
            rec.update(d_int=divide(P[t], r[:2]), d_int2=divide(P[t], r[1:3]),
                       orient3=''.join('o' if x[0] > 0 else 'i' for x in r[:3]))
        rows.append(rec)
R = pd.DataFrame(rows)
R = R.merge(pd.DataFrame({'time': tt, 'Ubar': Ubar, 'Hsec': Hsec}), on='time')
R['f_int'] = R.d_int / R.Hsec
dmax = R[R.part == 'all'].set_index('time').Ubar.abs().groupby(lambda x: x.normalize()).transform('max')
stage = pd.cut(R[R.part == 'all'].set_index('time').Ubar / dmax, [-1.01, -0.5, -0.15, 0.15, 0.5, 1.01],
               labels=['strong flood', 'weak flood', 'slack', 'weak ebb', 'strong ebb'])
R = R.merge(stage.rename('stage').reset_index(), on='time')
BF = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
F = pd.DataFrame({nm: BF['is_' + nm].values for nm in ['spring', 'neap', 'tropic', 'equatorial']}, index=pd.DatetimeIndex(BF.time.values).normalize())
R = R.merge(F, left_on=R.time.dt.normalize(), right_index=True, how='left').drop(columns='key_0', errors='ignore')
SEAS = {12: 'Dec-Mar', 1: 'Dec-Mar', 2: 'Dec-Mar', 3: 'Dec-Mar', 4: 'Apr-Jul', 5: 'Apr-Jul', 6: 'Apr-Jul', 7: 'Apr-Jul',
        8: 'Aug-Nov', 9: 'Aug-Nov', 10: 'Aug-Nov', 11: 'Aug-Nov'}
R['season'] = R.time.dt.month.map(SEAS)
R.to_csv(out_dir / 'hourly_interface.csv', index=False, float_format='%.4f')

q = lambda s: '%.1f m [%.1f-%.1f]' % (s.median(), s.quantile(0.25), s.quantile(0.75))
rep = ['divide depth below the moving surface: median [IQR]; two-layer hours only']
for nm in ['all', 'north', 'south']:
    A = R[R.part == nm]
    T2 = A[A.NL == 2]
    rep.append('%-5s: two-layer %.0f%% of hours; divide %s (%.2f of the water depth); out-over-in %.0f%% of two-layer hours; mean section depth %.1f m'
               % (nm, 100 * (A.NL == 2).mean(), q(T2.d_int), T2.f_int.median(), 100 * (T2.orient == 'out over in').mean(), A.Hsec.mean()))
A = R[R.part == 'all']
T2 = A[A.NL == 2]
rep.append('by orientation: ' + '; '.join('%s %s, upper %+.1f / lower %+.1f cm/s' % (o, q(g.d_int), 100 * g.U_up.mean(), 100 * g.U_lo.mean())
                                         for o, g in T2.groupby('orient')))
rep.append('by stage:')
for s_, g in T2.groupby('stage', observed=True):
    rep.append('   %-12s divide %s, out-over-in %.0f%%, upper %+.1f / lower %+.1f cm/s'
               % (s_, q(g.d_int), 100 * (g.orient == 'out over in').mean(), 100 * g.U_up.mean(), 100 * g.U_lo.mean()))
for b in ['neap', 'spring', 'equatorial', 'tropic']:
    g = T2[T2[b] == 1]
    rep.append('   %-12s divide %s, out-over-in %.0f%%' % (b, q(g.d_int), 100 * (g.orient == 'out over in').mean()))
for s_, g in T2.groupby('season'):
    rep.append('   %-12s divide %s, out-over-in %.0f%%' % (s_, q(g.d_int), 100 * (g.orient == 'out over in').mean()))
T3 = A[A.NL == 3]
rep.append('three-layer hours %.0f%%: upper divide %s, lower divide %s; patterns %s'
           % (100 * len(T3) / len(A), q(T3.d_int), q(T3.d_int2), T3.orient3.value_counts(normalize=True).round(2).to_dict()))
(out_dir / 'report.txt').write_text('\n'.join(rep) + '\n')
print('\n'.join(rep))
