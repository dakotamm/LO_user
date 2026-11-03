"""
Which way does BOTTOM water go at the Penn Cove mouth? Bottom 3 m vs top 3 m,
section means and north/south halves, from the pc_cove box mouth section (last
rho column west of pc_lp), hourly total velocity, along-cove (+ = OUT).

  mean      2-year means: bottom, top, section; bottom north vs south half
  stage     by tidal stage (slab velocity / that day's peak, as in
            20261027_pc_mouth_layers.py): bottom and top means, % hours the
            bottom flows in, % hours bottom and top flow opposite ways
  season    Godin-filtered bottom/top means and % of days the bottom flows in,
            for the trimesters Dec-Mar, Apr-Jul, Aug-Nov

Output: LO_output/DM_outs/20261028_pc_bottom_direction/{mean,stage,season}.csv
Feeds 20261028_pc_bottom_water_schematic.py.

  python 20261028_pc_bottom_direction.py
"""
import numpy as np, pandas as pd, xarray as xr
from lo_tools import Lfun, zfun
Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_bottom_direction'
Lfun.make_dir(out_dir)
d = xr.open_dataset('/Users/dakotamascarenas/LO_output/extract/wb1_t0_xn11abbur00/box/pc_cove_2024.01.01_2025.12.31.nc')
tt = pd.DatetimeIndex(d.ocean_time.values); lon = d.lon_rho.values; lat = d.lat_rho.values; mask = d.mask_rho.values == 1
LON_LP = pd.read_pickle('/Users/dakotamascarenas/LO_output/section_lines/pc_lp.p').x.mean()
i = int(np.flatnonzero(lon[0] < LON_LP)[-1]); jw = np.array([j for j in np.flatnonzero(mask[:, i]) if 0 < j < mask.shape[0] - 1])
u = np.nan_to_num(d.u[:, :, :, i-1:i+1].values).mean(-1)[:, :, jw]; v = np.nan_to_num(d.v[:, :, :, i].values); v = 0.5*(v[:, :, jw-1]+v[:, :, jw])
zw = d.z_w[:, :, :, i].values[:, :, jw]
uo = 0.9789*u + 0.2043*v; dz = np.diff(zw, axis=1); zm = 0.5*(zw[:, :-1]+zw[:, 1:])
def lay(sel, cols=slice(None)):
    w = np.where(sel, dz, 0)[:, :, cols]; return (uo[:, :, cols]*w).sum((1, 2))/w.sum((1, 2))
bot = zm - zw[:, :1] <= 3.0; top = zw[:, -1:] - zm <= 3.0
ub, us = lay(bot), lay(top); Ubar = (uo*dz).sum((1, 2))/dz.sum((1, 2))
lat_c = lat[jw, i]; north = lat_c > np.median(lat_c)
ubN, ubS = lay(bot, north), lay(bot, ~north)
H = pd.DataFrame({'ub': ub, 'us': us, 'U': Ubar, 'ubN': ubN, 'ubS': ubS}, index=tt)
x = H.U / H.U.abs().groupby(H.index.normalize()).transform('max')
H['stage'] = pd.cut(x, [-1.01, -0.5, -0.15, 0.15, 0.5, 1.01], labels=['strong flood', 'weak flood', 'slack', 'weak ebb', 'strong ebb'])
print('sign: + = OUT of the cove. bottom = bottom 3 m, section mean; top = top 3 m')
print('2-yr mean: bottom %+.2f cm/s (%+.2f km/day), top %+.2f, section %+.3f | bottom north half %+.2f, south half %+.2f cm/s'
      % (100*H.ub.mean(), H.ub.mean()*86.4, 100*H.us.mean(), 100*H.U.mean(), 100*H.ubN.mean(), 100*H.ubS.mean()))
print('%% hours bottom flows IN: %.0f%% (north half %.0f%%, south half %.0f%%)' % (100*(H.ub < 0).mean(), 100*(H.ubN < 0).mean(), 100*(H.ubS < 0).mean()))
g = H.groupby('stage', observed=True)
T = pd.DataFrame({'bottom IN %': 100*g.ub.apply(lambda s: (s < 0).mean()), 'bottom mean cm/s': 100*g.ub.mean(), 'top mean cm/s': 100*g.us.mean(),
                  'slab mean cm/s': 100*g.U.mean(), 'bottom opposite to top %': 100*g.apply(lambda d: (np.sign(d.ub) != np.sign(d.us)).mean())})
print(T.round(1).to_string())
T.to_csv(out_dir / 'stage.csv', float_format='%.3f')
pd.Series({'bottom': 100*H.ub.mean(), 'top': 100*H.us.mean(), 'section': 100*H.U.mean(), 'bottom_north': 100*H.ubN.mean(), 'bottom_south': 100*H.ubS.mean(),
           'bottom_in_pct': 100*(H.ub < 0).mean(), 'bottom_north_in_pct': 100*(H.ubN < 0).mean(), 'bottom_south_in_pct': 100*(H.ubS < 0).mean()}).to_csv(out_dir / 'mean.csv', header=['cm/s or %'])
SE = []
lp = pd.DataFrame({c: zfun.lowpass(H[c].values.astype(float), f='godin') for c in ['ub', 'us', 'U']}, index=tt)
for nm, mo in [('Dec-Mar', [12, 1, 2, 3]), ('Apr-Jul', [4, 5, 6, 7]), ('Aug-Nov', [8, 9, 10, 11])]:
    q = lp[lp.index.month.isin(mo)]
    print('%s subtidal: bottom %+.2f cm/s (IN on %.0f%% of days), top %+.2f' % (nm, 100*q.ub.mean(), 100*(q.ub.resample('D').mean() < 0).mean(), 100*q.us.mean()))
    SE.append(dict(season=nm, bottom=100*q.ub.mean(), top=100*q.us.mean(), bottom_in_days_pct=100*(q.ub.resample('D').mean() < 0).mean()))
pd.DataFrame(SE).to_csv(out_dir / 'season.csv', index=False, float_format='%.3f')
