"""
Penn Cove environmental variable time series, built step by step, each saved as it is made.

All from box extractions of wb1_t0_xn11abbur00 ocean_his (2024-2025, hourly
instantaneous, -lt hourly0, same footprint as pc_cove). Cove only: rho columns
xi_rho <= 30; column 31 is Saratoga Passage.

STEP 1: hourly along- and cross-cove wind stress [Pa], from the pc_cove_mix box.
    sustr, svstr: the surface stress ROMS applied, on u / v faces.
    Cove faces: u faces with water on both sides and xi_u < 30 (xi_u 30 is the
    pc_lp mouth face); v faces with water on both sides in rho columns <= 30.
    Mean over cove faces, then projected on the cove axis (ax, ay) = (-0.9789, -0.2043),
    mouth -> head:
        tau_al = sustr*ax + svstr*ay    + toward the head (up-cove)
        tau_cr = sustr*ay - svstr*ax    + toward the north shore (right-hand normal)

run 20261112_pc_var_ts.py
"""
import pickle

import numpy as np
import xarray as xr

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
AX, AY = -0.9789, -0.2043

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_dir = Ldir['LOo'] / 'extract' / GTAGEX / 'box'
mix_fn = box_dir / ('pc_cove_mix_' + DS0 + '_' + DS1 + '.nc')

dm = xr.open_dataset(mix_fn)

T = dm.ocean_time.values
mask = dm.mask_rho.values == 1

mu = mask[:, :-1] & mask[:, 1:]
mu[:, 30:] = False
mv = mask[:-1, :] & mask[1:, :]
mv[:, 31:] = False

taux = dm.sustr.values[:, mu].mean(axis=1)
tauy = dm.svstr.values[:, mv].mean(axis=1)

tau_al = taux * AX + tauy * AY
tau_cr = taux * AY - tauy * AX

print('%d hours, %s to %s' % (len(T), str(T[0])[:16], str(T[-1])[:16]))
print('%d u faces, %d v faces' % (mu.sum(), mv.sum()))
print('tau_al mean %+.4f, std %.4f Pa; up-cove %.0f%% of hours' % (tau_al.mean(), tau_al.std(), 100 * np.mean(tau_al > 0)))
print('tau_cr mean %+.4f, std %.4f Pa; northward %.0f%% of hours' % (tau_cr.mean(), tau_cr.std(), 100 * np.mean(tau_cr > 0)))

"""
STEP 2: hourly east-west and north-south wind stress [Pa] over upper Saratoga
Passage, from the sp_stress box (-surf True).
    tau_ew  sustr, + toward the east
    tau_ns  svstr, + toward the north
    wb1 is a plaid lon/lat grid (no angle), so sustr is east and svstr is north.
    Saratoga Passage = the tef2 segment pc_lp_p (between pc_lp, sp_mid and
    skagit_sp, 2602 cells). Its (j, i) cells are placed in the box by the box's
    corner index on the full grid. Faces with segment cells on both sides only,
    the same rule as the cove in STEP 1.
"""
sp_fn = box_dir / ('sp_stress_surf_' + DS0 + '_' + DS1 + '.nc')
seg_fn = Ldir['LOo'] / 'extract' / 'tef2' / 'seg_info_dict_wb1_pc1_trapsN00.p'

ds = xr.open_dataset(sp_fn)
assert (ds.ocean_time.values == T).all()

G = xr.open_dataset(Ldir['grid'] / 'grid.nc')
i0 = np.argmin(np.abs(G.lon_rho.values[0, :] - ds.lon_rho.values[0, 0]))
j0 = np.argmin(np.abs(G.lat_rho.values[:, 0] - ds.lat_rho.values[0, 0]))

ji = np.array(pickle.load(open(seg_fn, 'rb'))['pc_lp_p']['ji_list'])
sp = np.zeros(ds.mask_rho.shape, dtype=bool)
sp[ji[:, 0] - j0, ji[:, 1] - i0] = True
assert (ds.mask_rho.values[sp] == 1).all()

su = sp[:, :-1] & sp[:, 1:]
sv = sp[:-1, :] & sp[1:, :]

tau_ew = ds.sustr.values[:, su].mean(axis=1)
tau_ns = ds.svstr.values[:, sv].mean(axis=1)

print('Saratoga: %d cells, %d u faces, %d v faces' % (sp.sum(), su.sum(), sv.sum()))
print('tau_ew mean %+.4f, std %.4f Pa; eastward %.0f%% of hours' % (tau_ew.mean(), tau_ew.std(), 100 * np.mean(tau_ew > 0)))
print('tau_ns mean %+.4f, std %.4f Pa; northward %.0f%% of hours' % (tau_ns.mean(), tau_ns.std(), 100 * np.mean(tau_ns > 0)))
