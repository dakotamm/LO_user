"""
Frozen-field null test for the Penn Cove exchange flow calculations.

Chen et al. 2012 (p. 752), restating MacCready 2011:

    "Note that, if the fluxes in and out of a cross section at a salinity range
    ds are identical, then dQ/ds is zero. MacCready (2011) gave an example that
    dQ/ds = 0 would occur for purely tidal advection of a 'frozen' salinity
    field, as a water parcel is advected in and out of a cross section without
    being modified."

That is the null hypothesis the whole isohaline method rests on: reversible
tidal sloshing must produce ZERO exchange flow, no matter how big the tidal
prism. This script builds a frozen field out of the real velocities and checks
whether the pipeline actually returns zero.

Construction. At each cell c take the real transport q_c and area dA_c, form
u_c = q_c/dA_c, and integrate to get the tidal displacement

    xi_c(t) = tidal part of  cumsum(u_c) dt

then assign salinity as a fixed linear function of displacement,

    s_c(t) = S0 + gamma xi_c(t)

so a parcel that is advected past the section and back returns with exactly the
salinity it left with. gamma is scaled so the synthetic tidal salinity swing
matches the real one (std ~0.54 psu at pc_lp).

This cancels analytically. In the salinity band [s, s+ds] the cell spends
dt = ds/(gamma |u_c|) on the way up carrying q_c = u_c dA_c, contributing
+dA_c ds/gamma, and the same on the way down carrying -|u_c| dA_c, contributing
-dA_c ds/gamma. They cancel exactly, independent of |u_c|. So the true answer
is Qin = 0, and whatever the pipeline returns instead is its numerical floor.

Two variants:
    A  strictly frozen -- one uniform S0, so the section has no subtidal
       salinity structure at all and both TEF and Eq. (9) must return zero.
    B  frozen tidal wiggle on top of the real per-cell subtidal mean, so the
       steady structure survives and only the tidal part is made reversible.

run 20260916_frozen_field_control.py
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-sect', default='pc_lp', type=str)
parser.add_argument('-S0', default=30.0, type=float)
args = parser.parse_args()

_fn = Path(__file__).parent / '20260916_exchange_fun.py'
_spec = importlib.util.spec_from_file_location('exchange_fun', _fn)
xfun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xfun)

Ldir = Lfun.Lstart(gridname='wb1')
S = xfun.load_section(args.sect, args.gtagex, args.ds0, args.ds1, Ldir=Ldir)

q, dA, salt = S['q'], S['dA'], S['salt']
u = q / dA
xi = np.cumsum(u, axis=0) * 3600.
xi = xi - zfun.lowpass(xi, f='godin', nanpad=False)
s_tid = salt - zfun.lowpass(salt, f='godin', nanpad=False)
gamma = np.std(s_tid) / np.std(xi)

print('%s: tidal salinity swing %.4f psu, per-cell excursion rms %.0f m, '
      'gamma %.2e psu/m' % (args.sect, np.std(s_tid), np.std(xi), gamma))

A = dict(S); A['salt'] = (args.S0 + gamma * xi).astype(np.float32)
B = dict(S); B['salt'] = (salt.mean(axis=0)[None, :, :] + gamma * xi).astype(np.float32)

print()
print('%-30s %10s %10s %10s' % ('', 'ds [psu]', 'TEF Qin', 'Eq9 Qin'))
for NS in [36, 360, 1000]:
    for nm, D in [('REAL field', S),
                  ('A: strictly frozen (truth=0)', A),
                  ('B: frozen wiggle + real mean', B)]:
        T = xfun.tef_bulk(D, NS=NS)
        E = xfun.eulerian_isohaline(D, NS=NS)
        print('%-30s %10.3f %10.1f %10.1f'
              % (nm, 36 / NS, np.nanmean(T['Qin']), np.nanmean(E['Qin'])))
    print()
