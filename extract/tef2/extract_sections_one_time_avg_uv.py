"""
Extract BOTH horizontal velocity components at every tef2 section face, for a
single average file.

The standard tef2 chain (extract_sections_one_time_avg.py) keeps only Huon and
Hvom, the volume flux THROUGH each face. That is all a TEF budget needs, but it
throws away half the velocity vector: the component running ALONG the section
never enters the extraction at all. So |u| from extractions_avg is a lower
bound on the true current speed, and any along-cove or rotary motion is
invisible.

Here each face gets both:

    u_norm   velocity normal to the section, sign-corrected by pm
    u_tan    velocity tangential to the section

A face is either a u-face or a v-face on the C-grid, so one component sits
exactly on the face and the other has to be interpolated from the four
staggered neighbours of the other kind:

    u-face (j,i)   normal   = u[j,i]
                   tangent  = mean of v[j-1,i], v[j,i], v[j-1,i+1], v[j,i+1]
    v-face (j,i)   normal   = v[j,i]
                   tangent  = mean of u[j,i-1], u[j,i], u[j+1,i-1], u[j+1,i]

Only unmasked neighbours are averaged, so a face against the coast uses the
two or three wet neighbours rather than pulling a zero in off the land.

SIGN CONVENTION
n_hat is the section normal after pm is applied. t_hat = z_hat cross n_hat, the
normal rotated 90 degrees counterclockwise, which gives
    u-face:  u_norm = pm*u,  u_tan =  pm*v
    v-face:  u_norm = pm*v,  u_tan = -pm*u
Multiplying both components by pm is a 180 degree rotation, so handedness is
preserved and (u_norm, u_tan) is a proper right-handed pair at every face
regardless of which way the section was drawn.

q is carried through as well, straight from Huon/Hvom, so that
q/(dd*DZ) can be checked against u_norm -- they should agree to roundoff.

Called by extract_sections_avg_uv.py.
"""

from argparse import ArgumentParser
from xarray import open_dataset, Dataset
import numpy as np
from pandas import read_pickle

parser = ArgumentParser()
parser.add_argument('-sect_df_fn', type=str)   # path to sect_df
parser.add_argument('-in_fn', type=str)        # path to average file
parser.add_argument('-out_fn', type=str)       # path to outfile (temp directory)
args = parser.parse_args()

sect_df = read_pickle(args.sect_df_fn)
ds = open_dataset(args.in_fn, decode_times=False)
# decode_times=False is important for correct treatment of the time axis when
# the calling function concatenates with ncrcat

# grid info
DX = 1 / ds.pm.values
DY = 1 / ds.pn.values
dxv = DX[:-1, :] + np.diff(DX, axis=0) / 2     # DX on the v-grid
dyu = DY[:, :-1] + np.diff(DY, axis=1) / 2     # DY on the u-grid

u_df = sect_df[sect_df.uv == 'u']
v_df = sect_df[sect_df.uv == 'v']

CC = dict()
h = ds.h.values
CC['h'] = (h[sect_df.jrp, sect_df.irp] + h[sect_df.jrm, sect_df.irm]) / 2
dd = np.nan * np.ones(CC['h'].shape)
dd[v_df.index] = dxv[v_df.j, v_df.i]
dd[u_df.index] = dyu[u_df.j, u_df.i]
CC['dd'] = dd

aa = ds.zeta.values.squeeze()
CC['zeta'] = (aa[sect_df.jrp, sect_df.irp] + aa[sect_df.jrm, sect_df.irm]) / 2

# velocity on the two staggered grids, packed (z, eta, xi)
U = ds.u.values.squeeze()
V = ds.v.values.squeeze()
mask_u = ds.mask_u.values
mask_v = ds.mask_v.values
NZ, NJU, NIU = U.shape
_, NJV, NIV = V.shape
NP = len(sect_df)


def neighbour_mean(A, mask, jj, ii, offsets):
    """Mean of A over the four staggered neighbours, wet points only.

    A is (z, eta, xi) on one velocity grid, jj/ii index faces on the OTHER
    velocity grid, and offsets are the (dj, di) pairs that step from a face to
    its neighbours. Out-of-range and masked neighbours are dropped rather than
    counted as zero, which matters at the coast where a dry point would
    otherwise bias the tangential velocity toward zero.
    """
    NJ, NI = A.shape[1], A.shape[2]
    acc = np.zeros((A.shape[0], len(jj)))
    wsum = np.zeros((A.shape[0], len(jj)))
    for dj, di in offsets:
        jn, iN = jj + dj, ii + di
        good = (jn >= 0) & (jn < NJ) & (iN >= 0) & (iN < NI)
        jc, ic = np.clip(jn, 0, NJ - 1), np.clip(iN, 0, NI - 1)
        val = A[:, jc, ic]
        wet = good & (mask[jc, ic] > 0)
        w = np.broadcast_to(wet, val.shape)
        acc += np.where(w, np.nan_to_num(val, nan=0.0), 0.0)
        wsum += w
    return np.where(wsum > 0, acc / np.maximum(wsum, 1), np.nan)


u_norm = np.nan * np.ones((NZ, NP))
u_tan = np.nan * np.ones((NZ, NP))
q = np.nan * np.ones((NZ, NP))

if len(u_df) > 0:
    jj, ii = u_df.j.to_numpy(), u_df.i.to_numpy()
    pm = u_df.pm.to_numpy().reshape(1, -1)
    u_norm[:, u_df.index] = U[:, jj, ii] * pm
    # t_hat = z_hat x n_hat, so the tangent of a u-face is +v, times pm
    u_tan[:, u_df.index] = neighbour_mean(
        V, mask_v, jj, ii, [(-1, 0), (-1, 1), (0, 0), (0, 1)]) * pm
    q[:, u_df.index] = ds.Huon.values.squeeze()[:, jj, ii] * pm

if len(v_df) > 0:
    jj, ii = v_df.j.to_numpy(), v_df.i.to_numpy()
    pm = v_df.pm.to_numpy().reshape(1, -1)
    u_norm[:, v_df.index] = V[:, jj, ii] * pm
    # and the tangent of a v-face is -u, times pm
    u_tan[:, v_df.index] = -neighbour_mean(
        U, mask_u, jj, ii, [(0, -1), (0, 0), (1, -1), (1, 0)]) * pm
    q[:, v_df.index] = ds.Hvom.values.squeeze()[:, jj, ii] * pm

CC['u_norm'] = u_norm
CC['u_tan'] = u_tan
CC['q'] = q

ot = ds.ocean_time.values
attrs = {'units': ds.ocean_time.units}
ds1 = Dataset()
ds1['time'] = (('time'), ot, attrs)
ds1['h'] = (('p'), CC['h'])
ds1['dd'] = (('p'), CC['dd'])
ds1['zeta'] = (('time', 'p'), CC['zeta'].reshape(1, NP))
for vn in ['u_norm', 'u_tan', 'q']:
    # float32 halves the archive and halves what has to come back over scp.
    # Velocities are O(0.1) m s-1 and float32 carries ~7 significant digits,
    # so this costs nothing that any of these analyses can resolve.
    ds1[vn] = (('time', 'z', 'p'),
               CC[vn].reshape(1, NZ, NP).astype('float32'))
ds1.to_netcdf(args.out_fn, unlimited_dims='time')
