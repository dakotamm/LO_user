"""
Salinity along the centre of Penn Cove, from the head (west) to the mouth (east)
and one column into Saratoga Passage, averaged over flood, ebb and all hours, of
wb1_t0_xn11abbur00.

CENTRE LINE. In each rho column xi of the pc_cove box, the cell(s) midway between
the north and south shore: the wet cells run j0..j1, the midpoint is (j0 + j1)/2,
and the centre is that cell, or the mean of the two cells either side when the
midpoint falls between two. This is the same rule as the two-face "centre" columns
of 20261103_pc_section_profiles.py, so at pc_cp and pc_lp it picks the same cells.
(The hand-drawn pc_ew.p line runs diagonally across cells, so it is not used.)
Distance is measured along the line, from the head, with dx = 1/pm.

FLOOD / EBB from the section-total transport at the MOUTH, Q(t) = int u dA at
pc_lp (hourly): flood Q > 0 (into the cove), ebb Q < 0; "all" = every hour. The
cove floods and ebbs together (Q at all three sections has corr +0.998 with the
mouth-mean d(zeta)/dt), so one reference serves the whole line.

AVERAGING per cell (k, xi), over the hours of a phase, weighted by cell thickness:
    S = sum_t salt dz / sum_t dz
drawn between the phase-mean cell interfaces. Bottom row: flood minus ebb.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc, hourly ocean_his
       SNAPSHOTS (salt, z_w, and u at pc_lp for Q).

OUTPUT LO_output/DM_outs/20261103_pc_alongchannel_salt/
    alongchannel_salt.png   rows flood | ebb | all | flood - ebb

CYCLE (-cycle springneap | tropeq | strat | wind | windhs | windls | cross | stratcalm; see pc_tidal_cycles.py): the same four rows, with two COLUMNS, the high
and low third of the cycle (pc_tidal_cycles.py); one color scale per row type
across both columns. Saved as alongchannel_salt_springneap.png / _tropeq.png.
Default (none): unchanged.

run 20261103_pc_alongchannel_salt.py [-cycle springneap|tropeq|strat|wind|windhs|windls|cross|stratcalm]
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cmocean

from lo_tools import Lfun
import argparse
from pc_tidal_cycles import cycle_masks, CYCLES
parser = argparse.ArgumentParser()
parser.add_argument('-cycle', default='none', choices=['none', 'springneap', 'tropeq', 'strat', 'wind', 'windhs', 'windls', 'cross', 'stratcalm'])   # split each phase by the cycle
CYCLE = parser.parse_args().cycle

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                  # section-positive (east) is out of the cove
MOUTH = dict(XI_U=30, ETA=np.arange(19, 7, -1))              # pc_lp u-faces (global i 67, j 228..217)
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}           # box xi_u of each section: between rho xi and xi+1
PHASES = ['flood', 'ebb', 'all']

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261103_pc_alongchannel_salt'
out_dir.mkdir(parents=True, exist_ok=True)


ds = xr.open_dataset(in_fn)

# ------------------------------------------------------------------ centre line: the cell(s) midway between the shores, per column
mask = ds.mask_rho.values == 1
pm = ds.pm.values
XI, JC = [], []                           # columns on the line, and the centre cell(s) of each
for i in range(mask.shape[1]):
    jw = np.flatnonzero(mask[:, i])
    if len(jw) == 0:
        continue
    mid = 0.5 * (jw.min() + jw.max())
    XI.append(i)
    JC.append(np.unique([int(np.floor(mid)), int(np.ceil(mid))]))   # one cell, or the two either side
XI = np.array(XI)
dx = np.array([np.mean(1 / pm[j, i]) for i, j in zip(XI, JC)])      # (n,) along-line cell length [m]
xe = np.concatenate([[0], np.cumsum(dx)]) / 1000                      # km from the head, cell edges
print('centre line: %d columns, %.2f km from the head (xi %d) to xi %d' % (len(XI), xe[-1], XI[0], XI[-1]))

# ------------------------------------------------------------------ flood / ebb from the transport through the mouth
u = SIGN * ds.u.isel(xi_u=MOUTH['XI_U'], eta_u=MOUTH['ETA']).to_numpy().astype(float)                 # (t, k, p) + into the cove
zwm = ds.z_w.isel(eta_rho=MOUTH['ETA'], xi_rho=[MOUTH['XI_U'], MOUTH['XI_U'] + 1]).to_numpy().mean(axis=-1)
dym = (1 / ds.pn.isel(eta_rho=MOUTH['ETA'], xi_rho=[MOUTH['XI_U'], MOUTH['XI_U'] + 1]).to_numpy()).mean(axis=-1)
Q = (u * np.diff(zwm, axis=1) * dym[None, None, :]).sum(axis=(1, 2))                                  # (t,)
MASK = {'flood': Q > 0, 'ebb': Q < 0, 'all': np.ones(len(Q), bool)}
if CYCLE != 'none':                      # each phase also split by the cycle, e.g. 'flood, spring'
    CM = cycle_masks(ds.ocean_time.values)
    MASK.update({'%s, %s' % (ph, half): MASK[ph] & CM[half] for ph in PHASES for half in CYCLES[CYCLE]})
    PHASES = list(MASK)
print('mouth: flood %.1f%% of hours, ebb %.1f%%' % (100 * MASK['flood'].mean(), 100 * MASK['ebb'].mean()))
del u, zwm

# ------------------------------------------------------------------ thickness-weighted mean salinity over each phase
# The box is stored in compressed chunks of ONE HOUR x the whole box, so reading a single
# column still decompresses every chunk. Read the whole box once, a month of hours at a time,
# and accumulate the sums for every centre-line column from each block.
NK = ds.sizes['s_rho']
NT = ds.sizes['ocean_time']
JJ = np.concatenate(JC)                   # every centre cell, flattened
II = np.concatenate([[i] * len(J) for i, J in zip(XI, JC)])
COLN = np.concatenate([[n] * len(J) for n, J in enumerate(JC)])       # which column each cell belongs to
SDZ = {ph: np.zeros((NK, len(XI))) for ph in PHASES}                  # sum salt * dz
DZ = {ph: np.zeros((NK, len(XI))) for ph in PHASES}                   # sum dz
ZWS = {ph: np.zeros((NK + 1, len(XI))) for ph in PHASES}              # sum z_w
CNT = {ph: np.zeros(len(XI)) for ph in PHASES}                        # cell-hours
BLK = 744                                 # hours per block (31 days)
for t0 in range(0, NT, BLK):
    tt = slice(t0, min(t0 + BLK, NT))
    s = ds.salt.isel(ocean_time=tt).to_numpy().astype(float)[:, :, JJ, II]   # (tb, k, cells)
    zw = ds.z_w.isel(ocean_time=tt).to_numpy()[:, :, JJ, II]                  # (tb, w, cells)
    dz = np.diff(zw, axis=1)
    for ph in PHASES:
        m = MASK[ph][tt]
        for n in range(len(XI)):
            c = COLN == n
            SDZ[ph][:, n] += (s[m][:, :, c] * dz[m][:, :, c]).sum(axis=(0, 2))
            DZ[ph][:, n] += dz[m][:, :, c].sum(axis=(0, 2))
            ZWS[ph][:, n] += zw[m][:, :, c].sum(axis=(0, 2))
            CNT[ph][n] += m.sum() * c.sum()
    print('  hours %d-%d of %d' % (tt.start, tt.stop, NT))
SAL = {ph: SDZ[ph] / DZ[ph] for ph in PHASES}                         # (k, n) mean salinity
ZW = {ph: ZWS[ph] / CNT[ph][None, :] for ph in PHASES}                # (k+1, n) mean interfaces
ds.close()
for ph in PHASES:
    print('%-5s surface %.2f to %.2f, bottom %.2f to %.2f g/kg (head to mouth)'
          % (ph, SAL[ph][-1, 0], SAL[ph][-1, -1], SAL[ph][0, 0], SAL[ph][0, -1]))


# ------------------------------------------------------------------ plot
def draw(ax, F, zw, cmap, vmin, vmax, dlev, zero=False):
    """Each centre-line column drawn between its mean interfaces, contours at the cell centres."""
    for n in range(F.shape[1]):
        cs = ax.pcolormesh(xe[n:n + 2], zw[:, n], F[:, n:n + 1], cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    X = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), F.shape)
    Z = 0.5 * (zw[1:] + zw[:-1])
    lev = np.arange(np.floor(vmin / dlev) * dlev, vmax + dlev, dlev)
    cc = ax.contour(X, Z, F, levels=lev[lev != 0] if zero else lev, colors='k', linewidths=0.5)
    if not zero:
        ax.clabel(cc, cc.levels[::2], fontsize=6, fmt='%.1f')
    else:
        ax.contour(X, Z, F, levels=[0], colors='k', linewidths=1.2)
    for sn, xu in SECT_XI_U.items():      # a section's u-face is the east edge of rho column xu
        xk = xe[np.flatnonzero(XI == xu)[0] + 1]
        ax.axvline(xk, color='w' if not zero else '0.3', lw=1, ls='--')
        ax.text(xk, 0.3, sn, ha='center', va='bottom', fontsize=8)
    ax.set_xlim(xe[0], xe[-1])            # west (head) at left, east (mouth) at right
    ax.set_ylim(None, 1.2)
    ax.grid(True, alpha=0.3)
    return cs


smin = np.floor(min(np.nanmin(SAL[ph]) for ph in PHASES) * 4) / 4
smax = np.ceil(max(np.nanmax(SAL[ph]) for ph in PHASES) * 4) / 4
key = lambda ph, half: ph if half == '' else '%s, %s' % (ph, half)
cols = [''] if CYCLE == 'none' else list(CYCLES[CYCLE])
D = {half: SAL[key('flood', half)] - SAL[key('ebb', half)] for half in cols}
dmax = max(np.nanpercentile(np.abs(D[half]), 99) for half in cols)
fig, axs = plt.subplots(4, len(cols), figsize=(12 * (1 if CYCLE == 'none' else 0.85) * len(cols), 13), sharex=True, sharey=True, layout='constrained', squeeze=False)
for j, half in enumerate(cols):
    for ax, ph in zip(axs[:3, j], ['flood', 'ebb', 'all']):
        k = key(ph, half)
        cs = draw(ax, SAL[k], ZW[k], cmocean.cm.haline, smin, smax, 0.25)
        ax.set_title('%s (%.0f%% of hours)' % (k, 100 * MASK[k].mean()), fontsize=10)
    cd = draw(axs[3, j], D[half], 0.5 * (ZW[key('flood', half)] + ZW[key('ebb', half)]), cmocean.cm.balance, -dmax, dmax, 0.05, zero=True)
    axs[3, j].set_title('flood minus ebb' + ('' if half == '' else ', ' + half), fontsize=10)
    axs[3, j].set_xlabel('distance along the centre of the cove from the head [km] (west at left, mouth at right)')
    print('%s flood - ebb: %.3f to %+.3f g/kg' % (half or 'all hours', np.nanmin(D[half]), np.nanmax(D[half])))
for ax in axs[:, 0]:
    ax.set_ylabel('z [m]')
fig.colorbar(cs, ax=axs[:3, :].ravel().tolist(), location='right', shrink=0.8, label='mean salinity [g/kg] (contours every 0.25)')
fig.colorbar(cd, ax=axs[3, :].ravel().tolist(), location='right', label='flood - ebb [g/kg]\n(contours every 0.05)')
fig.suptitle('Salinity along the centre of Penn Cove (cells midway between the shores), flood / ebb from the mouth transport;\nhourly snapshots, %s, %s to %s'
             % (GTAGEX, DS0, DS1), fontsize=10)
fig.savefig(out_dir / ('alongchannel_salt%s.png' % ('' if CYCLE == 'none' else '_' + CYCLE)), dpi=200, transparent=True)
plt.close(fig)
print('saved to ' + str(out_dir))
