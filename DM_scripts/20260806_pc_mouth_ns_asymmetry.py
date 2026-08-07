"""
North vs south asymmetry at the Penn Cove mouth over a tidal cycle.

Reads tidal_composite.csv written by 20260806_pc_mouth_salinity_tides.py and
puts the two points on the SAME axes, plus their difference, which is the only
way to see that the surface and the bed are doing two different things:

  BED      north and south are in phase (shape correlation +0.95), both peaking
           ~3 h after high water, with the north amplitude 1.75x the south.
           That is along-channel advection, amplified on the inflow side.

  SURFACE  north and south are in ANTIphase (shape correlation -0.87). North is
           saltiest at high water, south is saltiest at low water. Along-channel
           advection cannot do that -- it would move both sides the same way.
           A cross-mouth oscillation of the lateral salinity front can: the mean
           surface gradient across the mouth is ~0.4 g/kg (north fresher), and
           swinging that front north/south alternately freshens one side and
           salts the other.

The difference panels carry +/- 2 standard errors of the DIFFERENCE
(sqrt(se_n^2 + se_s^2)), so a band that clears zero is a real asymmetry at that
point in the cycle, not two noisy curves being over-read.

Runs on the mac -- it only needs the CSV.

run 20260806_pc_mouth_ns_asymmetry.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_pc_mouth_salinity_tides'
C = pd.read_csv(out_dir / 'tidal_composite.csv')
C['se'] = C['std'] / np.sqrt(C['n'])

LCOLOR = {'north': '#0072B2', 'south': '#D55E00'}
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
VARS = [('s_top', 'surface salinity'), ('s_bot', 'bottom salinity'),
        ('dstrat', 'bottom - surface')]


def get(vn, ph, loc):
    return (C[(C['var'] == vn) & (C.phase == ph) & (C.location == loc)]
            .sort_values('hrs_from_HW'))


fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True,
                        layout='constrained')

for j, (vn, lab) in enumerate(VARS):
    # ---- row 0: the two points overlaid
    ax = axs[0][j]
    for loc in ['north', 'south']:
        for ph, ls, a in [('spring', '-', 1.0), ('neap', '--', 0.55)]:
            g = get(vn, ph, loc)
            ax.plot(g.hrs_from_HW, g.mean_anomaly, ls, lw=2, alpha=a,
                    color=LCOLOR[loc], label='%s %s' % (loc, ph))
            if ph == 'spring':
                ax.fill_between(g.hrs_from_HW, g.mean_anomaly - 2 * g.se,
                                g.mean_anomaly + 2 * g.se, color=LCOLOR[loc],
                                alpha=0.15, lw=0)
    gN, gS = get(vn, 'spring', 'north'), get(vn, 'spring', 'south')
    r = np.corrcoef(gN.mean_anomaly, gS.mean_anomaly)[0, 1]
    pk = {loc: get(vn, 'spring', loc).set_index('hrs_from_HW')
          .mean_anomaly.idxmax() for loc in ['north', 'south']}
    amp = {loc: np.ptp(get(vn, 'spring', loc).mean_anomaly)
           for loc in ['north', 'south']}
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.grid(**GRID)
    ax.set_title('%s\nshape corr %+.2f   |   peak N %+.0f h, S %+.0f h   |   '
                 'amp N/S %.2f' % (lab, r, pk['north'], pk['south'],
                                   amp['north'] / amp['south']), fontsize=10)
    if j == 0:
        ax.set_ylabel('tidal anomaly (g kg$^{-1}$)\nsubtidal removed')
        ax.legend(fontsize=7, ncol=2)

    # ---- row 1: north minus south, with the error of the difference
    ax = axs[1][j]
    for ph, ls, a in [('spring', '-', 1.0), ('neap', '--', 0.55)]:
        gN, gS = get(vn, ph, 'north'), get(vn, ph, 'south')
        d = gN.mean_anomaly.values - gS.mean_anomaly.values
        se = np.hypot(gN.se.values, gS.se.values)
        x = gN.hrs_from_HW.values
        ax.plot(x, d, ls, lw=2, alpha=a, color='k', label=ph)
        if ph == 'spring':
            ax.fill_between(x, d - 2 * se, d + 2 * se, color='k', alpha=0.15,
                            lw=0)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.grid(**GRID)
    ax.set_xlabel('hours from high water')
    ax.set_title('north minus south', fontsize=10)
    if j == 0:
        ax.set_ylabel('N - S (g kg$^{-1}$)')
        ax.legend(fontsize=8)
    # flood half of the cycle
    y0, y1 = ax.get_ylim()
    ax.fill_between([-6.5, 0], y0, y1, color='#4565e8', alpha=0.07, lw=0,
                    zorder=0)
    ax.set_ylim(y0, y1)
    ax.set_xlim(-6.5, 6.5)

fig.suptitle('wb1_t0_xn11abbur00 -- Penn Cove mouth (pc_lp), north vs south '
             'through the tidal cycle\n'
             'bed: in phase, north amplified (along-channel).  '
             'surface: in ANTIphase (cross-mouth front).  '
             'blue = flood', fontsize=12)
fn = out_dir / 'fig5_north_south_asymmetry.png'
fig.savefig(fn, dpi=200, bbox_inches='tight')
print('saved %s' % fn)

# ------------------------------------------------------------------ table ---
rows = []
for vn, lab in VARS:
    for ph in ['spring', 'neap']:
        gN, gS = get(vn, ph, 'north'), get(vn, ph, 'south')
        d = gN.mean_anomaly.values - gS.mean_anomaly.values
        se = np.hypot(gN.se.values, gS.se.values)
        x = gN.hrs_from_HW.values
        rows.append(dict(
            var=vn, phase=ph,
            shape_corr=np.corrcoef(gN.mean_anomaly, gS.mean_anomaly)[0, 1],
            amp_north=np.ptp(gN.mean_anomaly), amp_south=np.ptp(gS.mean_anomaly),
            amp_ratio=np.ptp(gN.mean_anomaly) / np.ptp(gS.mean_anomaly),
            peak_h_north=x[np.argmax(gN.mean_anomaly.values)],
            peak_h_south=x[np.argmax(gS.mean_anomaly.values)],
            max_abs_diff=np.max(np.abs(d)),
            h_of_max_diff=x[np.argmax(np.abs(d))],
            n_bins_diff_sig=int(np.sum(np.abs(d) > 2 * se))))
R = pd.DataFrame(rows)
R.to_csv(out_dir / 'north_south_asymmetry.csv', index=False,
         float_format='%.4f')
print('\n' + R.round(3).to_string(index=False))
print('\nsaved %s' % (out_dir / 'north_south_asymmetry.csv'))
