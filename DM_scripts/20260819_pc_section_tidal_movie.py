"""
Tidal-cycle movie of the flow and density through the three Penn Cove
cross-sections, for an SSH-MATCHED pair of tidal days in September and December.

WHY MATCHED. Penn Cove's residual flow is ~1.8x faster in Nov-Dec than in
Aug-Oct, but wind, tide AND stratification all peak together in December and
all bottom out in September (daily partial correlations, all significant:
speed~strat|wind,tide +0.50; speed~tide|strat,wind +0.48; speed~wind|strat,tide
+0.42). A naive September-vs-December contrast therefore confounds all three.
Matching the two windows on hourly SSH removes the TIDAL part of that by
construction, so what is left between the columns is stratification and wind.
Scoring follows 20260811_pc_matched_weeks.py: cost = RMS(ssh_A - ssh_B) over
the window with each window's own mean removed (so the seasonal steric offset
drops out), and candidate windows may only start at a higher high water, so the
two are phase-locked rather than merely similar in range.

WHAT IS DRAWN, per frame
  rows 1-2   section-normal velocity u at pc_cp / pc_lj / pc_lp (head -> mouth),
             September on row 1, December on row 2. NEGATED so positive is INTO
             the cove, consistent with [[tef2-wb1-pc1-collection]].
  rows 3-4   potential density sigma0 at the same sections, same two months.
  row 5      left  hourly SSH for both windows, means removed, with a cursor.
                   This is the panel that shows the matching actually holds.
             right along-cove and cross-cove wind for both windows, same cursor.
                   Cross-cove sign is computed here from raw u_pc/v_pc, NOT
                   taken from the reduction -- see [[wcross-sign-bug]].

Velocity is on the native u faces and density at rho points; eta_u == eta_rho,
so the two rows share a latitude axis with no interpolation. Colour scales are
FIXED across the whole movie and shared between the two months, or the seasonal
difference the movie is about would be normalised away frame by frame.

COMPANION STILL. --tides writes a separate figure: a utide harmonic solve on
the 2-year cove-mean SSH (constituent amplitudes and phases), plus the
reconstruction over the two chosen windows with the major constituents drawn
separately, so the movie's forcing can be read as a sum of knowns.

run 20260819_pc_section_tidal_movie.py
run 20260819_pc_section_tidal_movie.py --hours 49 --vformat prores
"""
import argparse
import warnings

import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-job', default='pc_cove', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('--hours', default=25, type=int, help='window length (25 = one tidal day)')
p.add_argument('--monthA', default=9, type=int)
p.add_argument('--monthB', default=12, type=int)
p.add_argument('--dpi', default=110, type=int)
p.add_argument('--fps', default=4, type=int)
p.add_argument('--vformat', default='mp4', choices=['mp4', 'prores', 'qtrle'])
p.add_argument('--tides', default=True, type=Lfun.boolean_string)
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname='wb1')
box_fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'box' /
          ('%s_%s_%s.nc' % (args.job, args.ds0, args.ds1)))
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind' /
           ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260819_pc_section_tidal_movie'
Lfun.make_dir(out_dir)

CB = dict(blue='#0072B2', red='#CC0000', green='#009E73', orange='#D55E00',
          purple='#CC79A7', grey='#7f7f7f')
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 12
SECTS = [('pc_cp (Coupeville)', -122.7218), ('pc_lj (mid)', -122.6936),
         ('pc_lp (mouth)', -122.6534)]
# along-cove unit vector (mouth -> head, 255 deg true) and cross-cove (-> N shore)
EA = np.array([-0.966, -0.259])
EC = np.array([-0.259, 0.966])

ds = xr.open_dataset(box_fn)
lon_r = ds.lon_rho.values[0, :]
lon_u = ds.lon_u.values[0, :]
lat_r = ds.lat_rho.values[:, 0]
mask = ds.mask_rho.values.astype(bool)
tt = pd.to_datetime(ds.ocean_time.values)
print('box %s : %d times' % (box_fn.name, len(tt)))

# ---------------------------------------------------------------- ssh matching
ssh = pd.Series(np.nanmean(np.where(mask, ds.zeta.values, np.nan), axis=(1, 2)), index=tt)
N = args.hours


def higher_high_starts(month):
    """Indices of higher high waters in `month`, >= 20 h apart (keeps the larger
    of each day's pair), with a full window available after each."""
    s = ssh.values
    cand = []
    for k in range(1, len(s) - 1):
        if tt[k].month != month or k + N > len(s):
            continue
        if s[k] > s[k - 1] and s[k] >= s[k + 1]:
            cand.append(k)
    keep = []
    for k in cand:
        if keep and (tt[k] - tt[keep[-1]]) < pd.Timedelta(hours=20):
            if s[k] > s[keep[-1]]:
                keep[-1] = k
        else:
            keep.append(k)
    return keep


A_starts, B_starts = higher_high_starts(args.monthA), higher_high_starts(args.monthB)
print('candidate starts: month %d -> %d, month %d -> %d'
      % (args.monthA, len(A_starts), args.monthB, len(B_starts)))
best = None
for ia in A_starts:
    a = ssh.values[ia:ia + N]
    a = a - a.mean()
    for ib in B_starts:
        b = ssh.values[ib:ib + N]
        b = b - b.mean()
        cost = np.sqrt(((a - b) ** 2).mean())
        if best is None or cost < best[0]:
            best = (cost, ia, ib)
cost, iA, iB = best
WIN = {'A': (iA, tt[iA]), 'B': (iB, tt[iB])}
print('MATCHED: %s (month %d) vs %s (month %d), RMS ssh difference %.4f m over %d h'
      % (tt[iA], args.monthA, tt[iB], args.monthB, cost, N))
print('   window ssh range A %.2f m, B %.2f m'
      % (np.ptp(ssh.values[iA:iA + N]), np.ptp(ssh.values[iB:iB + N])))

# ---------------------------------------------------------------- section data
DAT = {}
for lab, xl in SECTS:
    iu = int(np.argmin(np.abs(lon_u - xl)))
    ir = int(np.argmin(np.abs(lon_r - xl)))
    wet = np.flatnonzero(mask[:, ir])
    y = (lat_r[wet] - lat_r[wet].mean()) * 111.32
    d = {'y': y, 'iu': iu, 'ir': ir, 'wet': wet}
    for tag, i0 in (('A', iA), ('B', iB)):
        sl = slice(i0, i0 + N)
        u = -ds.u.isel(xi_u=iu, eta_u=wet).values[sl]          # + = INTO the cove
        zwr = ds.z_w.isel(xi_rho=ir, eta_rho=wet).values[sl]
        zr = 0.5 * (zwr[:, :-1, :] + zwr[:, 1:, :])
        salt = ds.salt.isel(xi_rho=ir, eta_rho=wet).values[sl]
        temp = ds.temp.isel(xi_rho=ir, eta_rho=wet).values[sl]
        SA = gsw.SA_from_SP(salt, -zr, lon_r[ir], lat_r[wet].mean())
        sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, temp))
        d[tag] = dict(u=u, sig=sig, z=zr)
    DAT[lab] = d
    print('  %-20s u face %d, rho col %d, %d wet rows' % (lab, iu, ir, len(wet)))

ULIM = max(np.nanmax(np.abs(DAT[l][t]['u'])) for l, _ in SECTS for t in 'AB')
SIGALL = np.concatenate([DAT[l][t]['sig'].ravel() for l, _ in SECTS for t in 'AB'])
SLO, SHI = np.nanpercentile(SIGALL, [1, 99])
print('fixed scales: |u| <= %.3f m/s ; sigma0 %.2f - %.2f kg/m3' % (ULIM, SLO, SHI))

W = pd.read_pickle(wind_fn)['W'].reindex(tt)
wa = W.u_pc.values * EA[0] + W.v_pc.values * EA[1]
wc = W.u_pc.values * EC[0] + W.v_pc.values * EC[1]
SER = {t: dict(ssh=ssh.values[i:i + N] - ssh.values[i:i + N].mean(),
               wa=wa[i:i + N], wc=wc[i:i + N], t=tt[i:i + N])
       for t, (i, _) in (('A', WIN['A']), ('B', WIN['B']))}

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16.5, 15.5))
gs = GridSpec(5, 3, height_ratios=[1, 1, 1, 1, 0.85], hspace=0.42, wspace=0.26,
              left=0.06, right=0.965, top=0.935, bottom=0.055)
axU = {'A': [fig.add_subplot(gs[0, c]) for c in range(3)],
       'B': [fig.add_subplot(gs[1, c]) for c in range(3)]}
axS = {'A': [fig.add_subplot(gs[2, c]) for c in range(3)],
       'B': [fig.add_subplot(gs[3, c]) for c in range(3)]}
axSSH = fig.add_subplot(gs[4, 0:2])
axW = fig.add_subplot(gs[4, 2])
MON = {'A': tt[iA].strftime('%b %Y'), 'B': tt[iB].strftime('%b %Y')}
COL = {'A': CB['orange'], 'B': CB['blue']}

art = {}
for tag in 'AB':
    for c, (lab, _) in enumerate(SECTS):
        d = DAT[lab]
        Y = np.tile(d['y'], (d[tag]['z'].shape[1], 1))
        a = axU[tag][c]
        art[('u', tag, c)] = a.pcolormesh(Y, d[tag]['z'][0], d[tag]['u'][0],
                                          cmap='RdBu_r', vmin=-ULIM, vmax=ULIM,
                                          shading='gouraud')
        a.set_title('%s  |  %s' % (lab, MON[tag]), fontsize=FS - 1)
        a.set_ylabel('z (m)' if c == 0 else '')
        b = axS[tag][c]
        art[('s', tag, c)] = b.pcolormesh(Y, d[tag]['z'][0], d[tag]['sig'][0],
                                          cmap='viridis', vmin=SLO, vmax=SHI,
                                          shading='gouraud')
        b.set_title('%s  |  %s' % (lab, MON[tag]), fontsize=FS - 1)
        b.set_ylabel('z (m)' if c == 0 else '')
        for ax_ in (a, b):
            ax_.set_xlabel('km north of section centre', fontsize=FS - 2)
plt.colorbar(art[('u', 'A', 2)], ax=[axU['A'][2], axU['B'][2]],
             label='u into cove (m s$^{-1}$)', shrink=0.85)
plt.colorbar(art[('s', 'A', 2)], ax=[axS['A'][2], axS['B'][2]],
             label=r'$\sigma_0$ (kg m$^{-3}$)', shrink=0.85)

hrs = np.arange(N)
for tag in 'AB':
    axSSH.plot(hrs, SER[tag]['ssh'], '-', color=COL[tag], lw=2, label=MON[tag])
    axW.plot(hrs, SER[tag]['wa'], '-', color=COL[tag], lw=2, label='%s along' % MON[tag])
    axW.plot(hrs, SER[tag]['wc'], '--', color=COL[tag], lw=1.6, label='%s cross' % MON[tag])
axSSH.set_ylabel('ssh anomaly (m)'); axSSH.set_xlabel('hours from higher high water')
axSSH.set_title('SSH, means removed -- RMS difference %.3f m (this is the matching)' % cost,
                fontsize=FS)
axSSH.grid(**GRID); axSSH.legend(fontsize=9, ncol=2)
axW.axhline(0, color='k', lw=0.8)
axW.set_ylabel('wind (m s$^{-1}$)'); axW.set_xlabel('hours from higher high water')
axW.set_title('along-cove (solid, + = toward head)\ncross-cove (dashed, + = toward N shore)',
              fontsize=FS - 1)
axW.grid(**GRID); axW.legend(fontsize=7.5, ncol=2)
curs = [axSSH.axvline(0, color='k', lw=1.4), axW.axvline(0, color='k', lw=1.4)]
sup = fig.suptitle('', fontsize=FS + 3)


def draw(k):
    for tag in 'AB':
        for c, (lab, _) in enumerate(SECTS):
            d = DAT[lab]
            art[('u', tag, c)].set_array(d[tag]['u'][k].ravel())
            art[('s', tag, c)].set_array(d[tag]['sig'][k].ravel())
    for cu in curs:
        cu.set_xdata([k, k])
    sup.set_text('Penn Cove sections, SSH-matched tidal day  |  hour %d of %d  |  %s  vs  %s'
                 % (k + 1, N, SER['A']['t'][k].strftime('%Y-%m-%d %H:%M'),
                    SER['B']['t'][k].strftime('%Y-%m-%d %H:%M')))
    return []


EVEN = ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']
VF = {'mp4': dict(ext='.mp4', kw=dict(codec='h264',
                                      extra_args=EVEN + ['-crf', '18', '-pix_fmt', 'yuv420p'])),
      'prores': dict(ext='.mov', kw=dict(codec='prores_ks',
                                         extra_args=['-pix_fmt', 'yuva444p10le',
                                                     '-profile:v', '4444'])),
      'qtrle': dict(ext='.mov', kw=dict(codec='qtrle', extra_args=['-pix_fmt', 'argb']))}
vf = VF[args.vformat]
stem = 'pc_sections_%s_vs_%s' % (tt[iA].strftime('%Y%m%d'), tt[iB].strftime('%Y%m%d'))
mov = out_dir / (stem + vf['ext'])
anim = FuncAnimation(fig, draw, frames=N, blit=False)
anim.save(str(mov), writer=FFMpegWriter(fps=args.fps, **vf['kw']), dpi=args.dpi,
          savefig_kwargs=dict(transparent=(args.vformat != 'mp4')))
print('wrote %s' % mov)
draw(0)
fig.savefig(out_dir / (stem + '_frame00.png'), dpi=args.dpi, transparent=True)
plt.close(fig)

# ---------------------------------------------------------------- tides still
# CONSTITUENTS ARE FIT HERE, NOT WITH UTIDE. utide 0.3.1 under numpy 2.3.2 is
# broken in this environment: constit='auto' returns ZERO constituents, and an
# explicit list returns amplitudes of 1e6-1e7 m explaining 0.1% of the variance.
# Checked against both matplotlib date2num and a days-since-year-0 datenum, so
# it is not the epoch. A plain least-squares fit at known frequencies is exact
# for this purpose and fully under our control; the only thing given up is
# nodal correction, which over a 2-year record is a small amplitude modulation
# and does not affect the phasing this movie needs.
CONS = {'M2': 0.0805114007, 'S2': 0.0833333333, 'N2': 0.0789992487,
        'K2': 0.0835614924, 'K1': 0.0417807462, 'O1': 0.0387306544,
        'P1': 0.0415525871, 'Q1': 0.0372185025, 'M4': 0.1610228013,
        'MS4': 0.1638447355, 'Mf': 0.0030500918, 'Mm': 0.0015121518}


def harm_design(t_hours, names):
    X = [np.ones_like(t_hours)]
    for n in names:
        w = 2 * np.pi * CONS[n]
        X += [np.cos(w * t_hours), np.sin(w * t_hours)]
    return np.column_stack(X)


if args.tides:
    names = list(CONS)
    th = (tt - tt[0]).total_seconds().values / 3600.0
    X = harm_design(th, names)
    beta, *_ = np.linalg.lstsq(X, ssh.values, rcond=None)
    fit = X @ beta
    amp = {n: float(np.hypot(beta[1 + 2 * k], beta[2 + 2 * k])) for k, n in enumerate(names)}
    pha = {n: float(np.degrees(np.arctan2(-beta[2 + 2 * k], beta[1 + 2 * k])) % 360)
           for k, n in enumerate(names)}
    res = ssh.values - fit
    vexp = 100 * (1 - np.var(res) / np.var(ssh.values))
    F = (amp['K1'] + amp['O1']) / (amp['M2'] + amp['S2'])
    tab = pd.DataFrame(dict(constituent=names, amplitude_m=[amp[n] for n in names],
                            phase_deg=[pha[n] for n in names])).sort_values(
        'amplitude_m', ascending=False).reset_index(drop=True)
    print('\nTIDAL CONSTITUENTS (cove-mean ssh, 2 yr, least-squares at known frequencies)')
    print(tab.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print('variance explained %.1f%% ; form factor (K1+O1)/(M2+S2) = %.2f' % (vexp, F))
    tab.to_csv(out_dir / 'tidal_constituents.csv', index=False)

    fig2, ax2 = plt.subplots(1, 3, figsize=(17, 4.6))
    t6 = tab.head(8)
    ax2[0].bar(t6.constituent, t6.amplitude_m, color=CB['blue'])
    ax2[0].set_ylabel('amplitude (m)')
    ax2[0].grid(**GRID, axis='y')
    ax2[0].set_title('constituents, cove-mean SSH\n%.1f%% of variance, F = %.2f (mixed)'
                     % (vexp, F), fontsize=FS)
    for j, tag in enumerate('AB'):
        i0w = WIN[tag][0]
        seg = slice(i0w, i0w + N)
        a = ax2[j + 1]
        a.plot(hrs, ssh.values[seg] - ssh.values[seg].mean(), 'k-', lw=2.4, label='model ssh')
        a.plot(hrs, fit[seg] - fit[seg].mean(), color=CB['red'], lw=1.5, ls='--',
               label='harmonic fit')
        for nm, colr in [('M2', CB['blue']), ('K1', CB['green']), ('O1', CB['orange']),
                         ('S2', CB['purple'])]:
            k = names.index(nm)
            w = 2 * np.pi * CONS[nm]
            one = beta[1 + 2 * k] * np.cos(w * th[seg]) + beta[2 + 2 * k] * np.sin(w * th[seg])
            a.plot(hrs, one - one.mean(), color=colr, lw=1.2, alpha=0.9,
                   label='%s (%.2f m)' % (nm, amp[nm]))
        a.set_title('%s window' % MON[tag], fontsize=FS)
        a.set_xlabel('hours from higher high water')
        a.set_ylabel('ssh anomaly (m)')
        a.grid(**GRID)
        a.legend(fontsize=8, ncol=2)
    fig2.suptitle('Penn Cove tidal forcing over the two SSH-matched windows', fontsize=FS + 2)
    fig2.tight_layout()
    fig2.savefig(out_dir / 'pc_tidal_constituents.png', dpi=200, transparent=True)
    plt.close(fig2)
    print('wrote %s' % (out_dir / 'pc_tidal_constituents.png'))
