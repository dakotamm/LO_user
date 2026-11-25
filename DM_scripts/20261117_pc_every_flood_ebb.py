"""
Penn Cove pc_lj (Lovejoy): every flood and every ebb of 2024 at its peak hour, as a cross-channel
averaged profile and as the depth average at each face, with the average greater (peak) and
average lesser flood / ebb drawn over them. Greater / lesser = the bigger / smaller flood (ebb) of
each lunar day (STEP 2); this is NOT the 18 h suppression picking of 20261116_pc_peak_flood_ebb.py
and 20261117_pc_lesser_flood_ebb.py, so the averages differ a little from theirs.

STEP 1  load pc_lj from the pc_cove box (hourly ocean_his snapshots, hh:00) for 2024: u normal to
        the faces (+ = into the cove), z_w at the faces (mean of the two rho cells either side),
        face width dy = 1/pn. PAD days past the year end are kept so the lunar-day picking of the
        last December tides sees the same neighbours as in the greater / lesser scripts.
STEP 2  half-cycles and their peak hours from the pc_lj section-total Q (+ = flood): runs of one sign
        of Q, runs shorter than MIN_RUN h absorbed into the run before, first and last (partial) runs
        dropped; peak = max |Q| hour of the run. Every half-cycle peaking in 2024 is one line.
        Lunar days: a flood start to the second next flood start, i.e. 4 consecutive half-cycles
        (flood, ebb, flood, ebb; the runs alternate in sign), tiled in steps of 4 from the baseline
        day's first flood (picks.csv, 2024-08-31 05:00), so the baseline day is one tile and every day
        has exactly 2 floods and 2 ebbs. (A fixed 24.84 h grid was tried first: tide times drift
        against it and flood peaks fell on its boundaries, 21 days with 1 flood and 16 with 3.)
        In each day the flood (ebb) with the larger |Q| at its peak is greater, the other lesser.
STEP 3  at every peak hour:
            profile   U_b = Q_b / A_b in EDGES (1 m) bins of depth below the moving surface, each
                      cell's transport split among the bins it overlaps (sum_b Q_b = Q exactly); a bin
                      is drawn where it is wetted by at least half a face (WET)
            faces     depth average sum_k u dz / sum_k dz at each face
        greater / lesser averages: sum_t Q_b / sum_t A_b and sum_t sum_k u dz / sum_t sum_k dz over
        their peak hours (transport-weighted, as STEP 12 of the greater / lesser scripts); a bin of
        the average profile is drawn only where it is wet in every hour averaged.
STEP 2b the floods and ebbs of the baseline day and of each variant day (lunar days, one axis moved at
        a time, tide order fEFe matched), from 20261113_pc_candidate_days/picks.csv: half-cycles whose
        peak hour is in [t0, t1) of the day, each one greater or lesser by STEP 2.
            baseline     2024-08-31 05:00   (= JJA seasonal baseline)
            spring       2024-09-02 07:00   neap        2024-09-08 22:00
            tropic       2024-08-01 04:00   equatorial  2024-09-07 21:00
            up-cove      2024-03-28 08:00   down-cove   2024-07-02 03:00
            high strat   2024-03-20 17:00   low strat   2024-10-05 20:00
STEP 4  one figure per day, 4 x 2: rows greater flood | lesser flood | greater ebb | lesser ebb,
        columns profile | depth average per face. In each row: that class's tides thin grey, their
        average thick (black greater / orange lesser), the baseline day's tide of that class dashed;
        on a variant figure the variant day's tide of that class in VCOL. + = into the cove drawn
        LEFT; profile depth axis down; faces against distance across from the north end, north at
        the TOP. One symmetric u scale per column, the same in every figure.
STEP 5  the same 4 x 2 layout with every tide of each row's class in its class colour, no averages or days.
STEP 6  class of every hour for four sets, exactly as 20261116_pc_peak_flood_ebb.py (a tide takes the
        class of its peak hour):
        springneap / tropeq  the D2 (M2 + S2 + N2 + K2) / D1 (K1 + O1 + P1 + Q1) tidal transport through
                  pc_lj reconstructed from the 20261112_pc_utide fits (fits.p, a refit template cell
                  for utide.reconstruct, its mean flow set to 0) at the 16 rho cells either side of the
                  faces, Q_band = sum_p u h dy; envelope = |hilbert(Q_band)|, EPAD days either side of
                  the year; spring / neap, tropic / equatorial = top / bottom third of hours within
                  each calendar month.
        wind      the surface stress ROMS applied (sustr, svstr; pc_cove_mix box) averaged over the
                  cove's water faces, onto the cove axis AXIS, hourly then Godin: up-cove = tau_al > TTH,
                  down-cove = tau_al < -TTH, calm = |tau_al| and |tau_cr| <= TTH; the rest (cross-cove
                  stress only) 'other'.
        strat     Godin drho = sigma0 (gsw) bottom - top s-level at pc_lj, width-weighted across the
                  faces; high / low = top / bottom third of the 2024 hours, year-wide.
        season    DJF (Jan, Feb, Dec 2024) / MAM / JJA / SON by the month of the peak hour, colours as
                  20261113_pc_baseline_profiles.py.
        daynight  day = the sun above the horizon at pc_lj at the peak hour (NOAA solar position at the
                  section's mean lon / lat), else night; follows the day length through the year.
STEP 7  one figure per set, the STEP 5 layout (4 x 2, no averages or days) with every tide coloured by
        its class (CCOL; the middle third / other light grey), drawn in time order so no class is
        painted over another. Then one figure per class of each set (STEP 7b), the same layout with that
        class's tides on top of the other classes' tides in light grey, so the spread within a class shows.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       LO_output/DM_outs/20261112_pc_utide/fits.p
       LO_output/extract/[gtagex]/box/pc_cove_mix_[ds0]_[ds1].nc
       LO_output/DM_outs/20261113_pc_candidate_days/picks.csv
OUTPUT LO_output/DM_outs/20261117_pc_every_flood_ebb/
       pc_lj_every_baseline.png
       pc_lj_every_[variant].png      spring, neap, tropic, equatorial, up-cove, down-cove, high_strat, low_strat
       pc_lj_every_by_class.png       STEP 5
       pc_lj_every_[set].png          STEP 7, set = springneap, tropeq, wind, strat, season, daynight
       pc_lj_every_[set]_[class].png  STEP 7b, one per class (spaces as _)

run 20261117_pc_every_flood_ebb.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gsw
import numpy as np
import pandas as pd
import pickle
import utide
import xarray as xr
from scipy.signal import hilbert

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'     # the box file
YEAR = 2024
PAD = pd.Timedelta(days=2)                # past the year end, so the last lunar days pick as in the greater / lesser scripts
SIGN = -1                                 # section-positive (east) is out of the cove
XI_U, ETA = 23, np.arange(14, 6, -1)      # pc_lj u faces in the box, north to south (global i 60, j 223..216)
MIN_RUN = 3                               # half-cycles shorter than this [h] are slack jitter
EDGES = np.arange(0, 31, 1.0)             # profile bins, depth below the surface [m]
COL = {'greater': 'k', 'lesser': '#EE7733'}
VARIANTS = ['spring', 'neap', 'tropic', 'equatorial', 'up-cove', 'down-cove', 'high strat', 'low strat']   # picks.csv cases
VCOL = '#0077BB'                          # variant day's tides
BAND = {'D2': ['M2', 'S2', 'N2', 'K2'],     # semidiurnal band
        'D1': ['K1', 'O1', 'P1', 'Q1']}     # diurnal band, S1 left out (mostly the daily wind)
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head (pc_forcing_states.py)
TTH = 0.01                                # wind stress threshold [Pa]
CCOL = {'spring': '#CC3311', 'neap': '#33BBEE', 'tropic': '#EE7733', 'equatorial': '#009988',
        'up-cove': '#EE3377', 'down-cove': '#0077BB', 'calm': 'k', 'high strat': '#AA3377', 'low strat': '#66CCEE',
        'DJF': '#4477AA', 'MAM': '#228833', 'JJA': '#EE6677', 'SON': '#CCBB44', 'day': '#DDAA33', 'night': '#004488',
        'middle third': '0.75', 'other': '0.75'}
EPAD = pd.Timedelta(days=30)              # reconstruction padding either side of the year, for hilbert

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
fit_fn = Ldir['LOo'] / 'DM_outs' / '20261112_pc_utide' / 'fits.p'
mix_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
pick_fn = Ldir['LOo'] / 'DM_outs' / '20261113_pc_candidate_days' / 'picks.csv'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_every_flood_ebb'
Lfun.make_dir(out_dir)

# ------------------------------------------------------------------ STEP 1: load pc_lj from the box
t0 = pd.Timestamp('%d-01-01' % YEAR)
t1 = pd.Timestamp('%d-01-01' % (YEAR + 1))
ds = xr.open_dataset(box_fn).sel(ocean_time=slice(t0, t1 + PAD))
ot = pd.DatetimeIndex(ds.ocean_time.values)
u = SIGN * ds['u'].isel(xi_u=XI_U, eta_u=ETA).values.astype(float)                   # (t, k, p) + into the cove
zw = ds['z_w'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)       # (t, w, p) at the face
dy = (1 / ds['pn'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values).mean(axis=-1)  # (p,)
h_f = ds['h'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)      # (p,) depth at the face (STEP 6)
ts = {vn: ds[vn].isel(s_rho=[0, -1], eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)
      for vn in ['salt', 'temp']}                                                    # (t, bot/top, p) for STEP 6 strat
lon_f, lat_f = ds['lon_u'].isel(eta_u=ETA, xi_u=XI_U).values, ds['lat_u'].isel(eta_u=ETA, xi_u=XI_U).values
ds.close()
dz = np.diff(zw, axis=1)                                                             # (t, k, p)
Q = (u * dz * dy).sum(axis=(1, 2))                                                   # (t,) + = flood

print('%s  %d hours  %s to %s' % (box_fn.name, len(ot), ot[0], ot[-1]))
print('pc_lj: %d faces, %d s-levels, width %.0f m; Q %+.0f to %+.0f m3/s'
      % (len(dy), u.shape[1], dy.sum(), Q.min(), Q.max()))

# ------------------------------------------------------------------ STEP 2: half-cycles, greater and lesser peak hours
def runs(x):
    """[(start, stop), ...] of constant x (stop exclusive)"""
    k = np.flatnonzero(np.diff(x) != 0) + 1
    return list(zip(np.r_[0, k], np.r_[k, len(x)]))


sg = np.sign(Q)
for a, b in runs(sg):                                    # short runs -> 0, then take the previous sign
    if b - a < MIN_RUN:
        sg[a:b] = 0
sg = pd.Series(sg).replace(0, np.nan).ffill().bfill().to_numpy()
HC = runs(sg)[1:-1]                                      # the first and last runs are partial
hc_sign = np.array([sg[a] for a, b in HC])
hc_pk = np.array([a + np.argmax(np.abs(Q[a:b])) for a, b in HC])    # peak hour of each half-cycle

picks = pd.read_csv(pick_fn, parse_dates=['t0', 't1']).set_index('case')
hc_t0 = ot[[a for a, b in HC]]                           # start hour of each half-cycle
ia = np.flatnonzero((hc_t0 == picks.loc['baseline', 't0']) & (hc_sign == 1))
assert len(ia) == 1, 'baseline t0 is not a flood start of this Q'
hc_day = (np.arange(len(HC)) - ia[0]) // 4               # lunar day of each half-cycle: F E F E from the baseline's first flood
dlen = pd.Series(hc_t0).groupby(hc_day).first().diff().shift(-1) / pd.Timedelta(hours=1)   # day lengths [h]
print('lunar days: length median %.0f h, 1-99%% %.0f-%.0f h, min %.0f, max %.0f h'
      % (dlen.median(), *dlen.quantile([.01, .99]), dlen.min(), dlen.max()))

in_yr = lambda k: k[(ot[k] >= t0) & (ot[k] < t1)]        # peaks in YEAR only
PK = {}                                                  # {phase: {'every' | 'greater' | 'lesser': peak hours}}
for phase, sgn in [('flood', 1), ('ebb', -1)]:
    m = hc_sign == sgn
    k, d = hc_pk[m], hc_day[m]
    gr = np.zeros(len(k), dtype=bool)
    for dd in np.unique(d):                              # the larger |Q| of each lunar day
        i = np.flatnonzero(d == dd)
        gr[i[np.argmax(np.abs(Q[k[i]]))]] = True
    PK[phase] = {'every': in_yr(k), 'greater': in_yr(k[gr]), 'lesser': in_yr(k[~gr])}
    print('%-5s %d half-cycles in %d: %d greater, %d lesser; |Q| at peak median: greater %.0f, lesser %.0f m3/s'
          % (phase, len(PK[phase]['every']), YEAR, len(PK[phase]['greater']), len(PK[phase]['lesser']),
             np.median(np.abs(Q[PK[phase]['greater']])), np.median(np.abs(Q[PK[phase]['lesser']]))))

# ------------------------------------------------------------------ STEP 2b: the floods and ebbs of the baseline and variant days
DAYS = ['baseline'] + VARIANTS
TIDES = {}                                               # {day: {phase: [(peak hour, 'greater' | 'lesser' | 'neither'), ...]}}
for day in DAYS:
    TIDES[day] = {}
    for phase in PK:
        k = PK[phase]['every']
        k = k[(ot[k] >= picks.loc[day, 't0']) & (ot[k] < picks.loc[day, 't1'])]
        TIDES[day][phase] = [(kk, 'greater' if kk in PK[phase]['greater'] else 'lesser' if kk in PK[phase]['lesser'] else 'neither')
                             for kk in k]
    print('%-10s %s  %s' % (day, picks.loc[day, 't0'].strftime('%Y-%m-%d %H'),
                            ';  '.join('%s %s %s Q %+5.0f' % (ph, ot[kk].strftime('%m-%d %H'), kind, Q[kk])
                                       for ph in PK for kk, kind in TIDES[day][ph])))

# ------------------------------------------------------------------ STEP 3: profile and face depth average at every peak hour
def binned_QA(u, zw, dy):
    """Q_b and A_b (t, b) in the EDGES bins of depth below the moving surface, top bin first: each cell's
    transport split among the bins it overlaps (sum_b Q_b = Q exactly). As 20261113_pc_baseline_profiles.py."""
    dz = np.diff(zw, axis=1)
    q = u * dz * dy
    dw = zw[:, -1:, :] - zw                                          # depth below the surface of each interface
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    assert d_bot.max() < EDGES[-1]
    nb = len(EDGES) - 1
    Qb, Ab = np.zeros((len(u), nb)), np.zeros((len(u), nb))
    for b in range(nb):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None)
        Qb[:, b] = (q * ov / dz).sum(axis=(1, 2))
        Ab[:, b] = (ov * dy).sum(axis=(1, 2))
    assert np.allclose(Qb.sum(axis=1), q.sum(axis=(1, 2)))
    return Qb, Ab


WET = 0.5 * dy.mean()                                                # a bin counts as wet with at least half a face
QA = {}                                                              # {phase: (Qb, Ab, UDZ, H)} at the peak hours PK[phase]['every']
L = {}                                                               # {phase: {kind: (profile, faces)}}


def avg(phase, hrs):
    """transport-weighted profile (b,) and face depth average (p,) over the peak hours hrs of phase"""
    Qb, Ab, UDZ, H = QA[phase]
    m = np.isin(PK[phase]['every'], hrs)
    prof = Qb[m].sum(axis=0) / np.maximum(Ab[m].sum(axis=0), 1e-12)
    prof = np.where((Ab[m] >= WET).all(axis=0), prof, np.nan)      # wet in every hour averaged
    return prof, UDZ[m].sum(axis=0) / H[m].sum(axis=0)


for phase in PK:
    it = PK[phase]['every']
    Qb, Ab = binned_QA(u[it], zw[it], dy)                            # (n, b)
    UDZ, H = (u[it] * dz[it]).sum(axis=1), dz[it].sum(axis=1)        # (n, p)
    QA[phase] = (Qb, Ab, UDZ, H)
    L[phase] = {'every': (np.where(Ab >= WET, Qb / np.maximum(Ab, 1e-12), np.nan), UDZ / H)}
    for kind in ['greater', 'lesser']:
        L[phase][kind] = prof, _ = avg(phase, PK[phase][kind])
        m = np.isin(it, PK[phase][kind])
        print('%-5s %-7s n %3d  profile %+5.1f to %+5.1f cm/s, faces %+5.1f to %+5.1f cm/s'
              % (phase, kind, m.sum(), 100 * np.nanmin(prof), 100 * np.nanmax(prof),
                 100 * L[phase][kind][1].min(), 100 * L[phase][kind][1].max()))

# ------------------------------------------------------------------ STEP 4: one 2 x 2 figure per day
xe = np.r_[0, np.cumsum(dy)] / 1000                                  # face edges [km] from the north end, north to south
xcf = 0.5 * (xe[1:] + xe[:-1])                                       # face centres
zcb = 0.5 * (EDGES[1:] + EDGES[:-1])                                 # bin centres, depth below the surface
XL = [np.ceil(2 * 1.05 * 100 * max(np.nanmax(np.abs(L[ph]['every'][i])) for ph in L)) / 2 for i in range(2)]   # per column, every figure
zbot = max(zcb[np.isfinite(L[ph]['every'][0]).any(axis=0)].max() for ph in L) + 1


ROWS = [(phase, kind) for phase in PK for kind in ['greater', 'lesser']]


def row_lines(axr, phase, kind):
    """profile and face lines of the tides of one class: (index into L[phase]['every'], prof, face)"""
    i = np.flatnonzero(np.isin(PK[phase]['every'], PK[phase][kind]))
    return i, L[phase]['every'][0][i], L[phase]['every'][1][i]


def day_line(axr, phase, kind, day, color, ls):
    """the tide of class kind on day, if the day has one"""
    for kk, kd in TIDES[day][phase]:
        if kd == kind:
            i = np.flatnonzero(PK[phase]['every'] == kk)[0]
            lab = '%s, %s UTC, Q %+.0f m$^3$ s$^{-1}$' % (day, ot[kk].strftime('%m-%d %H:%M'), Q[kk])
            axr[0].plot(100 * L[phase]['every'][0][i], zcb, ls, color=color, lw=2.5, label=lab)
            axr[1].plot(100 * L[phase]['every'][1][i], xcf, ls, color=color, lw=2.5, label=lab)


def dress(axs, legend=True):
    """axes limits, labels, titles and row legends of a 4 x 2 figure"""
    for r, (phase, kind) in enumerate(ROWS):
        axr = axs[r]
        for ax, xl in zip(axr, XL):
            ax.axvline(0, color='0.4', lw=0.8)
            ax.set_xlim(xl, -xl)                                     # + (into the cove) drawn LEFT
            ax.grid(color='lightgray', linestyle='--', alpha=0.5)
            if r == len(ROWS) - 1:
                ax.set_xlabel('u [cm s$^{-1}$]   \u2190 IN | OUT \u2192')
        axr[0].set_ylim(zbot, 0)
        axr[0].set_ylabel('%s %s\ndepth below the surface [m]' % (kind.upper(), phase.upper()), fontweight='bold')
        axr[1].set_ylim(xe[-1], xe[0])                               # north at the top
        axr[1].set_ylabel('distance across from the north end [km]\nS \u2190 | \u2192 N')
        if legend:
            axr[1].legend(*axr[0].get_legend_handles_labels(), loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)
    axs[0, 0].set_title('Cross-channel averaged profile', fontweight='bold')
    axs[0, 1].set_title('Depth average at each face', fontweight='bold')


plt.rcParams.update({'font.size': 12})
for day in DAYS:
    fig, axs = plt.subplots(len(ROWS), 2, figsize=(17, 22), layout='constrained', sharex='col')
    for r, (phase, kind) in enumerate(ROWS):
        axr = axs[r]
        i, prof, face = row_lines(axr, phase, kind)
        axr[0].plot(100 * prof.T, zcb, '-', color='0.5', lw=0.5, alpha=0.08)
        axr[1].plot(100 * face.T, xcf, '-', color='0.5', lw=0.5, alpha=0.08)
        axr[0].plot([], [], '-', color='0.5', lw=1.5, label='every %s %s (n = %d)' % (kind, phase, len(i)))
        lab = 'average %s %s' % (kind, phase)
        axr[0].plot(100 * L[phase][kind][0], zcb, '-', color=COL[kind], lw=3, label=lab)
        axr[1].plot(100 * L[phase][kind][1], xcf, '-', color=COL[kind], lw=3, label=lab)
        day_line(axr, phase, kind, 'baseline', COL[kind], '--')
        if day != 'baseline':
            day_line(axr, phase, kind, day, VCOL, '-')
    dress(axs)
    t0b, t1b = picks.loc['baseline', 't0'], picks.loc['baseline', 't1']
    sub = 'baseline (dashed) lunar day %s to %s UTC' % (t0b.strftime('%Y-%m-%d %H:%M'), t1b.strftime('%m-%d %H:%M'))
    if day != 'baseline':
        sub += ';  %s (blue) lunar day %s to %s UTC' % (day, picks.loc[day, 't0'].strftime('%Y-%m-%d %H:%M'),
                                                        picks.loc[day, 't1'].strftime('%m-%d %H:%M'))
    fig.suptitle('pc_lj (Lovejoy), every flood and ebb of %d at its peak hour, greater / lesser of each lunar day, %s, hourly ocean_his\n%s'
                 % (YEAR, GTAGEX, sub), fontsize=13)
    out_fn = out_dir / ('pc_lj_every_%s.png' % day.replace(' ', '_'))
    fig.savefig(out_fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % out_fn)

# ------------------------------------------------------------------ STEP 5: every tide coloured by class
fig, axs = plt.subplots(len(ROWS), 2, figsize=(17, 22), layout='constrained', sharex='col')
for r, (phase, kind) in enumerate(ROWS):
    axr = axs[r]
    i, prof, face = row_lines(axr, phase, kind)
    axr[0].plot(100 * prof.T, zcb, '-', color=COL[kind], lw=0.6, alpha=0.15)
    axr[1].plot(100 * face.T, xcf, '-', color=COL[kind], lw=0.6, alpha=0.15)
    axr[0].plot([], [], '-', color=COL[kind], lw=2, label='%s %s (n = %d)' % (kind, phase, len(i)))
dress(axs)
fig.suptitle('pc_lj (Lovejoy), every flood and ebb of %d at its peak hour, greater / lesser of each lunar day, %s, hourly ocean_his'
             % (YEAR, GTAGEX), fontsize=13)
out_fn = out_dir / 'pc_lj_every_by_class.png'
fig.savefig(out_fn, dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % out_fn)

def sun_elevation(t, lat, lon):
    """solar elevation [deg] at the UTC times t (DatetimeIndex), NOAA general solar position approximation"""
    hr = t.hour.to_numpy() + t.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (t.dayofyear.to_numpy() - 1 + (hr - 12) / 24)          # fractional year
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((hr * 60 + eqt + 4 * lon) / 4 - 180)                         # hour angle (true solar time)
    la = np.radians(lat)
    return 90 - np.degrees(np.arccos(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))


# ------------------------------------------------------------------ STEP 6: class of every hour for each set (as 20261116)
yr = (ot >= t0) & (ot < t1)
CLS = {}                                                 # {set: (hourly class labels (t,), [classes], subtitle)}

# springneap / tropeq: band envelopes from the utide fits
F = pickle.load(open(fit_fn, 'rb'))
UV_OUT = ['Lsmaj', 'Lsmin', 'theta', 'g']
jt, it_ = 10, XI_U                                       # template coef: refit one cell (box rho j 10, i 23) as 20261112_pc_utide.py
kt = np.flatnonzero((F['jj'] == jt) & (F['ii'] == it_))[0]
dsf = xr.open_dataset(box_fn)
TF = dsf.ocean_time.values
ut = np.nan_to_num(dsf['ubar'].isel(eta_u=jt, xi_u=[it_ - 1, it_]).values, nan=0.0).mean(axis=1)
vt = np.nan_to_num(dsf['vbar'].isel(eta_v=[jt - 1, jt], xi_v=it_).values, nan=0.0).mean(axis=1)
dsf.close()
coef = utide.solve(TF, ut, vt, lat=F['lat'][kt], constit=F['cons'], **F['kw'])
io = [F['cons'].index(n) for n in coef.name]             # fits.p order -> coef order
for vn in UV_OUT:
    assert np.allclose(coef[vn], F['bar'][vn][kt, io]), vn
print('template refit reproduces fits.p at box rho (%d, %d)' % (jt, it_))

tr = pd.date_range(t0 - EPAD, t1 + EPAD, freq='h')
for sn, band, (c_hi, c_lo) in [('springneap', 'D2', ('spring', 'neap')), ('tropeq', 'D1', ('tropic', 'equatorial'))]:
    ub = np.zeros((len(tr), len(ETA)))                   # (t, p) east u of the band at the faces
    for p, j in enumerate(ETA):
        for i in [XI_U, XI_U + 1]:
            k = np.flatnonzero((F['jj'] == j) & (F['ii'] == i))[0]
            c = coef.copy()
            for vn in UV_OUT:
                c[vn] = F['bar'][vn][k, io]
            c['umean'], c['vmean'] = 0.0, 0.0            # reconstruct adds the fit's mean flow back: tide only
            ub[:, p] += 0.5 * utide.reconstruct(tr, c, constit=BAND[band], verbose=False).u
    Qband = SIGN * (ub * h_f * dy).sum(axis=1)           # + into the cove
    env = pd.Series(np.abs(hilbert(Qband)), index=tr).reindex(ot)
    assert env.notna().all()
    cl = pd.Series('middle third', index=ot)
    for mo, e in env[yr].groupby(ot[yr].month):          # terciles within each month
        lo, hi = e.quantile([1 / 3, 2 / 3])
        cl[e.index[e >= hi]] = c_hi
        cl[e.index[e <= lo]] = c_lo
    CLS[sn] = (cl.to_numpy(), [c_hi, c_lo, 'middle third'],
               '%s / %s = top / bottom third of hours by the %s envelope of the pc_lj tidal transport, within each month' % (c_hi, c_lo, band))
    print('Q_%s envelope in %d: %.0f to %.0f m3/s (median %.0f)' % (band, YEAR, env[yr].min(), env[yr].max(), env[yr].median()))

# wind: Godin applied stress over the cove
dm = xr.open_dataset(mix_fn).sel(ocean_time=slice(t0, t1 + PAD))
assert (pd.DatetimeIndex(dm.ocean_time.values) == ot).all()
mask = dm['mask_rho'].values == 1
mu = mask[:, :-1] & mask[:, 1:]
mu[:, 30:] = False                                       # cove u faces: west of the mouth face (box xi_u 30)
mv = mask[:-1, :] & mask[1:, :]
mv[:, 31:] = False                                       # cove v faces: rho columns <= 30
tx = np.nanmean(dm['sustr'].values[:, mu], axis=1)
ty = np.nanmean(dm['svstr'].values[:, mv], axis=1)
dm.close()
tau_al = zfun.lowpass(tx * AXIS[0] + ty * AXIS[1], f='godin')   # + toward the head
tau_cr = zfun.lowpass(tx * AXIS[1] - ty * AXIS[0], f='godin')   # + toward the north shore
WC = np.full(len(ot), 'other', dtype=object)
WC[tau_al > TTH] = 'up-cove'
WC[tau_al < -TTH] = 'down-cove'
WC[(np.abs(tau_al) <= TTH) & (np.abs(tau_cr) <= TTH)] = 'calm'
CLS['wind'] = (WC, ['up-cove', 'down-cove', 'calm', 'other'],
               'Godin cove-mean applied stress: up-cove / down-cove = along > +%.2f / < -%.2f Pa, calm = |along| and |cross| <= %.2f Pa, '
               'other = cross-cove only' % (TTH, TTH, TTH))

# strat: Godin drho at pc_lj
SA = gsw.SA_from_SP(ts['salt'], 0, lon_f, lat_f)
sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, ts['temp']))      # (t, bot/top, p)
drho = zfun.lowpass(((sig[:, 0] - sig[:, 1]) * dy).sum(axis=1) / dy.sum(), f='godin')   # width-weighted
lo, hi = np.nanpercentile(drho[yr], [100 / 3, 200 / 3])
SC = np.full(len(ot), 'middle third', dtype=object)
SC[drho >= hi] = 'high strat'
SC[drho <= lo] = 'low strat'
CLS['strat'] = (SC, ['high strat', 'low strat', 'middle third'],
                'Godin drho = sigma0 bottom - top at pc_lj, width-weighted: high / low = top / bottom third of %d hours (> %.2f / < %.2f kg m$^{-3}$)'
                % (YEAR, hi, lo))

# season: by month
SEAS = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
CLS['season'] = (np.array([SEAS[m] for m in ot.month], dtype=object), ['DJF', 'MAM', 'JJA', 'SON'],
                 'season of the peak hour: DJF = Jan, Feb, Dec %d, MAM, JJA, SON' % YEAR)

# day / night: sun above the horizon at pc_lj
elev = sun_elevation(ot, lat_f.mean(), lon_f.mean())
CLS['daynight'] = (np.where(elev > 0, 'day', 'night').astype(object), ['day', 'night'],
                   'day = sun above the horizon at pc_lj at the peak hour (NOAA solar position), night = below')
print('day hours: %.0f%% of %d; Jun 21 %.1f h, Dec 21 %.1f h' % (100 * (elev[yr] > 0).mean(), YEAR,
      (elev[ot.normalize() == pd.Timestamp('%d-06-21' % YEAR)] > 0).sum(), (elev[ot.normalize() == pd.Timestamp('%d-12-21' % YEAR)] > 0).sum()))

for sn, (cl, order, _) in CLS.items():
    for phase, kind in ROWS:
        c = cl[PK[phase][kind]]
        print('  %-10s %-7s %-5s %s' % (sn, kind, phase, ', '.join('%s %d' % (x, (c == x).sum()) for x in order)))

# ------------------------------------------------------------------ STEP 7: every tide coloured by class, one figure per set (7) and per class (7b)
def class_figure(cl, show, subtitle, out_fn, alpha, grey, bg=False):
    """4 x 2 figure of the tides whose class (cl, hourly) is in show, coloured by class (grey classes drawn in grey);
    bg: the tides of the other classes behind, light grey"""
    fig, axs = plt.subplots(len(ROWS), 2, figsize=(17, 22), layout='constrained', sharex='col')
    for r, (phase, kind) in enumerate(ROWS):
        axr = axs[r]
        i, prof, face = row_lines(axr, phase, kind)
        cs = cl[PK[phase]['every'][i]]                   # class of each tide, time order
        if bg:
            nb = ~np.isin(cs, show)
            axr[0].plot(100 * prof[nb].T, zcb, '-', color='0.8', lw=0.5, alpha=0.3)
            axr[1].plot(100 * face[nb].T, xcf, '-', color='0.8', lw=0.5, alpha=0.3)
            axr[0].plot([], [], '-', color='0.8', lw=2, label='other classes (n = %d)' % nb.sum())
        for j in np.flatnonzero(np.isin(cs, show)):
            c = grey if CCOL[cs[j]] == '0.75' else CCOL[cs[j]]
            axr[0].plot(100 * prof[j], zcb, '-', color=c, lw=0.6, alpha=alpha)
            axr[1].plot(100 * face[j], xcf, '-', color=c, lw=0.6, alpha=alpha)
        for c in show:
            axr[0].plot([], [], '-', color=grey if CCOL[c] == '0.75' else CCOL[c], lw=2,
                        label='%s %s %s (n = %d)' % (c, kind, phase, (cs == c).sum()))
    dress(axs)
    fig.suptitle('pc_lj (Lovejoy), every flood and ebb of %d at its peak hour, greater / lesser of each lunar day, %s, hourly ocean_his\n%s'
                 % (YEAR, GTAGEX, subtitle), fontsize=13)
    fig.savefig(out_fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % out_fn)


for sn, (cl, order, subtitle) in CLS.items():
    class_figure(cl, order, subtitle, out_dir / ('pc_lj_every_%s.png' % sn), 0.25, '0.75')
    for c in order:                                      # STEP 7b: one class per figure
        class_figure(cl, [c], subtitle + '\nthis figure: %s (other classes light grey)' % c,
                     out_dir / ('pc_lj_every_%s_%s.png' % (sn, c.replace(' ', '_'))), 0.4, '0.4', bg=True)
