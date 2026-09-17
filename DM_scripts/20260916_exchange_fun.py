"""
Shared machinery for the Eulerian vs isohaline (TEF) exchange flow comparison
at the wb1_pc1 sections, following

  Chen, S.-N., W. R. Geyer, D. K. Ralston, and J. A. Lerczak, 2012: Estuarine
  Exchange Flow Quantified with Isohaline Coordinates: Contrasting Long and
  Short Estuaries. J. Phys. Oceanogr., 42(5), 748-763.
  doi:10.1175/JPO-D-11-086.1

The isohaline method itself is MacCready (2011, doi:10.1175/2011JPO4517.1),
which is what LO/extract/tef2 already implements.

Everything here runs off the hourly section extractions in

    LO_output/extract/[gtagex]/tef2/extractions_avg_[dates]/[sn].nc

which carry q [m3 s-1, already a volume flux in the section frame], salt, DZ,
dd and zeta at every stairstep point p and every s-level z. The pc sections are
8-12 points wide, so the whole two-year record is a few hundred MB and all of
this is cheap to do on the mac -- no apogee round trip.

THE ONE RULE: tidal averaging is always applied to q, never to a velocity and a
cell area separately. q is stored hourly as Huon/Hvom, so <q> already carries
the Stokes-type <u' dA'> correlation. Reconstructing u = q/(dd*DZ), filtering
that, and multiplying by a mean area throws that term away and gets the
subtidal transport wrong.

Tidal averaging follows bulk_calc_avg.py exactly -- Godin filter, then
[pad:-pad+1:24] with pad=36 -- so everything lands on the same daily time axis
as the existing bulk_avg_* files and can be compared to them point for point.

Definitions
-----------
Write <> for the Godin average, and index section cells by c = (z, p).

isohaline (TEF):  bin q and q*s hourly into NS salinity classes, Godin average
    each class, cumulate from the salty end to get Q(S), then hand Q(S) to
    Marvin Lorenz' multi-layer divider (tef_fun_lorenz) and collapse to two
    layers. Satisfies the Knudsen relations by construction.

Eulerian:  split the section by the sign of the subtidal transport <q>_c and
    carry the subtidal salinity <s>_c through it,

        Qin_E  = sum over {<q>_c > 0} of <q>_c
        sin_E  = sum over {<q>_c > 0} of <q>_c <s>_c  /  Qin_E

    Pairing <q>_c with <s>_c rather than with <q s>_c is the whole point: it is
    what makes the Eulerian estimate blind to tidal pumping, which is the
    quantity Chen et al. are after.

    Two variants:
      'cell'     sign taken per (z, p) cell. The section is 2D, so these
                 "layers" are not horizontal.
      'vertical' sum over p first, then split by sign of <q>(z). This is the
                 textbook two-layer Eulerian. cell - vertical isolates the
                 lateral mode.

    At pc_lp 'cell' is the one to believe. The worry with a per-cell sign split
    is that it has no noise floor -- every cell with <q> > 0 counts, so with
    360 cells it could inflate Qin out of speckle. It does not: the time-mean
    sign map is a clean north/south bisection (p 0-5 in, p 6-11 out, p=0 being
    the north end), 9 of 12 columns never change sign with depth, and the top
    20 of 360 cells carry only 25% of Qin, so no handful of cells dominates.
    That is a real lateral gyre, not noise. 'vertical' sums over p first and so
    cancels the north inflow against the south outflow before it ever takes a
    sign, throwing away most of the actual exchange -- which is why it comes
    out smaller. Treat 'vertical' as the textbook-comparable number, not the
    physically right one for this section.

salt flux decomposition:  with Q = sum_c <q>_c, A = sum_c <dA>_c and
    s0 = sum_c <s>_c <dA>_c / A, the identity

        F_total = F_mean + F_exch + F_tidal
        F_mean  = Q s0
        F_exch  = sum_c <q>_c (<s>_c - s0)
        F_tidal = sum_c <q s>_c - sum_c <q>_c <s>_c

    is exact. F_mean + F_exch is the Eulerian salt flux; F_tidal is everything
    the Eulerian analysis misses.
"""
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun
import extract.tef2.archive.tef_fun_lorenz as tfl

# matches bulk_calc_avg.py: more than the 35 nans the Godin filter leaves, and
# chosen so that subsampling by 24 lands at noon of each day
GODIN_PAD = 36

# Sign convention. create_sect_df gives every pc section u-faces with pm=+1, so
# their positive direction is EASTWARD. Penn Cove opens to the east into
# Saratoga Passage and the cove interior is to the west (pc_cp at i=52 is the
# innermost, pc_lj at i=60, pc_lp at i=67 is the mouth), so the section's own
# positive direction points OUT of the cove. Flip it, so that everywhere below
# "in" means into Penn Cove and sin > sout reads the normal estuarine way.
# sp_mid is a v-face line with pm=+1, i.e. positive is northward, which already
# points up Saratoga Passage toward the cove.
INFLOW_SIGN = {'pc_lp': -1, 'pc_cp': -1, 'pc_lj': -1, 'sp_mid': 1}

# matches process_sections_avg.py
NS_DEFAULT = 1000
S_LOW_DEFAULT = 0.0
S_HI_DEFAULT = 36.0


def section_dir(gtagex, ds0, ds1, Ldir=None, kind='extractions_avg'):
    """Path to one of the tef2 output folders for a run."""
    if Ldir is None:
        Ldir = Lfun.Lstart(gridname='wb1')
    return Ldir['LOo'] / 'extract' / gtagex / 'tef2' / (kind + '_' + ds0 + '_' + ds1)


def load_section(sn, gtagex, ds0, ds1, Ldir=None, orient=True):
    """
    Load the hourly extraction for one section.

    With orient=True (the default) q is flipped so that positive is INTO Penn
    Cove, using INFLOW_SIGN. Pass orient=False to keep the raw section frame,
    which is what you want if you are comparing against bulk_avg_* files.

    Returns a dict of numpy arrays. q, salt and DZ are (NT, NZ, NP); dd and h
    are (NP,); zeta is (NT, NP); time is (NT,).
    """
    in_dir = section_dir(gtagex, ds0, ds1, Ldir=Ldir, kind='extractions_avg')
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    S = dict()
    S['sn'] = sn
    S['time'] = ds.time.to_numpy()
    if orient:
        if sn not in INFLOW_SIGN:
            raise KeyError('no INFLOW_SIGN entry for section ' + sn
                           + ' -- work out which way its positive direction points'
                           + ' before trusting any in/out label')
        S['sign'] = INFLOW_SIGN[sn]
    else:
        S['sign'] = 1
    S['q'] = S['sign'] * ds.q.to_numpy()
    S['salt'] = ds.salt.to_numpy().astype(float)
    S['DZ'] = ds.DZ.to_numpy()
    S['dd'] = ds.dd.to_numpy()
    S['h'] = ds.h.to_numpy()
    S['zeta'] = ds.zeta.to_numpy().astype(float)
    ds.close()
    # cell face area, broadcast dd over z
    S['dA'] = S['DZ'] * S['dd'][np.newaxis, np.newaxis, :]
    return S


def godin_daily(x, pad=GODIN_PAD):
    """
    Godin filter along axis 0, then subsample to daily, exactly as
    bulk_calc_avg.py does. Works for any number of dimensions -- zfun.lowpass
    flattens in Fortran order so the convolution runs along axis 0, and the
    nan-contaminated ends of every column sit inside the pad we cut off.
    """
    return zfun.lowpass(np.asarray(x, dtype=float), f='godin')[pad:-pad+1:24, ...]


def daily_time(time, pad=GODIN_PAD):
    """The daily time axis that godin_daily() produces."""
    return time[pad:-pad+1:24]


def qprism_series(qnet, pad=GODIN_PAD):
    """
    Tidal prism transport, as defined in bulk_calc_avg.py: strip the subtidal
    part off qnet, take the absolute value, Godin average, halve it.
    """
    qnet = np.asarray(qnet, dtype=float)
    qabs = np.abs(qnet - zfun.lowpass(qnet, f='godin', nanpad=False))
    return godin_daily(qabs, pad=pad) / 2


# ----------------------------------------------------------------- isohaline

def tef_bulk(S, NS=NS_DEFAULT, S_low=S_LOW_DEFAULT, S_hi=S_HI_DEFAULT,
             nlay=31, pad=GODIN_PAD, min_trans=1):
    """
    Isohaline (TEF) exchange flow for one section.

    Reproduces process_sections_avg.py + bulk_calc_avg.py in one pass, so the
    result can be checked against the existing bulk_avg_* files. The binning is
    done with np.add.at rather than binned_statistic, which is much faster and
    numerically identical for a 'sum' statistic on a uniform grid.

    Returns a dict with the daily two-layer bulk values and the multi-layer
    arrays behind them.
    """
    q = S['q']
    salt = S['salt']
    NT, NZ, NP = q.shape

    sedges = np.linspace(S_low, S_hi, NS + 1)
    sbins = sedges[:-1] + np.diff(sedges) / 2
    DS = sedges[1] - sedges[0]

    # hourly transport in each salinity class
    tef_q = np.zeros((NT, NS))
    tef_s = np.zeros((NT, NS))
    # bin index of every cell, clipped into range so that the rare cell outside
    # [S_low, S_hi] lands in the end bin rather than being dropped
    ib = np.clip(((salt - S_low) / DS).astype(int), 0, NS - 1)
    for tt in range(NT):
        np.add.at(tef_q[tt, :], ib[tt].ravel(), q[tt].ravel())
        np.add.at(tef_s[tt, :], ib[tt].ravel(), (q[tt] * salt[tt]).ravel())

    qnet = np.nansum(q, axis=(1, 2))

    # tidally average the salinity classes, then cumulate from the salty end so
    # that Q(S) is the transport of water saltier than S
    TEF_lp = {'q': godin_daily(tef_q, pad=pad), 'salt': godin_daily(tef_s, pad=pad)}
    NTd = TEF_lp['q'].shape[0]

    MLO, Q_dict = divide_Q_of_S(TEF_lp['q'], TEF_lp['salt'], sedges,
                                nlay=nlay, min_trans=min_trans)

    out = dict()
    out['time'] = daily_time(S['time'], pad=pad)
    out['q_layer'] = MLO['q']
    out['salt_layer'] = MLO['salt']
    out['sbins'] = sbins
    out['tef_q'] = TEF_lp['q']
    out['Q_of_S'] = Q_dict['q']
    out['sedges'] = sedges
    out['qnet'] = godin_daily(qnet, pad=pad)
    out['qprism'] = qprism_series(qnet, pad=pad)
    out.update(two_layer(MLO['q'], MLO['salt'], prefix=''))
    return out


def divide_Q_of_S(binned_q, binned_s, sedges, nlay=31, min_trans=1):
    """
    Shared back end for both isohaline calculations.

    Takes transport already binned into salinity classes -- binned_q[t, n] is
    the transport in class n at time t, binned_s[t, n] the salt transport --
    cumulates from the salty end to build Q(S), and hands each time step to
    Marvin Lorenz' multi-layer divider.

    Returns (MLO, Q_dict): the multi-layer bulk values and the Q(S) curves.
    """
    NT, NS = binned_q.shape
    TEF = {'q': binned_q, 'salt': binned_s}
    vn_list = ['q', 'salt']

    Q_dict = dict()
    for vn in vn_list:
        Q = np.zeros((NT, NS + 1))
        Q[:, :-1] = np.fliplr(np.cumsum(np.fliplr(TEF[vn]), axis=1))
        Q_dict[vn] = Q

    # Collect first, size the layer axis afterwards. bulk_calc_avg.py preallocates
    # 31 layers and raises a broadcast error if the divider ever finds more; the
    # subtidal salinity field in eulerian_isohaline() does exactly that (33 at
    # pc_cp), so grow to fit rather than guess. nlay is the floor, not the cap.
    rows = []
    for dd in range(NT):
        thisQ = {vn: Q_dict[vn][dd, :] for vn in vn_list}
        in_dict, out_dict, div_sal, ind, minmax = tfl.calc_bulk_values(
            sedges, thisQ, vn_list, print_info=False, min_trans=min_trans)
        bulk = {vn: np.array(in_dict[vn] + out_dict[vn]) for vn in vn_list}
        ii = np.argsort(bulk['salt'])
        rows.append({vn: bulk[vn][ii] for vn in vn_list})

    NL = max(nlay, max(len(r['q']) for r in rows))
    MLO = {vn: np.nan * np.ones((NT, NL)) for vn in vn_list}
    for dd, r in enumerate(rows):
        n = len(r['q'])
        if n > 0:
            for vn in vn_list:
                MLO[vn][dd, :n] = r[vn]
    return MLO, Q_dict


def eulerian_isohaline(S, NS=NS_DEFAULT, S_low=S_LOW_DEFAULT, S_hi=S_HI_DEFAULT,
                       nlay=31, pad=GODIN_PAD, min_trans=1):
    """
    Chen et al. (2012) Eq. (9): the Eulerian exchange flow evaluated in
    isohaline coordinates, which is the comparison their paper actually makes.

        QEu(s) = < int int over {A : (s0+s1) > s} of (u0 + u1) dA >

    Because q = u dA exactly, <u dA>_c = Q_c and <dA>_c = a_c, so their
    integrand (u0 + u1)_c <dA>_c = (u0 + Q_c/a_c - u0) a_c = Q_c -- the u0
    cancels and Eq. (9) is just the subtidal cell transport Q_c binned by the
    subtidal cell salinity S_c = <s a>/<a> = s0 + s1.

    So this runs the same Q(S) machinery as tef_bulk(), but fed the SUBTIDAL
    fields instead of the hourly ones. The difference between the two is
    exactly the tidal flux FT, which is the point of the comparison.

    This is NOT the same as eulerian_bulk(), which splits on the sign of Q_c.
    Sorting into salinity classes lets an inflowing cell cancel an outflowing
    cell at the same subtidal salinity; a sign split never cancels. At the Penn
    Cove mouth, where the inflow and outflow sit side by side at nearly equal
    salinity, the two differ a lot. Eq. (9) is the net measure and the one to
    quote against the paper; eulerian_bulk() is a gross measure.

    Chen et al. use ds = 1 psu bins; the default here is much finer, which
    matters because Penn Cove's whole salinity range is a few psu.
    """
    q_lp = godin_daily(S['q'], pad=pad)
    dA_lp = godin_daily(S['dA'], pad=pad)
    sdA_lp = godin_daily(S['salt'] * S['dA'], pad=pad)
    with np.errstate(invalid='ignore', divide='ignore'):
        s_lp = sdA_lp / dA_lp

    NTd = q_lp.shape[0]
    sedges = np.linspace(S_low, S_hi, NS + 1)
    DS = sedges[1] - sedges[0]

    qq = q_lp.reshape(NTd, -1)
    ss = s_lp.reshape(NTd, -1)
    ib = np.clip(((ss - S_low) / DS).astype(int), 0, NS - 1)

    binned_q = np.zeros((NTd, NS))
    binned_s = np.zeros((NTd, NS))
    for tt in range(NTd):
        np.add.at(binned_q[tt, :], ib[tt], qq[tt])
        np.add.at(binned_s[tt, :], ib[tt], qq[tt] * ss[tt])

    MLO, Q_dict = divide_Q_of_S(binned_q, binned_s, sedges,
                                nlay=nlay, min_trans=min_trans)
    out = {'time': daily_time(S['time'], pad=pad),
           'q_layer': MLO['q'], 'salt_layer': MLO['salt'],
           'Q_of_S': Q_dict['q'], 'sedges': sedges}
    out.update(two_layer(MLO['q'], MLO['salt']))
    return out


def two_layer(q_layer, salt_layer, prefix=''):
    """
    Collapse Lorenz multi-layer output to two layers: add up all the inflowing
    layers and all the outflowing ones, salinity transport weighted.

    Note the layer index is rebuilt at every time step by calc_bulk_values, so
    layer 0 is not the same water on consecutive days -- only the sign matters.
    """
    q = np.asarray(q_layer, dtype=float)
    s = np.asarray(salt_layer, dtype=float)
    fin = np.where(q > 0, q, np.nan)
    fout = np.where(q < 0, q, np.nan)
    Qin = np.nansum(fin, axis=1)
    Qout = np.nansum(fout, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        sin = np.nansum(fin * s, axis=1) / Qin
        sout = np.nansum(fout * s, axis=1) / Qout
    # a day with no inflowing layer at all is a missing value, not a zero
    Qin[Qin == 0] = np.nan
    Qout[Qout == 0] = np.nan
    return {prefix + 'Qin': Qin, prefix + 'Qout': Qout,
            prefix + 'sin': sin, prefix + 'sout': sout}


# ------------------------------------------------------------------ Eulerian

def eulerian_bulk(S, mode='cell', pad=GODIN_PAD):
    """
    Eulerian exchange flow for one section: split by the sign of the subtidal
    transport and carry the subtidal salinity through it. See the module
    docstring for what 'cell' and 'vertical' mean.
    """
    q_lp = godin_daily(S['q'], pad=pad)            # (NTd, NZ, NP)
    dA_lp = godin_daily(S['dA'], pad=pad)
    # subtidal salinity, area weighted so that a cell that is thin at low tide
    # does not count as much as a thick one
    qs_lp = godin_daily(S['q'] * S['salt'], pad=pad)
    sdA_lp = godin_daily(S['salt'] * S['dA'], pad=pad)
    with np.errstate(invalid='ignore', divide='ignore'):
        s_lp = sdA_lp / dA_lp

    if mode == 'cell':
        qq = q_lp.reshape(q_lp.shape[0], -1)
        ss = s_lp.reshape(s_lp.shape[0], -1)
    elif mode == 'vertical':
        # collapse the lateral dimension first; salinity becomes the
        # area weighted mean across the section at that level
        qq = np.nansum(q_lp, axis=2)
        with np.errstate(invalid='ignore', divide='ignore'):
            ss = np.nansum(sdA_lp, axis=2) / np.nansum(dA_lp, axis=2)
    else:
        raise ValueError("mode must be 'cell' or 'vertical'")

    fin = np.where(qq > 0, qq, 0.0)
    fout = np.where(qq < 0, qq, 0.0)
    Qin = np.nansum(fin, axis=1)
    Qout = np.nansum(fout, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        sin = np.nansum(fin * ss, axis=1) / Qin
        sout = np.nansum(fout * ss, axis=1) / Qout
    Qin[Qin == 0] = np.nan
    Qout[Qout == 0] = np.nan

    out = {'Qin': Qin, 'Qout': Qout, 'sin': sin, 'sout': sout}
    if mode == 'cell':
        out.update(flux_decomp(q_lp, s_lp, dA_lp, qs_lp))
    return out


def flux_decomp(q_lp, s_lp, dA_lp, qs_lp):
    """
    Exact three-way split of the subtidal salt flux into mean, steady exchange
    and tidal parts. See the module docstring for the algebra.
    """
    ax = (1, 2)
    Q = np.nansum(q_lp, axis=ax)
    A = np.nansum(dA_lp, axis=ax)
    s0 = np.nansum(s_lp * dA_lp, axis=ax) / A
    F_total = np.nansum(qs_lp, axis=ax)
    F_eul = np.nansum(q_lp * s_lp, axis=ax)
    F_mean = Q * s0
    F_exch = F_eul - F_mean
    F_tidal = F_total - F_eul
    return {'F_total': F_total, 'F_mean': F_mean, 'F_exch': F_exch,
            'F_tidal': F_tidal, 's0': s0, 'Qnet': Q, 'area': A}
