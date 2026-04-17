"""
Tidal phase detection functions.

Pure functions that operate on 1D numpy/xarray arrays — no file I/O,
no hardcoded section or grid names.

Two detection approaches:
    1. Signal-based  (qnet sign, d(zeta)/dt, qprism percentile)
    2. Harmonic      (UTide solve/reconstruct)

Typical usage
-------------
    import tide_phase_fun as tpf

    # signal-based
    is_flood, is_ebb = tpf.detect_flood_ebb(qnet, time)
    is_spring, is_neap = tpf.detect_spring_neap(qprism)
    slack_hi, slack_lo = tpf.detect_slack(qnet, time)

    # harmonic
    phase_dict = tpf.detect_phases_utide(zeta, time, lat=48.0)

    # grouping
    labels = tpf.get_phase_labels(time, is_flood, is_spring,
                                   slack_hi, slack_lo)
    mask = tpf.get_phase_mask(labels, 'spring_flood')

Dependencies: numpy, pandas, scipy.signal
Optional: utide  (only for detect_phases_utide)
Optional: lo_tools.zfun  (only for compute_qprism)
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# ---------------------------------------------------------------------------
# Signal-based detection
# ---------------------------------------------------------------------------

def detect_flood_ebb(signal, time, method='qnet'):
    """
    Classify each timestep as flood or ebb.

    Parameters
    ----------
    signal : 1-D array
        qnet [m3/s] if method='qnet', or zeta [m] if method='zeta'.
    time : 1-D array of datetime64
        Timestamps matching *signal*.
    method : str, {'qnet', 'zeta'}
        'qnet' — flood where signal > 0, ebb where signal < 0.
        'zeta' — flood where d(zeta)/dt > 0 (rising), ebb where < 0.

    Returns
    -------
    is_flood, is_ebb : boolean arrays (same length as signal)
    """
    signal = np.asarray(signal, dtype=float)

    if method == 'qnet':
        is_flood = signal > 0
        is_ebb = signal < 0
    elif method == 'zeta':
        # central difference for interior, forward/backward at edges
        dt_sec = _dt_seconds(time)
        dzdt = np.gradient(signal, dt_sec)
        is_flood = dzdt > 0
        is_ebb = dzdt < 0
    else:
        raise ValueError(f"method must be 'qnet' or 'zeta', got '{method}'")

    return is_flood, is_ebb


def detect_slack(signal, time, method='qnet'):
    """
    Identify high-slack and low-slack water times.

    Parameters
    ----------
    signal : 1-D array
        qnet or zeta (see *method*).
    time : 1-D array of datetime64
    method : str, {'qnet', 'zeta'}
        'qnet' — slack at zero-crossings of qnet.
            high-slack = sign change + → − (flood→ebb)
            low-slack  = sign change − → + (ebb→flood)
        'zeta' — slack at local extrema of zeta.
            high-slack = local max  (peak high tide)
            low-slack  = local min  (peak low tide)

    Returns
    -------
    slack_hi, slack_lo : boolean arrays (same length as signal)
        True at timesteps identified as high-slack or low-slack.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    slack_hi = np.zeros(n, dtype=bool)
    slack_lo = np.zeros(n, dtype=bool)

    if method == 'qnet':
        sign = np.sign(signal)
        for i in range(1, n):
            if sign[i - 1] > 0 and sign[i] <= 0:
                slack_hi[i] = True
            elif sign[i - 1] < 0 and sign[i] >= 0:
                slack_lo[i] = True
    elif method == 'zeta':
        # Use local extrema of zeta
        hi_idx = argrelextrema(signal, np.greater, order=3)[0]
        lo_idx = argrelextrema(signal, np.less, order=3)[0]
        slack_hi[hi_idx] = True
        slack_lo[lo_idx] = True
    else:
        raise ValueError(f"method must be 'qnet' or 'zeta', got '{method}'")

    return slack_hi, slack_lo


def detect_spring_neap(qprism, percentile_thresh=50):
    """
    Classify each timestep as spring or neap based on qprism amplitude.

    Parameters
    ----------
    qprism : 1-D array
        Tidal prism time series (typically Godin-filtered, daily subsampled).
        Can have NaNs at edges from filtering.
    percentile_thresh : float, default 50
        Percentile of qprism used as the spring/neap boundary.

    Returns
    -------
    is_spring, is_neap : boolean arrays (same length as qprism)
        NaN entries in qprism map to False in both arrays.
    """
    qprism = np.asarray(qprism, dtype=float)
    valid = ~np.isnan(qprism)
    thresh = np.nanpercentile(qprism, percentile_thresh)

    is_spring = valid & (qprism >= thresh)
    is_neap = valid & (qprism < thresh)
    return is_spring, is_neap


def compute_qprism(qnet, pad=36):
    """
    Compute qprism from a raw hourly qnet time series.

    Follows the bulk_calc_avg.py pattern:
        1. Godin low-pass qnet to get subtidal flow
        2. |qnet − qnet_lp| = tidal amplitude
        3. Godin low-pass the amplitude
        4. qprism = 0.5 * <|qnet − qnet_lp|>

    Parameters
    ----------
    qnet : 1-D array
        Hourly net volume transport [m3/s].
    pad : int, default 36
        Padding for Godin filter edge effects + noon subsampling.

    Returns
    -------
    qprism : 1-D array (shorter than input by 2*pad − 1)
    time_indices : 1-D integer array
        Indices into the original qnet array corresponding to qprism values
        (noon of each day, excluding first/last days).
    """
    from lo_tools import zfun

    qnet = np.asarray(qnet, dtype=float)
    qnet_lp = zfun.lowpass(qnet, f='godin', nanpad=False)
    qabs = np.abs(qnet - qnet_lp)
    qabs_lp = zfun.lowpass(qabs, f='godin')[pad:-pad + 1:24]
    qprism = qabs_lp / 2.0
    time_indices = np.arange(pad, len(qnet) - pad + 1, 24)
    return qprism, time_indices


# ---------------------------------------------------------------------------
# Harmonic (UTide) detection
# ---------------------------------------------------------------------------

def detect_phases_utide(zeta, time, lat):
    """
    Classify tidal phases using UTide harmonic analysis.

    Parameters
    ----------
    zeta : 1-D array
        Sea surface height [m].
    time : 1-D array of datetime64 or array of matplotlib datenums
        Timestamps corresponding to *zeta*.
    lat : float
        Latitude of the observation (needed by UTide).

    Returns
    -------
    phase_dict : dict with keys
        'is_flood'     : bool array — rising predicted tide
        'is_ebb'       : bool array — falling predicted tide
        'is_spring'    : bool array — large tidal range (envelope > median)
        'is_neap'      : bool array — small tidal range
        'slack_hi'     : bool array — predicted high-water slack
        'slack_lo'     : bool array — predicted low-water slack
        'pred'         : float array — UTide tidal prediction
        'coef'         : UTide coefficient object
    """
    import utide

    zeta = np.asarray(zeta, dtype=float)

    # Convert datetime64 to matplotlib datenums for UTide
    time_num = _to_datenum(time)

    coef = utide.solve(time_num, zeta, lat=lat,
                       method='ols',
                       conf_int='none',
                       verbose=False)

    pred = utide.reconstruct(time_num, coef, verbose=False).h

    # Flood/ebb from derivative of predicted tide
    dt_sec = _dt_seconds(time)
    dpdt = np.gradient(pred, dt_sec)
    is_flood = dpdt > 0
    is_ebb = dpdt < 0

    # Slack from predicted tide extrema
    slack_hi_idx = argrelextrema(pred, np.greater, order=3)[0]
    slack_lo_idx = argrelextrema(pred, np.less, order=3)[0]
    slack_hi = np.zeros(len(pred), dtype=bool)
    slack_lo = np.zeros(len(pred), dtype=bool)
    slack_hi[slack_hi_idx] = True
    slack_lo[slack_lo_idx] = True

    # Spring/neap from tidal range envelope
    is_spring, is_neap = _spring_neap_from_envelope(pred)

    return {
        'is_flood': is_flood,
        'is_ebb': is_ebb,
        'is_spring': is_spring,
        'is_neap': is_neap,
        'slack_hi': slack_hi,
        'slack_lo': slack_lo,
        'pred': pred,
        'coef': coef,
    }


# ---------------------------------------------------------------------------
# Grouping / labeling utilities
# ---------------------------------------------------------------------------

def get_phase_labels(time, is_flood, is_spring, slack_hi, slack_lo):
    """
    Assemble a DataFrame of phase labels aligned to *time*.

    Parameters
    ----------
    time : 1-D array of datetime64
    is_flood, is_spring : boolean arrays
    slack_hi, slack_lo : boolean arrays

    Returns
    -------
    df : pandas DataFrame with columns:
        time, is_flood, is_ebb, is_spring, is_neap,
        is_slack_high, is_slack_low, phase
        where *phase* is one of:
        'spring_flood', 'spring_ebb', 'neap_flood', 'neap_ebb',
        'slack_high', 'slack_low'
    """
    is_ebb = ~is_flood & ~slack_hi & ~slack_lo
    is_neap = ~is_spring

    df = pd.DataFrame({
        'time': np.asarray(time),
        'is_flood': np.asarray(is_flood, dtype=bool),
        'is_ebb': np.asarray(is_ebb, dtype=bool),
        'is_spring': np.asarray(is_spring, dtype=bool),
        'is_neap': np.asarray(is_neap, dtype=bool),
        'is_slack_high': np.asarray(slack_hi, dtype=bool),
        'is_slack_low': np.asarray(slack_lo, dtype=bool),
    })

    # Composite label
    phase = np.full(len(df), 'unclassified', dtype=object)
    phase[df.is_slack_high.values] = 'slack_high'
    phase[df.is_slack_low.values] = 'slack_low'
    phase[df.is_spring.values & df.is_flood.values] = 'spring_flood'
    phase[df.is_spring.values & df.is_ebb.values] = 'spring_ebb'
    phase[df.is_neap.values & df.is_flood.values] = 'neap_flood'
    phase[df.is_neap.values & df.is_ebb.values] = 'neap_ebb'
    df['phase'] = phase

    return df


def get_phase_mask(labels_df, phase_name):
    """
    Return a boolean mask for a given phase name.

    Parameters
    ----------
    labels_df : DataFrame from get_phase_labels()
    phase_name : str
        One of: 'spring_flood', 'spring_ebb', 'neap_flood', 'neap_ebb',
                'slack_high', 'slack_low', 'flood', 'ebb', 'spring', 'neap'

    Returns
    -------
    mask : boolean array
    """
    if phase_name in ('spring_flood', 'spring_ebb', 'neap_flood', 'neap_ebb',
                       'slack_high', 'slack_low'):
        return (labels_df['phase'] == phase_name).values
    elif phase_name == 'flood':
        return labels_df['is_flood'].values
    elif phase_name == 'ebb':
        return labels_df['is_ebb'].values
    elif phase_name == 'spring':
        return labels_df['is_spring'].values
    elif phase_name == 'neap':
        return labels_df['is_neap'].values
    else:
        raise ValueError(f"Unknown phase_name '{phase_name}'. "
                         "Valid: spring_flood, spring_ebb, neap_flood, neap_ebb, "
                         "slack_high, slack_low, flood, ebb, spring, neap")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dt_seconds(time):
    """Convert a datetime64 array to elapsed seconds (float) for np.gradient."""
    t = np.asarray(time, dtype='datetime64[ns]')
    dt_ns = (t - t[0]).astype(float)  # nanoseconds
    return dt_ns / 1e9  # seconds


def _to_datenum(time):
    """Convert datetime64 to matplotlib date numbers (days since 0001-01-01)."""
    import matplotlib.dates as mdates
    t = pd.DatetimeIndex(np.asarray(time))
    return mdates.date2num(t.to_pydatetime())


def _spring_neap_from_envelope(pred, window_hours=25):
    """
    Classify spring/neap from the tidal range envelope.

    Computes a running tidal range (max − min over ~25 h window),
    then splits at the median.
    """
    n = len(pred)
    half = window_hours // 2
    tidal_range = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        tidal_range[i] = np.nanmax(pred[lo:hi]) - np.nanmin(pred[lo:hi])

    thresh = np.nanmedian(tidal_range)
    valid = ~np.isnan(tidal_range)
    is_spring = valid & (tidal_range >= thresh)
    is_neap = valid & (tidal_range < thresh)
    return is_spring, is_neap
