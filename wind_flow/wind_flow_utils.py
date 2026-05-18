"""
Helpers for wind / flow / stratification diagnostics at a single mooring.

Designed for hourly LO mooring extractions (e.g. M1 in pc0/wb1_r0_xn11b).
All time-series helpers expect 1-D arrays sampled on a uniform hourly grid;
the caller is responsible for ensuring that.

LO conventions used here:
  - s_rho index 0 = bottom, s_rho index -1 = surface
    (same as lo_tools / find_hypoxia_events.py)
  - oxygen units in mooring file: mmol/m^3 (== uM)
"""

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import zfun

try:
    import gsw
    _HAS_GSW = True
except ImportError:
    _HAS_GSW = False


# ---- physical constants ----
DO_UM_TO_MGL = 32.0 / 1000.0          # mmol/m^3 -> mg/L
RHO_AIR = 1.22                        # kg/m^3
CD_WIND = 1.3e-3                      # neutral 10-m drag coefficient


def load_moor(moor_fn):
    """Open a mooring NetCDF and return the xarray Dataset."""
    return xr.open_dataset(moor_fn)


# ----------------------------------------------------------------------------
# Wind
# ----------------------------------------------------------------------------

def wind_stress_bulk(uwind, vwind, rho_air=RHO_AIR, cd=CD_WIND):
    """
    Bulk wind-stress estimate from 10-m wind components.

        tau = rho_air * Cd * |U| * U

    Returns (tau_x, tau_y, |tau|) in N/m^2.
    """
    speed = np.hypot(uwind, vwind)
    tau_x = rho_air * cd * speed * uwind
    tau_y = rho_air * cd * speed * vwind
    tau_mag = np.hypot(tau_x, tau_y)
    return tau_x, tau_y, tau_mag


def get_wind(ds):
    """
    Return a DataFrame of hourly wind diagnostics indexed by ocean_time.

    Columns: Uwind, Vwind, wind_speed, wind_dir_from_deg (met convention
    = direction the wind comes FROM, 0=N), tau_x, tau_y, tau_mag.
    """
    times = pd.to_datetime(ds.ocean_time.values)
    u = ds.Uwind.values
    v = ds.Vwind.values
    speed = np.hypot(u, v)
    # Meteorological "from" direction
    dir_to = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0   # direction TO
    dir_from = (dir_to + 180.0) % 360.0
    tx, ty, tmag = wind_stress_bulk(u, v)
    return pd.DataFrame(
        {
            'Uwind': u, 'Vwind': v,
            'wind_speed': speed,
            'wind_dir_from_deg': dir_from,
            'tau_x': tx, 'tau_y': ty, 'tau_mag': tmag,
        },
        index=times,
    )


# ----------------------------------------------------------------------------
# Flow profile
# ----------------------------------------------------------------------------

def depth_average(field, z_w):
    """
    Vertical mean weighted by layer thickness.

    Parameters
    ----------
    field : (time, s_rho) array
    z_w   : (time, s_rho+1) array, monotonically increasing with index

    Returns
    -------
    (time,) array of depth-averaged values.
    """
    dz = np.diff(z_w, axis=1)               # (time, s_rho)
    H = dz.sum(axis=1)                      # (time,)
    return (field * dz).sum(axis=1) / H


def get_flow_profile(ds):
    """
    Extract flow at surface / depth-averaged / bottom from an M1-style
    mooring file. Returns a DataFrame indexed by ocean_time with columns:

        u_surface, v_surface, u_bottom, v_bottom,
        u_depthavg, v_depthavg

    LO s_rho convention: index 0 = bottom, index -1 = surface.
    """
    times = pd.to_datetime(ds.ocean_time.values)
    u = ds.u.values     # (time, s_rho)
    v = ds.v.values
    z_w = ds.z_w.values

    u_avg = depth_average(u, z_w)
    v_avg = depth_average(v, z_w)

    return pd.DataFrame(
        {
            'u_surface':  u[:, -1],
            'v_surface':  v[:, -1],
            'u_bottom':   u[:,  0],
            'v_bottom':   v[:,  0],
            'u_depthavg': u_avg,
            'v_depthavg': v_avg,
        },
        index=times,
    )


# ----------------------------------------------------------------------------
# Principal axis & rotation
# ----------------------------------------------------------------------------

def principal_axis(u, v):
    """
    Principal-axis angle of a 2-D vector time series.

    Returns
    -------
    theta_rad : float
        Angle (radians, math convention CCW from +x/east) of the major
        variance axis. The "along" component is obtained by rotating
        (u, v) by -theta_rad.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    mask = np.isfinite(u) & np.isfinite(v)
    u = u[mask] - np.nanmean(u[mask])
    v = v[mask] - np.nanmean(v[mask])
    cov = np.cov(u, v)
    vals, vecs = np.linalg.eigh(cov)
    # eigh returns ascending; pick the major axis (last column)
    major = vecs[:, -1]
    return float(np.arctan2(major[1], major[0]))


def rotate(u, v, theta_rad):
    """
    Rotate (u, v) by -theta so that the major axis aligns with +x.

    Returns (along, across).
    """
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    along  =  c * u + s * v
    across = -s * u + c * v
    return along, across


# ----------------------------------------------------------------------------
# Stratification
# ----------------------------------------------------------------------------

def sigma0_profile(ds):
    """
    Return potential density anomaly sigma0 (kg/m^3) on (time, s_rho).

    Uses GSW with the mooring's lon/lat. Salinity in mooring is SP
    (practical salinity from ROMS); we convert to absolute via SA_from_SP
    using surface pressure ~0 for simplicity at this depth.
    """
    if not _HAS_GSW:
        raise ImportError('gsw is required for sigma0_profile; pip install gsw')
    SP = ds.salt.values                     # (time, s_rho)
    t = ds.temp.values
    z = ds.z_rho.values                     # (time, s_rho), negative down
    lon = float(ds.lon_rho.values)
    lat = float(ds.lat_rho.values)
    # Pressure (dbar) from depth (m); z is negative below MSL
    p = gsw.p_from_z(z, lat)
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    CT = gsw.CT_from_pt(SA, t)              # ROMS temp = potential temp
    return gsw.sigma0(SA, CT)               # kg/m^3


def strat_delta_rho(ds):
    """
    Return DataFrame with d_rho = rho_bottom - rho_top, indexed by time.
    Positive = stably stratified (denser at bottom).
    """
    times = pd.to_datetime(ds.ocean_time.values)
    sig = sigma0_profile(ds)
    drho = sig[:, 0] - sig[:, -1]          # bottom - surface
    return pd.DataFrame({'d_rho': drho}, index=times)


# ----------------------------------------------------------------------------
# Bottom DO
# ----------------------------------------------------------------------------

def get_bottom_do_mgl(ds):
    """Return bottom DO in mg/L as a pandas Series indexed by ocean_time."""
    times = pd.to_datetime(ds.ocean_time.values)
    do = ds.oxygen.values[:, 0] * DO_UM_TO_MGL
    return pd.Series(do, index=times, name='bot_DO_mgL')


# ----------------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------------

def godin_lowpass(series):
    """
    Apply the LO Godin 24-24-25 tidal-averaging filter to a 1-D
    hourly array or pandas Series, returning the same type. NaN padding
    follows the lo_tools convention.
    """
    if isinstance(series, pd.Series):
        out = zfun.lowpass(series.values.astype(float), f='godin')
        return pd.Series(out, index=series.index, name=series.name)
    return zfun.lowpass(np.asarray(series, dtype=float), f='godin')


def godin_lowpass_df(df):
    """Apply Godin lowpass to every column of a DataFrame."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = godin_lowpass(df[c])
    return out
