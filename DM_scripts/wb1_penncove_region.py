"""
Shared region definition for the Penn Cove zoomed wb1 plots.

Holds the zoom box, exclude/include masking polygons, the Penn Cove SSH-average
box, the DO/hypoxia conventions, and small helpers. Imported by
wb1_penncove_salinity.py and wb1_penncove_multivar.py so the spatial extent and
masking stay IDENTICAL across all of the Penn Cove plots -- edit here once.
"""
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from lo_tools import plotting_functions as pfun

# default zoom box (lon0, lon1, lat0, lat1)
ZOOM = dict(lon0=-122.78, lon1=-122.40, lat0=48.15, lat1=48.40)

# Cells inside any EXCLUDE polygon are removed -- UNLESS also inside an INCLUDE
# polygon, which wins. Vertices are (lon, lat).
EXCLUDE_POLYS = [
    # western channel west of Whidbey
    [(-122.79, 48.41), (-122.61, 48.41), (-122.68, 48.33),
     (-122.72, 48.27), (-122.71, 48.20), (-122.62, 48.155), (-122.79, 48.145)],
    # bottom yellow strip
    [(-122.71, 48.165), (-122.59, 48.165), (-122.59, 48.143), (-122.71, 48.143)],
    # western tip of Penn Cove -- remove explicitly
    [(-122.775, 48.188), (-122.708, 48.188), (-122.708, 48.232), (-122.775, 48.232)],
]
INCLUDE_POLYS = [
    # western edge + neck of Penn Cove to add back
    [(-122.732, 48.210), (-122.683, 48.210), (-122.683, 48.248), (-122.732, 48.248)],
]
# Box over which SSH (zeta) is averaged for the tidal-phase timeseries panel.
PENN_COVE_BOX = [-122.73, -122.60, 48.215, 48.250]   # lon0, lon1, lat0, lat1

# DO / hypoxia conventions (match the obs/model DO scripts)
DO_MMOL_TO_MGL = 32.0 / 1000.0   # mmol m-3 (uM) -> mg L-1
HYPOXIC_MGL = 2.0                # hypoxic threshold
LOWDO_MGL = 5.0                  # low-DO threshold (second layer-thickness panel)

_EXCL = [MplPath(p) for p in EXCLUDE_POLYS]
_INCL = [MplPath(p) for p in INCLUDE_POLYS]


def region_remove_mask(lon, lat):
    """Boolean array shaped like lon: True where the cell should be removed."""
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    remove = np.zeros(lon.size, dtype=bool)
    for ep in _EXCL:
        remove |= ep.contains_points(pts)
    for ip in _INCL:
        remove &= ~ip.contains_points(pts)
    return remove.reshape(lon.shape)


def mask_field(field, lon, lat, mask_rho, apply_region=True):
    """NaN out land (mask_rho==0) and, optionally, the exclude/include region."""
    out = np.where(mask_rho == 0, np.nan, field)
    if apply_region:
        out = np.where(region_remove_mask(lon, lat), np.nan, out)
    return out


def get_ssh_series(fn_list, box=PENN_COVE_BOX):
    """Penn Cove box-mean SSH (zeta) per file. Returns (times_local, ssh_array)."""
    t, v = [], []
    for fn in fn_list:
        ds = xr.open_dataset(fn)
        lon = ds.lon_rho.values
        lat = ds.lat_rho.values
        zeta = ds.zeta[0, :, :].values
        mask = ds.mask_rho.values
        inbox = ((lon >= box[0]) & (lon <= box[1]) &
                 (lat >= box[2]) & (lat <= box[3]) & (mask == 1))
        v.append(np.nanmean(np.where(inbox, zeta, np.nan)))
        t_utc = pd.Timestamp(ds.ocean_time.values[0]).to_pydatetime()
        t.append(pfun.get_dt_local(t_utc).replace(tzinfo=None))  # naive local (PST)
        ds.close()
    return t, np.array(v)
