"""
Shared helpers for the 2026.06.11 KC + Ecology obs-model validation scripts
(profiles and lowpassed time series), for wb1_t0_xn11abbur00 over 2024-2025.
"""

import numpy as np
import gsw

# name of the dated figure output folder
OUT_FOLDER = '20260611_kcecology_compare'
DEFAULT_GTX = 'wb1_t0_xn11abbur00'
MOOR_JOB = 'KCEcology_2024_2025'

# obs sources -> pretty label
SOURCES = {
    'kc_whidbeyBasin': 'King County',
    'ecology_nc': 'Ecology',
}

# the 15 in-wb1-domain stations (obs 'name' values). Used to restrict plots to
# the wb1 station set even when working with a larger grid (e.g. cas7).
WB1_STATIONS = [
    'ADM001', 'ADM003', 'PSS019', 'PTH005', 'SAR003', 'SKG003',
    'PENNCOVEENT', 'PENNCOVEWEST', 'PSUSANBUOY', 'PSUSANENT', 'PSUSANKP',
    'Poss DO-2', 'SARATOGACH', 'SARATOGAOP', 'SARATOGARP',
]
# same names with spaces -> underscores, to match mooring filenames
WB1_STATIONS_SAFE = set(s.replace(' ', '_') for s in WB1_STATIONS)

# umol/L (uM) -> mg/L for dissolved oxygen
DO_UM_TO_MGL = 32.0 / 1000.0

# Variables to plot.  For each: display name, obs column, and how to pull the
# model value out of a cast/mooring xarray Dataset (function of the Dataset).
# Model 'salt'/'temp' are handled specially (converted to SA/CT with gsw); DO is
# converted from uM to mg/L.
VARS = ['CT', 'SA', 'DO (mg L-1)', 'NO3 (uM)', 'NH4 (uM)', 'DIN (uM)',
        'Chl (mg m-3)', 'TA (uM)', 'DIC (uM)']

# model data_var name for each display variable (None => derived / special)
MOD_VARNAME = {
    'DO (uM)': 'oxygen',
    'NO3 (uM)': 'NO3',
    'NH4 (uM)': 'NH4',
    'Chl (mg m-3)': 'chlorophyll',
    'TA (uM)': 'alkalinity',
    'DIC (uM)': 'TIC',
}

# consistent depth (z, m) axis for all profile figures (deepest of the 15
# wb1 stations is ADM003 at ~-211 m)
DEPTH_LIM = (-230, 5)

LIMS = {
    'SA': (14, 34), 'CT': (4, 20), 'DO (uM)': (0, 450), 'DO (mg L-1)': (0, 15),
    'NO3 (uM)': (0, 45), 'NH4 (uM)': (0, 10), 'DIN (uM)': (0, 50),
    'Chl (mg m-3)': (0, 30), 'TA (uM)': (1500, 2400), 'DIC (uM)': (1500, 2400),
}


def out_dir(Ldir):
    """Dated output folder for figures."""
    return Ldir['LOo'] / 'obsmod_val_plots' / OUT_FOLDER


def model_SA_CT(ds, lon, lat):
    """Convert a cast/mooring Dataset's salt(SP) & temp(PT) to SA & CT.
    Returns (SA, CT) arrays shaped like ds.salt (z[, time])."""
    SP = ds['salt'].values
    PT = ds['temp'].values
    z = ds['z_rho'].values
    p = gsw.p_from_z(z, lat)
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    CT = gsw.CT_from_pt(SA, PT)
    return SA, CT


def model_var(ds, vn, SA=None, CT=None):
    """Return the model value array for display variable vn, or None if the
    underlying data_var is not present. SA/CT may be precomputed and passed in."""
    if vn == 'SA':
        return SA
    if vn == 'CT':
        return CT
    if vn == 'DO (mg L-1)':
        return ds['oxygen'].values * DO_UM_TO_MGL if 'oxygen' in ds else None
    if vn == 'DIN (uM)':
        if 'NO3' in ds and 'NH4' in ds:
            return ds['NO3'].values + ds['NH4'].values
        return None
    name = MOD_VARNAME.get(vn)
    if name is not None and name in ds:
        return ds[name].values
    return None


def obs_var(df, vn):
    """Return the obs column for display variable vn (deriving DIN / DO), or NaNs."""
    if vn == 'DO (mg L-1)':
        if 'DO (uM)' in df:
            return (df['DO (uM)'] * DO_UM_TO_MGL).to_numpy()
        return np.full(len(df), np.nan)
    if vn == 'DIN (uM)':
        if 'NO3 (uM)' in df and 'NH4 (uM)' in df:
            return (df['NO3 (uM)'] + df['NH4 (uM)']).to_numpy()
        return np.full(len(df), np.nan)
    if vn in df:
        return df[vn].to_numpy()
    return np.full(len(df), np.nan)


def has_data(arr):
    """True if arr has at least one finite value."""
    return arr is not None and np.isfinite(np.asarray(arr, dtype=float)).any()
