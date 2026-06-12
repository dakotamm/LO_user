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

# Variables to plot.  For each: display name, obs column, and how to pull the
# model value out of a cast/mooring xarray Dataset (function of the Dataset).
# Model 'salt'/'temp' are handled specially (converted to SA/CT with gsw).
VARS = ['CT', 'SA', 'DO (uM)']

# model data_var name for each display variable (None => derived / special)
MOD_VARNAME = {
    'DO (uM)': 'oxygen',
    'NO3 (uM)': 'NO3',
    'NH4 (uM)': 'NH4',
    'Chl (mg m-3)': 'chlorophyll',
    'TA (uM)': 'alkalinity',
    'DIC (uM)': 'TIC',
}

LIMS = {
    'SA': (14, 34), 'CT': (4, 20), 'DO (uM)': (0, 450),
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
    if vn == 'DIN (uM)':
        if 'NO3' in ds and 'NH4' in ds:
            return ds['NO3'].values + ds['NH4'].values
        return None
    name = MOD_VARNAME.get(vn)
    if name is not None and name in ds:
        return ds[name].values
    return None


def obs_var(df, vn):
    """Return the obs column for display variable vn (deriving DIN), or NaNs."""
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
