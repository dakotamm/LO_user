"""
Compare Penn Cove Dec 2025 survey (CTD towyo + ADCP) against
wb1_t0_xn11abbur00 model output.

Designed to run on apogee where model data lives in roms_out2.
On mac: set ROMS_OUT manually or test with locally available runs.

Outputs section comparison PNGs to LO_output/DM_outs/pcDec2025_obs_mod/.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4 as nc4
import xarray as xr
from pathlib import Path
from datetime import datetime, timezone, timedelta
from scipy.interpolate import griddata

from lo_tools import Lfun, zfun, zrfun

# ── Configuration ────────────────────────────────────────────────────────────
GTAGEX = 'wb1_t0_xn11abbur00'
Ldir = Lfun.Lstart()

if 'mac' in Ldir['lo_env']:
    ROMS_OUT = Path('/dat2/dakotamm/LO_roms') / GTAGEX
    OBS_DIR  = Ldir['data'] / 'obs' / 'X - pcDec2025Recon '
    OUT_DIR  = Ldir['LOo'] / 'obsmod_val_plots' / 'pcDec2025_obs_mod'
else:
    ROMS_OUT = Ldir['roms_out2'] / GTAGEX
    OBS_DIR  = Ldir['data'] / 'obs' / 'X - pcDec2025Recon '
    OUT_DIR  = Ldir['LOo'] / 'obsmod_val_plots' / 'pcDec2025_obs_mod'

OUT_DIR.mkdir(parents=True, exist_ok=True)

CTD_FN  = OBS_DIR / 'CTD_towyo_PennCove_Dec2025.nc'
ADCP_FN = OBS_DIR / 'ADCP_PennCove_Dec2025.nc'

DO_UM_TO_MGL = 32.0 / 1000.0  # model oxygen: µM → mg/L

LAPS = ['lap1', 'lap2', 'lap3']
SEGS = ['entrance', 'south', 'back', 'north']

# colormap limits per variable (obs and model use same scale)
CLIMS = {
    'salt': (18, 30),
    'temp': (8, 12),
    'do':   (3, 10),
    'spd':  (0, 0.5),  # ADCP speed (m/s)
}
CMAPS = {
    'salt': 'viridis',
    'temp': 'plasma',
    'do':   'RdYlBu',
    'spd':  'YlOrRd',
}
LABELS = {
    'salt': 'Salinity (PSU)',
    'temp': 'Temperature (°C)',
    'do':   'DO (mg/L)',
    'spd':  'Speed (m/s)',
}

# ── Grid ────────────────────────────────────────────────────────────────────
print('Loading model grid...')
ref_fn = ROMS_OUT / 'f2025.12.04' / 'ocean_his_0001.nc'
G, S, _ = zrfun.get_basic_info(ref_fn)
lon_vec  = G['lon_rho'][0, :]   # 1-D, length L
lat_vec  = G['lat_rho'][:, 0]   # 1-D, length M
mask_rho = G['mask_rho']         # (M, L)
h_bathy  = G['h']                # (M, L)


# ── Model file lookup and cache ──────────────────────────────────────────────
_mod_cache: dict = {}

def _model_fn(unix_time: float) -> Path:
    """Return path to the model history file nearest in time to unix_time."""
    dt = datetime.fromtimestamp(unix_time, tz=timezone.utc).replace(tzinfo=None)
    # round to nearest hour
    if dt.minute >= 30:
        dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=0, second=0, microsecond=0)
    if dt.hour == 0:
        # midnight: use 0025 of previous f-day
        dt_prev = dt - timedelta(days=1)
        return ROMS_OUT / ('f' + dt_prev.strftime('%Y.%m.%d')) / 'ocean_his_0025.nc'
    else:
        return ROMS_OUT / ('f' + dt.strftime('%Y.%m.%d')) / f'ocean_his_{dt.hour + 1:04d}.nc'


def get_model_ds(unix_time: float):
    """Return a cached xarray Dataset for the model file nearest to unix_time."""
    fn = _model_fn(unix_time)
    key = str(fn)
    if key not in _mod_cache:
        if fn.is_file():
            _mod_cache[key] = xr.open_dataset(fn)
        else:
            print(f'  WARNING: model file not found: {fn}')
            _mod_cache[key] = None
    return _mod_cache[key]


def extract_mod_column(ds, iy: int, ix: int):
    """
    Extract (z_rho, salt, temp, oxygen_mgl, u_rho, v_rho) at grid cell (iy, ix).
    z_rho is negative, increasing toward 0 at surface.
    u_rho and v_rho are interpolated from u/v stagger grids to rho point.
    Returns NaN arrays if ds is None or point is masked.
    """
    nz = S['N']
    nan_vec = np.full(nz, np.nan)
    if ds is None or mask_rho[iy, ix] == 0:
        return nan_vec, nan_vec, nan_vec, nan_vec, nan_vec, nan_vec

    zeta = float(ds['zeta'].values[0, iy, ix])
    h_col = float(h_bathy[iy, ix])
    z_rho = zrfun.get_z(np.array([[h_col]]), np.array([[zeta]]), S, only_rho=True)

    salt = ds['salt'].values[0, :, iy, ix]
    temp = ds['temp'].values[0, :, iy, ix]
    oxy  = ds['oxygen'].values[0, :, iy, ix] * DO_UM_TO_MGL

    # u: average from u-stagger to rho point
    if ix > 0:
        u_col = 0.5 * (ds['u'].values[0, :, iy, ix] + ds['u'].values[0, :, iy, ix - 1])
    else:
        u_col = ds['u'].values[0, :, iy, ix]

    # v: average from v-stagger to rho point
    if iy > 0:
        v_col = 0.5 * (ds['v'].values[0, :, iy, ix] + ds['v'].values[0, :, iy - 1, ix])
    else:
        v_col = ds['v'].values[0, :, iy, ix]

    return z_rho, salt, temp, oxy, u_col, v_col


# ── Obs loaders ──────────────────────────────────────────────────────────────
def _fill(ma):
    """Convert masked array to float ndarray, replacing masked values with NaN."""
    import numpy.ma as npm
    return np.array(npm.filled(ma, np.nan), dtype=float)


def load_ctd(lap: str, seg: str) -> dict:
    """Load CTD obs for one lap/segment. depth is positive (m below surface)."""
    with nc4.Dataset(CTD_FN) as ds:
        g = ds[lap][seg]
        return {
            'time':        _fill(g['time'][:]),          # (nt,) unix s
            'lat':         _fill(g['lat'][:]),            # (nt,)
            'lon':         _fill(g['lon'][:]),            # (nt,)
            'depth':       _fill(g['depth'][:]),          # (nt, nz) positive m
            'salt':        _fill(g['salinity'][:]),       # (nt, nz)
            'temp':        _fill(g['temperature'][:]),    # (nt, nz)
            'do':          _fill(g['DO'][:]),             # (nt, nz) mg/L
            'bottomtrack': _fill(g['bottomtrack'][:]),   # (nt,)
        }


def load_adcp(lap: str, seg: str) -> dict:
    """Load ADCP obs for one lap/segment. depthBins positive (m below surface)."""
    with nc4.Dataset(ADCP_FN) as ds:
        g = ds[lap][seg]
        return {
            'time':        _fill(g['time'][:]),       # (nt,) unix s
            'lat':         _fill(g['lat'][:]),         # (nt,)
            'lon':         _fill(g['lon'][:]),         # (nt,)
            'depthBins':   _fill(g['depthBins'][:]),  # (nz,) positive m
            'vel_east':    _fill(g['vel_east'][:]),   # (nz, nt)
            'vel_north':   _fill(g['vel_north'][:]),  # (nz, nt)
            'bottomtrack': _fill(g['bottomtrack'][:]), # (nt,)
        }


# ── Along-track distance ─────────────────────────────────────────────────────
def along_track_km(lats, lons):
    """Cumulative along-track distance in km from the first point."""
    x, y = zfun.ll2xy(lons, lats, lons[0], lats[0])
    dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return dist / 1000.0


# ── Section gridding ─────────────────────────────────────────────────────────
def make_section_grid(x_pts, z_pts, val_pts, nx=80, nz=60):
    """
    Interpolate scattered (x, z, val) onto a regular (nx, nz) grid.
    x_pts: along-track distance (km)
    z_pts: depth (negative m)
    val_pts: variable values
    Returns xi, zi, Vi (regular grid arrays) for use with pcolormesh.
    """
    mask = np.isfinite(x_pts) & np.isfinite(z_pts) & np.isfinite(val_pts)
    if mask.sum() < 4:
        return None, None, None
    xi = np.linspace(x_pts[mask].min(), x_pts[mask].max(), nx)
    zi = np.linspace(z_pts[mask].min(), z_pts[mask].max(), nz)
    Xi, Zi = np.meshgrid(xi, zi)
    Vi = griddata(
        np.column_stack([x_pts[mask], z_pts[mask]]),
        val_pts[mask],
        (Xi, Zi),
        method='linear',
    )
    return xi, zi, Vi


# ── Build scatter data for one CTD segment ───────────────────────────────────
def build_ctd_section(lap: str, seg: str):
    """
    For one CTD lap/segment, collect scattered (x_dist, z, salt, temp, do)
    for obs and model.  Returns obs dict and mod dict.
    """
    obs = load_ctd(lap, seg)
    nt = len(obs['time'])
    nz_obs = obs['depth'].shape[1]
    x_km = along_track_km(obs['lat'], obs['lon'])

    # Flatten obs into scattered arrays
    x_obs = np.repeat(x_km, nz_obs)
    z_obs = -obs['depth'].ravel()   # negative convention
    s_obs = obs['salt'].ravel()
    t_obs = obs['temp'].ravel()
    d_obs = obs['do'].ravel()

    # Collect model columns at each obs position
    x_mod_list, z_mod_list, s_mod_list, t_mod_list, d_mod_list = [], [], [], [], []
    for i in range(nt):
        lon_i, lat_i, time_i = obs['lon'][i], obs['lat'][i], obs['time'][i]
        ix = int(zfun.find_nearest_ind(lon_vec, lon_i))
        iy = int(zfun.find_nearest_ind(lat_vec, lat_i))
        ds = get_model_ds(time_i)
        z_rho, salt_m, temp_m, oxy_m, _, _ = extract_mod_column(ds, iy, ix)
        nz_m = len(z_rho)
        x_mod_list.append(np.full(nz_m, x_km[i]))
        z_mod_list.append(z_rho)
        s_mod_list.append(salt_m)
        t_mod_list.append(temp_m)
        d_mod_list.append(oxy_m)

    x_mod = np.concatenate(x_mod_list)
    z_mod = np.concatenate(z_mod_list)
    s_mod = np.concatenate(s_mod_list)
    t_mod = np.concatenate(t_mod_list)
    d_mod = np.concatenate(d_mod_list)

    obs_pts = dict(x=x_obs, z=z_obs, salt=s_obs, temp=t_obs, do=d_obs)
    mod_pts = dict(x=x_mod, z=z_mod, salt=s_mod, temp=t_mod, do=d_mod)
    return obs_pts, mod_pts


def build_adcp_section(lap: str, seg: str):
    """
    For one ADCP lap/segment, collect scattered (x_dist, z, speed)
    for obs and model.
    """
    obs = load_adcp(lap, seg)
    nt = len(obs['time'])
    nz_obs = len(obs['depthBins'])
    x_km = along_track_km(obs['lat'], obs['lon'])
    z_bins = -obs['depthBins']   # negative convention

    # ADCP: vel dims are (nz, nt)
    spd_obs = np.sqrt(obs['vel_east'] ** 2 + obs['vel_north'] ** 2)

    x_obs = np.tile(x_km, (nz_obs, 1)).ravel()        # (nz*nt,)
    z_obs = np.tile(z_bins[:, np.newaxis], (1, nt)).ravel()
    s_obs = spd_obs.ravel()

    x_mod_list, z_mod_list, spd_mod_list = [], [], []
    for i in range(nt):
        lon_i, lat_i, time_i = obs['lon'][i], obs['lat'][i], obs['time'][i]
        ix = int(zfun.find_nearest_ind(lon_vec, lon_i))
        iy = int(zfun.find_nearest_ind(lat_vec, lat_i))
        ds = get_model_ds(time_i)
        z_rho, _, _, _, u_m, v_m = extract_mod_column(ds, iy, ix)
        spd_m = np.sqrt(u_m ** 2 + v_m ** 2)
        nz_m = len(z_rho)
        x_mod_list.append(np.full(nz_m, x_km[i]))
        z_mod_list.append(z_rho)
        spd_mod_list.append(spd_m)

    x_mod = np.concatenate(x_mod_list)
    z_mod = np.concatenate(z_mod_list)
    spd_mod = np.concatenate(spd_mod_list)

    obs_pts = dict(x=x_obs, z=z_obs, spd=s_obs)
    mod_pts = dict(x=x_mod, z=z_mod, spd=spd_mod)
    return obs_pts, mod_pts


# ── Matched-depth pairs for property-property plots ──────────────────────────
def build_ctd_matchup(lap: str, seg: str):
    """
    For each CTD obs sample, interpolate the model profile to the obs depth.
    Returns a dict of 1-D matched arrays: obs_salt, mod_salt, obs_temp, mod_temp,
    obs_do, mod_do, z (negative m).  NaN where model is out of range.
    """
    obs = load_ctd(lap, seg)
    nt = len(obs['time'])

    obs_s, mod_s = [], []
    obs_t, mod_t = [], []
    obs_d, mod_d = [], []
    z_out = []

    for i in range(nt):
        lon_i, lat_i, time_i = obs['lon'][i], obs['lat'][i], obs['time'][i]
        ix = int(zfun.find_nearest_ind(lon_vec, lon_i))
        iy = int(zfun.find_nearest_ind(lat_vec, lat_i))
        ds = get_model_ds(time_i)
        z_rho, salt_m, temp_m, oxy_m, _, _ = extract_mod_column(ds, iy, ix)

        # obs depths for this cast (negative convention)
        z_obs = -obs['depth'][i, :]   # (nz_obs,)
        valid = np.isfinite(z_obs) & np.isfinite(obs['salt'][i, :])

        # model z_rho is already sorted bottom→surface (increasing); interp needs that
        for arr_obs, arr_mod, mod_vals in [
            (obs['salt'][i, :],  mod_s, salt_m),
            (obs['temp'][i, :],  mod_t, temp_m),
            (obs['do'][i, :],    mod_d, oxy_m),
        ]:
            interped = np.full(z_obs.shape, np.nan)
            if valid.any() and np.isfinite(z_rho).all():
                interped[valid] = np.interp(
                    z_obs[valid], z_rho, mod_vals,
                    left=np.nan, right=np.nan,
                )
            arr_mod.append(interped)

        obs_s.append(obs['salt'][i, :])
        obs_t.append(obs['temp'][i, :])
        obs_d.append(obs['do'][i, :])
        z_out.append(z_obs)

    return {
        'z':       np.concatenate(z_out),
        'obs_salt': np.concatenate(obs_s),
        'mod_salt': np.concatenate(mod_s),
        'obs_temp': np.concatenate(obs_t),
        'mod_temp': np.concatenate(mod_t),
        'obs_do':   np.concatenate(obs_d),
        'mod_do':   np.concatenate(mod_d),
    }


def build_adcp_matchup(lap: str, seg: str):
    """
    For each ADCP depth bin, interpolate the model speed to that depth.
    Returns matched obs_spd, mod_spd, z arrays.
    """
    obs = load_adcp(lap, seg)
    nt = len(obs['time'])
    z_bins = -obs['depthBins']   # (nz,) negative

    obs_spd_list, mod_spd_list, z_list = [], [], []

    for i in range(nt):
        lon_i, lat_i, time_i = obs['lon'][i], obs['lat'][i], obs['time'][i]
        ix = int(zfun.find_nearest_ind(lon_vec, lon_i))
        iy = int(zfun.find_nearest_ind(lat_vec, lat_i))
        ds = get_model_ds(time_i)
        z_rho, _, _, _, u_m, v_m = extract_mod_column(ds, iy, ix)

        spd_obs_col = np.sqrt(obs['vel_east'][:, i] ** 2 + obs['vel_north'][:, i] ** 2)
        spd_mod_col = np.sqrt(u_m ** 2 + v_m ** 2)

        mod_interped = np.full(z_bins.shape, np.nan)
        if np.isfinite(z_rho).all():
            mod_interped = np.interp(z_bins, z_rho, spd_mod_col,
                                     left=np.nan, right=np.nan)
        obs_spd_list.append(spd_obs_col)
        mod_spd_list.append(mod_interped)
        z_list.append(z_bins)

    return {
        'z':       np.concatenate(z_list),
        'obs_spd': np.concatenate(obs_spd_list),
        'mod_spd': np.concatenate(mod_spd_list),
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_section_pair(ax_obs, ax_mod, obs_pts, mod_pts,
                      x_key, z_key, v_key,
                      cmap, vmin, vmax, lap_label=''):
    """
    Fill ax_obs and ax_mod with pcolormesh sections.
    Returns the last pcm object (for a shared colorbar), or None.
    """
    pcm_out = None
    for ax, pts, title_prefix in [
        (ax_obs, obs_pts, 'Obs'),
        (ax_mod, mod_pts, 'Model'),
    ]:
        xi, zi, Vi = make_section_grid(pts[x_key], pts[z_key], pts[v_key])
        if Vi is None:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9)
        else:
            pcm = ax.pcolormesh(xi, zi, Vi, cmap=cmap, vmin=vmin, vmax=vmax,
                                shading='auto')
            pcm_out = pcm
        ax.set_box_aspect(1)
        ax.set_xlabel('Distance (km)', fontsize=8)
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f'{title_prefix} — {lap_label}', fontsize=9)
    return pcm_out


# ── Prop-prop panel helper ────────────────────────────────────────────────────
LAP_COLORS = {'lap1': 'tab:blue', 'lap2': 'tab:orange', 'lap3': 'tab:green'}

def _draw_prop_panel(ax, pairs, label):
    """
    pairs: {lap: (obs_1d, mod_1d)} — pre-collected matched arrays.
    Draws scatter colored by lap, 1:1 line, and stats box.
    """
    all_o, all_m = [], []
    for lap in LAPS:
        if lap not in pairs:
            continue
        o, m = pairs[lap]
        mask = np.isfinite(o) & np.isfinite(m)
        if mask.sum() < 2:
            continue
        ax.scatter(o[mask], m[mask], s=4, alpha=0.35,
                   color=LAP_COLORS[lap], edgecolors='none', label=lap)
        all_o.append(o[mask])
        all_m.append(m[mask])
    if not all_o:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=9)
        return
    o_cat = np.concatenate(all_o)
    m_cat = np.concatenate(all_m)
    lo = min(o_cat.min(), m_cat.min())
    hi = max(o_cat.max(), m_cat.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1)
    bias = float(np.mean(m_cat - o_cat))
    rmse = float(np.sqrt(np.mean((m_cat - o_cat) ** 2)))
    ax.text(0.05, 0.95,
            f'N={len(o_cat)}\nbias={bias:+.3f}\nRMSE={rmse:.3f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect('equal')
    ax.set_xlabel(f'Obs {label}', fontsize=9)
    ax.set_ylabel(f'Model {label}', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, markerscale=2, framealpha=0.8)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():

    # ── Section plots ────────────────────────────────────────────────────────
    # One figure per (variable, segment): 3 laps × 2 cols (obs | model)
    sec_specs = [
        ('salt', 'salt', False),
        ('temp', 'temp', False),
        ('do',   'do',   False),
        ('spd',  'spd',  True),
    ]
    for var, v_key, is_adcp in sec_specs:
        cmap       = CMAPS[var]
        vmin, vmax = CLIMS[var]
        label      = LABELS[var]
        for seg in SEGS:
            print(f'  sections {var}/{seg}')
            fig, axes = plt.subplots(
                len(LAPS), 2,
                figsize=(8, 4 * len(LAPS)),
                squeeze=False,
            )
            last_pcm = None
            for r, lap in enumerate(LAPS):
                try:
                    if is_adcp:
                        obs_pts, mod_pts = build_adcp_section(lap, seg)
                    else:
                        obs_pts, mod_pts = build_ctd_section(lap, seg)
                except Exception as e:
                    print(f'    ERROR: {e}')
                    continue
                pcm = plot_section_pair(
                    axes[r, 0], axes[r, 1],
                    obs_pts, mod_pts,
                    x_key='x', z_key='z', v_key=v_key,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    lap_label=lap,
                )
                if pcm is not None:
                    last_pcm = pcm
            if last_pcm is not None:
                fig.colorbar(last_pcm, ax=axes.ravel().tolist(),
                             shrink=0.35, pad=0.04, label=label)
            fig.suptitle(
                f'Penn Cove Dec 2025 — {seg}\n{label}  |  {GTAGEX}',
                fontsize=10,
            )
            fig.tight_layout()
            out_fn = OUT_DIR / f'pcDec2025_{var}_{seg}_sections.png'
            fig.savefig(out_fn, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'    Saved: {out_fn}')

    # ── Collect all matchup data once ────────────────────────────────────────
    print('Collecting matchup data...')
    ctd_mu  = {}   # (lap, seg) -> dict
    adcp_mu = {}   # (lap, seg) -> dict
    for lap in LAPS:
        for seg in SEGS:
            print(f'  matchup CTD  {lap}/{seg}')
            try:
                ctd_mu[(lap, seg)] = build_ctd_matchup(lap, seg)
            except Exception as e:
                print(f'    ERROR: {e}')
            print(f'  matchup ADCP {lap}/{seg}')
            try:
                adcp_mu[(lap, seg)] = build_adcp_matchup(lap, seg)
            except Exception as e:
                print(f'    ERROR: {e}')

    def _pairs(data, obs_k, mod_k):
        """Aggregate matched pairs by lap from the matchup cache."""
        result = {}
        for lap in LAPS:
            os_, ms_ = [], []
            for seg in SEGS:
                mu = data.get((lap, seg))
                if mu is not None:
                    os_.append(mu[obs_k])
                    ms_.append(mu[mod_k])
            if os_:
                result[lap] = (np.concatenate(os_), np.concatenate(ms_))
        return result

    # ── 2×2 property-property figure ─────────────────────────────────────────
    print('Building property-property figure...')
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), squeeze=False)
    _draw_prop_panel(axes[0, 0], _pairs(ctd_mu,  'obs_salt', 'mod_salt'), LABELS['salt'])
    _draw_prop_panel(axes[0, 1], _pairs(ctd_mu,  'obs_temp', 'mod_temp'), LABELS['temp'])
    _draw_prop_panel(axes[1, 0], _pairs(ctd_mu,  'obs_do',   'mod_do'),   LABELS['do'])
    _draw_prop_panel(axes[1, 1], _pairs(adcp_mu, 'obs_spd',  'mod_spd'),  LABELS['spd'])
    fig.suptitle(f'Penn Cove Dec 2025 — obs vs model\n{GTAGEX}', fontsize=10)
    fig.tight_layout()
    out_fn = OUT_DIR / 'pcDec2025_prop_prop.png'
    fig.savefig(out_fn, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_fn}')

    # ── T-S diagram ──────────────────────────────────────────────────────────
    print('Building T-S diagram...')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), squeeze=False)
    for lap in LAPS:
        c = LAP_COLORS[lap]
        so, to, sm, tm = [], [], [], []
        for seg in SEGS:
            mu = ctd_mu.get((lap, seg))
            if mu is None:
                continue
            so.append(mu['obs_salt']); to.append(mu['obs_temp'])
            sm.append(mu['mod_salt']); tm.append(mu['mod_temp'])
        if not so:
            continue
        s_o = np.concatenate(so); t_o = np.concatenate(to)
        s_m = np.concatenate(sm); t_m = np.concatenate(tm)
        ok_o = np.isfinite(s_o) & np.isfinite(t_o)
        ok_m = np.isfinite(s_m) & np.isfinite(t_m)
        axes[0, 0].scatter(s_o[ok_o], t_o[ok_o], s=4, alpha=0.3,
                           color=c, edgecolors='none', label=lap)
        axes[0, 1].scatter(s_m[ok_m], t_m[ok_m], s=4, alpha=0.3,
                           color=c, edgecolors='none', label=lap)
    for ax, title in zip(axes[0], ['Obs', 'Model']):
        ax.set_xlabel('Salinity (PSU)', fontsize=9)
        ax.set_ylabel('Temperature (°C)', fontsize=9)
        ax.set_title(f'{title}', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, markerscale=3, framealpha=0.8)
        ax.grid(alpha=0.3)
    fig.suptitle(f'T-S diagram — Penn Cove Dec 2025\n{GTAGEX}', fontsize=10)
    fig.tight_layout()
    out_fn = OUT_DIR / 'pcDec2025_TS_diagram.png'
    fig.savefig(out_fn, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_fn}')

    print('\nDone. Output in:', OUT_DIR)


if __name__ == '__main__':
    main()
