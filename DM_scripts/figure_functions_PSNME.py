#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepared for publication: 2026/01/07

Author: Dakota Mascarenas

Plotting code for: "Century-Scale Changes in Dissolved Oxygen, Temperature, and Salinity in Puget Sound" (Mascarenas et al., in review; submitted 2026/01/09 to Estuaries & Coasts)

This script provides functions for data analysis and figure plotting in corresponding manuscript. Please reach out to the author at dakotamm@uw.edu for any questions.

"""

# import modules
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import gsw

# get unique cast locations for all casts in Puget Sound given locations on LiveOcean grid (MacCready et al., 2021; see text for more details and citations)
def get_cast_locations(df):
    df['ix_iy'] = df['ix'].astype(str).apply(lambda x: x.zfill(4)) + '_' + df['iy'].astype(str).apply(lambda x: x.zfill(4))
    cast_locations = df.groupby(['ix_iy']).first().reset_index()
    return cast_locations

# plot coast; adapted from Parker MacCready's public LO github repository (https://github.com/parkermac/LO/blob/main/lo_tools/lo_tools/plotting_functions.py)
def add_coast(ax, df_directory, color='k', linewidth=0.5):
    '''
    adapted from (https://github.com/parkermac/LO/blob/main/lo_tools/lo_tools/plotting_functions.py)
    '''
    fn = df_directory + '/coast_pnw.p'
    C = pd.read_pickle(fn)
    ax.plot(C['lon'].values, C['lat'].values, '-', color=color, linewidth=linewidth)

# make plot aspect ratio locally Cartesian; adapted from Parker MacCready's public LO github repository (https://github.com/parkermac/LO/blob/main/lo_tools/lo_tools/plotting_functions.py)
def dar(ax):
    '''
    adapted from (https://github.com/parkermac/LO/blob/main/lo_tools/lo_tools/plotting_functions.py)
    '''
    yl = ax.get_ylim()
    yav = (yl[0] + yl[1])/2
    ax.set_aspect(1/np.cos(np.pi*yav/180))


# apply DO filtering to casts in the bottom 50th percentile of annual seasonal bottom DO values
def filter_DO(site_depth_avg_var_DF):
    df_deep_DO = site_depth_avg_var_DF[(site_depth_avg_var_DF['var'] == 'DO_mg_L') & (site_depth_avg_var_DF['surf_deep'] == 'deep')]
    df_deep_DO['year_adjusted'] = df_deep_DO['year']
    df_deep_DO.loc[df_deep_DO['month'] == 12, 'year_adjusted'] = df_deep_DO['year'] + 1 #since trimesters do not evenly bisect one calendar year, this incorporates december into the following calendar year !!!!!*****
    df_deep_DO_q50 = df_deep_DO[['site', 'year_adjusted', 'season','val']].groupby(['site', 'year_adjusted', 'season']).quantile(0.5)
    df_deep_DO_q50 = df_deep_DO_q50.rename(columns={'val':'deep_DO_q50'})
    df_deep_DO_w_q50 = pd.merge(df_deep_DO, df_deep_DO_q50, how='left', on=['site','season','year_adjusted'])
    df_deep_DO_leq_q50 = df_deep_DO_w_q50[df_deep_DO_w_q50['val'] <= df_deep_DO_w_q50['deep_DO_q50']]
    cid_DO_leq_q50 = df_deep_DO_leq_q50['cid'].unique()
    filter_DO_DF = site_depth_avg_var_DF[(site_depth_avg_var_DF['var'] == 'DO_mg_L') & (site_depth_avg_var_DF['cid'].isin(cid_DO_leq_q50))]
    return filter_DO_DF

# Mann-Kendall trend test (adapted from original Matlab function; see Mann 1945,
# Kendall 1975). Returns reject_null, two-tailed p-value, and Z statistic.
def mann_kendall(V, alpha=0.05):
    V = np.reshape(V, (len(V), 1))
    alpha = alpha/2
    n = len(V)
    S = 0
    for i in range(0, n-1):
        for j in range(i+1, n):
            if V[j]>V[i]:
                S = S+1
            if V[j]<V[i]:
                S = S-1
    VarS = (n*(n-1)*(2*n+5))/18
    StdS = np.sqrt(VarS)
    # Ties are not considered
    Kendall_Tau = S/(n*(n-1)/2)
    if S>=0:
        if S==0:
             Z = 0
        else:
            Z = ((S-1)/StdS)
    else:
        Z = (S+1)/StdS
    Zalpha = stats.norm.ppf(1-alpha,0,1)
    p_value = 2*(1-stats.norm.cdf(abs(Z), 0, 1)) #Two-tailed test p-value
    reject_null = abs(Z) > Zalpha # reject null hypothesis only if abs(Z) > Zalpha
    return reject_null, p_value, Z

# calculate Theil-Sen slopes with confidence intervals for given time series and values and specified alpha
def calc_ts_slopes(working_df, alpha):
    x = working_df['date_ordinal'].copy()
    x_working = working_df['datetime'].copy()
    y = working_df['val'].copy()
    result = stats.theilslopes(y,x,alpha=alpha)
    B1 = result.slope
    B0 = result.intercept
    concat_df = working_df.head(1).copy()
    concat_df['B1'] = B1
    concat_df['B0'] = B0
    high_sB1 = result.high_slope
    low_sB1 = result.low_slope
    slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_working.max().year - x_working.min().year)
    concat_df['slope_datetime'] = slope_datetime #per year
    slope_datetime_s_hi = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_working.max().year - x_working.min().year)
    slope_datetime_s_lo = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_working.max().year - x_working.min().year)
    concat_df['slope_datetime_s_hi'] = slope_datetime_s_hi #per year
    concat_df['slope_datetime_s_lo'] = slope_datetime_s_lo #per year
    # Mann-Kendall trend significance (p) and sample size (n) -- used by paper_1_table_3
    reject_null, p_value, Z = mann_kendall(y, alpha)
    concat_df['p'] = p_value
    concat_df['n'] = len(y)
    working_df_concat = concat_df[['site', 'season', 'surf_deep', 'var', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo', 'B1', 'B0', 'p', 'n']]
    return working_df_concat

# calculate Theil-Sen slopes for DO, absolute salinity, conservative temperature, and calculated DO saturation
def calc_slopes_var(alpha, site_depth_avg_var_DF, filter_DO_DF=None, DO_sat_DF=None):
    slope_DF = pd.DataFrame()
    if filter_DO_DF is not None:
        CT_SA_DF = site_depth_avg_var_DF[site_depth_avg_var_DF['var'].isin(['CT','SA'])]
        if DO_sat_DF is not None:
            big_df = pd.concat([CT_SA_DF, filter_DO_DF, DO_sat_DF])
        else:
            big_df = pd.concat([CT_SA_DF, filter_DO_DF])
    else:
        if DO_sat_DF is not None:
            big_df = pd.concat([site_depth_avg_var_DF, DO_sat_DF])
        else:
            big_df = site_depth_avg_var_DF
    for site in big_df['site'].unique():
        for season in big_df['season'].unique():
            for var in big_df['var'].unique():
                for depth in big_df['surf_deep'].unique():
                    working_df = big_df[(big_df['site'] == site) & (big_df['season'] == season) & (big_df['var'] == var) & (big_df['surf_deep'] == depth)]
                    #working_df['var'] = working_df['surf_deep'] + '_' + working_df['var']
                    working_df_concat = calc_ts_slopes(working_df, alpha)
                    slope_DF = pd.concat([slope_DF, working_df_concat])    
    slope_DF.loc[slope_DF['site'] == 'point_jefferson', 'site_label'] = 'PJ' #labels for plotting
    slope_DF.loc[slope_DF['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'
    slope_DF.loc[slope_DF['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'
    slope_DF.loc[slope_DF['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'
    slope_DF.loc[slope_DF['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'
    slope_DF.loc[slope_DF['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'
    slope_DF.loc[slope_DF['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'
    slope_DF.loc[slope_DF['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'
    slope_DF.loc[slope_DF['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'
    slope_DF.loc[slope_DF['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'
    slope_DF.loc[slope_DF['site'] == 'point_jefferson', 'site_num'] = 1
    slope_DF.loc[slope_DF['site'] == 'near_seattle_offshore', 'site_num'] = 2
    slope_DF.loc[slope_DF['site'] == 'carr_inlet_mid', 'site_num'] = 3
    slope_DF.loc[slope_DF['site'] == 'saratoga_passage_mid', 'site_num'] = 4
    slope_DF.loc[slope_DF['site'] == 'lynch_cove_mid', 'site_num'] = 5
    slope_DF.loc[slope_DF['season'] == 'grow', 'season_label'] = 'Spring (Apr-Jul)'
    slope_DF.loc[slope_DF['season'] == 'loDO', 'season_label'] = 'Low-DO (Aug-Nov)'
    slope_DF.loc[slope_DF['season'] == 'winter', 'season_label'] = 'Winter (Dec-Mar)'
    slope_DF.loc[slope_DF['surf_deep'] == 'surf', 'depth_label'] = 'Surface'
    slope_DF.loc[slope_DF['surf_deep'] == 'deep', 'depth_label'] = 'Bottom'
    slope_DF.loc[slope_DF['var'] == 'CT', 'var_label'] = '[°C]'
    slope_DF.loc[slope_DF['var'] == 'SA', 'var_label'] = '[g/kg]'
    slope_DF.loc[slope_DF['var'] == 'DO_mg_L', 'var_label'] = '[mg/L]'
    slope_DF.loc[slope_DF['var'] == 'DO_sol', 'var_label'] = '[mg/L]'
    return slope_DF

# --- helpers for trend-shape classification (used by calc_trend_shapes) ---

# least-squares non-decreasing (isotonic) fit to y, via the pool-adjacent-violators algorithm
def pava_increasing(y):
    means = []  # running mean of each pooled block
    counts = []  # number of points in each block
    for value in y:
        block_mean, block_count = float(value), 1
        # merge with the previous block while it would break the non-decreasing order
        while means and means[-1] >= block_mean:
            prev_mean = means.pop()
            prev_count = counts.pop()
            block_mean = (prev_mean * prev_count + block_mean * block_count) / (prev_count + block_count)
            block_count += prev_count
        means.append(block_mean)
        counts.append(block_count)
    # expand each block's mean back out to one value per original point
    fit = np.empty(len(y))
    start = 0
    for block_mean, block_count in zip(means, counts):
        fit[start:start + block_count] = block_mean
        start += block_count
    return fit

# best monotone fit to y: the increasing or decreasing isotonic fit with the smaller residual sum of squares
def best_monotone(y):
    increasing = pava_increasing(y)
    decreasing = -pava_increasing(-y)
    if np.sum((y - increasing) ** 2) <= np.sum((y - decreasing) ** 2):
        return increasing
    return decreasing

# how much a smoothed curve reverses: total variation minus net change (0 for a perfectly monotone curve)
def reversal_stat(curve):
    total_variation = np.sum(np.abs(np.diff(curve)))
    net_change = abs(curve[-1] - curve[0])
    return float(total_variation - net_change)

# resample residuals in contiguous moving blocks (preserves short-range autocorrelation of the series)
def moving_block_resample(residuals, rng):
    n = len(residuals)
    block = max(3, int(round(n ** (1 / 3))))  # block length grows slowly with sample size
    last_start = n - block
    out = np.empty(n)
    filled = 0
    while filled < n:
        start = rng.integers(0, last_start + 1)
        take = min(block, n - filled)
        out[filled:filled + take] = residuals[start:start + take]
        filled += take
    return out

# direction-agnostic monotonicity test. Compares the reversal in a LOWESS smooth of the data against
# reversals generated under a best-monotone null (residuals resampled in moving blocks). Returns
# (R_frac, p): R_frac is the reversal as a fraction of total variation; small p indicates a real reversal.
def mono_bootstrap(t, y, rng, n_boot):
    order = np.argsort(t, kind='mergesort')
    t_sorted, y_sorted = t[order], y[order]
    span = 0.6 if len(t) < 40 else 0.45  # wider LOWESS window for shorter records
    smooth = lowess(y_sorted, t_sorted, frac=span, return_sorted=True)[:, 1]
    observed_R = reversal_stat(smooth)
    total_variation = np.sum(np.abs(np.diff(smooth)))
    monotone_fit = best_monotone(y_sorted)
    residuals = y_sorted - monotone_fit
    n_exceed = 0
    for _ in range(n_boot):
        null_y = monotone_fit + moving_block_resample(residuals, rng)
        null_smooth = lowess(null_y, t_sorted, frac=span, return_sorted=True)[:, 1]
        if reversal_stat(null_smooth) >= observed_R:
            n_exceed += 1
    R_frac = observed_R / total_variation if total_variation > 0 else np.nan
    p_value = (1 + n_exceed) / (n_boot + 1)
    return R_frac, p_value

# least-squares quadratic fit; returns [b0, b1, b2] for y ~ b0 + b1*x + b2*x^2
def fit_quadratic(x, y):
    design = np.column_stack([np.ones_like(x), x, x * x])
    return np.linalg.lstsq(design, y, rcond=None)[0]

# unconditional linearity test. Compares the observed quadratic curvature |b2| against curvature generated
# under a linear null (residuals resampled in moving blocks). Returns (label, p): label is 'accelerating' or
# 'decelerating' relative to the trend direction trend_sign; small p indicates the series is nonlinear.
def curvature_bootstrap(t, y, trend_sign, rng, n_boot, days=365.25):
    order = np.argsort(t, kind='mergesort')
    years = (t[order] - t.mean()) / days  # center and scale time to years for numerical conditioning
    y_sorted = y[order]
    observed_b2 = fit_quadratic(years, y_sorted)[2]
    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones_like(years), years]), y_sorted, rcond=None)[0]
    linear_fit = intercept + slope * years
    residuals = y_sorted - linear_fit
    n_exceed = 0
    for _ in range(n_boot):
        null_y = linear_fit + moving_block_resample(residuals, rng)
        if abs(fit_quadratic(years, null_y)[2]) >= abs(observed_b2):
            n_exceed += 1
    if trend_sign == 0:
        label = ''
    elif np.sign(observed_b2) == trend_sign:
        label = 'accelerating'
    else:
        label = 'decelerating'
    p_value = (1 + n_exceed) / (n_boot + 1)
    return label, p_value

# Benjamini-Hochberg FDR q-values for an array of p-values (NaNs are passed through unchanged)
def benjamini_hochberg(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    p_valid = p[valid]
    m = len(p_valid)
    if m == 0:
        return q
    order = np.argsort(p_valid)
    p_sorted = p_valid[order]
    scaled = p_sorted * m / np.arange(1, m + 1)
    # step-up: q at each rank is the running minimum of the scaled p-values from the largest rank down
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q_valid = np.empty(m)
    q_valid[order] = np.clip(q_sorted, 0, 1)
    q[valid] = q_valid
    return q

# classify each seasonal time series' full-record trend shape as reversal / curved / linear / no-trend.
# For every site/season/depth/variable series this computes: full-record Theil-Sen slope and significance
# (CI excludes 0); a direction-agnostic monotonicity test (mono_bootstrap); and a linearity test
# (curvature_bootstrap). Both tests are put on the same inferential footing: Benjamini-Hochberg FDR applied
# over the CT/SA/DO measured series (90) and the DO_sat solubility series (30) separately. Because each BH
# threshold (~alpha/90) sits at or below the first-pass bootstrap floor 1/(B+1), the near-threshold
# candidates of BOTH tests are re-run at B_high before FDR is applied. The linearity-first hierarchy assigns
# 'group': reversal ('rev') if the monotonicity FDR rejects, else curved ('curve') if the curvature FDR
# rejects, else linear ('lin') if the full-record slope is significant, else no trend ('no'). A raw
# curvature flag (nonlin_uncond) is also returned for sensitivity reporting.
def calc_trend_shapes(alpha, site_depth_avg_var_DF, filter_DO_DF=None, DO_sat_DF=None,
                      B=1000, B_high=10000, min_n=20, cand_thresh=0.02, seed=20260723):
    # build the working set: measured DO gets the bottom-water filter, DO_sat is the calculated solubility
    if filter_DO_DF is not None:
        parts = [site_depth_avg_var_DF[site_depth_avg_var_DF['var'].isin(['CT', 'SA'])], filter_DO_DF]
    else:
        parts = [site_depth_avg_var_DF]
    if DO_sat_DF is not None:
        parts.append(DO_sat_DF)
    big_df = pd.concat(parts)
    # fixed iteration order so the shared random-number stream (and thus the bootstrap p-values) is reproducible
    sites = ['point_jefferson', 'near_seattle_offshore', 'carr_inlet_mid', 'saratoga_passage_mid', 'lynch_cove_mid']
    seasons = ['winter', 'grow', 'loDO']
    depths = ['surf', 'deep']
    variables = ['CT', 'SA', 'DO_mg_L']
    if DO_sat_DF is not None:
        variables.append('DO_sat')  # DO_sat is the calculated-solubility series (Table 3)
    rng = np.random.default_rng(seed)
    rows = []
    series_xy = {}  # cache each series' (x, y) so near-threshold candidates can be re-run at higher B
    for var in variables:
        for site in sites:
            for season in seasons:
                for depth in depths:
                    series = big_df[(big_df['var'] == var) & (big_df['site'] == site) &
                                    (big_df['season'] == season) & (big_df['surf_deep'] == depth)]
                    series = series.dropna(subset=['val', 'date_ordinal'])
                    x = series['date_ordinal'].to_numpy(float)
                    y = series['val'].to_numpy(float)
                    record = dict(site=site, season=season, surf_deep=depth, var=var, n=len(x))
                    if len(x) < min_n:  # too few points to test shape -> treated as no trend downstream
                        rows.append({**record, 'full_slope': np.nan, 'full_sig': False, 'R_frac': np.nan,
                                     'mono_p': np.nan, 'lin_p': np.nan, 'accel': ''})
                        continue
                    series_xy[(var, site, season, depth)] = (x, y)
                    ts = stats.theilslopes(y, x, alpha=alpha)
                    full_slope = float(ts.slope) * 365.25 * 100  # per-century change in the variable's units
                    full_sig = bool(ts.low_slope > 0 or ts.high_slope < 0)
                    R_frac, mono_p = mono_bootstrap(x, y, rng, B)
                    accel, lin_p = curvature_bootstrap(x, y, np.sign(full_slope), rng, B)
                    rows.append({**record, 'full_slope': full_slope, 'full_sig': full_sig, 'R_frac': R_frac,
                                 'mono_p': mono_p, 'lin_p': lin_p, 'accel': accel})
    shape_DF = pd.DataFrame(rows)
    # both tests are FDR-corrected below, and each BH threshold (~alpha/90) sits at or below the first-pass
    # bootstrap floor 1/(B+1), so the near-threshold candidates of both are re-run at B_high before FDR can
    # certify them. R_frac and the accel label are deterministic, so only the p-values are updated.
    hi_rng = np.random.default_rng(seed + 1)
    for i in shape_DF.index[shape_DF['mono_p'] < cand_thresh]:
        row = shape_DF.loc[i]
        x, y = series_xy[(row['var'], row['site'], row['season'], row['surf_deep'])]
        _, mono_p = mono_bootstrap(x, y, hi_rng, B_high)
        shape_DF.loc[i, 'mono_p'] = mono_p
    for i in shape_DF.index[shape_DF['lin_p'] < cand_thresh]:
        row = shape_DF.loc[i]
        x, y = series_xy[(row['var'], row['site'], row['season'], row['surf_deep'])]
        _, lin_p = curvature_bootstrap(x, y, np.sign(row['full_slope']), hi_rng, B_high)
        shape_DF.loc[i, 'lin_p'] = lin_p
    # Benjamini-Hochberg over the measured CT/SA/DO series (90) and the DO_sat solubility series (30)
    # separately, applied to both the reversal (mono) and curvature (lin) p-values
    shape_DF['mono_q'] = np.nan
    shape_DF['lin_q'] = np.nan
    for family in (shape_DF['var'].isin(['CT', 'SA', 'DO_mg_L']), shape_DF['var'] == 'DO_sat'):
        shape_DF.loc[family, 'mono_q'] = benjamini_hochberg(shape_DF.loc[family, 'mono_p'].values)
        shape_DF.loc[family, 'lin_q'] = benjamini_hochberg(shape_DF.loc[family, 'lin_p'].values)
    shape_DF['nonmono_BH'] = shape_DF['mono_q'] < alpha  # significant reversal after FDR
    shape_DF['nonlin_BH'] = shape_DF['lin_q'] < alpha  # significant curvature after FDR
    shape_DF['nonlin_uncond'] = shape_DF['lin_p'] < alpha  # raw curvature flag, kept for sensitivity reporting
    # linearity-first hierarchy: reversal > curved > linear > no trend
    def group(row):
        if bool(row['nonmono_BH']):
            return 'rev'
        if bool(row['nonlin_BH']):
            return 'curve'
        if bool(row['full_sig']):
            return 'lin'
        return 'no'
    shape_DF['group'] = shape_DF.apply(group, axis=1)
    shape_DF.loc[shape_DF['site'] == 'point_jefferson', 'site_label'] = 'PJ' #labels for plotting
    shape_DF.loc[shape_DF['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'
    shape_DF.loc[shape_DF['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'
    shape_DF.loc[shape_DF['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'
    shape_DF.loc[shape_DF['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'
    shape_DF.loc[shape_DF['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'
    shape_DF.loc[shape_DF['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'
    shape_DF.loc[shape_DF['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'
    shape_DF.loc[shape_DF['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'
    shape_DF.loc[shape_DF['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'
    shape_DF.loc[shape_DF['site'] == 'point_jefferson', 'site_num'] = 1
    shape_DF.loc[shape_DF['site'] == 'near_seattle_offshore', 'site_num'] = 2
    shape_DF.loc[shape_DF['site'] == 'carr_inlet_mid', 'site_num'] = 3
    shape_DF.loc[shape_DF['site'] == 'saratoga_passage_mid', 'site_num'] = 4
    shape_DF.loc[shape_DF['site'] == 'lynch_cove_mid', 'site_num'] = 5
    shape_DF.loc[shape_DF['season'] == 'grow', 'season_label'] = 'Spring (Apr-Jul)'
    shape_DF.loc[shape_DF['season'] == 'loDO', 'season_label'] = 'Low-DO (Aug-Nov)'
    shape_DF.loc[shape_DF['season'] == 'winter', 'season_label'] = 'Winter (Dec-Mar)'
    shape_DF.loc[shape_DF['surf_deep'] == 'surf', 'depth_label'] = 'Surface'
    shape_DF.loc[shape_DF['surf_deep'] == 'deep', 'depth_label'] = 'Bottom'
    shape_DF.loc[shape_DF['var'] == 'CT', 'var_label'] = '[°C]'
    shape_DF.loc[shape_DF['var'] == 'SA', 'var_label'] = '[g/kg]'
    shape_DF.loc[shape_DF['var'] == 'DO_mg_L', 'var_label'] = '[mg/L]'
    shape_DF.loc[shape_DF['var'] == 'DO_sat', 'var_label'] = '[mg/L]'
    return shape_DF

# median pairwise (Theil-Sen) slope of y vs x, used by the era-difference bootstrap below.
# Kept separate from calc_ts_slopes so the null-shifted bootstrap can call it thousands of times cheaply.
def ts_slope_pairwise(x, y):
    if len(x) < 3:
        return np.nan
    i, j = np.triu_indices(len(x), k=1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    m = dx != 0
    return float(np.median(dy[m] / dx[m])) if m.any() else np.nan

# compare pre-1999 vs post-1999 Theil-Sen slopes for every site/season/depth/variable series (1999 =
# onset of the modern WA Dept. of Ecology CTD program). For each series this returns the full-record,
# pre-1999, and post-1999 slopes with 95% CIs (per century, reusing calc_slopes_var so the estimator
# matches the manuscript's reported trends) alongside a test of the era difference. The era difference
# delta = post_slope - pre_slope (per century) is tested with a null-shifted bootstrap: casts are
# resampled with replacement within each era (B draws), the bootstrap delta distribution is recentered
# on zero to form the null, and the two-sided p-value is the null probability of a delta at least as
# extreme as observed. p-values are Benjamini-Hochberg-corrected over the testable series (era_q) and a
# series is significant at era_q < alpha (era_sig). Runs the (slow) bootstrap once; downstream scripts
# read the saved data frame instead of re-running it. DO uses the DO-filtered casts (filter_DO_DF).
def calc_era_slope_diff(alpha, site_depth_avg_var_DF, filter_DO_DF, split_year=1999, B=10000, seed=20260630):
    # split each source into pre-/post-split_year eras
    pre_DF  = site_depth_avg_var_DF[site_depth_avg_var_DF['year'] < split_year].copy()
    post_DF = site_depth_avg_var_DF[site_depth_avg_var_DF['year'] >= split_year].copy()
    filter_DO_pre_DF  = filter_DO_DF[filter_DO_DF['year'] < split_year].copy()
    filter_DO_post_DF = filter_DO_DF[filter_DO_DF['year'] >= split_year].copy()

    # full-record and per-era Theil-Sen slopes with 95% CIs (per century), reusing the manuscript's estimator
    slope_full_DF = calc_slopes_var(alpha, site_depth_avg_var_DF, filter_DO_DF)
    slope_pre_DF  = calc_slopes_var(alpha, pre_DF,  filter_DO_pre_DF)
    slope_post_DF = calc_slopes_var(alpha, post_DF, filter_DO_post_DF)
    keys = ['var', 'site', 'season', 'surf_deep']
    labels = keys + ['site_num', 'site_label', 'site_type', 'season_label', 'depth_label', 'var_label']
    slope_cols = {'slope_datetime': 'slope_cent', 'slope_datetime_s_lo': 'slope_cent_lo', 'slope_datetime_s_hi': 'slope_cent_hi'}

    def era_slopes(slope_DF, suffix):
        out = slope_DF[keys + list(slope_cols)].copy()
        for src, dst in slope_cols.items():
            out[dst + suffix] = out.pop(src) * 100  # per century
        return out

    era_DF = slope_full_DF[labels].copy()
    era_DF = era_DF.merge(era_slopes(slope_full_DF, '_full'), on=keys, how='left')
    era_DF = era_DF.merge(era_slopes(slope_pre_DF,  '_pre'),  on=keys, how='left')
    era_DF = era_DF.merge(era_slopes(slope_post_DF, '_post'), on=keys, how='left')
    # connector direction in the figure: sign of the plotted post-minus-pre change
    era_DF['delta_cent'] = era_DF['slope_cent_post'] - era_DF['slope_cent_pre']

    # null-shifted bootstrap of the era difference, resampling casts within each era with replacement
    DAYS_PER_YR = 365.25
    src_pre  = {'CT': pre_DF,  'SA': pre_DF,  'DO_mg_L': filter_DO_pre_DF}
    src_post = {'CT': post_DF, 'SA': post_DF, 'DO_mg_L': filter_DO_post_DF}
    variables = ['CT', 'SA', 'DO_mg_L']
    sites = ['point_jefferson', 'near_seattle_offshore', 'carr_inlet_mid', 'saratoga_passage_mid', 'lynch_cove_mid']
    seasons = ['winter', 'grow', 'loDO']
    depths = ['surf', 'deep']
    rng = np.random.default_rng(seed)  # fixed stream so the bootstrap p-values are reproducible
    boot_rows = []
    for var in variables:
        for site in sites:
            for season in seasons:
                for depth in depths:
                    def xy(src):
                        s = src[(src['var'] == var) & (src['site'] == site) & (src['season'] == season) &
                                (src['surf_deep'] == depth)].dropna(subset=['val', 'date_ordinal'])
                        return s['date_ordinal'].to_numpy(float), s['val'].to_numpy(float)
                    xp, yp = xy(src_pre[var])
                    xo, yo = xy(src_post[var])
                    rec = dict(var=var, site=site, season=season, surf_deep=depth,
                               n_pre=len(xp), n_post=len(xo))
                    if len(xp) < 3 or len(xo) < 3:  # too few casts in an era to test the difference
                        boot_rows.append({**rec, 'era_delta': np.nan, 'era_p': np.nan})
                        continue
                    d_obs = (ts_slope_pairwise(xo, yo) - ts_slope_pairwise(xp, yp)) * DAYS_PER_YR * 100
                    ip = rng.integers(0, len(xp), size=(B, len(xp)))
                    io = rng.integers(0, len(xo), size=(B, len(xo)))
                    bd = np.array([(ts_slope_pairwise(xo[io[b]], yo[io[b]]) - ts_slope_pairwise(xp[ip[b]], yp[ip[b]])) * DAYS_PER_YR * 100
                                   for b in range(B)])
                    bd = bd[np.isfinite(bd)]
                    p = float(min(np.mean(np.abs(bd - bd.mean()) >= abs(d_obs)), 1.0)) if len(bd) >= 10 else np.nan
                    boot_rows.append({**rec, 'era_delta': d_obs, 'era_p': p})
    boot_DF = pd.DataFrame(boot_rows)
    boot_DF['era_q'] = benjamini_hochberg(boot_DF['era_p'].values)  # BH over the testable series
    boot_DF['era_sig'] = boot_DF['era_q'] < alpha

    era_DF = era_DF.merge(boot_DF, on=keys, how='left')
    return era_DF

# calculate time series median and interquartile range (IQR) values for each season
def calc_seasonal_series_med_iqr(site_depth_avg_var_DF, filter_DO_DF=None):
    if filter_DO_DF is not None:
        CT_SA_DF = site_depth_avg_var_DF[site_depth_avg_var_DF['var'].isin(['CT','SA'])]
        big_df = pd.concat([CT_SA_DF, filter_DO_DF])
    else:
        big_df = site_depth_avg_var_DF
    series_counts = (big_df
                     .dropna()
                     .groupby(['site', 'season', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                     .reset_index()
                     .rename(columns={'cid':'cid_count'})
                     )
    means_DF = big_df.groupby(['site', 'surf_deep', 'season','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
    means_DF.columns = means_DF.columns.to_flat_index().map('_'.join)
    means_DF = means_DF.reset_index().dropna() 
    means_DF = (means_DF
                    .rename(columns={'date_ordinal_mean':'date_ordinal'})
                    .dropna()
                    .assign(
                            datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                            )
                    )
    means_DF = pd.merge(means_DF, series_counts, how='left', on=['site','surf_deep','season','var'])
    means_DF = means_DF[means_DF['cid_count'] >1] #redundant but fine (see note line 234)
    pctl = (big_df
            .dropna()
            .groupby(['site', 'surf_deep', 'season', 'var'])['val']
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .rename(columns={0.25: 'val_q25', 0.5: 'val_median', 0.75: 'val_q75'})
            .reset_index()
            )
    means_DF = pd.merge(means_DF, pctl, how='left', on=['site','surf_deep','season','var'])
    means_DF.loc[means_DF['site'] == 'point_jefferson', 'site_label'] = 'PJ' #labels for plotting
    means_DF.loc[means_DF['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'
    means_DF.loc[means_DF['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'
    means_DF.loc[means_DF['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'
    means_DF.loc[means_DF['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'
    means_DF.loc[means_DF['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'
    means_DF.loc[means_DF['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'
    means_DF.loc[means_DF['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'
    means_DF.loc[means_DF['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'
    means_DF.loc[means_DF['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'
    means_DF.loc[means_DF['site'] == 'point_jefferson', 'site_num'] = 1
    means_DF.loc[means_DF['site'] == 'near_seattle_offshore', 'site_num'] = 2
    means_DF.loc[means_DF['site'] == 'carr_inlet_mid', 'site_num'] = 3
    means_DF.loc[means_DF['site'] == 'saratoga_passage_mid', 'site_num'] = 4
    means_DF.loc[means_DF['site'] == 'lynch_cove_mid', 'site_num'] = 5
    means_DF.loc[means_DF['season'] == 'grow', 'season_label'] = 'Spring (Apr-Jul)'
    means_DF.loc[means_DF['season'] == 'loDO', 'season_label'] = 'Low-DO (Aug-Nov)'
    means_DF.loc[means_DF['season'] == 'winter', 'season_label'] = 'Winter (Dec-Mar)'
    means_DF.loc[means_DF['surf_deep'] == 'surf', 'depth_label'] = 'Surface'
    means_DF.loc[means_DF['surf_deep'] == 'deep', 'depth_label'] = 'Bottom'
    means_DF.loc[means_DF['var'] == 'CT', 'var_label'] = '[°C]'
    means_DF.loc[means_DF['var'] == 'SA', 'var_label'] = '[g/kg]'
    means_DF.loc[means_DF['var'] == 'DO_mg_L', 'var_label'] = '[mg/L]'
    return means_DF

    
# calculate DO saturation using individual cast, depth-binned conservative temperature and absolute salinity; NOTE: this only uses casts that go to bottom water to avoid oversampling in surface waters
def calc_DO_sat(site_depth_avg_var_DF):
    cid_deep = site_depth_avg_var_DF.loc[site_depth_avg_var_DF['surf_deep'] == 'deep', 'cid'] #find unique casts that exceed the threshold depth for bottom water depth-binning
    df_deep = site_depth_avg_var_DF[site_depth_avg_var_DF['cid'].isin(cid_deep)] #filter dataframe to bottom water casts
    df_calc = df_deep.pivot(index = ['site', 'year', 'month', 'season','date_ordinal','cid'], columns = ['surf_deep', 'var'], values ='val')
    df_calc.columns = df_calc.columns.to_flat_index().map('_'.join)
    df_calc = df_calc.reset_index()
    df_calc['surf_dens'] = gsw.density.sigma0(df_calc['surf_SA'], df_calc['surf_CT'])
    df_calc['deep_dens'] = gsw.density.sigma0(df_calc['deep_SA'], df_calc['deep_CT'])
    A_0 = 5.80818 #all in umol/kg, from Garcia & Gordon (1992)
    A_1 = 3.20684
    A_2 = 4.11890
    A_3 = 4.93845
    A_4 = 1.01567
    A_5 = 1.41575
    B_0 = -7.01211e-3
    B_1 = -7.25958e-3
    B_2 = -7.93334e-3
    B_3 = -5.54491e-3
    C_0 = -1.32412e-7
    df_calc['surf_T_s'] = np.log((298.15 - df_calc['surf_CT'])/(273.15 + df_calc['surf_CT']))
    df_calc['surf_C_o_*'] = np.exp(A_0 + A_1*df_calc['surf_T_s'] + A_2*df_calc['surf_T_s']**2 + A_3*df_calc['surf_T_s']**3 + A_4*df_calc['surf_T_s']**4 + A_5*df_calc['surf_T_s']**5 + 
                           df_calc['surf_SA']*(B_0 + B_1*df_calc['surf_T_s'] + B_2*df_calc['surf_T_s']**2 + B_3*df_calc['surf_T_s']**3) + C_0*df_calc['surf_SA']**2)
    df_calc['surf_DO_sat'] =  df_calc['surf_C_o_*']*(df_calc['surf_dens']/1000 + 1)*32/1000
    df_calc['deep_T_s'] = np.log((298.15 - df_calc['deep_CT'])/(273.15 + df_calc['deep_CT']))
    df_calc['deep_C_o_*'] = np.exp(A_0 + A_1*df_calc['deep_T_s'] + A_2*df_calc['deep_T_s']**2 + A_3*df_calc['deep_T_s']**3 + A_4*df_calc['deep_T_s']**4 + A_5*df_calc['deep_T_s']**5 + 
                           df_calc['deep_SA']*(B_0 + B_1*df_calc['deep_T_s'] + B_2*df_calc['deep_T_s']**2 + B_3*df_calc['deep_T_s']**3) + C_0*df_calc['deep_SA']**2)
    df_calc['deep_DO_sat'] =  df_calc['deep_C_o_*']*(df_calc['deep_dens']/1000 + 1)*32/1000
    DO_sat_DF = pd.melt(df_calc, id_vars = ['site', 'year', 'month', 'season', 'date_ordinal','cid'], value_vars=['surf_DO_sat', 'deep_DO_sat'], var_name='var', value_name='val')
    DO_sat_DF.loc[DO_sat_DF['var'] == 'surf_DO_sat', 'surf_deep'] = 'surf' # match formatting
    DO_sat_DF.loc[DO_sat_DF['var'] == 'deep_DO_sat', 'surf_deep'] = 'deep'
    DO_sat_DF.loc[DO_sat_DF['var'] == 'surf_DO_sat', 'var'] = 'DO_sat'
    DO_sat_DF.loc[DO_sat_DF['var'] == 'deep_DO_sat', 'var'] = 'DO_sat'
    DO_sat_DF = DO_sat_DF.dropna().assign(datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))
    return DO_sat_DF



# clean specific air temperature dataset for Figure 10; averages daily maximum and minimum for monthly average
def get_clean_temps(temp_temp_df):
    temp_df = temp_temp_df.copy()
    temp_df['site'] = 'seatac'
    temp_df['datetime'] = pd.to_datetime(temp_df['DATE'])
    temp_df['year'] = pd.DatetimeIndex(temp_df['datetime']).year
    temp_df['month'] = pd.DatetimeIndex(temp_df['datetime']).month
    temp_df['year_month'] = temp_df['year'].astype(str) + '_' + temp_df['month'].astype(str).apply(lambda x: x.zfill(2))
    temp_df['date_ordinal'] = temp_df['datetime'].apply(lambda x: x.toordinal())
    temp_df.loc[temp_df['month'].isin([4,5,6,7]), 'season'] = 'grow'
    temp_df.loc[temp_df['month'].isin([8,9,10,11]), 'season'] = 'loDO'
    temp_df.loc[temp_df['month'].isin([12,1,2,3]), 'season'] = 'winter'
    temp_df['TAVG'] = temp_df[['TMAX', 'TMIN']].mean(axis=1)
    temp_df['year_season'] = temp_df['year'].astype(str) + '_' + temp_df['season']
    temp_df = pd.melt(temp_df, id_vars =['site', 'STATION', 'NAME', 'DATE', 'datetime', 'year', 'month', 'year_month', 'date_ordinal', 'season', 'year_season'], value_vars=['PRCP', 'TMAX', 'TMIN', 'TSUN', 'TAVG'], var_name='var', value_name='val')
    return temp_df

# calculate monthly average air temperatures for specific dataset for Figure 10 with 95% confidence intervals
def get_monthly_temps(temp_df):
    monthly_counts = (temp_df
                          .dropna()
                          .groupby(['site','year_month', 'var']).agg({'val' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'val':'val_count'})
                          )
    temp_monthly_avg_df = temp_df[['site', 'datetime', 'date_ordinal', 'year', 'month', 'year_month', 'season', 'year_season', 'var', 'val']].groupby(['site', 'year','month','year_month','season', 'year_season', 'var']).agg({'val':['mean', 'std'], 'date_ordinal':['mean']})
    temp_monthly_avg_df.columns = temp_monthly_avg_df.columns.to_flat_index().map('_'.join)
    temp_monthly_avg_df = temp_monthly_avg_df.reset_index().dropna()
    temp_monthly_avg_df = (temp_monthly_avg_df
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))
                      )
    temp_monthly_avg_df = pd.merge(temp_monthly_avg_df, monthly_counts, how='left', on=['site', 'year_month', 'var'])
    temp_monthly_avg_df = temp_monthly_avg_df[temp_monthly_avg_df['val_count'] >1] #redundant but sanity check
    temp_monthly_avg_df['val_ci95hi'] = temp_monthly_avg_df['val_mean'] + 1.96*temp_monthly_avg_df['val_std']/np.sqrt(temp_monthly_avg_df['val_count'])
    temp_monthly_avg_df['val_ci95lo'] = temp_monthly_avg_df['val_mean'] - 1.96*temp_monthly_avg_df['val_std']/np.sqrt(temp_monthly_avg_df['val_count'])
    temp_monthly_avg_df['val'] = temp_monthly_avg_df['val_mean']
    temp_monthly_DF = temp_monthly_avg_df.copy()
    return temp_monthly_DF

# calculate annual average air temperatures for specific dataset for Figure 10 with 95% confidence intervals
def get_annual_temps(temp_df):
    annual_counts = (temp_df
                          .dropna()
                          .groupby(['site','year', 'var']).agg({'val' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'val':'val_count'})
                          )
    temp_annual_avg_df = temp_df[['site', 'datetime', 'date_ordinal', 'year','var','val']].groupby(['site', 'year', 'var']).agg({'val':['mean', 'std'], 'date_ordinal':['mean']})
    temp_annual_avg_df.columns = temp_annual_avg_df.columns.to_flat_index().map('_'.join)
    temp_annual_avg_df = temp_annual_avg_df.reset_index().dropna()
    temp_annual_avg_df = (temp_annual_avg_df
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))
                      )
    temp_annual_avg_df = pd.merge(temp_annual_avg_df, annual_counts, how='left', on=['site', 'year', 'var'])
    temp_annual_avg_df = temp_annual_avg_df[temp_annual_avg_df['val_count'] >1] #redundant but sanity check
    temp_annual_avg_df['val_ci95hi'] = temp_annual_avg_df['val_mean'] + 1.96*temp_annual_avg_df['val_std']/np.sqrt(temp_annual_avg_df['val_count'])
    temp_annual_avg_df['val_ci95lo'] = temp_annual_avg_df['val_mean'] - 1.96*temp_annual_avg_df['val_std']/np.sqrt(temp_annual_avg_df['val_count'])
    temp_annual_avg_df['val'] = temp_annual_avg_df['val_mean']
    temp_annual_DF = temp_annual_avg_df.copy()
    return temp_annual_DF

# calculate depth-binned, yearly seasonal average DO, absolute salinity, conservative temperature, and calculated DO saturation with 95% confidence intervals
def get_seasonal_vars(site_depth_avg_var_DF, filter_DO_DF=None):
    if filter_DO_DF is not None:
        CT_SA_DF = site_depth_avg_var_DF[site_depth_avg_var_DF['var'].isin(['CT','SA'])]
        big_df = pd.concat([CT_SA_DF, filter_DO_DF])
    else:
        big_df = site_depth_avg_var_DF
    seasonal_counts = (big_df
                          .dropna()
                          .groupby(['site','year','surf_deep', 'season', 'var']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )
    seasonal_avg_df = big_df.groupby(['site', 'surf_deep', 'season', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
    seasonal_avg_df.columns = seasonal_avg_df.columns.to_flat_index().map('_'.join)
    seasonal_avg_df = seasonal_avg_df.reset_index().dropna()
    seasonal_avg_df = (seasonal_avg_df
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .dropna()
                      .assign(datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))
                      )
    seasonal_avg_df = pd.merge(seasonal_avg_df, seasonal_counts, how='left', on=['site','surf_deep', 'season', 'year','var'])
    seasonal_avg_df = seasonal_avg_df[seasonal_avg_df['cid_count'] >1] #redundant but sanity check
    seasonal_avg_df['val_ci95hi'] = seasonal_avg_df['val_mean'] + 1.96*seasonal_avg_df['val_std']/np.sqrt(seasonal_avg_df['cid_count'])
    seasonal_avg_df['val_ci95lo'] = seasonal_avg_df['val_mean'] - 1.96*seasonal_avg_df['val_std']/np.sqrt(seasonal_avg_df['cid_count'])
    seasonal_avg_df['val'] = seasonal_avg_df['val_mean']
    seasonal_DF = seasonal_avg_df.copy()
    return seasonal_DF