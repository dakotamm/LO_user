"""
Extract DFO CTD and bottle casts near the CFMETR Nanoose Bay station for HD.

Source:
    Local LO archive of the DFO/CIOOS Pacific compilation maintained by
    Susan Allen's group (Elise Olson). Files at:
        LO_output/obs/dfo1/ctd/{info_,}YYYY.p
        LO_output/obs/dfo1/bottle/{info_,}YYYY.p
    Original data: https://data.cioospacific.ca/erddap/index.html
    Citation: see https://cioospacific.ca/about/data-management/
    See also LO/obs/README.md and LO/obs/dfo1/README.md.

Station:
    The CFMETR (Canadian Forces Maritime Experimental and Test Ranges)
    Nanoose Bay station ("Whisky Gulf") near 49.32 N, -124.13 W in the
    Strait of Georgia, as used by:
        Masson & Cummins (2007), "Temperature trends and interannual
        variability in the Strait of Georgia, British Columbia",
        Cont. Shelf Res., doi:10.1016/j.csr.2006.10.009.

Outputs (LO_output/extract/nanoose_for_HD/):
    nanoose_ctd_info.{p,csv}        - one row per cast (lon, lat, time, ...)
    nanoose_ctd_profiles.{p,csv}    - all depth samples for those casts
    nanoose_bottle_info.{p,csv}
    nanoose_bottle_profiles.{p,csv}
    nanoose_summary.txt             - search params, counts/year, var coverage
    nanoose_map.png                 - map + casts-per-year bar chart

Run:
    python extract_nanoose.py
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

# ---------------------------------------------------------------------------
# Search parameters (edit here)
# ---------------------------------------------------------------------------
NANOOSE_LAT = 49.32      # deg N  (CFMETR Nanoose offshore measurement area;
NANOOSE_LON = -124.13    # deg E   ~8 km NE of DFO/IOS station 7930 in the bay)
SEARCH_RADIUS_KM = 20.0  # km from the station center
NAME_FILTER = "nanoose"  # also include casts whose `name` contains this (case-insensitive)
                         # set to None to disable the name-based OR filter

YEARS_CTD = list(range(1965, 2022))     # dfo1 ctd coverage
YEARS_BOTTLE = list(range(1930, 2022))  # dfo1 bottle coverage

OTYPES = ["ctd", "bottle"]
YEAR_LISTS = {"ctd": YEARS_CTD, "bottle": YEARS_BOTTLE}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km. Inputs may be scalars or arrays (degrees)."""
    R = 6371.0088
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon1r = np.radians(lon1); lat1r = np.radians(lat1)
    lon2r = np.radians(lon2); lat2r = np.radians(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def select_casts_for_year(in_dir: Path, year: int):
    """
    Load info_YEAR.p, return (info_subset, cid_list, distances_km) for casts
    that match the spatial radius OR the name filter. Returns (None, [], None)
    if no info file or no matches.
    """
    info_fn = in_dir / f"info_{year}.p"
    if not info_fn.is_file():
        return None, [], None
    info = pd.read_pickle(info_fn)
    if info.empty:
        return None, [], None

    dist = haversine_km(info["lon"].values, info["lat"].values,
                        NANOOSE_LON, NANOOSE_LAT)
    spatial_mask = dist <= SEARCH_RADIUS_KM

    if NAME_FILTER is not None and "name" in info.columns:
        name_series = info["name"].astype("string").str.lower()
        name_mask = name_series.str.contains(NAME_FILTER, na=False).values
    else:
        name_mask = np.zeros_like(spatial_mask, dtype=bool)

    keep_mask = spatial_mask | name_mask
    if not keep_mask.any():
        return None, [], None

    info_sel = info.loc[keep_mask].copy()
    info_sel["distance_km"] = dist[keep_mask]
    info_sel["matched_by_name"] = name_mask[keep_mask] & ~spatial_mask[keep_mask]
    info_sel["year"] = year
    cid_list = info_sel.index.tolist()
    return info_sel, cid_list, dist[keep_mask]


def extract_otype(otype: str, in_root: Path):
    """Loop years, return concatenated (info_df, profiles_df) with global ids."""
    in_dir = in_root / "dfo1" / otype
    info_chunks = []
    prof_chunks = []
    years = YEAR_LISTS[otype]

    for year in years:
        info_sel, cid_list, _ = select_casts_for_year(in_dir, year)
        if info_sel is None:
            continue

        # Build globally-unique cast ids: "<year>_<original_cid>"
        # info_sel index is named 'cid' (per LO convention); move it to a column.
        info_sel = info_sel.reset_index()  # creates a 'cid' column
        info_sel = info_sel.rename(columns={"cid": "cid_orig"})
        info_sel["nanoose_cid"] = (
            info_sel["year"].astype(str) + "_" + info_sel["cid_orig"].astype(str)
        )

        # Load profiles for this year and subset
        prof_fn = in_dir / f"{year}.p"
        if not prof_fn.is_file():
            print(f"  [{otype} {year}] info had hits but no profile file; skipping")
            continue
        prof = pd.read_pickle(prof_fn)
        prof_sel = prof[prof["cid"].isin(cid_list)].copy()
        if prof_sel.empty:
            print(f"  [{otype} {year}] profile file had no rows for matched cids")
            continue

        # Map original cid -> global id
        cid_to_global = dict(zip(info_sel["cid_orig"], info_sel["nanoose_cid"]))
        prof_sel["nanoose_cid"] = prof_sel["cid"].map(cid_to_global)
        prof_sel["year"] = year

        info_chunks.append(info_sel)
        prof_chunks.append(prof_sel)
        print(f"  [{otype} {year}] casts={len(info_sel):4d}  rows={len(prof_sel):6d}")

    if not info_chunks:
        return pd.DataFrame(), pd.DataFrame()

    info_all = pd.concat(info_chunks, ignore_index=True)
    prof_all = pd.concat(prof_chunks, ignore_index=True)

    # Reorder columns: put ids/coords first
    front_info = ["nanoose_cid", "year", "cid_orig", "time", "lon", "lat",
                  "name", "cruise", "distance_km", "matched_by_name"]
    info_cols = [c for c in front_info if c in info_all.columns] + \
                [c for c in info_all.columns if c not in front_info]
    info_all = info_all[info_cols]

    front_prof = ["nanoose_cid", "year", "cid", "time", "lon", "lat",
                  "name", "cruise", "z"]
    prof_cols = [c for c in front_prof if c in prof_all.columns] + \
                [c for c in prof_all.columns if c not in front_prof]
    prof_all = prof_all[prof_cols]

    return info_all, prof_all


def write_outputs(otype, info_df, prof_df, out_dir: Path):
    if info_df.empty:
        print(f"[{otype}] no casts found; skipping write.")
        return
    info_p = out_dir / f"nanoose_{otype}_info.p"
    info_csv = out_dir / f"nanoose_{otype}_info.csv"
    prof_p = out_dir / f"nanoose_{otype}_profiles.p"
    prof_csv = out_dir / f"nanoose_{otype}_profiles.csv"

    info_df.to_pickle(info_p)
    info_df.to_csv(info_csv, index=False)
    prof_df.to_pickle(prof_p)
    prof_df.to_csv(prof_csv, index=False)
    print(f"[{otype}] wrote {info_p.name}, {info_csv.name}, {prof_p.name}, {prof_csv.name}")


def write_summary(out_dir, results):
    """results = {otype: (info_df, prof_df)}"""
    lines = []
    lines.append("Nanoose Bay extraction (for HD)")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    lines.append("")
    lines.append("Source: LO_output/obs/dfo1 (DFO/CIOOS Pacific compilation)")
    lines.append("Reference station: Masson & Cummins (2007),")
    lines.append("  doi:10.1016/j.csr.2006.10.009")
    lines.append("")
    lines.append("Search parameters:")
    lines.append(f"  center lon, lat = {NANOOSE_LON}, {NANOOSE_LAT}")
    lines.append(f"  radius          = {SEARCH_RADIUS_KM} km (great-circle)")
    lines.append(f"  name filter     = {NAME_FILTER!r} (case-insensitive substring; OR)")
    lines.append("")
    for otype, (info, prof) in results.items():
        lines.append(f"--- {otype} ---")
        if info.empty:
            lines.append("  no casts found")
            lines.append("")
            continue
        lines.append(f"  total casts : {len(info)}")
        lines.append(f"  total rows  : {len(prof)}")
        lines.append(f"  year range  : {int(info['year'].min())}-{int(info['year'].max())}")
        lines.append(f"  matched_by_name only : {int(info['matched_by_name'].sum())}")
        lines.append(f"  lon range   : {info['lon'].min():.4f} to {info['lon'].max():.4f}")
        lines.append(f"  lat range   : {info['lat'].min():.4f} to {info['lat'].max():.4f}")
        lines.append(f"  dist (km)   : min {info['distance_km'].min():.2f}, "
                     f"max {info['distance_km'].max():.2f}, "
                     f"median {info['distance_km'].median():.2f}")
        lines.append("  casts per year:")
        per_year = info.groupby("year").size()
        for y, n in per_year.items():
            lines.append(f"    {int(y)}: {int(n)}")
        # variable coverage in profiles: non-null fraction
        lines.append("  profile variable non-null fraction:")
        skip_cols = {"nanoose_cid", "cid", "year", "time", "lon", "lat",
                     "name", "cruise"}
        for c in prof.columns:
            if c in skip_cols:
                continue
            frac = prof[c].notna().mean()
            lines.append(f"    {c:24s} {frac*100:6.2f} %")
        lines.append("")

    fn = out_dir / "nanoose_summary.txt"
    fn.write_text("\n".join(lines))
    print(f"wrote {fn.name}")


def write_map(out_dir, results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Map
    ax = axes[0]
    colors = {"ctd": "tab:blue", "bottle": "tab:orange"}
    for otype, (info, _) in results.items():
        if info.empty:
            continue
        ax.scatter(info["lon"], info["lat"], s=12, alpha=0.5,
                   c=colors[otype], label=f"{otype} (n={len(info)})")
    ax.scatter([NANOOSE_LON], [NANOOSE_LAT], marker="*", s=220, c="red",
               edgecolor="k", zorder=5, label="Nanoose station")
    # Approx search radius circle (degrees, rough)
    theta = np.linspace(0, 2 * np.pi, 200)
    dlat = SEARCH_RADIUS_KM / 111.0
    dlon = SEARCH_RADIUS_KM / (111.0 * np.cos(np.radians(NANOOSE_LAT)))
    ax.plot(NANOOSE_LON + dlon * np.cos(theta),
            NANOOSE_LAT + dlat * np.sin(theta), "r--", lw=1,
            label=f"{SEARCH_RADIUS_KM} km radius")
    # Coastline (LO PNW coast file)
    pfun.add_coast(ax, color="0.4", linewidth=0.6)
    # Set explicit limits based on the search circle + cast cluster (not the
    # coastline, which would otherwise blow the view out to all of PNW).
    lon_min = NANOOSE_LON - dlon
    lon_max = NANOOSE_LON + dlon
    lat_min = NANOOSE_LAT - dlat
    lat_max = NANOOSE_LAT + dlat
    for otype, (info, _) in results.items():
        if info.empty:
            continue
        lon_min = min(lon_min, info["lon"].min())
        lon_max = max(lon_max, info["lon"].max())
        lat_min = min(lat_min, info["lat"].min())
        lat_max = max(lat_max, info["lat"].max())
    pad = 0.05
    ax.set_xlim(lon_min - pad, lon_max + pad)
    ax.set_ylim(lat_min - pad, lat_max + pad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1.0 / np.cos(np.radians(NANOOSE_LAT)))
    ax.set_title("Nanoose Bay extracted cast locations")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Casts per year bar chart
    ax = axes[1]
    width = 0.4
    for i, (otype, (info, _)) in enumerate(results.items()):
        if info.empty:
            continue
        per_year = info.groupby("year").size()
        ax.bar(per_year.index + (i - 0.5) * width, per_year.values,
               width=width, label=otype, color=colors[otype], alpha=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of casts")
    ax.set_title("Casts per year")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fn = out_dir / "nanoose_map.png"
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print(f"wrote {fn.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    Ldir = Lfun.Lstart()
    in_root = Ldir["LOo"] / "obs"
    out_dir = Ldir["LOo"] / "extract" / "nanoose_for_HD"
    Lfun.make_dir(out_dir)

    print(f"Search center : ({NANOOSE_LON}, {NANOOSE_LAT})  radius {SEARCH_RADIUS_KM} km")
    print(f"Name filter   : {NAME_FILTER!r}")
    print(f"Input root    : {in_root}")
    print(f"Output dir    : {out_dir}")
    print()

    results = {}
    for otype in OTYPES:
        print(f"--- extracting {otype} ---")
        info_df, prof_df = extract_otype(otype, in_root)
        results[otype] = (info_df, prof_df)
        write_outputs(otype, info_df, prof_df, out_dir)
        print()

    write_summary(out_dir, results)
    write_map(out_dir, results)

    print()
    print("Done. Summary:")
    for otype, (info, prof) in results.items():
        if info.empty:
            print(f"  {otype}: 0 casts")
        else:
            print(f"  {otype}: {len(info)} casts, {len(prof)} rows, "
                  f"{int(info['year'].min())}-{int(info['year'].max())}")


if __name__ == "__main__":
    main()
