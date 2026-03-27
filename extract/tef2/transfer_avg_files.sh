#!/bin/bash
#
# Transfer ocean_avg files from this (source) machine to a remote destination
# for a date range.
#
# Run this on the machine that has the avg files.
#
# Usage:
#   bash transfer_avg_files.sh ds0 ds1 /local/path/to/gtagex user@dest:/remote/path/to/gtagex
#
# Example:
#   bash transfer_avg_files.sh 2017.07.04 2017.07.06 \
#       /dat1/dakotamm/LO_roms/cas7_trapsV00_meV00 \
#       dakotamm@apogee:/dat1/dakotamm/LO_roms/cas7_trapsV00_meV00

set -e

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 ds0 ds1 local_gtagex_dir dest_base"
    echo "  ds0, ds1:          date range in YYYY.MM.DD format"
    echo "  local_gtagex_dir:  local path to gtagex (e.g. /dat1/.../cas7_trapsV00_meV00)"
    echo "  dest_base:         user@host:/path/to/gtagex"
    exit 1
fi

DS0=$1
DS1=$2
SRC=$3
DST=$4

# Convert YYYY.MM.DD to a date we can iterate
d=$(echo "$DS0" | tr '.' '-')
end=$(echo "$DS1" | tr '.' '-')

while [[ "$d" < "$end" || "$d" == "$end" ]]; do
    # Convert back to YYYY.MM.DD for folder name
    ds=$(echo "$d" | tr '-' '.')
    fdir="f${ds}"

    echo "=== Transferring ${fdir}/ocean_avg_*.nc ==="
    # Create destination directory if needed
    # Extract user@host and path from DST
    DST_HOST="${DST%%:*}"
    DST_PATH="${DST#*:}"
    ssh "$DST_HOST" "mkdir -p ${DST_PATH}/${fdir}"
    scp "${SRC}/${fdir}"/ocean_avg_*.nc "${DST}/${fdir}/"

    # Increment date by 1 day
    d=$(date -j -v+1d -f "%Y-%m-%d" "$d" "+%Y-%m-%d" 2>/dev/null \
        || date -d "$d + 1 day" "+%Y-%m-%d")
done

echo "Done."
