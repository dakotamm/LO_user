#!/bin/bash
# Run on apogee. Makes an hourly surface salt+temp movie for the
# wb1_t0_xn11abbur00 run, Dec 3-6 2025, then prints where it landed.
#
#   bash plot_wb1_surface_salt.sh
#
set -euo pipefail

# ---- settings -------------------------------------------------------------
GTX="wb1_t0_xn11abbur00"
DS0="2025.12.03"
DS1="2025.12.06"
PT="P_basic"        # surface salt + temp
LT="hourly0"        # start at ocean_his_0001.nc on DS0 (clean hour-0 start)
RO=2                # /dat2/dakotamm/LO_roms
# ---------------------------------------------------------------------------

# Activate the loenv conda environment (try common conda locations).
if ! command -v conda >/dev/null 2>&1; then
    for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        if [ -f "$c/etc/profile.d/conda.sh" ]; then
            source "$c/etc/profile.d/conda.sh"
            break
        fi
    done
fi
conda activate loenv 2>/dev/null || echo "WARN: could not 'conda activate loenv' -- assuming it is already active."

# The run lives on /dat2/dakotamm/LO_roms (-ro 2).
DISK="/dat2/dakotamm/LO_roms"
if [ ! -d "$DISK/$GTX" ]; then
    echo "ERROR: could not find $GTX under $DISK" >&2
    exit 1
fi
echo "Using run at $DISK/$GTX  ->  -ro $RO"

# Sanity-check the four day-folders have hourly history files.
for d in 03 04 05 06; do
    f="$DISK/$GTX/f2025.12.$d/ocean_his_0001.nc"
    if [ ! -f "$f" ]; then
        echo "WARN: missing $f (movie may be short or fail)."
    fi
done

# Make the movie.
cd ~/LO/plotting
python pan_plot.py -gtx "$GTX" -ro "$RO" \
    -0 "$DS0" -1 "$DS1" -lt "$LT" -pt "$PT" \
    -avl False -mov True -save True

OUTDIR="$HOME/LO_output/plots/${LT}_${PT}_${GTX}"
echo ""
echo "Done. Output here on apogee:"
echo "  $OUTDIR/movie.mp4   (+ plot_*.png frames)"
echo ""
echo "Pull it to your Mac with (run THIS on the Mac):"
echo "  rsync -av dakotamm@apogee.ocean.washington.edu:$OUTDIR/ ~/LO_output/plots/${LT}_${PT}_${GTX}/"
