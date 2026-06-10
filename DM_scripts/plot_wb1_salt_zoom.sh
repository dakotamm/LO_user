#!/bin/bash
# Run on apogee. Activates loenv and makes the zoomed surface-salinity
# movie (Penn Cove / Saratoga Passage) for wb1_t0_xn11abbur00, Dec 3-6 2025.
#
#   bash plot_wb1_salt_zoom.sh
#
# Any extra args are passed through to the python script, e.g. to retune:
#   bash plot_wb1_salt_zoom.sh --smin 20 --smax 30
#   bash plot_wb1_salt_zoom.sh --lon0 -122.8 --lon1 -122.4 --lat0 48.1 --lat1 48.4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate loenv (source conda.sh even if (base) is active; non-interactive shell).
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate loenv 2>/dev/null || echo "WARN: could not 'conda activate loenv' -- assuming the active env has the LO packages."

python "$SCRIPT_DIR/plot_wb1_salt_zoom.py" "$@"

OUTDIR="$PARENT/LO_output/plots/saltzoom_wb1_t0_xn11abbur00"
echo ""
echo "Done. Output here on apogee:"
echo "  $OUTDIR/movie.mp4   (+ plot_*.png frames)"
echo ""
echo "Pull it to your Mac with (run THIS on the Mac):"
echo "  rsync -av dakotamm@apogee.ocean.washington.edu:$OUTDIR/ ~/LO_output/plots/saltzoom_wb1_t0_xn11abbur00/"
