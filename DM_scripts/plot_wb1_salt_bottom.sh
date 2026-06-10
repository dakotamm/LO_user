#!/bin/bash
# Run on apogee. Bottom-salinity version of the zoomed Penn Cove movie for
# wb1_t0_xn11abbur00, Dec 3-6 2025, locked to the SAME 15-25 g/kg color range
# as the surface plot. Uses the same zoom box, exclude/include polygons, and
# Penn Cove SSH tidal-phase panel as plot_wb1_salt_zoom.py.
#
#   bash plot_wb1_salt_bottom.sh
#
# Extra args pass through, e.g.:  bash plot_wb1_salt_bottom.sh --debug-polys
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate loenv (source conda.sh even if (base) is active; non-interactive shell).
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate loenv 2>/dev/null || echo "WARN: could not 'conda activate loenv' -- assuming the active env has the LO packages."

python "$SCRIPT_DIR/plot_wb1_salt_zoom.py" --bottom --smin 15 --smax 25 "$@"

OUTDIR="$PARENT/LO_output/plots/saltbot_wb1_t0_xn11abbur00"
echo ""
echo "Done. Output here on apogee:"
echo "  $OUTDIR/movie.mp4   (+ plot_*.png frames)"
echo ""
echo "Pull it to your Mac with (run THIS on the Mac):"
echo "  rsync -av dakotamm@apogee.ocean.washington.edu:$OUTDIR/ ~/LO_output/plots/saltbot_wb1_t0_xn11abbur00/"
