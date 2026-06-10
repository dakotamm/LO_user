#!/bin/bash
# Run on apogee. Zoomed Penn Cove BOTTOM salinity movie (+ tidal-phase SSH
# panel) for wb1_t0_xn11abbur00, Dec 3-6 2025. Color range locked to 15-25,
# matching the surface plot. Same zoom box / polygons as the surface version.
# Calls the shared engine wb1_penncove_salinity.py with --bottom.
#
#   bash wb1_penncove_salinity_bottom.sh
#
# Extra args pass through, e.g.:
#   bash wb1_penncove_salinity_bottom.sh --debug-polys
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate loenv (source conda.sh even if (base) is active; non-interactive shell).
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate loenv 2>/dev/null || echo "WARN: could not 'conda activate loenv' -- assuming the active env has the LO packages."

python "$SCRIPT_DIR/wb1_penncove_salinity.py" --bottom --smin 15 --smax 25 "$@"

OUTDIR="$PARENT/LO_output/plots/penncove_salt_bottom_wb1_t0_xn11abbur00"
echo ""
echo "Done. Output here on apogee:"
echo "  $OUTDIR/movie.mp4   (+ plot_*.png frames)"
echo ""
echo "Pull it to your Mac with (run THIS on the Mac):"
echo "  rsync -av dakotamm@apogee.ocean.washington.edu:$OUTDIR/ ~/LO_output/plots/penncove_salt_bottom_wb1_t0_xn11abbur00/"
