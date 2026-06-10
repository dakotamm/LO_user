#!/bin/bash
# Run on apogee. Zoomed Penn Cove SURFACE salinity movie (+ tidal-phase SSH
# panel) for wb1_t0_xn11abbur00, Dec 3-6 2025. Color range locked to 15-25.
# Calls the shared engine wb1_penncove_salinity.py.
#
#   bash wb1_penncove_salinity_surface.sh
#
# Extra args pass through, e.g.:
#   bash wb1_penncove_salinity_surface.sh --debug-polys
#   bash wb1_penncove_salinity_surface.sh --smin 18 --smax 28
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate loenv (source conda.sh even if (base) is active; non-interactive shell).
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate loenv 2>/dev/null || echo "WARN: could not 'conda activate loenv' -- assuming the active env has the LO packages."

python "$SCRIPT_DIR/wb1_penncove_salinity.py" --smin 15 --smax 25 "$@"

OUTDIR="$PARENT/LO_output/plots/penncove_salt_surface_wb1_t0_xn11abbur00"
echo ""
echo "Done. Output here on apogee:"
echo "  $OUTDIR/movie.mp4   (+ plot_*.png frames)"
echo ""
echo "Pull it to your Mac with (run THIS on the Mac):"
echo "  rsync -av dakotamm@apogee.ocean.washington.edu:$OUTDIR/ ~/LO_output/plots/penncove_salt_surface_wb1_t0_xn11abbur00/"
