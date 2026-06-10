#!/bin/bash
# Run on apogee. Three-panel zoomed Penn Cove movie (surface salinity, bottom
# DO, hypoxic layer depth) + a shared Penn Cove SSH tidal-phase strip, for
# wb1_t0_xn11abbur00, Sept 1-3 2025. Color ranges auto-scale per field.
# Output folder includes the date range. Calls wb1_penncove_multivar.py.
#
#   bash wb1_penncove_multivar.sh
#
# Extra args pass through, e.g. a different range or fixed DO scale:
#   bash wb1_penncove_multivar.sh --ds0 2025.08.15 --ds1 2025.08.18
#   bash wb1_penncove_multivar.sh --do-min 0 --do-max 10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate loenv (source conda.sh even if (base) is active; non-interactive shell).
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate loenv 2>/dev/null || echo "WARN: could not 'conda activate loenv' -- assuming the active env has the LO packages."

python "$SCRIPT_DIR/wb1_penncove_multivar.py" "$@"

OUTDIR="$PARENT/LO_output/plots/penncove_multivar_2025.09.01_2025.09.03_wb1_t0_xn11abbur00"
echo ""
echo "Done. Output here on apogee (default Sept 1-3 run):"
echo "  $OUTDIR/movie.mp4   (+ plot_*.png frames)"
echo ""
echo "Pull it to your Mac with (run THIS on the Mac):"
echo "  rsync -av dakotamm@apogee.ocean.washington.edu:$OUTDIR/ ~/LO_output/plots/$(basename "$OUTDIR")/"
