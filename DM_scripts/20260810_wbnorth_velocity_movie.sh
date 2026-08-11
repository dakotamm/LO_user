#!/bin/bash
# Run on apogee. Tidal-cycle movie of the DEPTH-AVERAGED CURRENT over wb_north
# (speed shaded, arrows over the top) with the pc_lp velocity cross-section and
# Penn Cove SSH on the left. Same figure format as the salinity movie
# (20260806_wbnorth_tidal_movie.sh), which is untouched and can still be run.
# Defaults to the first week of Sept 2025 (169 hourly frames) of
# wb1_t0_xn11abbur00.
#
#   bash 20260810_wbnorth_velocity_movie.sh
#
# Extra args pass through:
#   bash 20260810_wbnorth_velocity_movie.sh --test               # one still, fast
#   bash 20260810_wbnorth_velocity_movie.sh --quiver-step 5      # denser arrows
#   bash 20260810_wbnorth_velocity_movie.sh --vmax 0.8           # fix the speed scale
#   bash 20260810_wbnorth_velocity_movie.sh --vscale 0.6         # fix the section scale
#   bash 20260810_wbnorth_velocity_movie.sh --sect-var salt      # salt section instead
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate loenv (source conda.sh even if (base) is active; non-interactive shell).
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate loenv 2>/dev/null || echo "WARN: could not 'conda activate loenv' -- assuming the active env has the LO packages."

python "$SCRIPT_DIR/20260810_wbnorth_velocity_movie.py" "$@"

OUTDIR="$PARENT/LO_output/DM_outs/20260810_wbnorth_velocity_movie"
echo ""
echo "Done. Output here on apogee:"
echo "  $OUTDIR/  (mp4 + a frame0.png still)"
echo ""
echo "Pull it to your Mac with (run THIS on the Mac):"
echo "  rsync -av dakotamm@apogee.ocean.washington.edu:$OUTDIR/ ~/LO_output/DM_outs/20260810_wbnorth_velocity_movie/"
