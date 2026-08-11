#!/bin/bash
# Run on apogee. Seasonal movie of BOTTOM hypoxia over Whidbey Basin, one frame
# per day off lowpassed.nc, with the hypoxic-area series running beside it (the
# whole `wb` polygon on top, Penn Cove as a percent of its own floor below).
# Defaults to May 1 - Nov 30 2025 of wb1_t0_xn11abbur00 = 214 daily frames.
#
#   bash 20260810_hypoxia_movie.sh
#
# Extra args pass through:
#   bash 20260810_hypoxia_movie.sh --test              # one still at the peak, fast
#   bash 20260810_hypoxia_movie.sh --stride 3          # every third day
#   bash 20260810_hypoxia_movie.sh --region wb_north   # zoom to north Whidbey
#   bash 20260810_hypoxia_movie.sh --thresh 0.5 2 5    # different bands
#   bash 20260810_hypoxia_movie.sh --cmap oxy          # continuous cmocean map
#   bash 20260810_hypoxia_movie.sh --fps 12            # faster season
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate loenv (source conda.sh even if (base) is active; non-interactive shell).
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate loenv 2>/dev/null || echo "WARN: could not 'conda activate loenv' -- assuming the active env has the LO packages."

python "$SCRIPT_DIR/20260810_hypoxia_movie.py" "$@"

OUTDIR="$PARENT/LO_output/DM_outs/20260810_hypoxia_movie"
echo ""
echo "Done. Output here on apogee:"
echo "  $OUTDIR/  (mp4 + frame0/peak stills + the series png and csv)"
echo ""
echo "Pull it to your Mac with (run THIS on the Mac):"
echo "  rsync -av dakotamm@apogee.ocean.washington.edu:$OUTDIR/ ~/LO_output/DM_outs/20260810_hypoxia_movie/"
