#!/bin/bash
# Run on apogee. Tidal-cycle movie of surface salinity over wb_north, with
# surface+bottom salinity at two points across the Penn Cove mouth (pc_lp) and
# Penn Cove box-mean SSH running underneath it. Defaults to Sept 1-2 2025
# (49 hourly frames, ~2 diurnal / 4 semidiurnal cycles) of wb1_t0_xn11abbur00.
#
#   bash 20260806_wbnorth_tidal_movie.sh
#
# Extra args pass through:
#   bash 20260806_wbnorth_tidal_movie.sh --test              # one still, fast
#   bash 20260806_wbnorth_tidal_movie.sh --ds0 2025.07.15 --ds1 2025.07.16
#   bash 20260806_wbnorth_tidal_movie.sh --var temp
#   bash 20260806_wbnorth_tidal_movie.sh --region skagit_delta --sect skagit_sp
#   bash 20260806_wbnorth_tidal_movie.sh --fracs 0.15,0.85   # closer to the shores
#   bash 20260806_wbnorth_tidal_movie.sh --vmin 24 --vmax 30 # force the color range
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate loenv (source conda.sh even if (base) is active; non-interactive shell).
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate loenv 2>/dev/null || echo "WARN: could not 'conda activate loenv' -- assuming the active env has the LO packages."

python "$SCRIPT_DIR/20260806_wbnorth_tidal_movie.py" "$@"

OUTDIR="$PARENT/LO_output/DM_outs/20260806_wbnorth_tidal_movie"
echo ""
echo "Done. Output here on apogee:"
echo "  $OUTDIR/  (mp4 + a frame0.png still)"
echo ""
echo "Pull it to your Mac with (run THIS on the Mac):"
echo "  rsync -av dakotamm@apogee.ocean.washington.edu:$OUTDIR/ ~/LO_output/DM_outs/20260806_wbnorth_tidal_movie/"
