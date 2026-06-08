#!/bin/bash
# Quick memory check before running compare_pcDec2025_obs_mod.py.
# Run on apogee: bash LO_user/obsmod/check_apogee_memory.sh

echo "=== System memory ==="
free -h

echo ""
echo "=== /proc/meminfo (key fields) ==="
grep -E "^(MemTotal|MemFree|MemAvailable|Cached|Buffers):" /proc/meminfo

echo ""
echo "=== Your current processes ==="
ps -u "$USER" -o pid,rss,vsz,comm --sort=-rss | head -15

echo ""
echo "=== Estimated model cache size ==="
# Survey spans ~10 hours -> ~10 files; each ocean_his is ~250 MB
# Report actual file sizes if path is available
ROMS_DIR="${1:-/data2/dakotamm/LO_roms/wb1_t0_xn11abbur00/f2025.12.04}"
if [ -d "$ROMS_DIR" ]; then
    echo "Checking files in: $ROMS_DIR"
    ls -lh "$ROMS_DIR"/ocean_his_001{7,8,9}.nc \
           "$ROMS_DIR"/ocean_his_002{0,1,2,3,4,5}.nc 2>/dev/null | \
        awk '{print $5, $9}'
    echo ""
    TOTAL=$(du -sh "$ROMS_DIR"/ocean_his_*.nc 2>/dev/null | \
            awk '{sum+=$1} END {print sum}')
    echo "Approx total for all his files in that day: ${TOTAL} MB"
    echo "(Cache only loads ~10 hourly files covering the survey)"
else
    echo "ROMS dir not found: $ROMS_DIR"
    echo "Pass the correct path as argument: bash check_apogee_memory.sh /path/to/run/fYYYY.MM.DD"
fi

echo ""
echo "=== Rule of thumb ==="
echo "Need ~10 x (size of one ocean_his file) of free RAM for the model cache."
echo "MemAvailable above should comfortably exceed that."
