#!/usr/bin/env bash
# Run the full pipeline. Raw downloads go to RAW_DIR (default data/raw).
# Python runs in isolated mode (-I) so that nothing in the download folder
# or the current directory can shadow an installed module.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
RAW_DIR="${RAW_DIR:-data/raw}"
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  "$PY" -I -B scripts/00_download_data.py --raw-dir "$RAW_DIR"
fi
"$PY" -I -B scripts/01_clean_data.py --raw-dir "$RAW_DIR"
"$PY" -I -B scripts/02_trends.py
"$PY" -I -B scripts/03_ml_model.py
"$PY" -I -B scripts/04_carbonate_box_model.py
"$PY" -I -B scripts/05_loads.py
echo "All steps finished. Tables in results/tables, figures in results/figures."
