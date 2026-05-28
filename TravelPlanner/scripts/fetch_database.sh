#!/usr/bin/env bash
# Download the upstream TravelPlanner database (327 MB, Google Drive).
# Idempotent: skips download if database/ already populated.
#
# Run from the TravelPlanner/ directory after env_setup.sh:
#   bash scripts/fetch_database.sh
set -euo pipefail

GDRIVE_FILE_ID="1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE"
DEST_DIR="$(pwd)"   # script expects to be invoked from TravelPlanner/

if [ ! -f "scripts/fetch_database.sh" ]; then
  echo "ERROR: run from TravelPlanner/ directory" >&2
  exit 1
fi

# Sentinel: one of the required jsonl files. Skip if already present.
if [ -f "database/validation_ref_info.jsonl" ] \
   && [ -d "database/accommodations" ] \
   && [ -d "database/flights" ]; then
  echo "database/ already populated — skipping download"
  exit 0
fi

# Ensure gdown is available.
if ! python -c "import gdown" 2>/dev/null; then
  echo "installing gdown into active python env"
  python -m pip install --quiet gdown
fi

mkdir -p .tmp_db
ZIP=".tmp_db/travelplanner_database.zip"

echo "downloading database from Google Drive (id=$GDRIVE_FILE_ID)"
python -m gdown --id "$GDRIVE_FILE_ID" -O "$ZIP"

echo "unzipping to $DEST_DIR"
# Upstream zip contains a top-level `database/` directory.
unzip -q -o "$ZIP" -d "$DEST_DIR"

rm -rf .tmp_db

# Sanity check.
if [ ! -f "database/validation_ref_info.jsonl" ]; then
  echo "ERROR: expected database/validation_ref_info.jsonl after unzip — layout changed?" >&2
  exit 2
fi

echo "fetch_database.sh OK — $(du -sh database/ | cut -f1) in database/"
