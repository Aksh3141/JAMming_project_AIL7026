#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DB_FEAT="$1"
Q_FEAT="$2"
OUT="$3"

if [ -z "$DB_FEAT" ] || [ -z "$Q_FEAT" ] || [ -z "$OUT" ]; then
    echo "Usage: bash generate_candidates.sh <db_features.npy> <q_features.npy> <out_candidates.dat>"
    exit 1
fi

echo "=== generate_candidates.sh ==="
echo "  DB features:  $DB_FEAT"
echo "  Q  features:  $Q_FEAT"
echo "  Output:       $OUT"

python3 "$SCRIPT_DIR/generate_candidates.py" "$DB_FEAT" "$Q_FEAT" "$OUT"
