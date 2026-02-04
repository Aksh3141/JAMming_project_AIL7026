#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="$1"
OUT_PKL="$2"

if [ -z "$DATASET" ] || [ -z "$OUT_PKL" ]; then
    echo "Usage: bash identify.sh <path_graph_dataset> <path_discriminative_subgraphs>"
    exit 1
fi

# Create output directory if it doesn't exist
OUT_DIR="$(dirname "$OUT_PKL")"
if [ ! -d "$OUT_DIR" ]; then
    echo "Creating output directory: $OUT_DIR"
    mkdir -p "$OUT_DIR"
fi

# Gaston binary lives in the same directory as this script (placed by env.sh)
GASTON="$SCRIPT_DIR/gaston"
if [ ! -x "$GASTON" ]; then
    echo "ERROR: gaston binary not found at $GASTON"
    echo "  Run 'bash env.sh' first, or place the compiled binary there."
    exit 1
fi

echo "=== identify.sh ==="
echo "  Dataset:   $DATASET"
echo "  Output:    $OUT_PKL"
echo "  Gaston:    $GASTON"

python3 "$SCRIPT_DIR/identify_subgraphs.py" "$GASTON" "$DATASET" "$OUT_PKL"

echo "Discriminative subgraphs saved to: $OUT_PKL"