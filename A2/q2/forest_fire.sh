#!/bin/bash
# forest_fire.sh
# Usage:
#   bash forest_fire.sh <graph_path> <seed_path> <output_path> <k> <n_random_instances> <hops>

GRAPH_PATH="$1"
SEED_PATH="$2"
OUTPUT_PATH="$3"
K="$4"
N_RANDOM="$5"
HOPS="$6"

# Get directory of this script so we can find forest_fire.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python solver
python3 "$SCRIPT_DIR/forest_fire.py" \
    "$GRAPH_PATH" \
    "$SEED_PATH" \
    "$OUTPUT_PATH" \
    "$K" \
    "$N_RANDOM" \
    "$HOPS"