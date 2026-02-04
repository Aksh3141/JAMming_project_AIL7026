#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GRAPHS="$1"
SUBS="$2"
FEATS="$3"

if [ -z "$GRAPHS" ] || [ -z "$SUBS" ] || [ -z "$FEATS" ]; then
    echo "Usage: bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>"
    exit 1
fi

echo "=== convert.sh ==="
echo "  Graphs:      $GRAPHS"
echo "  Subgraphs:   $SUBS"
echo "  Output .npy: $FEATS"

python3 "$SCRIPT_DIR/convert_features.py" "$GRAPHS" "$SUBS" "$FEATS"
