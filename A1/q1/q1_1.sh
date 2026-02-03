#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: bash q1_1.sh <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out>"
    exit 1
fi

APRIORI_EXEC="$1"
FP_EXEC="$2"
DATASET="$3"
OUTPUT_DIR="$4"

mkdir -p "$OUTPUT_DIR"

SUPPORT_THRESHOLDS=(5 10 25 50 90)

TIMEOUT=3600

# Create temporary directory for timing data
TEMP_DIR=$(mktemp -d)
APRIORI_TIMES="$TEMP_DIR/apriori_times.txt"
FP_TIMES="$TEMP_DIR/fp_times.txt"

# Clear timing files
> "$APRIORI_TIMES"
> "$FP_TIMES"

echo "Starting experiments..."

# Run experiments for each support threshold
for SUPPORT in "${SUPPORT_THRESHOLDS[@]}"; do
    echo "Running experiments with support threshold: ${SUPPORT}%"
    
    APRIORI_OUT="$OUTPUT_DIR/ap${SUPPORT}"
    FP_OUT="$OUTPUT_DIR/fp${SUPPORT}"
    
    # Run Apriori algorithm with timeout
    echo -n "  Running Apriori... "
    START_TIME=$(date +%s.%N)
    timeout $TIMEOUT "$APRIORI_EXEC" -s"${SUPPORT}" -m2 -q-1 -y "$DATASET" - 2>/dev/null > "$APRIORI_OUT"
    EXIT_CODE=$?
    END_TIME=$(date +%s.%N)
    APRIORI_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    
    if [ $EXIT_CODE -eq 124 ]; then
        echo "TIMEOUT (3600 s)"
        echo "$SUPPORT $TIMEOUT" >> "$APRIORI_TIMES"
        # Create output file with timeout message
        echo "TIMEOUT - 3600s" > "$APRIORI_OUT"
    else
        echo "${APRIORI_TIME}s"
        echo "$SUPPORT $APRIORI_TIME" >> "$APRIORI_TIMES"
    fi
    
    # Run FP-Tree algorithm with timeout
    echo -n "  Running FP-Growth... "
    START_TIME=$(date +%s.%N)
    timeout $TIMEOUT "$FP_EXEC" -s"${SUPPORT}" -m2 -q-1 "$DATASET" - 2>/dev/null > "$FP_OUT"
    EXIT_CODE=$?
    END_TIME=$(date +%s.%N)
    FP_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    
    if [ $EXIT_CODE -eq 124 ]; then
        echo "TIMEOUT (3600 s)"
        echo "$SUPPORT $TIMEOUT" >> "$FP_TIMES"
        # Create output file with timeout message
        echo "TIMEOUT - 3600s" > "$FP_OUT"
    else
        echo "${FP_TIME}s"
        echo "$SUPPORT $FP_TIME" >> "$FP_TIMES"
    fi
done

echo "Generating plot..."
python3 "$(dirname "$0")/plot_results.py" "$TEMP_DIR" "$OUTPUT_DIR"

# Clean up temporary directory
rm -rf "$TEMP_DIR"

echo "Done! Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
ls -1 "$OUTPUT_DIR"