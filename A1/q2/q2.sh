#!/bin/bash

# q2.sh - Main script for Question 2
# Usage: bash q2.sh <path_gspan_executable> <path_fsg_executable> <path_gaston_executable> <path_dataset> <path_out>

# Check arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: bash q2.sh <path_gspan_executable> <path_fsg_executable> <path_gaston_executable> <path_dataset> <path_out>"
    exit 1
fi

GSPAN_EXEC=$1
FSG_EXEC=$2
GASTON_EXEC=$3
DATASET=$4
OUTPUT_DIR=$5

TIME_LIMIT="1h"

mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Question 2: Frequent Subgraph Mining Comparison"
echo "================================================"
echo ""

echo "Converting dataset formats..."
GSPAN_DATA="dataset_gspan.dat"
GASTON_DATA="dataset_gaston.dat"
FSG_DATA="dataset_fsg.dat"

python3 convert_data.py "$DATASET" "$GSPAN_DATA" "$GASTON_DATA" "$FSG_DATA"
echo ""

TOTAL_GRAPHS=$(grep -c "^t #" "$GASTON_DATA")
echo "Total graphs in dataset: $TOTAL_GRAPHS"
echo ""

SUPPORTS=(5 10 25 50 95)

declare -A gspan_times
declare -A fsg_times
declare -A gaston_times

echo "================================================"
echo "Running algorithms at different support thresholds"
echo "================================================"
echo ""

for SUP in "${SUPPORTS[@]}"; do
    echo "----------------------------------------"
    echo "Running with minimum support: ${SUP}%"
    echo "----------------------------------------"

    PERCENT_SUP=$SUP
    ABS_SUP=$(echo "scale=0; ($TOTAL_GRAPHS * $SUP) / 100" | bc)
    DEC_SUP=$(echo "scale=4; $SUP / 100" | bc)

    #############################
    # gSpan
    #############################
    echo "Running gSpan..."
    START=$(date +%s.%N)

    # gSpan prints FP only into dataset_gspan.dat.fp
    timeout "$TIME_LIMIT" "$GSPAN_EXEC" -f "$GSPAN_DATA" -s "$DEC_SUP" -o >/dev/null 2>/dev/null
    STATUS=$?
    END=$(date +%s.%N)
    
    GSPAN_FP="${GSPAN_DATA}.fp"
    TARGET_GSPAN="${OUTPUT_DIR}/gspan${SUP}"

    if [ -f "$GSPAN_FP" ]; then
        mv "$GSPAN_FP" "$TARGET_GSPAN"
        echo "Saved gSpan output → $TARGET_GSPAN"
    fi

    if [ $STATUS -eq 124 ]; then
        echo "WARNING: gSpan timed out"
        gspan_times[$SUP]="3600"
    elif [ $STATUS -ne 0 ]; then
        echo "WARNING: gSpan failed"
        gspan_times[$SUP]="N/A"
    else
        gspan_times[$SUP]=$(echo "$END - $START" | bc)
        echo "gSpan completed in ${gspan_times[$SUP]} seconds"
    fi
    echo ""

    #############################
    # FSG
    #############################
    echo "Running FSG..."
    START=$(date +%s.%N)

    timeout "$TIME_LIMIT" "$FSG_EXEC" -s "$SUP" "$FSG_DATA" >/dev/null 2>/dev/null
    STATUS=$?
    END=$(date +%s.%N)

    FSG_FP="${FSG_DATA}.fp"
    TARGET_FSG="${OUTPUT_DIR}/fsg${SUP}"

    if [ -f "${FSG_DATA}.fp" ]; then
        mv "${FSG_DATA}.fp" "${OUTPUT_DIR}/fsg${SUP}"
    elif [ -f "${FSG_DATA%.*}.fp" ]; then
        mv "${FSG_DATA%.*}.fp" "${OUTPUT_DIR}/fsg${SUP}"
    fi

    if [ $STATUS -eq 124 ]; then
        echo "WARNING: FSG timed out"
        fsg_times[$SUP]="3600"
    elif [ $STATUS -ne 0 ]; then
        echo "WARNING: FSG failed"
        fsg_times[$SUP]="N/A"
    else
        fsg_times[$SUP]=$(echo "$END - $START" | bc)
        echo "FSG completed in ${fsg_times[$SUP]} seconds"
    fi
    echo ""

    #############################
    # Gaston
    #############################
    echo "Running Gaston..."
    GASTON_OUT="${OUTPUT_DIR}/gaston${SUP}"
    START=$(date +%s.%N)

    timeout "$TIME_LIMIT" "$GASTON_EXEC" "$ABS_SUP" "$GASTON_DATA" "$GASTON_OUT" >/dev/null 2>/dev/null
    STATUS=$?
    END=$(date +%s.%N)

    if [ $STATUS -eq 124 ]; then
        echo "WARNING: Gaston timed out"
        gaston_times[$SUP]="3600"
    elif [ $STATUS -ne 0 ]; then
        echo "WARNING: Gaston failed"
        gaston_times[$SUP]="N/A"
    else
        gaston_times[$SUP]=$(echo "$END - $START" | bc)
        echo "Gaston completed in ${gaston_times[$SUP]} seconds"
    fi
    echo ""

done

echo "================================================"
echo "Summary of Runtimes"
echo "================================================"
printf "%-10s %-15s %-15s %-15s\n" "Support(%)" "gSpan(s)" "FSG(s)" "Gaston(s)"
printf "%-10s %-15s %-15s %-15s\n" "----------" "----------" "----------" "----------"

for SUP in "${SUPPORTS[@]}"; do
    printf "%-10s %-15s %-15s %-15s\n" "$SUP" "${gspan_times[$SUP]}" "${fsg_times[$SUP]}" "${gaston_times[$SUP]}"
done
echo ""

###############################################
# Plotting
###############################################

echo "Generating plot..."

cat > plot_script_temp.py << 'EOF'
import matplotlib.pyplot as plt
import sys

supports = [5, 10, 25, 50, 95]

def to_float(x):
    try:
        return float(x)
    except:
        return None

output_path = sys.argv[1]

gspan = [to_float(sys.argv[i]) for i in range(2, 7)]
fsg   = [to_float(sys.argv[i]) for i in range(7, 12)]
gaston= [to_float(sys.argv[i]) for i in range(12, 17)]

plt.figure(figsize=(10,6))

def plot_with_gap(supp, times, label, marker):
    xs = []
    ys = []
    for s,t in zip(supp, times):
        if t is not None:
            xs.append(s)
            ys.append(t)
    if xs:
        plt.plot(xs, ys, marker=marker, label=label, linewidth=2, markersize=8)

plot_with_gap(supports, gspan, 'gSpan', 'o')
plot_with_gap(supports, fsg, 'FSG', 's')
plot_with_gap(supports, gaston, 'Gaston', '^')

plt.xlabel('Minimum Support (%)')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison: gSpan vs FSG vs Gaston')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(output_path, dpi=300, bbox_inches='tight')
EOF

python3 plot_script_temp.py "${OUTPUT_DIR}/plot.png" \
    "${gspan_times[5]}" "${gspan_times[10]}" "${gspan_times[25]}" "${gspan_times[50]}" "${gspan_times[95]}" \
    "${fsg_times[5]}"   "${fsg_times[10]}"   "${fsg_times[25]}"   "${fsg_times[50]}"   "${fsg_times[95]}" \
    "${gaston_times[5]}" "${gaston_times[10]}" "${gaston_times[25]}" "${gaston_times[50]}" "${gaston_times[95]}"

rm -f plot_script_temp.py

echo ""
echo "================================================"
echo "All tasks completed!"
echo "================================================"
echo "Final outputs:"
echo "  gSpan  → ${OUTPUT_DIR}/gspan95"
echo "  FSG    → ${OUTPUT_DIR}/fsg95"
echo "  Gaston → ${OUTPUT_DIR}/gaston95"
echo "Plot    → ${OUTPUT_DIR}/plot.png"
echo ""
