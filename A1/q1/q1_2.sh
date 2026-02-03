#!/bin/bash

# Check if the correct number of arguments is provided 
# use universal_itemset = 300 to reproduce results
if [ "$#" -ne 2 ]; then
    echo "Usage: bash q1_2.sh <universal_itemset> <num_transactions>"
    exit 1
fi

UNIVERSAL_ITEMSET=$1
NUM_TRANSACTIONS=$2
# Output filename must be generated_transactions.dat 
OUTPUT_FILE="generated_transactions.dat"

# Execute the python script with the provided arguments
# Ensure your python script is named q1_2.py or update accordingly [cite: 73]
python3 generate_dataset.py "$UNIVERSAL_ITEMSET" "$NUM_TRANSACTIONS" "$OUTPUT_FILE"

# Check if the python script executed successfully
if [ $? -eq 0 ]; then
    echo "Dataset generation successful: $OUTPUT_FILE"
else
    echo "Error: Dataset generation failed."
    exit 1
fi