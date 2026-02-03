#!/usr/bin/env python3
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_timing_data(filepath):
    supports = []
    times = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                supports.append(float(parts[0]))
                times.append(float(parts[1]))
    return supports, times

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 plot_results.py <temp_directory> <output_directory>")
        sys.exit(1)
    
    temp_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Read timing data from temporary directory
    apriori_file = os.path.join(temp_dir, "apriori_times.txt")
    fp_file = os.path.join(temp_dir, "fp_times.txt")
    
    apriori_supports, apriori_times = read_timing_data(apriori_file)
    fp_supports, fp_times = read_timing_data(fp_file)
    
    # Create plot with fixed style for reproducibility
    plt.figure(figsize=(10, 6))
    plt.plot(apriori_supports, apriori_times, marker='o', linewidth=2, 
             markersize=8, label='Apriori', color='#1f77b4')
    plt.plot(fp_supports, fp_times, marker='s', linewidth=2, 
             markersize=8, label='FP-Growth', color='#ff7f0e')
    
    plt.xlabel('Support Threshold (%)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Performance Comparison: Apriori vs FP-Growth', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot as plot.png in output directory
    plot_path = os.path.join(output_dir, "plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()