#!/usr/bin/env python3
"""
Data conversion script for Q2
Converts Yeast format to gSpan/Gaston and FSG formats
"""

import sys

def convert_yeast_to_gspan(input_file, output_file):
    """
    Convert Yeast format to gSpan format using the SAME numeric mapping as Gaston.

    gSpan format (numeric labels):
    t # 0
    v 0 1
    v 1 1
    e 0 1 3
    """

    # FIXED, CONSISTENT MAPPING (same as Gaston)
    label_map = {
        'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4,
        'I': 5, 'N': 6, 'O': 7, 'P': 8, 'S': 9, 'Si': 10
    }

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        graph_id = 0
        lines = fin.readlines()
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()

            # Graph header
            if line.startswith('#'):
                fout.write(f"t # {graph_id}\n")
                graph_id += 1
                i += 1

                # Number of nodes
                num_nodes = int(lines[i].strip())
                i += 1

                # Node labels
                for node_idx in range(num_nodes):
                    node_label = lines[i].strip()

                    if node_label not in label_map:
                        raise ValueError(f"Unknown label '{node_label}' not in fixed mapping.")

                    numeric_label = label_map[node_label]
                    fout.write(f"v {node_idx} {numeric_label}\n")
                    i += 1

                # Number of edges
                num_edges = int(lines[i].strip())
                i += 1

                # Edges
                for _ in range(num_edges):
                    src, dst, edge_label = lines[i].strip().split()
                    fout.write(f"e {src} {dst} {edge_label}\n")
                    i += 1

            else:
                i += 1

    print(f"Converted {graph_id} graphs to gSpan format: {output_file}")
    print("Used fixed label mapping:", label_map)

def convert_yeast_to_gaston(input_file, output_file):
    """
    Convert Yeast format to Gaston format using FIXED numeric label mapping.

    Required mapping:
        'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4,
        'I': 5, 'N': 6, 'O': 7, 'P': 8, 'S': 9, 'Si': 10
    """

    # Predefined fixed mapping
    label_map = {
        'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4,
        'I': 5, 'N': 6, 'O': 7, 'P': 8, 'S': 9, 'Si': 10
    }

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        graph_id = 0
        lines = fin.readlines()
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()

            # Graph header
            if line.startswith('#'):
                fout.write(f't # {graph_id}\n')
                graph_id += 1
                i += 1

                # Number of nodes
                num_nodes = int(lines[i].strip())
                i += 1

                # Node labels
                for node_idx in range(num_nodes):
                    node_label = lines[i].strip()

                    if node_label not in label_map:
                        raise ValueError(f"Unknown atom label '{node_label}' not in fixed mapping.")

                    numeric_label = label_map[node_label]
                    fout.write(f'v {node_idx} {numeric_label}\n')
                    i += 1

                # Number of edges
                num_edges = int(lines[i].strip())
                i += 1

                # Edges
                for _ in range(num_edges):
                    src, dst, edge_lbl = lines[i].strip().split()
                    fout.write(f'e {src} {dst} {edge_lbl}\n')
                    i += 1

            else:
                i += 1

    print(f"Converted {graph_id} graphs to Gaston format: {output_file}")
    print("Used fixed label mapping:", label_map)


def convert_yeast_to_fsg(input_file, output_file):
    """
    Convert Yeast format to FSG format
    
    FSG format uses 'u' for undirected edges instead of 'e'
    """
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        graph_id = 0
        lines = fin.readlines()
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Graph ID line (starts with #)
            if line.startswith('#'):
                fout.write(f't # {graph_id}\n')
                graph_id += 1
                i += 1
                
                # Read number of nodes
                num_nodes = int(lines[i].strip())
                i += 1
                
                # Read node labels and write vertex lines
                for node_idx in range(num_nodes):
                    node_label = lines[i].strip()
                    fout.write(f'v {node_idx} {node_label}\n')
                    i += 1
                
                # Read number of edges
                num_edges = int(lines[i].strip())
                i += 1
                
                # Read edges and write edge lines (using 'u' for FSG)
                for _ in range(num_edges):
                    edge_parts = lines[i].strip().split()
                    src, dst, edge_label = edge_parts[0], edge_parts[1], edge_parts[2]
                    fout.write(f'u {src} {dst} {edge_label}\n')
                    i += 1
            else:
                i += 1
    
    print(f"Converted {graph_id} graphs to FSG format: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python convert_data.py <input_yeast_file> <output_gspan_file> <output_gaston_file> <output_fsg_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_gspan = sys.argv[2]
    output_gaston = sys.argv[3]
    output_fsg = sys.argv[4]
    
    convert_yeast_to_gspan(input_file, output_gspan)
    convert_yeast_to_gaston(input_file, output_gaston)
    convert_yeast_to_fsg(input_file, output_fsg)