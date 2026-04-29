"""
train.py  –  COL761 Assignment 3 unified training entry point

Usage
-----
    python train.py --dataset A|B|C --task node|link \
        --data_dir /absolute/path/to/datasets \
        --model_dir /path/to/save/models \
        --kerberos YOUR_KERBEROS

This script dispatches to the appropriate dataset-specific training script:
  - train_A.py for Dataset A (node classification, 7 classes, Accuracy)
  - train_B.py for Dataset B (binary node classification, AUC-ROC)
  - train_C.py for Dataset C (link prediction, Hits@50)
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="COL761 A3 – Train GNN models for node/link prediction"
    )
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"],
                        help="Which dataset to train on")
    parser.add_argument("--task", required=True, choices=["node", "link"],
                        help="Task type: node classification or link prediction")
    parser.add_argument("--data_dir", required=True,
                        help="Absolute path to datasets directory")
    parser.add_argument("--model_dir", required=True,
                        help="Directory where trained model will be saved")
    parser.add_argument("--kerberos", required=True,
                        help="Your Kerberos ID")
    args = parser.parse_args()

    # Validate task ↔ dataset
    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(
            f"--task {args.task} is not valid for --dataset {args.dataset}. "
            f"Expected dataset in {valid[args.task]}."
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, f"train_{args.dataset}.py")

    if not os.path.isfile(train_script):
        raise FileNotFoundError(f"Training script not found: {train_script}")

    cmd = [
        sys.executable, train_script,
        "--data_dir", os.path.abspath(args.data_dir),
        "--kerberos", args.kerberos,
        "--output_dir", os.path.abspath(args.model_dir),
    ]

    print(f"Dispatching to {train_script} ...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=script_dir)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
