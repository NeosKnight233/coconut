#!/usr/bin/env python3
"""
Single-GPU or Multi-GPU training script for Coconut with Llama models
This is a Python wrapper that can be easily modified for different setups
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train Coconut with Llama models on GSM8k")
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama3.2-1b", "llama3.2-3b", "llama3.1-8b", "all"],
        default="llama3.2-3b",
        help="Model to train (default: llama3.2-3b)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="Number of GPUs to use (default: 4)"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for distributed training (default: 29500)"
    )
    args = parser.parse_args()

    # Configuration mapping
    config_map = {
        "llama3.2-1b": "args/gsm_coconut_llama3.2_1b.yaml",
        "llama3.2-3b": "args/gsm_coconut_llama3.2_3b.yaml",
        "llama3.1-8b": "args/gsm_coconut_llama3.1_8b.yaml",
    }

    # Check data files
    if not os.path.exists("data/gsm_train.json") or not os.path.exists("data/gsm_valid.json"):
        print("Error: GSM8k data files not found!")
        print("Please ensure data/gsm_train.json and data/gsm_valid.json exist.")
        sys.exit(1)

    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)

    # Determine which models to train
    if args.model == "all":
        models_to_train = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
    else:
        models_to_train = [args.model]

    # Train each model
    for model in models_to_train:
        config_file = config_map[model]
        
        print("=" * 60)
        print(f"Training Coconut on GSM8k with {model}")
        print(f"Config: {config_file}")
        print(f"GPUs: {args.num_gpus}")
        print("=" * 60)
        print()

        # Build torchrun command
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.num_gpus}",
            f"--master_port={args.master_port}",
            "run.py",
            config_file
        ]

        # Execute training
        try:
            subprocess.run(cmd, check=True)
            print(f"\nCompleted training {model}")
            print(f"Checkpoints saved in: checkpoints/gsm-coconut-{model}/\n")
        except subprocess.CalledProcessError as e:
            print(f"Error training {model}: {e}")
            sys.exit(1)

    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
