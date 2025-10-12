#!/usr/bin/env python3
"""
Main entry point for the CardioEquation project.
This script provides a unified interface to run different components of the system.
"""

import os
import sys
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ecg_generator import main as generator_main
from ecg_model_trainer import main as trainer_main


def main():
    """Main function to run the CardioEquation system."""
    parser = argparse.ArgumentParser(description="CardioEquation: AI-Generated Personalized ECG Equation System")
    parser.add_argument(
        "mode",
        choices=["generate", "train", "demo"],
        help="Mode to run: generate (ECG signals), train (neural network), demo (interactive demonstration)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (optional)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        print("🏃 Running ECG Generator...")
        generator_main()
    elif args.mode == "train":
        print("🏋️ Running ECG Model Trainer...")
        trainer_main()
    elif args.mode == "demo":
        print("🎮 Running Interactive Demo...")
        # For now, we'll just run the generator demo
        generator_main()
    else:
        print("Invalid mode. Use 'generate', 'train', or 'demo'")
        sys.exit(1)


if __name__ == "__main__":
    main()