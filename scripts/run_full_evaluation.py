"""
Complete evaluation script that runs the full evaluation pipeline.

This script:
1. Loads the model
2. Evaluates on the dataset
3. Generates comprehensive benchmarks
4. Creates evaluation report
5. Saves all results

Usage:
    python scripts/run_full_evaluation.py [--sample-size N]
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ModelConfig
from src.evaluate import run_full_evaluation, save_evaluation_results, print_evaluation_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)

def main():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument("--data", default="banner_data_train.csv", help="Path to training CSV")
    parser.add_argument("--sample-size", type=int, help="Limit evaluation samples (for quick testing)")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--report", default="evaluation_report.txt", help="Human-readable report")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Banner Classification System - Full Evaluation")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create model config
    config = ModelConfig(
        use_quantization=True,
        quantization_bits=4,
        use_few_shot=True
    )
    
    print("Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Quantization: {config.use_quantization} ({config.quantization_bits}-bit)")
    print(f"  Few-shot: {config.use_few_shot}")
    print()
    
    try:
        # Run evaluation
        print("Starting evaluation...")
        print("  This will:")
        print("    1. Load the model (may take 1-2 minutes first time)")
        print("    2. Load and split the dataset")
        print("    3. Evaluate on test set")
        print("    4. Generate benchmarks")
        print()
        
        results = run_full_evaluation(
            config,
            args.data,
            None,  # No separate test CSV
            args.test_split,
            args.sample_size
        )
        
        # Save results
        save_evaluation_results(results, args.output)
        print(f"\nResults saved to: {args.output}")
        
        # Print summary
        print_evaluation_summary(results)
        
        # Save human-readable report
        with open(args.report, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Banner Classification System - Evaluation Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary
            print_evaluation_summary(results, file=f)
            
        print(f"\nHuman-readable report saved to: {args.report}")
        
        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n[ERROR] Evaluation failed: {e}")
        print("Check evaluation.log for details")
        sys.exit(1)

if __name__ == "__main__":
    main()

