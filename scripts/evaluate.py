"""
Script to evaluate the banner classifier on the training dataset.

Usage:
    python scripts/evaluate.py --data banner_data_train.csv --output results.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ModelConfig
from src.evaluate import run_full_evaluation, save_evaluation_results, print_evaluation_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate banner classifier")
    parser.add_argument("--data", required=True, help="Path to training CSV file")
    parser.add_argument("--test", help="Path to separate test CSV file (optional)")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio if no test file")
    parser.add_argument("--sample-size", type=int, help="Limit evaluation to N samples (for quick testing)")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--no-quantization", action="store_true", help="Disable quantization")
    parser.add_argument("--quantization-bits", type=int, default=4, choices=[4, 8], help="Quantization bits")
    parser.add_argument("--no-few-shot", action="store_true", help="Disable few-shot prompting")
    
    args = parser.parse_args()
    
    # Create model config
    config = ModelConfig(
        use_quantization=not args.no_quantization,
        quantization_bits=args.quantization_bits,
        use_few_shot=not args.no_few_shot
    )
    
    # Run evaluation
    results = run_full_evaluation(
        config,
        args.data,
        args.test,
        args.test_split,
        args.sample_size
    )
    
    # Save and print results
    save_evaluation_results(results, args.output)
    print_evaluation_summary(results)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()

