"""
Main evaluation script for banner classification.

Supports both TinyLlama (few-shot) and RoBERTa (research paper) approaches.

Usage:
    python scripts/evaluate.py --data banner_data_train.csv --model roberta --sample-size 500
    python scripts/evaluate.py --data banner_data_train.csv --model tinylama --sample-size 500
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ModelConfig, BannerClassifier
from src.model_roberta import RobertaBannerClassifier
from src.data_loader import load_dataset, prepare_data_for_training
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
    parser = argparse.ArgumentParser(description="Evaluate banner classifier")
    parser.add_argument("--data", default="banner_data_train.csv", help="Path to training CSV")
    parser.add_argument("--model", choices=["roberta", "tinylama"], default="roberta", 
                       help="Model approach: roberta (research paper) or tinylama (few-shot)")
    parser.add_argument("--test", help="Path to separate test CSV (optional)")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--sample-size", type=int, help="Limit evaluation samples")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune RoBERTa (only for roberta model)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (if fine-tuning)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Banner Classification Evaluation")
    print(f"Model: {args.model.upper()}")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    print("Loading dataset...")
    df = load_dataset(args.data)
    texts, labels = prepare_data_for_training(df)
    print(f"Loaded {len(texts)} samples")
    print()
    
    # Create model config
    if args.model == "roberta":
        from src.evaluate import evaluate_model, benchmark_throughput, BenchmarkResults
        
        # Initialize RoBERTa classifier
        print("Initializing RoBERTa classifier...")
        classifier = RobertaBannerClassifier(model_name="distilroberta-base")
        classifier.load_model()
        
        # Fine-tune if requested
        if args.fine_tune:
            print("Fine-tuning model...")
            split_idx = int(len(texts) * (1 - args.test_split))
            train_texts = texts[:split_idx]
            train_labels = labels[:split_idx]
            
            # Use subset for faster training
            if len(train_texts) > 10000:
                train_texts = train_texts[:10000]
                train_labels = train_labels[:10000]
            
            classifier.fine_tune(train_texts, train_labels, num_epochs=args.epochs)
            print()
        
        # Split test data
        split_idx = int(len(texts) * (1 - args.test_split))
        test_texts = texts[split_idx:]
        test_labels = labels[split_idx:]
        
        if args.sample_size:
            test_texts = test_texts[:args.sample_size]
            test_labels = test_labels[:args.sample_size]
        
        # Evaluate
        print(f"Evaluating on {len(test_texts)} samples...")
        results = evaluate_model(classifier, test_texts, test_labels)
        throughput_results = benchmark_throughput(classifier, test_texts[:100])
        
        # Compile final results
        final_results = {
            "model_config": {
                "model_type": "roberta",
                "model_name": "distilroberta-base",
                "fine_tuned": args.fine_tune,
                "epochs": args.epochs if args.fine_tune else 0
            },
            "dataset_info": {
                "total_samples": len(test_texts),
                "evaluated_samples": len(results.predictions),
                "class_distribution": {cat: test_labels.count(cat) for cat in set(test_labels)}
            },
            "latency_stats": results.get_latency_stats(),
            "accuracy_metrics": results.get_accuracy_metrics(),
            "throughput_benchmarks": throughput_results,
            "errors": results.errors[:10] if results.errors else []
        }
        
    else:
        # TinyLlama approach
        config = ModelConfig(
            use_quantization=True,
            quantization_bits=4,
            use_few_shot=True
        )
        final_results = run_full_evaluation(
            config,
            args.data,
            args.test,
            args.test_split,
            args.sample_size
        )
    
    # Save results
    save_evaluation_results(final_results, args.output)
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print_evaluation_summary(final_results)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
