"""
Evaluation and benchmarking utilities for the banner classifier.

Provides comprehensive evaluation metrics, performance benchmarks,
and error analysis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from typing import List, Dict, Tuple, Optional
import time
import logging
from pathlib import Path
import json

from src.model import BannerClassifier, ModelConfig
from src.data_loader import load_dataset, prepare_data_for_training

logger = logging.getLogger(__name__)


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.predictions: List[str] = []
        self.true_labels: List[str] = []
        self.errors: List[str] = []
        self.memory_usage: Optional[float] = None
        
    def add_prediction(self, prediction: str, true_label: str, latency: float):
        """Add a prediction result."""
        self.predictions.append(prediction)
        self.true_labels.append(true_label)
        self.latencies.append(latency)
    
    def add_error(self, error: str):
        """Add an error."""
        self.errors.append(error)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not self.latencies:
            return {}
        
        latencies = np.array(self.latencies)
        return {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "std": float(np.std(latencies))
        }
    
    def get_accuracy_metrics(self) -> Dict[str, any]:
        """Calculate accuracy metrics."""
        if not self.predictions or not self.true_labels:
            return {}
        
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predictions, average=None, zero_division=0
        )
        
        # Per-class metrics
        classes = sorted(set(self.true_labels + self.predictions))
        per_class = {
            cls: {
                "precision": float(precision[i]) if i < len(precision) else 0.0,
                "recall": float(recall[i]) if i < len(recall) else 0.0,
                "f1": float(f1[i]) if i < len(f1) else 0.0,
                "support": int(support[i]) if i < len(support) else 0
            }
            for i, cls in enumerate(classes)
        }
        
        # Macro averages
        macro_precision = np.mean([m["precision"] for m in per_class.values()])
        macro_recall = np.mean([m["recall"] for m in per_class.values()])
        macro_f1 = np.mean([m["f1"] for m in per_class.values()])
        
        return {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(self.true_labels, self.predictions, labels=classes).tolist(),
            "classes": classes
        }


def evaluate_model(
    classifier: BannerClassifier,
    test_texts: List[str],
    test_labels: List[str],
    sample_size: Optional[int] = None
) -> BenchmarkResults:
    """
    Evaluate the classifier on test data.
    
    Args:
        classifier: Loaded classifier instance
        test_texts: List of test banner texts
        test_labels: List of true labels
        sample_size: Optional limit on number of samples to evaluate
        
    Returns:
        BenchmarkResults with all metrics
    """
    results = BenchmarkResults()
    
    if sample_size:
        test_texts = test_texts[:sample_size]
        test_labels = test_labels[:sample_size]
    
    logger.info(f"Evaluating on {len(test_texts)} samples...")
    
    for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
        try:
            start_time = time.time()
            prediction = classifier.predict(text)
            latency = time.time() - start_time
            
            results.add_prediction(
                prediction['category'],
                true_label,
                latency
            )
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(test_texts)} samples")
                
        except Exception as e:
            logger.error(f"Error predicting sample {i}: {e}")
            results.add_error(str(e))
            results.add_prediction("other", true_label, 0.0)
    
    return results


def benchmark_throughput(
    classifier: BannerClassifier,
    test_texts: List[str],
    batch_sizes: List[int] = [1, 5, 10, 20, 50]
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark throughput for different batch sizes.
    
    Args:
        classifier: Loaded classifier instance
        test_texts: Test banner texts
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary mapping batch size to throughput metrics
    """
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(test_texts):
            continue
        
        logger.info(f"Benchmarking batch size {batch_size}...")
        batch_texts = test_texts[:batch_size]
        
        # Warmup
        classifier.predict_batch(batch_texts[:min(5, len(batch_texts))])
        
        # Measure
        start_time = time.time()
        classifier.predict_batch(batch_texts)
        elapsed = time.time() - start_time
        
        throughput = batch_size / elapsed
        
        results[batch_size] = {
            "total_time": elapsed,
            "throughput": throughput,  # predictions per second
            "avg_latency": elapsed / batch_size
        }
    
    return results


def run_full_evaluation(
    model_config: ModelConfig,
    train_csv: str,
    test_csv: Optional[str] = None,
    test_split: float = 0.2,
    sample_size: Optional[int] = None
) -> Dict[str, any]:
    """
    Run a full evaluation pipeline.
    
    Args:
        model_config: Model configuration
        train_csv: Path to training CSV
        test_csv: Optional path to separate test CSV
        test_split: Fraction to use for testing if no test_csv
        sample_size: Optional limit on evaluation samples
        
    Returns:
        Complete evaluation results dictionary
    """
    logger.info("Starting full evaluation...")
    
    # Load data
    df = load_dataset(train_csv)
    texts, labels = prepare_data_for_training(df)
    
    # Split data if needed
    if test_csv:
        test_df = load_dataset(test_csv)
        test_texts, test_labels = prepare_data_for_training(test_df)
    else:
        split_idx = int(len(texts) * (1 - test_split))
        test_texts = texts[split_idx:]
        test_labels = labels[split_idx:]
    
    # Load and evaluate model
    classifier = BannerClassifier(model_config)
    classifier.load_model()
    
    # Evaluate
    eval_results = evaluate_model(classifier, test_texts, test_labels, sample_size)
    
    # Benchmark throughput
    throughput_results = benchmark_throughput(classifier, test_texts[:100])
    
    # Compile results
    results = {
        "model_config": {
            "model_name": model_config.model_name,
            "use_quantization": model_config.use_quantization,
            "quantization_bits": model_config.quantization_bits,
            "use_few_shot": model_config.use_few_shot
        },
        "dataset_info": {
            "total_samples": len(test_texts),
            "evaluated_samples": len(eval_results.predictions),
            "class_distribution": pd.Series(test_labels).value_counts().to_dict()
        },
        "latency_stats": eval_results.get_latency_stats(),
        "accuracy_metrics": eval_results.get_accuracy_metrics(),
        "throughput_benchmarks": throughput_results,
        "errors": eval_results.errors[:10] if eval_results.errors else []  # First 10 errors
    }
    
    return results


def save_evaluation_results(results: Dict[str, any], output_path: str):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to {output_path}")


def print_evaluation_summary(results: Dict[str, any]):
    """Print a human-readable evaluation summary."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nModel: {results['model_config']['model_name']}")
    print(f"Quantization: {results['model_config']['use_quantization']} ({results['model_config']['quantization_bits']}-bit)")
    print(f"Few-shot: {results['model_config']['use_few_shot']}")
    
    print(f"\nDataset: {results['dataset_info']['evaluated_samples']} samples")
    print("\nClass Distribution:")
    for cls, count in results['dataset_info']['class_distribution'].items():
        print(f"  {cls}: {count}")
    
    print("\nAccuracy Metrics:")
    acc_metrics = results['accuracy_metrics']
    print(f"  Overall Accuracy: {acc_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {acc_metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {acc_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {acc_metrics['macro_recall']:.4f}")
    
    print("\nPer-Class Performance:")
    for cls, metrics in acc_metrics['per_class'].items():
        print(f"  {cls}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Support: {metrics['support']}")
    
    print("\nLatency Statistics (seconds):")
    latency = results['latency_stats']
    print(f"  Mean: {latency['mean']:.4f}")
    print(f"  Median (p50): {latency['median']:.4f}")
    print(f"  p95: {latency['p95']:.4f}")
    print(f"  p99: {latency['p99']:.4f}")
    print(f"  Min: {latency['min']:.4f}")
    print(f"  Max: {latency['max']:.4f}")
    
    print("\nThroughput Benchmarks:")
    for batch_size, metrics in results['throughput_benchmarks'].items():
        print(f"  Batch size {batch_size}: {metrics['throughput']:.2f} predictions/sec")
    
    if results['errors']:
        print(f"\nErrors encountered: {len(results['errors'])}")
        print("Sample errors:")
        for error in results['errors'][:5]:
            print(f"  {error}")
    
    print("\n" + "="*80)

