"""
Benchmark script for banner classifier performance.

Usage:
    python scripts/benchmark.py --data banner_data_train.csv --samples 100
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import BannerClassifier, ModelConfig
from src.data_loader import load_dataset, prepare_data_for_training

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def benchmark_single_predictions(classifier, test_texts, num_samples=100):
    """Benchmark single prediction latency."""
    print(f"\nBenchmarking single predictions ({num_samples} samples)...")
    
    latencies = []
    for i, text in enumerate(test_texts[:num_samples]):
        start = time.time()
        result = classifier.predict(text)
        latency = time.time() - start
        latencies.append(latency)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_samples}...")
    
    import numpy as np
    latencies = np.array(latencies)
    
    print(f"\nSingle Prediction Latency:")
    print(f"  Mean: {np.mean(latencies):.3f}s")
    print(f"  Median (p50): {np.median(latencies):.3f}s")
    print(f"  p95: {np.percentile(latencies, 95):.3f}s")
    print(f"  p99: {np.percentile(latencies, 99):.3f}s")
    print(f"  Min: {np.min(latencies):.3f}s")
    print(f"  Max: {np.max(latencies):.3f}s")
    
    return latencies


def benchmark_batch_predictions(classifier, test_texts, batch_sizes=[5, 10, 20, 50]):
    """Benchmark batch prediction throughput."""
    print(f"\nBenchmarking batch predictions...")
    
    results = {}
    for batch_size in batch_sizes:
        if batch_size > len(test_texts):
            continue
        
        batch = test_texts[:batch_size]
        
        # Warmup
        if len(batch) > 5:
            classifier.predict_batch(batch[:5])
        
        # Measure
        start = time.time()
        classifier.predict_batch(batch)
        elapsed = time.time() - start
        
        throughput = batch_size / elapsed
        avg_latency = elapsed / batch_size
        
        results[batch_size] = {
            "total_time": elapsed,
            "throughput": throughput,
            "avg_latency": avg_latency
        }
        
        print(f"\nBatch Size {batch_size}:")
        print(f"  Total Time: {elapsed:.3f}s")
        print(f"  Throughput: {throughput:.2f} predictions/sec")
        print(f"  Avg Latency: {avg_latency:.3f}s per prediction")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark banner classifier")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to benchmark")
    parser.add_argument("--no-quantization", action="store_true", help="Disable quantization")
    parser.add_argument("--quantization-bits", type=int, default=4, choices=[4, 8])
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = load_dataset(args.data)
    texts, labels = prepare_data_for_training(df)
    
    # Load model
    print("Loading model...")
    config = ModelConfig(
        use_quantization=not args.no_quantization,
        quantization_bits=args.quantization_bits
    )
    classifier = BannerClassifier(config)
    classifier.load_model()
    
    # Print model info
    info = classifier.get_model_info()
    print(f"\nModel Info:")
    print(f"  Model: {info['model_name']}")
    print(f"  Device: {info['device']}")
    print(f"  Quantization: {info['quantization']}")
    
    # Run benchmarks
    single_latencies = benchmark_single_predictions(classifier, texts, args.samples)
    batch_results = benchmark_batch_predictions(classifier, texts)
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

