#!/bin/bash
# Script to run full evaluation with the actual model

set -e

echo "=========================================="
echo "Running Banner Classification Evaluation"
echo "=========================================="
echo ""

# Check if dataset exists
if [ ! -f "banner_data_train.csv" ]; then
    echo "ERROR: banner_data_train.csv not found!"
    exit 1
fi

echo "Step 1: Running evaluation on dataset..."
echo "This will:"
echo "  - Load the model (may take 1-2 minutes first time)"
echo "  - Evaluate on test split"
echo "  - Generate benchmarks"
echo "  - Save results to evaluation_results.json"
echo ""

python scripts/evaluate.py \
    --data banner_data_train.csv \
    --test-split 0.2 \
    --sample-size 500 \
    --output evaluation_results.json

echo ""
echo "Step 2: Running benchmarks..."
python scripts/benchmark.py \
    --data banner_data_train.csv \
    --samples 50

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: evaluation_results.json"
echo "=========================================="

