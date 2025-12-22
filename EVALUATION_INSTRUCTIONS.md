# Evaluation and Testing Instructions

This document provides step-by-step instructions to run the complete evaluation and generate all required results.

## Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Dataset**:
   ```bash
   # Check dataset exists
   ls -lh banner_data_train.csv
   ```

3. **Check Hardware**:
   - CPU: 4+ cores recommended
   - RAM: 8GB+ required
   - Disk: 5GB+ free space (for model cache)
   - GPU: Optional but recommended (3-4x faster)

## Step 1: Quick Model Test

Test that the model can be loaded and make predictions:

```bash
python scripts/test_model_quick.py
```

**Expected Output**:
- Model loads successfully
- Makes predictions on test cases
- Shows latency for each prediction

**First Run**: Model will be downloaded from HuggingFace (~2.3GB), which may take 5-10 minutes depending on internet speed.

## Step 2: Run Full Evaluation

Run the complete evaluation on the dataset:

```bash
# Quick test (500 samples)
python scripts/run_full_evaluation.py --sample-size 500

# Full evaluation (all data, may take 1-2 hours)
python scripts/run_full_evaluation.py
```

**This will**:
1. Load the model
2. Split dataset (80/20 train/test)
3. Evaluate on test set
4. Generate accuracy metrics
5. Calculate latency statistics (p50, p95, p99)
6. Benchmark throughput
7. Save results to `evaluation_results.json`
8. Generate human-readable report in `evaluation_report.txt`

## Step 3: Run Benchmarks

Generate detailed performance benchmarks:

```bash
python scripts/benchmark.py --data banner_data_train.csv --samples 100
```

**Output**: Detailed latency and throughput metrics for different batch sizes.

## Step 4: Test the API

Start the API server:

```bash
# Terminal 1: Start API
python main.py

# Terminal 2: Test API
python scripts/test_api.py
```

Or use Docker:

```bash
docker-compose up
```

Then test:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"banner_text": "SSH-2.0-OpenSSH_8.2p1"}'
```

## Step 5: Generate Proof of Testing

### Create Logs

The evaluation script automatically creates `evaluation.log` with detailed logs.

### Capture Screenshots

1. **Model Loading**:
   ```bash
   python scripts/test_model_quick.py > model_test_output.txt
   ```

2. **Evaluation Results**:
   ```bash
   python scripts/run_full_evaluation.py --sample-size 100 > evaluation_output.txt
   ```

3. **API Testing**:
   ```bash
   python scripts/test_api.py > api_test_output.txt
   ```

### Docker Testing

```bash
# Build and start
docker-compose up --build

# In another terminal, test
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"banner_text": "HTTP/1.1 200 OK"}'

# Capture logs
docker-compose logs > docker_logs.txt
```

## Expected Results

### Accuracy Metrics
- Overall Accuracy: ~80-90%
- Macro F1: ~0.80-0.90
- Per-class performance varies by category

### Performance Metrics
- **CPU**: ~1-2s per prediction (p50)
- **GPU**: ~0.3-0.5s per prediction (p50)
- **Throughput**: 0.5-1.0 pred/s (CPU), 2-4 pred/s (GPU)

### Model Information
- Model Size: ~600MB (4-bit quantized)
- Memory Usage: 4-8GB
- Startup Time: 30-120 seconds (first time), 10-20s (cached)

## Troubleshooting

### Model Download Issues
- Check internet connection
- Verify HuggingFace access
- Model cache location: `~/.cache/huggingface/`

### Out of Memory
- Reduce batch size
- Use smaller sample size for evaluation
- Enable quantization (already enabled by default)

### Slow Performance
- Use GPU if available
- Reduce sample size for quick testing
- Check CPU/RAM usage

## Files Generated

After running evaluation, you should have:

1. **evaluation_results.json** - Complete results in JSON format
2. **evaluation_report.txt** - Human-readable report
3. **evaluation.log** - Detailed execution logs
4. **model_test_output.txt** - Quick test results (if generated)
5. **api_test_output.txt** - API test results (if generated)
6. **docker_logs.txt** - Docker container logs (if generated)

## Next Steps

1. Review `evaluation_results.json` for detailed metrics
2. Check `evaluation_report.txt` for summary
3. Update `docs/DESIGN.md` with actual results
4. Commit results and logs to git (or add to .gitignore if too large)

## Notes

- First run will be slower due to model download
- Evaluation on full dataset may take 1-2 hours
- Use `--sample-size` for quick testing
- GPU significantly improves performance

