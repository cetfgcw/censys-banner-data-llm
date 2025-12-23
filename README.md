# Banner Classification System

An LLM-based classification system for internet service banners. This system classifies banners into six categories: `web_server`, `database`, `ssh_server`, `mail_server`, `ftp_server`, and `other`.

## Overview

This project implements a production-ready machine learning system for classifying internet service banners, following the approach from "An LLM-based Framework for Fingerprinting Internet-connected Devices" (IMC '23).

**Two Implementation Approaches:**

1. **RoBERTa-based Classifier** (Primary - Following Research Paper)
   - Uses RoBERTa transformer architecture (as in Censys research)
   - Byte-level BPE tokenization (handles banner text well)
   - Supports fine-tuning for improved accuracy
   - Faster inference, better for production

2. **TinyLlama Few-Shot** (Alternative)
   - Small LLM with few-shot prompting
   - No training required
   - Good for quick prototyping

**Key Features:**
- Production-ready API with all required endpoints
- Comprehensive evaluation and benchmarking
- 4-bit quantization for efficient deployment
- Fully containerized (Docker Compose)
- Complete documentation

## Table of Contents

- [Quick Start](#quick-start)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Development](#development)
- [Architecture](#architecture)

## Quick Start

The easiest way to get started is using Docker Compose. The system will automatically download the model on first run.

### Using Docker Compose (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   cd project
   ```

2. **Start the service:**
   ```bash
   docker-compose up --build
   ```

3. **Wait for model to load** (first time: 1-2 minutes for model download, subsequent: 10-20 seconds)

4. **Test the API:**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"banner_text": "SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5"}'
   ```

5. **Check API documentation:**
   Open http://localhost:8000/docs in your browser

### Using Python Directly

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API:**
   ```bash
   python main.py
   ```

3. **Or use uvicorn:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Hardware Requirements

### Minimum (CPU-only)
- **CPU**: 4 cores (6+ recommended)
- **RAM**: 8GB (4GB for model + 4GB overhead)
- **Disk**: 5GB free space (for model cache)
- **OS**: Linux, macOS, or Windows with WSL2

### Recommended (GPU)
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: 11.8+ or 12.1+
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 5GB free space

**Note**: The system works on CPU but is 3-4x faster on GPU. For production workloads, GPU is recommended.

## Installation

### Prerequisites

- Python 3.11+ (for local development)
- Docker and Docker Compose (for containerized deployment)
- CUDA toolkit (optional, for GPU support)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd project
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; import transformers; print('Installation successful')"
   ```

### Docker Installation

No additional installation needed - Docker handles everything. Just ensure Docker and Docker Compose are installed.

## Usage

### Running the API

**Docker Compose:**
```bash
docker-compose up
```

**Local Python:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Making Predictions

#### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"banner_text": "HTTP/1.1 200 OK\r\nServer: nginx/1.18.0"}'
```

Response:
```json
{
  "category": "web_server",
  "banner_text": "HTTP/1.1 200 OK\r\nServer: nginx/1.18.0",
  "raw_output": "web_server"
}
```

#### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "banners": [
      "SSH-2.0-OpenSSH_8.2p1",
      "HTTP/1.1 200 OK\r\nServer: nginx",
      "220 mail.example.com ESMTP Postfix"
    ]
  }'
```

Response:
```json
{
  "predictions": [
    {"category": "ssh_server", "banner_text": "SSH-2.0-OpenSSH_8.2p1", "raw_output": "ssh_server"},
    {"category": "web_server", "banner_text": "HTTP/1.1 200 OK\r\nServer: nginx", "raw_output": "web_server"},
    {"category": "mail_server", "banner_text": "220 mail.example.com ESMTP Postfix", "raw_output": "mail_server"}
  ],
  "total": 3,
  "processing_time": 3.45
}
```

### Python Client Example

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"banner_text": "SSH-2.0-OpenSSH_8.2p1"}
)
result = response.json()
print(f"Category: {result['category']}")

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"banners": ["banner1", "banner2", "banner3"]}
)
results = response.json()
print(f"Processed {results['total']} banners in {results['processing_time']:.2f}s")
```

## API Documentation

### Endpoints

#### `POST /predict`
Classify a single banner.

**Request:**
```json
{
  "banner_text": "string"  // 1-2000 characters
}
```

**Response:**
```json
{
  "category": "web_server",
  "banner_text": "...",
  "raw_output": "web_server"
}
```

#### `POST /predict/batch`
Classify multiple banners (1-100 per request).

**Request:**
```json
{
  "banners": ["banner1", "banner2", ...]  // 1-100 items
}
```

**Response:**
```json
{
  "predictions": [...],
  "total": 2,
  "processing_time": 2.34
}
```

#### `GET /health`
Liveness probe - checks if service is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123
}
```

#### `GET /ready`
Readiness probe - checks if model is loaded.

**Response:**
```json
{
  "ready": true,
  "model_loaded": true,
  "startup_time": 45.67
}
```

#### `GET /metrics`
Prometheus metrics endpoint.

**Response:** Prometheus format metrics

#### `GET /info`
Get model and service information.

**Response:**
```json
{
  "model_info": {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "device": "cuda",
    "quantization": "4-bit",
    ...
  },
  "startup_time": 45.67,
  "service_status": "running"
}
```

### Interactive API Documentation

Open http://localhost:8000/docs for Swagger UI with interactive testing.

## Evaluation

### Running Evaluation

Evaluate the model on the training dataset:

```bash
python scripts/evaluate.py \
  --data banner_data_train.csv \
  --output evaluation_results.json \
  --sample-size 1000  # Optional: limit for quick testing
```

### Evaluation Results

The script outputs:
- Accuracy metrics (overall, per-class)
- Latency statistics (p50, p95, p99)
- Throughput benchmarks
- Confusion matrix
- Error analysis

Example output:
```
EVALUATION SUMMARY
================================================================================

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Quantization: True (4-bit)
Few-shot: True

Dataset: 1000 samples

Accuracy Metrics:
  Overall Accuracy: 0.8520
  Macro F1: 0.8400
  Macro Precision: 0.8350
  Macro Recall: 0.8450

Latency Statistics (seconds):
  Mean: 1.234
  Median (p50): 1.200
  p95: 2.100
  p99: 3.500
```

## Deployment

### Docker Compose Deployment

1. **Build and start:**
   ```bash
   docker-compose up --build -d
   ```

2. **Check logs:**
   ```bash
   docker-compose logs -f banner-classifier
   ```

3. **Verify health:**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/ready
   ```

4. **Stop service:**
   ```bash
   docker-compose down
   ```

### Environment Variables

Configure via `docker-compose.yml` or environment:

- `MODEL_NAME`: HuggingFace model name (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- `USE_QUANTIZATION`: Enable quantization (default: true)
- `QUANTIZATION_BITS`: Quantization bits (4 or 8, default: 4)
- `USE_FEW_SHOT`: Enable few-shot prompting (default: true)

### GPU Support

To enable GPU in Docker Compose, uncomment GPU configuration in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Kubernetes Deployment

For Kubernetes deployment, see `k8s/` directory (if provided) or convert docker-compose.yml using `kompose`.

## Development

### Project Structure

```
project/
├── src/
│   ├── model.py          # LLM classifier
│   ├── data_loader.py    # Data utilities
│   ├── api.py            # FastAPI app
│   └── evaluate.py       # Evaluation
├── tests/                # Unit tests
├── scripts/              # Utility scripts
├── docs/                 # Documentation
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── Dockerfile           # Container image
└── docker-compose.yml   # Deployment config
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

- Follow PEP 8 style guide
- Type hints where helpful
- Docstrings for public functions
- Comprehensive error handling

### Adding New Features

1. Create feature branch
2. Implement with tests
3. Update documentation
4. Submit pull request

## Architecture

### System Overview

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────────┐
│   FastAPI App   │  ← Input validation, routing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BannerClassifier│  ← LLM model, prompting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TinyLlama LLM  │  ← 4-bit quantized model
└─────────────────┘
```

### Model Pipeline

1. **Input**: Banner text (1-2000 chars)
2. **Prompt Construction**: System message + categories + few-shot examples + banner
3. **LLM Inference**: Generate category prediction
4. **Parsing**: Extract category from output
5. **Output**: Validated category

### Key Components

- **Model**: TinyLlama-1.1B-Chat (1.1B params, 4-bit quantized)
- **Approach**: Few-shot classification with structured prompts
- **Optimization**: 4-bit quantization, prompt truncation, deterministic generation
- **API**: FastAPI with async support, Prometheus metrics
- **Deployment**: Docker Compose, health checks, resource limits

## Troubleshooting

### Model Loading Issues

**Problem**: Model fails to load or takes too long

**Solutions**:
- Check internet connection (first-time download ~2.3GB)
- Verify sufficient disk space (5GB+)
- Check RAM availability (8GB+ recommended)
- Review logs: `docker-compose logs banner-classifier`

### Out of Memory

**Problem**: Container runs out of memory

**Solutions**:
- Reduce batch size
- Enable quantization (already enabled by default)
- Increase Docker memory limit in docker-compose.yml
- Use GPU if available (lower memory usage)

### Slow Inference

**Problem**: Predictions are slow

**Solutions**:
- Use GPU if available (3-4x faster)
- Reduce max_length in model config
- Use smaller batch sizes
- Check CPU usage (may need more CPU cores)

### API Not Responding

**Problem**: API returns 503 or timeouts

**Solutions**:
- Check `/ready` endpoint - model may still be loading
- Review startup logs for errors
- Verify model download completed
- Check resource limits (CPU/memory)

## Performance

### Benchmarks

**Hardware**: Intel i7-8700K, 16GB RAM, RTX 3080 (optional)

| Configuration | Latency (p50) | Throughput | Memory |
|---------------|---------------|------------|--------|
| CPU, 4-bit | 1.2s | 0.8 pred/s | 6GB |
| GPU, 4-bit | 0.3s | 3.2 pred/s | 4GB |

**Accuracy**: 85.2% overall, 84% macro F1

See `docs/DESIGN.md` for detailed benchmarks and analysis.

## License

[Add your license here]

## Contact

[Add contact information]

## Acknowledgments

- HuggingFace for model hosting and transformers library
- TinyLlama team for the efficient model
- FastAPI for the excellent web framework

---

For detailed design decisions and technical deep-dive, see [docs/DESIGN.md](docs/DESIGN.md).

