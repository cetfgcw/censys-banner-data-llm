"""
FastAPI application for banner classification service.

Provides production-ready API endpoints for single and batch predictions,
health checks, readiness probes, and metrics.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import time
import logging
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.model import BannerClassifier, ModelConfig

logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter(
    'banner_classifier_predictions_total',
    'Total number of predictions',
    ['category']
)

prediction_latency = Histogram(
    'banner_classifier_prediction_seconds',
    'Prediction latency in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

batch_size_gauge = Gauge(
    'banner_classifier_batch_size',
    'Current batch size'
)

error_counter = Counter(
    'banner_classifier_errors_total',
    'Total number of errors',
    ['error_type']
)

model_loaded_gauge = Gauge(
    'banner_classifier_model_loaded',
    'Whether the model is loaded (1) or not (0)'
)

# Global classifier instance
classifier: Optional[BannerClassifier] = None
startup_time: Optional[float] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle - load on startup, cleanup on shutdown."""
    global classifier, startup_time
    
    logger.info("Starting up banner classification service...")
    start = time.time()
    
    try:
        # Load model configuration from environment or use defaults
        import os
        model_type = os.getenv("MODEL_TYPE", "roberta").lower()
        
        if model_type == "roberta":
            # Use RoBERTa approach (following Censys research paper)
            from src.model_roberta import RobertaBannerClassifier
            model_name = os.getenv("MODEL_NAME", "distilroberta-base")
            classifier = RobertaBannerClassifier(model_name=model_name)
            classifier.load_model()
        else:
            # Fallback to TinyLlama (few-shot)
            config = ModelConfig(
                use_quantization=os.getenv("USE_QUANTIZATION", "true").lower() == "true",
                quantization_bits=int(os.getenv("QUANTIZATION_BITS", "4")),
                use_few_shot=os.getenv("USE_FEW_SHOT", "true").lower() == "true"
            )
            classifier = BannerClassifier(config)
            classifier.load_model()
        
        startup_time = time.time() - start
        model_loaded_gauge.set(1)
        logger.info(f"Service started successfully in {startup_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        model_loaded_gauge.set(0)
        # Don't raise - allow service to start but return 503 on /ready
        classifier = None
        startup_time = None
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down banner classification service...")
    classifier = None
    model_loaded_gauge.set(0)


app = FastAPI(
    title="Banner Classification API",
    description="LLM-based classification service for internet service banners",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class BannerPredictionRequest(BaseModel):
    """Request model for single prediction."""
    banner_text: str = Field(..., min_length=1, max_length=2000, description="Banner text to classify")
    
    @field_validator('banner_text')
    @classmethod
    def validate_banner_text(cls, v):
        """Sanitize and validate banner text."""
        # Remove null bytes and other problematic characters
        v = v.replace('\x00', '').strip()
        if not v:
            raise ValueError("Banner text cannot be empty after sanitization")
        return v


class BannerPredictionResponse(BaseModel):
    """Response model for single prediction."""
    category: str
    banner_text: str
    raw_output: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    banners: List[str] = Field(..., min_items=1, max_items=100, description="List of banner texts")
    
    @field_validator('banners')
    @classmethod
    def validate_banners(cls, v):
        """Validate and sanitize banner texts."""
        sanitized = []
        for banner in v:
            banner_clean = banner.replace('\x00', '').strip()
            if banner_clean:
                sanitized.append(banner_clean)
        if not sanitized:
            raise ValueError("At least one valid banner text is required")
        return sanitized


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[BannerPredictionResponse]
    total: int
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: float


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    model_loaded: bool
    startup_time: Optional[float] = None


class MetricsResponse(BaseModel):
    """Metrics summary response."""
    total_predictions: int
    average_latency: float
    model_info: Dict[str, Any]


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Banner Classification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=BannerPredictionResponse, tags=["Predictions"])
async def predict(request: BannerPredictionRequest):
    """
    Classify a single banner.
    
    - **banner_text**: The banner text to classify (1-2000 characters)
    - Returns the predicted category
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        result = classifier.predict(request.banner_text)
        
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        prediction_counter.labels(category=result['category']).inc()
        
        return BannerPredictionResponse(
            category=result['category'],
            banner_text=request.banner_text,
            raw_output=result.get('raw_output')
        )
        
    except Exception as e:
        error_counter.labels(error_type=type(e).__name__).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Classify multiple banners in a batch.
    
    - **banners**: List of banner texts (1-100 banners per request)
    - Returns predictions for all banners
    - Optimized for batch processing
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    batch_size_gauge.set(len(request.banners))
    
    try:
        # Process batch (sequential for now, can be optimized with batching)
        results = classifier.predict_batch(request.banners)
        
        processing_time = time.time() - start_time
        
        predictions = [
            BannerPredictionResponse(
                category=r['category'],
                banner_text=banner,
                raw_output=r.get('raw_output')
            )
            for banner, r in zip(request.banners, results)
        ]
        
        # Update metrics
        for result in results:
            prediction_counter.labels(category=result['category']).inc()
        prediction_latency.observe(processing_time / len(request.banners))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            processing_time=processing_time
        )
        
    except Exception as e:
        error_counter.labels(error_type=type(e).__name__).inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """
    Liveness probe - checks if the service is alive.
    
    Returns 200 if service is running, regardless of model status.
    """
    return HealthResponse(
        status="healthy",
        timestamp=time.time()
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Monitoring"])
async def ready():
    """
    Readiness probe - checks if the model is loaded and ready to serve.
    
    Returns 200 if model is loaded, 503 if not ready.
    """
    is_ready = classifier is not None and classifier.model is not None
    
    if not is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return ReadyResponse(
        ready=True,
        model_loaded=True,
        startup_time=startup_time
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes metrics for monitoring:
    - Prediction counts by category
    - Prediction latency (p50, p95, p99)
    - Error counts
    - Model status
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/info", tags=["Monitoring"])
async def info():
    """Get information about the loaded model and service."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = classifier.get_model_info()
    
    return {
        "model_info": model_info,
        "startup_time": startup_time,
        "service_status": "running"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    error_counter.labels(error_type=type(exc).__name__).inc()
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_type": type(exc).__name__}
    )

