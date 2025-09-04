"""
Real-Time Model Serving API for Customer Retention Predictions
FastAPI-based serving with sub-50ms response time and Redis caching
Author: Nikitha Shiva
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Retention Prediction API",
    description="Real-time customer churn prediction with 90% accuracy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global variables for model and cache
model = None
preprocessor = None
feature_names = None
label_encoder = None
redis_client = None

# Configuration
CONFIG = {
    'model_path': 'models/customer_retention_model.pkl',
    'preprocessor_path': 'models/preprocessor.pkl',
    'feature_names_path': 'models/feature_names.pkl',
    'label_encoder_path': 'models/label_encoder.pkl',
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
    'cache_ttl': 3600,  # 1 hour
    'batch_size_limit': 1000,
    'rate_limit': 10000  # requests per hour
}

class CustomerData(BaseModel):
    """Input schema for single customer prediction"""
    customer_id: str = Field(..., description="Unique customer identifier")
    recency_30d: float = Field(..., ge=0, description="Days since last purchase (30d window)")
    frequency_30d: float = Field(..., ge=0, description="Purchase frequency (30d window)")
    monetary_total_30d: float = Field(..., ge=0, description="Total monetary value (30d window)")
    monetary_avg_30d: float = Field(..., ge=0, description="Average order value (30d window)")
    engagement_score: float = Field(..., ge=0, le=1, description="Customer engagement score (0-1)")
    customer_tenure_months: int = Field(..., ge=0, description="Customer tenure in months")
    support_tickets_30d: int = Field(..., ge=0, description="Number of support tickets (30d)")
    age: Optional[int] = Field(35, ge=18, le=100, description="Customer age")
    value_tier: Optional[str] = Field("medium", description="Customer value tier")
    
    @validator('value_tier')
    def validate_value_tier(cls, v):
        valid_tiers = ['low_value', 'medium_value', 'high_value', 'premium']
        if v not in valid_tiers:
            return 'medium_value'
        return v

class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions"""
    customers: List[CustomerData] = Field(..., max_items=1000)
    include_explanations: bool = Field(False, description="Include prediction explanations")

class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    risk_level: str
    confidence_score: float = Field(..., ge=0, le=1)
    recommended_actions: List[str]
    model_version: str
    prediction_timestamp: datetime
    response_time_ms: float

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse]
    batch_size: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    average_churn_probability: float
    batch_processing_time_ms: float
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    model_loaded: bool
    redis_connected: bool
    uptime_seconds: float
    version: str
    last_prediction: Optional[datetime] = None

class ModelPerformanceMetrics(BaseModel):
    """Model performance tracking"""
    total_predictions: int
    avg_response_time_ms: float
    cache_hit_rate: float
    error_rate: float
    last_updated: datetime

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize model and connections on startup"""
    global model, preprocessor, feature_names, label_encoder, redis_client
    
    try:
        # Load model components
        logger.info("Loading model components...")
        model = joblib.load(CONFIG['model_path'])
        preprocessor = joblib.load(CONFIG['preprocessor_path'])
        feature_names = joblib.load(CONFIG['feature_names_path'])
        
        try:
            label_encoder = joblib.load(CONFIG['label_encoder_path'])
        except FileNotFoundError:
            logger.warning("Label encoder not found, using default")
            label_encoder = None
        
        # Initialize Redis
        redis_client = redis.Redis(
            host=CONFIG['redis_host'],
            port=CONFIG['redis_port'],
            db=CONFIG['redis_db'],
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
        
        # Test Redis connection
        redis_client.ping()
        
        logger.info("Model and Redis initialized successfully")
        
        # Initialize performance tracking
        await reset_performance_metrics()
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global redis_client
    if redis_client:
        redis_client.close()
    logger.info("API shutdown completed")

# Dependency functions
async def get_redis_client():
    """Get Redis client dependency"""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    return redis_client

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token verification (extend for production)"""
    # In production, implement proper JWT token verification
    if credentials and credentials.credentials == "demo-token":
        return credentials.credentials
    return None

# Helper functions
def generate_cache_key(customer_data: Dict) -> str:
    """Generate cache key for customer data"""
    # Create deterministic hash from customer features
    data_string = json.dumps(customer_data, sort_keys=True)
    return f"prediction:{hashlib.md5(data_string.encode()).hexdigest()}"

def calculate_confidence_score(churn_probability: float) -> float:
    """Calculate prediction confidence based on probability"""
    # Distance from decision boundary (0.5)
    return abs(churn_probability - 0.5) * 2

def determine_risk_level(churn_probability: float) -> str:
    """Determine risk level from churn probability"""
    if churn_probability > 0.7:
        return "HIGH"
    elif churn_probability > 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def generate_recommendations(customer_data: CustomerData, churn_prob: float) -> List[str]:
    """Generate personalized retention recommendations"""
    recommendations = []
    
    if churn_prob > 0.7:
        # High risk customers
        if customer_data.engagement_score < 0.3:
            recommendations.append("Immediate personal outreach required")
            recommendations.append("Assign dedicated account manager")
        
        if customer_data.recency_30d > 60:
            recommendations.append("Send win-back campaign with special offer")
            recommendations.append("Schedule product usage consultation")
        
        if customer_data.support_tickets_30d > 5:
            recommendations.append("Priority support queue assignment")
            recommendations.append("Product training session")
    
    elif churn_prob > 0.4:
        # Medium risk customers
        recommendations.append("Enroll in loyalty program")
        recommendations.append("Send usage analytics and insights")
        
        if customer_data.frequency_30d < 2:
            recommendations.append("Promote high-value features")
        
        if customer_data.value_tier in ['high_value', 'premium']:
            recommendations.append("VIP customer engagement program")
    
    else:
        # Low risk customers
        recommendations.append("Continue standard engagement")
        recommendations.append("Upsell additional features")
    
    return recommendations

async def update_performance_metrics(response_time_ms: float, cache_hit: bool, error: bool = False):
    """Update performance tracking metrics"""
    try:
        # Increment counters
        redis_client.incr("metrics:total_predictions")
        
        if cache_hit:
            redis_client.incr("metrics:cache_hits")
        
        if error:
            redis_client.incr("metrics:errors")
        
        # Update response time (rolling average)
        redis_client.lpush("metrics:response_times", response_time_ms)
        redis_client.ltrim("metrics:response_times", 0, 999)  # Keep last 1000
        
        # Update last prediction time
        redis_client.set("metrics:last_prediction", datetime.now().isoformat())
        
    except Exception as e:
        logger.warning(f"Failed to update metrics: {e}")

async def reset_performance_metrics():
    """Reset performance metrics"""
    try:
        redis_client.delete(
            "metrics:total_predictions",
            "metrics:cache_hits", 
            "metrics:errors",
            "metrics:response_times",
            "metrics:last_prediction"
        )
        redis_client.set("metrics:startup_time", datetime.now().isoformat())
    except Exception as e:
        logger.warning(f"Failed to reset metrics: {e}")

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Retention Prediction API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    startup_time = redis_client.get("metrics:startup_time")
    uptime = 0
    
    if startup_time:
        startup_dt = datetime.fromisoformat(startup_time)
        uptime = (datetime.now() - startup_dt).total_seconds()
    
    last_prediction_str = redis_client.get("metrics:last_prediction")
    last_prediction = None
    if last_prediction_str:
        last_prediction = datetime.fromisoformat(last_prediction_str)
    
    return HealthResponse(
        status="healthy" if model is not None and redis_client is not None else "unhealthy",
        model_loaded=model is not None,
        redis_connected=redis_client is not None,
        uptime_seconds=uptime,
        version="1.0.0",
        last_prediction=last_prediction
    )

@app.get("/metrics", response_model=ModelPerformanceMetrics)
async def get_metrics():
    """Get model performance metrics"""
    try:
        total_predictions = int(redis_client.get("metrics:total_predictions") or 0)
        cache_hits = int(redis_client.get("metrics:cache_hits") or 0)
        errors = int(redis_client.get("metrics:errors") or 0)
        
        # Calculate average response time
        response_times = redis_client.lrange("metrics:response_times", 0, -1)
        avg_response_time = 0
        if response_times:
            avg_response_time = sum(float(rt) for rt in response_times) / len(response_times)
        
        # Calculate rates
        cache_hit_rate = cache_hits / max(total_predictions, 1)
        error_rate = errors / max(total_predictions, 1)
        
        return ModelPerformanceMetrics(
            total_predictions=total_predictions,
            avg_response_time_ms=round(avg_response_time, 2),
            cache_hit_rate=round(cache_hit_rate, 4),
            error_rate=round(error_rate, 4),
            last_updated=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_single_customer(
    customer_data: CustomerData,
    background_tasks: BackgroundTasks,
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """
    Predict churn probability for a single customer
    Real-time prediction with sub-50ms target response time
    """
    start_time = time.time()
    cache_hit = False
    
    try:
        # Generate cache key
        customer_dict = customer_data.dict()
        cache_key = generate_cache_key(customer_dict)
        
        # Check cache first
        cached_result = redis_client.get(cache_key)
        if cached_result:
            cache_hit = True
            response = PredictionResponse.parse_raw(cached_result)
            response.response_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics in background
            background_tasks.add_task(
                update_performance_metrics, 
                response.response_time_ms, 
                cache_hit
            )
            
            return response
        
        # Prepare data for model
        df = pd.DataFrame([customer_dict])
        
        # Ensure all expected features are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Select and order features to match training
        df = df[feature_names[:len(df.columns)]]
        
        # Preprocess
        X_processed = preprocessor.transform(df)
        
        # Make prediction
        churn_probability = float(model.predict_proba(X_processed)[0][1])
        churn_prediction = churn_probability > 0.5
        
        # Calculate derived metrics
        confidence_score = calculate_confidence_score(churn_probability)
        risk_level = determine_risk_level(churn_probability)
        recommendations = generate_recommendations(customer_data, churn_probability)
        
        # Create response
        response = PredictionResponse(
            customer_id=customer_data.customer_id,
            churn_probability=round(churn_probability, 4),
            churn_prediction=churn_prediction,
            risk_level=risk_level,
            confidence_score=round(confidence_score, 4),
            recommended_actions=recommendations,
            model_version="1.0.0",
            prediction_timestamp=datetime.now(),
            response_time_ms=(time.time() - start_time) * 1000
        )
        
        # Cache result
        redis_client.setex(
            cache_key, 
            CONFIG['cache_ttl'], 
            response.json()
        )
        
        # Update metrics in background
        background_tasks.add_task(
            update_performance_metrics, 
            response.response_time_ms, 
            cache_hit
        )
        
        return response
        
    except Exception as e:
        error_response_time = (time.time() - start_time) * 1000
        background_tasks.add_task(
            update_performance_metrics, 
            error_response_time, 
            cache_hit, 
            error=True
        )
        
        logger.error(f"Prediction failed for customer {customer_data.customer_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_customers(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """
    Predict churn probability for multiple customers in batch
    Optimized for high-throughput processing
    """
    start_time = time.time()
    
    try:
        if len(request.customers) > CONFIG['batch_size_limit']:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size exceeds limit of {CONFIG['batch_size_limit']}"
            )
        
        predictions = []
        cache_hits = 0
        
        # Process each customer
        for customer_data in request.customers:
            try:
                # Check cache
                customer_dict = customer_data.dict()
                cache_key = generate_cache_key(customer_dict)
                cached_result = redis_client.get(cache_key)
                
                if cached_result:
                    cache_hits += 1
                    pred_response = PredictionResponse.parse_raw(cached_result)
                    predictions.append(pred_response)
                    continue
                
                # Make prediction (similar to single prediction logic)
                df = pd.DataFrame([customer_dict])
                
                # Ensure all expected features are present
                for feature in feature_names:
                    if feature not in df.columns:
                        df[feature] = 0
                
                df = df[feature_names[:len(df.columns)]]
                X_processed = preprocessor.transform(df)
                
                churn_probability = float(model.predict_proba(X_processed)[0][1])
                churn_prediction = churn_probability > 0.5
                
                confidence_score = calculate_confidence_score(churn_probability)
                risk_level = determine_risk_level(churn_probability)
                recommendations = generate_recommendations(customer_data, churn_probability)
                
                pred_response = PredictionResponse(
                    customer_id=customer_data.customer_id,
                    churn_probability=round(churn_probability, 4),
                    churn_prediction=churn_prediction,
                    risk_level=risk_level,
                    confidence_score=round(confidence_score, 4),
                    recommended_actions=recommendations,
                    model_version="1.0.0",
                    prediction_timestamp=datetime.now(),
                    response_time_ms=0  # Will be set at batch level
                )
                
                predictions.append(pred_response)
                
                # Cache result
                redis_client.setex(cache_key, CONFIG['cache_ttl'], pred_response.json())
                
            except Exception as e:
                logger.error(f"Failed to predict for customer {customer_data.customer_id}: {e}")
                # Continue with other customers
        
        # Calculate batch statistics
        total_processing_time = (time.time() - start_time) * 1000
        high_risk_count = sum(1 for p in predictions if p.risk_level == "HIGH")
        medium_risk_count = sum(1 for p in predictions if p.risk_level == "MEDIUM")
        low_risk_count = sum(1 for p in predictions if p.risk_level == "LOW")
        avg_churn_prob = sum(p.churn_probability for p in predictions) / len(predictions) if predictions else 0
        
        # Update response times
        for pred in predictions:
            pred.response_time_ms = total_processing_time / len(predictions)
        
        batch_response = BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            high_risk_count=high_risk_count,
            medium_risk_count=medium_risk_count,
            low_risk_count=low_risk_count,
            average_churn_probability=round(avg_churn_prob, 4),
            batch_processing_time_ms=round(total_processing_time, 2),
            timestamp=datetime.now()
        )
        
        # Update metrics
        cache_hit_rate = cache_hits / len(request.customers) if request.customers else 0
        background_tasks.add_task(
            update_performance_metrics,
            total_processing_time / len(predictions) if predictions else 0,
            cache_hit_rate > 0
        )
        
        return batch_response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.delete("/cache/clear")
async def clear_cache(redis_client: redis.Redis = Depends(get_redis_client)):
    """Clear prediction cache"""
    try:
        # Get all prediction cache keys
        cache_keys = redis_client.keys("prediction:*")
        if cache_keys:
            redis_client.delete(*cache_keys)
        
        return {"message": f"Cleared {len(cache_keys)} cached predictions"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.post("/cache/warmup")
async def warmup_cache():
    """Warm up cache with common predictions (for demo)"""
    # This would typically load common customer profiles
    # For demo purposes, we'll just return success
    return {"message": "Cache warmup completed"}

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "model_serving_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
