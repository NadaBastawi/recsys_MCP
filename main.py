"""
TORCO Pest Control — Recommendation API
FastAPI service exposing the hybrid recommender
"""
import logging
import os
import sys
from typing import List, Optional

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
import sentry_sdk

from recommender import HybridRecommender
from monitoring.logger import PredictionLogger

load_dotenv()

sentry_sdk.init(
    dsn="https://976a0582ae46c3a6817571521e4e2e98@o4511707022688256.ingest.de.sentry.io/4511707097006160",
    send_default_pii=True,
    traces_sample_rate=1.0,
)

logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
API_KEY = os.getenv("API_KEY", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_PORT = int(os.getenv("API_PORT", "8000"))

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Add model directory to path for imports
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

app = FastAPI(
    title="TORCO Pest Control Recommendation API",
    description="AI-powered service recommendations for pest control customers",
    version="1.0.0"
)

# Model management
class RecommenderManager:
    """Manages the recommender model loading and access."""
    
    def __init__(self):
        self._recommender = None
        self._load_model()
    
    def _load_model(self):
        """Load or train the recommender model."""
        model_path = os.path.join(MODEL_DIR, "torco_recommender.pkl")
        
        try:
            if os.path.exists(model_path):
                self._recommender = joblib.load(model_path)
                logger.info("Loaded saved model")
            else:
                logger.info("Saved model not found. Training new model for API startup...")
                customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
                interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
                services = pd.read_csv(os.path.join(DATA_DIR, "services.csv"))
                self._recommender = HybridRecommender()
                self._recommender.fit(customers, interactions, services)
                os.makedirs(MODEL_DIR, exist_ok=True)
                joblib.dump(self._recommender, model_path)
                logger.info("Trained and saved new model")
        except Exception as e:
            logger.exception("Error loading or training model: %s", e)
            raise
    
    @property
    def recommender(self):
        """Get the recommender instance."""
        return self._recommender

# Initialize recommender manager
recommender_manager = RecommenderManager()


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """Ensure protected endpoints receive a valid API key."""
    expected_key = API_KEY
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if not expected_key and x_api_key is None:
        return None
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return None

# ── SCHEMAS ───────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    customer_id: str
    season: Optional[str] = "summer"
    days_since_last_service: Optional[int] = 180
    top_n: Optional[int] = 5

class NewCustomerRequest(BaseModel):
    property_type: str          # e.g. "residential_house"
    zip_code: str               # e.g. "85743"
    building_age_years: int
    season: Optional[str] = "summer"
    days_since_last_service: Optional[int] = 365
    top_n: Optional[int] = 5

class ServiceRecommendation(BaseModel):
    service_id: str
    service_name: str
    category: str
    price_usd: float
    score: float
    urgency_score: float
    urgency_label: str

class RecommendResponse(BaseModel):
    customer_id: str
    recommendations: List[ServiceRecommendation]
    model_version: str = "1.0.0"

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "TORCO Recommendation API",
        "status": "running",
        "endpoints": ["/recommend", "/recommend/new-customer", "/health", "/services"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": recommender_manager.recommender is not None}

@app.get("/services")
def list_services():
    return [dict(service_id=k, **v) for k, v in recommender_manager.recommender.services_dict.items()]

@app.post("/recommend", response_model=RecommendResponse)
def recommend_for_existing_customer(request: RecommendRequest, _=Depends(require_api_key)):
    """
    Get service recommendations for an existing customer by ID.
    Uses their full history and profile for personalized recommendations.
    """
    if request.customer_id not in recommender_manager.recommender.customers_df.index:
        raise HTTPException(status_code=404, detail=f"Customer {request.customer_id} not found")

    recs = recommender_manager.recommender.recommend(
        customer_id=request.customer_id,
        top_n=request.top_n,
        season=request.season,
        days_since_last=request.days_since_last_service
    )

    return RecommendResponse(
        customer_id=request.customer_id,
        recommendations=[ServiceRecommendation(**r) for r in recs]
    )

@app.post("/recommend/new-customer")
def recommend_for_new_customer(request: NewCustomerRequest, _=Depends(require_api_key)):
    """
    Get recommendations for a new customer with no history.
    Falls back to content-based filtering using property profile.
    Cold-start solution using property features only.
    """
    recs = recommender_manager.recommender.recommend_cold_start(
        property_type=request.property_type,
        zip_code=request.zip_code,
        building_age_years=request.building_age_years,
        season=request.season,
        days_since_last=request.days_since_last_service,
        top_n=request.top_n
    )

    return {
        "customer_type": "new",
        "profile": {
            "property_type": request.property_type,
            "zip_code": request.zip_code,
            "building_age_years": request.building_age_years,
            "season": request.season,
            "days_since_last_service": request.days_since_last_service,
        },
        "recommendations": recs,
    }