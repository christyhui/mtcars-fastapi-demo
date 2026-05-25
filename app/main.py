"""
app/main.py

FastAPI application that serves predictions from a trained linear regression
model built on the MTCARS dataset.

Endpoints:
    GET  /health   — liveness check
    GET  /ready    — readiness check (model loaded?)
    POST /predict  — predict mpg given wt and hp
"""

import pandas as pd
import logging
import os
import pathlib
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Resolve relative to this file so pytest and uvicorn both find it
_HERE = pathlib.Path(__file__).parent.parent
MODEL_PATH = pathlib.Path(os.getenv("MODEL_PATH", str(_HERE / "models" / "model.pkl")))

# ── Model loading ─────────────────────────────────────────────────────────────

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, clean up on shutdown."""
    global model
    if not MODEL_PATH.exists():
        logger.warning(f"Model file not found at {MODEL_PATH}. /predict will be unavailable.")
    else:
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
    yield
    # shutdown: nothing to clean up for a simple model
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MTCARS MPG Predictor",
    description="Predicts fuel efficiency (mpg) from vehicle weight and horsepower.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Schemas ───────────────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    wt: float = Field(..., gt=0, description="Vehicle weight in 1000 lbs (e.g. 2.62)")
    hp: float = Field(..., gt=0, description="Gross horsepower (e.g. 110)")

    model_config = {"json_schema_extra": {"examples": [{"wt": 2.62, "hp": 110}]}}


class PredictionResponse(BaseModel):
    predicted_mpg: float
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Liveness check — returns ok if the API is running."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["Monitoring"])
def ready() -> ReadyResponse:
    """Readiness check — returns ok only if the model is loaded."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run train_model.py first.",
        )
    return ReadyResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict fuel efficiency (mpg) given vehicle weight and horsepower.

    - **wt**: weight in 1000 lbs (e.g. 2.62)
    - **hp**: gross horsepower (e.g. 110)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run train_model.py first.",
        )
    try:
        features = pd.DataFrame([{"wt": request.wt, "hp": request.hp}])
        prediction = model.predict(features)[0]
        logger.info(f"Prediction: wt={request.wt}, hp={request.hp} → mpg={prediction:.2f}")
        return PredictionResponse(predicted_mpg=round(float(prediction), 2))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
