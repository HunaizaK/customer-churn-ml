"""REST API: `uvicorn churn.serve:app` (or `make serve`)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn import __version__
from churn.predict import load_artifact, predict

YesNo = Literal["Yes", "No"]
YesNoNoSvc = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoSvc
    OnlineBackup: YesNoNoSvc
    DeviceProtection: YesNoNoSvc
    TechSupport: YesNoNoSvc
    StreamingTV: YesNoNoSvc
    StreamingMovies: YesNoNoSvc
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold: float


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["artifact"] = load_artifact(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    yield
    state.clear()


app = FastAPI(title="Churn prediction API", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    artifact = state.get("artifact")
    return {"status": "ok" if artifact else "no-model", "model": artifact["model_name"] if artifact else None}


@app.post("/predict", response_model=list[Prediction])
def predict_endpoint(customers: list[Customer]) -> list[dict]:
    if not customers:
        raise HTTPException(status_code=422, detail="Send at least one customer")
    if len(customers) > 1000:
        raise HTTPException(status_code=413, detail="Max 1000 customers per request")
    return predict(state["artifact"], [c.model_dump() for c in customers])
