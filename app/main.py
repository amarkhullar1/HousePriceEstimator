
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, numpy as np

app = FastAPI(title="London House Price Estimator", version="1.0")

class HouseFeatures(BaseModel):
    property_type: str = Field(..., description="D, S, T, F (detached/semidetached/terraced/flat)")
    tenure: str = Field(..., description="F or L")
    postcode: str
    borough: str
    floor_area: float = Field(..., gt=5, lt=1000)
    bedrooms: int = Field(..., ge=0, le=15)
    bathrooms: int = Field(..., ge=0, le=10)
    distance_to_station_km: float = Field(..., ge=0, lt=20)
    sale_year: int = Field(..., ge=1995, le=2100)
    sale_quarter: int = Field(..., ge=1, le=4)
    lat: float
    lon: float

@app.on_event("startup")
def load_model():
    global ART
    ART = joblib.load("model/house_price_catboost.joblib")

@app.post("/predict")
def predict(x: HouseFeatures):
    try:
        feats = ART["features"]
        row = {k: getattr(x, k) for k in feats}
        X = [list(row.values())]  # CatBoost handles order by column index
        # CatBoost in joblib has the column order it saw; we pass in same order:
        pred_log = ART["model"].predict(X)[0]
        price = float(np.expm1(pred_log))
        return {
            "estimated_price_gbp": round(price, 2),
            "explain": {
                "area_note": "Estimates reflect borough + proximity to public transport more than fine-grained street effects.",
                "currency": "GBP",
                "model": "CatBoostRegressor (MAE on CV shown during training)"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
