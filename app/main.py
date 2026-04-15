from fastapi import FastAPI
import joblib
from app.schemas import PredictionRequest

app = FastAPI(title="Breast Cancer Prediction API")

model = joblib.load("artifacts/model.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])[0]
    return {"prediction": int(prediction)}