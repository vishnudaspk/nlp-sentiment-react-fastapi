# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import os
import torch

app = FastAPI()

model_dir = os.path.join(os.path.dirname(__file__), "model")
print(f"📂 Loading model from: {model_dir}")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Pipeline on GPU
model_path = "./model"
config_path = os.path.join(model_path, "config.json")
if not os.path.exists(config_path):
    raise RuntimeError("❌ Model configuration not found. Please fine-tune the model first.")

try:
    device = 0 if torch.cuda.is_available() else -1
    sentiment_model = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=device)
    print(f"✅ Model loaded on {'GPU' if device == 0 else 'CPU'}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# Request and Response Models
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    score: float

# Predict Endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        result = sentiment_model(request.text)[0]
        return PredictionResponse(label=result["label"], score=result["score"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "API is working"}