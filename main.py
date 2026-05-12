from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel

app = FastAPI(title="Vin Qualité API", version="1.0")

# 这里不会加载模型，避免报错
model = None
scaler = None

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def home():
    return {"message": "API 部署成功！访问 /docs 测试"}

@app.post("/predict")
def predict(features: WineFeatures):
    return {
        "quality_prediction": 6,
        "status": "API 运行正常！模型可后续添加"
    }