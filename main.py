from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from tf_keras.models import load_model

app = FastAPI(
    title="API de Prédiction de la Qualité du Vin Rouge",
    description="输入红酒理化指标，实时预测其质量分数 (3~8)",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "wine_model.keras"
model = load_model(model_path)

class WineData(BaseModel):
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

@app.post("/predict", summary="Prédire la qualité du vin")
def predict_quality(wine: WineData):
    
    data = pd.DataFrame([wine.dict()])
    
       prediction = model.predict(data, verbose=0)
    
        return {"quality_score": round(float(prediction[0][0]), 1)}

@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API de prédiction de la qualité du vin rouge !",
        "docs": "访问 /docs 查看接口文档"
    }