from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Wine Quality API")

# 加载模型
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 输入结构
class WineInput(BaseModel):
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

# 测试接口
@app.get("/")
def home():
    return {"status": "API 运行正常 ✅"}

# 预测接口
@app.post("/predict")
def predict(input: WineInput):
    df = pd.DataFrame([input.dict()])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)
    return {"quality_prediction": round(float(prediction[0]), 2)}
