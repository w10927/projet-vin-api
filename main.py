from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel

# 初始化 FastAPI
app = FastAPI(title="Vin Qualité API", version="1.0")

# 加载轻量级模型（scikit-learn，几MB）
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    model = None
    scaler = None

# 定义红酒特征格式
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

# 首页测试
@app.get("/")
def home():
    return {"message": "API 部署成功！访问 /docs 测试"}

# 预测接口
@app.post("/predict")
def predict(features: WineFeatures):
    if model is None or scaler is None:
        return {"error": "模型未加载"}
    
    # 转换数据
    data = pd.DataFrame([features.dict()])
    data_scaled = scaler.transform(data)
    
    # 预测
    prediction = model.predict(data_scaled)
    
    return {
        "quality_prediction": int(prediction[0]),
        "status": "成功"
    }