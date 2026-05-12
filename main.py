# main.py 最终完美版
from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# 加载我们刚生成的模型
try:
    with open("wine_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    model = None
    scaler = None

# 输入数据结构
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

# 首页
@app.get("/")
def home():
    return {"status": "运行正常 ✅"}

# 预测接口
@app.post("/predict")
def predict(data: WineFeatures):
    if not model or not scaler:
        return {"error": "模型未加载"}
    
    df = pd.DataFrame([data.dict()])
    scaled = scaler.transform(df)
    result = model.predict(scaled)[0]
    
    return {"红酒质量预测": round(float(result), 2)}
