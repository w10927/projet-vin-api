from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Prédiction de la Qualité du Vin Rouge",
    description="输入红酒理化指标，实时预测其质量分数 (3~8)",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "wine_model.keras"
model = None

try:
    logger.info(f"当前工作目录: {BASE_DIR}")
    logger.info(f"模型文件路径: {model_path}")
    logger.info(f"模型文件是否存在: {model_path.exists()}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    model = load_model(model_path)
    logger.info("✅ 模型加载成功！")

except Exception as e:
    logger.error(f"❌ 模型加载失败: {str(e)}", exc_info=True)

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

@app.post("/predict")
def predict_quality(wine: WineData):
    try:
        if model is None:
            return {"status": "error", "message": "模型未加载，请检查部署日志"}
        
        # 将请求数据转为 DataFrame
        data = pd.DataFrame([wine.dict()])
        # 预测并返回结果
        prediction = model.predict(data, verbose=0)
        return {
            "status": "success",
            "quality_score": round(float(prediction[0][0]), 1)
        }
    
    except Exception as e:
        logger.error(f"❌ 预测失败: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.get("/")
def root():
    return {
        "status": "API 运行正常",
        "model_status": "已加载" if model else "未加载",
        "docs_url": "/docs"
    }