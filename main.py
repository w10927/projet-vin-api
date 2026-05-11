from fastapi import FastAPI
from pydantic import BaseModel
from predict import predire_qualite_vin

app = FastAPI(
    title="API de Prédiction de la Qualité du Vin Rouge",
    description="Déploiement cloud d'un modèle de réseau de neurones pour prédire la qualité du vin rouge. （云端部署的红酒质量预测API，支持实时预测）",
    version="1.0"
)

class CaracteristiquesVin(BaseModel):
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

@app.post("/predict", summary="Prédire la qualité du vin", description="输入红酒理化指标，实时预测其质量分数（3~8）")
def predict(vin: CaracteristiquesVin):
    note = predire_qualite_vin(vin.dict())
    return {
        "qualite_predite": note,
        "intervalle_de_note": "3~8",
        "statut": "Prédiction en temps réel réussie"
    }