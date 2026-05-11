import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

model = load_model("wine_model.keras")
df = pd.read_csv("cleaned_wine_fixed.csv")
X = df.drop("quality", axis=1)
scaler = StandardScaler()
scaler.fit(X)

def predire_qualite_vin(data):
    df_input = pd.DataFrame([data])
    df_scaled = scaler.transform(df_input)
    prediction = model.predict(df_scaled, verbose=0)
    classe = np.argmax(prediction)
    note = classe + 3
    return note

if __name__ == "__main__":
    exemple = {
        "fixed_acidity": 8.0,
        "volatile_acidity": 0.5,
        "citric_acid": 0.3,
        "residual_sugar": 2.0,
        "chlorides": 0.08,
        "free_sulfur_dioxide": 15,
        "total_sulfur_dioxide": 60,
        "density": 0.996,
        "pH": 3.3,
        "sulphates": 0.65,
        "alcohol": 10.5
    }
    print("✅ Qualité prédite :", predire_qualite_vin(exemple))