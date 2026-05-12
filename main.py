# 葡萄酒质量预测 - Streamlit 部署版（兼容你的FastAPI模型）
import streamlit as st
import pandas as pd
import joblib

# 加载模型
model = joblib.load("wine_model.pkl")
scaler = joblib.load("scaler.pkl")

# 标题
st.title("🍷 葡萄酒质量预测系统")
st.subheader("Projet Big Data S6 - Déploiement Cloud")

# 输入参数
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar", 0.0, 20.0, 2.0)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.05)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 70.0, 20.0)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 5.0, 300.0, 80.0)
density = st.slider("Density", 0.9900, 1.0030, 0.9960)
pH = st.slider("pH", 2.7, 4.0, 3.2)
sulphates = st.slider("Sulphates", 0.3, 1.5, 0.6)
alcohol = st.slider("Alcohol", 8.0, 15.0, 10.5)

# 预测按钮
if st.button("🔍 Prédire la qualité"):
    input_data = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"🎯 Qualité prédite : {int(prediction[0])} / 10")
