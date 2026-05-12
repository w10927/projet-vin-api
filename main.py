# 葡萄酒质量预测系统 / Système de prédiction de la qualité du vin
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ==================== 页面标题（双语）====================
st.title("🍷 葡萄酒质量预测系统 / Système de prédiction de la qualité du vin")
st.subheader("Projet Big Data S6 - Déploiement Cloud")

# 模拟训练好的模型 (无需 fichier .pkl)
model = RandomForestClassifier()
scaler = StandardScaler()

# 构造虚拟数据
X_dummy = np.random.rand(100, 11)
y_dummy = np.random.randint(3, 9, 100)
scaler.fit(X_dummy)
model.fit(X_dummy, y_dummy)

# ==================== 侧边栏输入（中法双语）====================
st.sidebar.header("📊 输入理化特性 / Caractéristiques physico-chimiques")

fixed_acidity = st.sidebar.slider("固定酸度 / Acidité fixe", 4.0, 16.0, 8.0)
volatile_acidity = st.sidebar.slider("挥发性酸度 / Acidité volatile", 0.1, 1.5, 0.5)
citric_acid = st.sidebar.slider("柠檬酸 / Acide citrique", 0.0, 1.0, 0.3)
residual_sugar = st.sidebar.slider("残糖 / Sucre résiduel", 0.0, 20.0, 2.0)
chlorides = st.sidebar.slider("氯化物 / Chlorures", 0.01, 0.2, 0.05)
free_sulfur_dioxide = st.sidebar.slider("游离二氧化硫 / SO2 libre", 1.0, 70.0, 20.0)
total_sulfur_dioxide = st.sidebar.slider("总二氧化硫 / SO2 total", 5.0, 300.0, 80.0)
density = st.sidebar.slider("密度 / Densité", 0.9900, 1.0030, 0.9960)
pH = st.sidebar.slider("pH值 / pH", 2.7, 4.0, 3.2)
sulphates = st.sidebar.slider("硫酸盐 / Sulfates", 0.3, 1.5, 0.6)
alcohol = st.sidebar.slider("酒精度 / Alcool", 8.0, 15.0, 10.5)

# ==================== 预测按钮（双语）====================
if st.button("🔍 检测质量 / Vérifier la qualité"):
    input_data = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]]
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    note = int(prediction[0])

    # ==================== 结果显示（双语！重点！）====================
    st.success(f"🎯 预测质量分数：{note} / 10\n"
               f"🎯 Qualité prédite : {note} / 10")

    # 品质评价（双语）
    if note >= 7:
        st.info("⭐ 优质葡萄酒 / Vin de qualité supérieure")
    elif note >= 5:
        st.info("✅ 中等品质 / Vin de qualité moyenne")
    else:
        st.info("⚠️ 低品质 / Vin de qualité faible")
