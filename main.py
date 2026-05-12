# 葡萄酒质量预测 - 无本地模型版本 (直接部署)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 页面配置
st.title("🍷 葡萄酒质量预测系统")
st.subheader("Big Data Project - Cloud Deployment")

# 模拟训练好的模型与标准化器 (无需pkl文件)
model = RandomForestClassifier()
scaler = StandardScaler()

# 构造虚拟数据用于拟合 (仅演示)
X_dummy = np.random.rand(100, 11)
y_dummy = np.random.randint(3, 9, 100)
scaler.fit(X_dummy)
model.fit(X_dummy, y_dummy)

# 输入参数
st.sidebar.header("输入理化特性")
fixed_acidity = st.sidebar.slider("固定酸度", 4.0, 16.0, 8.0)
volatile_acidity = st.sidebar.slider("挥发性酸度", 0.1, 1.5, 0.5)
citric_acid = st.sidebar.slider("柠檬酸", 0.0, 1.0, 0.3)
residual_sugar = st.sidebar.slider("残糖", 0.0, 20.0, 2.0)
chlorides = st.sidebar.slider("氯化物", 0.01, 0.2, 0.05)
free_sulfur_dioxide = st.sidebar.slider("游离二氧化硫", 1.0, 70.0, 20.0)
total_sulfur_dioxide = st.sidebar.slider("总二氧化硫", 5.0, 300.0, 80.0)
density = st.sidebar.slider("密度", 0.9900, 1.0030, 0.9960)
pH = st.sidebar.slider("pH值", 2.7, 4.0, 3.2)
sulphates = st.sidebar.slider("硫酸盐", 0.3, 1.5, 0.6)
alcohol = st.sidebar.slider("酒精度", 8.0, 15.0, 10.5)

# 预测
if st.button("🔍 预测质量"):
    input_data = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"🎯 预测质量分数：{int(prediction[0])} 分")
