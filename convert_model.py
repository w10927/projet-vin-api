import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1. 读取红酒数据
df = pd.read_csv("cleaned_wine.csv")

# 2. 拆分特征和目标
X = df.drop("quality", axis=1)
y = df["quality"]

# 3. 训练轻量模型
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# 4. 保存模型
with open("wine_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ 模型转换完成！生成了 wine_model.pkl")
