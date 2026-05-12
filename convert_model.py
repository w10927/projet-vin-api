import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1. 读取数据
df = pd.read_csv("cleaned_wine.csv")

# 2. 自动找目标列（兼容大小写）
target_col = None
for col in df.columns:
    if 'quality' in col.lower():
        target_col = col
        break

if target_col is None:
    print("❌ 没找到质量列，列名有：", df.columns.tolist())
    exit()

print(f"✅ 找到目标列：{target_col}")

# 3. 拆分特征和目标
X = df.drop(target_col, axis=1)
y = df[target_col]

# 4. 训练模型
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# 5. 保存模型
with open("wine_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ 模型转换完成！生成了 wine_model.pkl")
