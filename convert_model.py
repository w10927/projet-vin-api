import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 读取你最新上传的正确CSV文件
df = pd.read_csv("cleaned_wine_fixed.csv")

# 自动取最后一列为质量分数
X = df.drop(df.columns[-1], axis=1)
y = df.iloc[:, -1]

# 训练模型
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# 生成模型文件
with open("wine_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ 模型生成成功！已生成 wine_model.pkl")
