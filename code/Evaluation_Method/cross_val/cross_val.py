from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# データのロード
data = fetch_california_housing()
X = data.data
y = data.target

# モデルの作成
model = LinearRegression()

# 交差検証の実行
# ここではk=5のk-分割交差検証を使用
scores = cross_val_score(model, X, y, cv=5)

# 各フォールドのスコアを表示
print("Cross-validation scores:", scores)

# スコアの平均と標準偏差を表示
print("Mean score:", np.mean(scores))
print("Standard deviation:", np.std(scores))
