from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np

# データのロード
data = fetch_california_housing()
X = data.data
y = data.target

# モデルの作成
model = Ridge()

# ハイパーパラメータの設定
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

# グリッドサーチの設定
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

# グリッドサーチの実行
grid_search.fit(X, y)

# 最適なパラメータの表示
print("Best parameters:", grid_search.best_params_)

# 最適なパラメータでのスコアの表示
print("Best score:", grid_search.best_score_)
