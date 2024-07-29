import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データの読み込み
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# 検証セットをトレーニングセットから分割
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Gradient Boosting Classifierのモデルを作成し、アーリーストッピングを設定,
# n_iter_no_change: 検証セットでの損失が指定したエポック数（この場合10エポック）以上改善しなかった場合にトレーニングを停止
model = GradientBoostingClassifier(n_estimators=1000, validation_fraction=0.1, n_iter_no_change=10, random_state=42)

# モデルの学習
model.fit(X_train, y_train)

# 検証セットでの評価
val_accuracy = accuracy_score(y_val, model.predict(X_val))
print(f'Validation Accuracy: {val_accuracy:.2f}')

# テストセットでの評価
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f'Test Accuracy: {test_accuracy:.2f}')

# 特徴量の重要度を取得
importances = model.feature_importances_

# 重要度を降順に並べるためのインデックスを取得
indices = np.argsort(importances)[::-1]

# 特徴量名を取得
feature_names = data.feature_names

# 特徴量の重要度をDataFrameに変換
importance_df = pd.DataFrame({
    'Feature': [feature_names[i] for i in indices],
    'Importance': importances[indices]
})

# 横の棒グラフを作成
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Ranking')
plt.gca().invert_yaxis()  # 重要度が高いものが上に来るように順序を反転
plt.show()
