from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# ランダムフォレストのモデルを作成
# 複数のモデルを組み合わせて予測を行うアンサンブル学習の一つ
# ブートストラップ法と特徴量のランダムサンプリングを行い、複数の決定木を作成
# それらの決定木の多数決で予測を行う

model = RandomForestClassifier()
# モデルの学習
model.fit(X_train, y_train)
# モデルの予測
yield_pred = model.predict(X_test)
# モデルの評価
accuracy = accuracy_score(y_test, yield_pred)
print(f'Accuracy: {accuracy:.2f}')# 特徴量の重要度を取得
importances = model.feature_importances_

# 重要度を降順に並べるためのインデックスを取得
indices = np.argsort(importances)[::-1]

# 特徴量名を取得
feature_names = data.feature_names

# 重要度の高い順に並べて表示
print("Feature ranking:")
for i in range(X_train.shape[1]):
    print(f"{i + 1}. feature {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

# 特徴量の重要度をDataFrameに変換
importance_df = pd.DataFrame({
    'Feature': [feature_names[i] for i in indices],
    'Importance': importances[indices]
})

# データフレームの表示
print("\nFeature Importance DataFrame:")
print(importance_df)
# 棒グラフの表示
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance of Random Forest')
plt.gca().invert_yaxis()
plt.show()
"""
ランダムフォレスト
ランダムフォレストは、複数の決定木を組み合わせたアンサンブル学習手法である。各決定木は異なる部分集合のデータを用いて学習し、多数決によって最終的な予測を行う。過学習のリスクが低く、精度が高い。
勾配ブースティングなどの手法がある。複数のモデルを活用することで汎化性能を上げる。分類と回帰どちらにも活用できる。多数決で学習結果が決まる。
学習データを条件分岐によって分割していくことで分類問題を解く手法である。不純度という乱雑さを数値化したものを活用する。ラベルが多い→不純度は小さい。指標はジニ係数。不純度が小さくなるように分割する。
ブートストラップ法と特徴量のランダムな選択という工夫がある。

"""