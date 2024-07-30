from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# データの読み込み
data = load_breast_cancer()

# 特徴量データフレームの作成
X = pd.DataFrame(data.data, columns=data.feature_names)
print("特徴量データフレーム情報:")
print(X.info())  # 569行30列
print(X.head())  # 最初の5行のデータを表示

# 目標変数のデータフレームの作成
y = pd.DataFrame(data.target, columns=['species'])
print("\n目標変数データフレーム情報:")
print(y.info())  # 569行1列
print(y.head())  # 最初の5行のデータを表示

# 最初の10個の特徴量を選択
X = X.iloc[:, :10]
print("\n最初の10個の特徴量:")
print(X.head(10))  # 最初の10行のデータを表示

# ロジスティック回帰モデルの定義
model = LogisticRegression(solver='liblinear', max_iter=1000)

# モデルの学習
model.fit(X, y.values.ravel())  # yを1次元配列に変換

# 予測
y_pred = model.predict(X)

# 正解率の算出
score = accuracy_score(y, y_pred)
print("\nモデルの正解率:")
print(score)  # 0.9595782073813708

# 混同行列の算出
cm = confusion_matrix(y, y_pred)
print(cm)
"""
モデルの正解率:
0.9086115992970123
[[176  36]
 [ 16 341]]
これは
TN: 176, FP: 36
FN: 16, TP: 341
を表している。

- TN (True Negative) = 176:
  - 実際に悪性でない患者を正しく悪性でないと判定した数。

- FP (False Positive) = 36:
  - 実際に良性である患者を誤って悪性と判定した数。

- FN (False Negative) = 16:
  - 実際に悪性である患者を誤って良性と判定した数。

- TP (True Positive) = 341:
  - 実際に悪性である患者を正しく悪性と判定した数。
- FP = 36:
  - 36人の良性の患者を悪性と誤診。誤診による不必要な治療や心理的負担の可能性がある。

- FN = 16:
  - 16人の悪性の患者を良性と誤診。必要な治療を見逃し、患者の健康リスクを高める可能性がある。
 """