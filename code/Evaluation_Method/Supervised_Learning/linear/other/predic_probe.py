from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

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
# 予測確率
y_pred_proba = model.predict_proba(X)
print("\n予測確率:")
print(y_pred_proba)

"""
[[9.95581869e-01 4.41813062e-03]
 [9.99512682e-01 4.87318123e-04]
 [9.99668936e-01 3.31064282e-04]
 ...
 [9.73718065e-01 2.62819355e-02]
 [9.99994906e-01 5.09374719e-06]
 [2.59312243e-02 9.74068776e-01]]

 1行目の[9.95581869e-01 4.41813062e-03]は、ラベル0（良性）の確率が99.56%、ラベル1（悪性）の確率が0.44%を示している。
"""