from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


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

# 偽陽性率、真陽性率、閾値の算出
probas = model.predict_proba(X)

# roc   
roc_auc_score (y, probas[:, 1])

fpr, tpr, thresholds = roc_curve(y, probas[:, 1])
# ROC曲線の描画
plt.style.use('fivethirtyeight')

fig, ax = plt.subplots()
fig.set_size_inches(4.8, 5)

ax.step(fpr, tpr, 'gray')
ax.fill_between(fpr, tpr, 0, color='skyblue', alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_facecolor('xkcd:white')
plt.show()



