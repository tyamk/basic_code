# データセット
from sklearn.datasets import load_breast_cancer
import pandas as pd
# ロジステック回帰を用いて乳がんデータを二値分類する
from sklearn.linear_model import LogisticRegression
# 評価方法
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

model = LogisticRegression(solver='liblinear', max_iter=1000) # モデルの定義
model.fit(X, y) # モデルの学習

y_pred = model.predict(X) # 予測

score = accuracy_score(y, y_pred) # 正解率の算出 y:正解デークータ y_pred:予測データ
print(score) # 0.9595782073813708 # データのバランスが取れていることが重要　
# もし、データのバランスが悪いと、正解率の信憑性は低くなる




