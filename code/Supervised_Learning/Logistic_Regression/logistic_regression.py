import numpy as np
from sklearn.linear_model import LogisticRegression

X_train = np.r_[np.random.normal(3, 1, size=50),  # 平均3、標準偏差1の正規分布から50個の乱数生成(クラスター0)
          np.random.normal(-1, 1, size=50)].reshape((100, -1)) # 平均 -1、標準偏差 1 の正規分布に従う 50 個の乱数を生成(クラスター1)、100行と列数は自動的に計算することで行列に変換
        # sizeの合計が100になるように設定している
y_train = np.r_[np.zeros(50), np.ones(50)] # 0が50個、1が50個の配列を結合 # np_r_は行列を結合する関数
# ロジスティック回帰モデルの学習
model = LogisticRegression()
model.fit(X_train, y_train)
probabilities = model.predict_proba([[0], [1], [2]])[:, 1] # [:, 1] は2列目の要素を取り出す
print(probabilities)
# 出力
# [0.96456321 0.68389748 0.146735  ]
