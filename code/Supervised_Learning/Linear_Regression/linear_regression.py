from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データの生成
np.random.seed(42)
data = {
    'X': np.arange(11) * np.random.rand(11) * np.random.rand(11),
    'y': np.arange(11) * np.random.rand(11)
} 
df = pd.DataFrame(data)
# print(df.head())
# 散布図
"""df.plot.scatter(x = 'x', y = 'y', color = 'r')
plt.show()"""
# モデルの作成
# 線形回帰
model = LinearRegression()
# 説明変数と目的変数
# 学習
model.fit(df[['X']], df['y'])
# 切片と傾き
print(model.intercept_)
print(model.coef_)
# 予測
y_pred = model.predict([[10],[15]])
print(y_pred)
"""
線形回帰
線形回帰は、連続的な数値データを予測するための回帰分析手法である。入力変数と出力変数の関係を線形関数でモデル化し、最小二乗法を用いて回帰係数を推定する。単回帰と重回帰がある。
ある説明変数が大きくなるにつれて、目的変数も大きく、または、小さくなっていく。通常、１つ以上の説明変数を使ってモデルを作成する。一直線上にないデータ点から学習パラメータを求める
必要があり、異なる直線を引き、それぞれのデータからの差（目的変数と直線の差）を二乗して平均することで、適切な直線を選択する。この評価方法を平均二乗誤差という。誤差と学習パラメータ
との関係を表した関数を誤差関数という。誤差関数の値が最も最小になるパラメータを求める。

"""
