import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# データの生成
train_size = 20
test_size = 12  
train_X = np.random.uniform(low=0, high=1.2, size=train_size)
test_X = np.random.uniform(low=0, high=1.2, size=test_size)
train_y = np.sin(train_X * 2 * np.pi) + np.random.normal(0, 0.2, train_size)
test_y = np.sin(test_X * 2 * np.pi) + np.random.normal(0, 0.2, test_size)

# 6次式
degree = 6
poly = PolynomialFeatures(degree)
train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))
test_poly_X = poly.fit_transform(test_X.reshape(test_size, 1))

# アルファの範囲を設定
alphas = np.linspace(0, 2.0, 3) 

# グラフの設定
x_fit = np.linspace(0, 1.2, 100)
x_fit_poly = poly.transform(x_fit.reshape(-1, 1))

results = []

plt.figure(figsize=(12, 8))

for i in alphas:
    model = Ridge(alpha=i)
    model.fit(train_poly_X, train_y)
    train_pred_y = model.predict(train_poly_X)
    test_pred_y = model.predict(test_poly_X)
    train_mse = mean_squared_error(train_y, train_pred_y)
    test_mse = mean_squared_error(test_y, test_pred_y)
    results.append({'alpha': i, 'train_mse': train_mse, 'test_mse': test_mse})

# リストをDataFrameに変換
df = pd.DataFrame(results)
print(df)

# データの散布をプロット
plt.scatter(train_X, train_y, color='blue', label='Training Data')
plt.scatter(test_X, test_y, color='red', label='Test Data')

# 真の関数をプロット
plt.plot(x_fit, np.sin(2 * np.pi * x_fit), color='black', linestyle='--', label='True Function')

# グラフの設定
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
plt.title(f'6th Degree Polynomial Regression with Various Alphas')
plt.legend()
plt.show()

"""
正則化
正則化は、モデルの過学習を防ぐための手法である。L1正則化（ラッソ）とL2正則化（リッジ）があり、前者はモデルの一部の係数をゼロにし、後者は全ての係数を小さくする。正則化項を目的関数に追加することで実現される。
ほかのアルゴリズムと併用する。損失関数に罰則項を与えて、モデルの汎化能力を高めることができる。過学習は複雑なモデルに起きやすく、正則化によって複雑さを緩和し、汎化性能を向上させる。
cf. 過学習：学習誤差と検証誤差の差が著しく大きいこと。罰則項=正則化項。RSSに加える。
"""