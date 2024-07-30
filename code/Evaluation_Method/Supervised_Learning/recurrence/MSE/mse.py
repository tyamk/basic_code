from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# カリフォルニア住宅データセットをロード
data = fetch_california_housing()
X = data.data[:, [5]]
y = data.target

# 線形回帰モデルの作成と適合
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 傾きと切片を取得
a = model.coef_
b = model.intercept_
print(a, b)

# プロットの作成
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='data', marker='o')
ax.plot(X, y_pred, color='red', label='regression curve')
ax.legend()
plt.show()
