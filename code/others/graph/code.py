import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

# データの生成
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 2, size=x.shape)

# 単回帰
linear_model = LinearRegression()
linear_model.fit(x.reshape(-1, 1), y)
y_pred_linear = linear_model.predict(x.reshape(-1, 1))

# 重回帰の例として、x^2も使う
X_poly = np.column_stack((x, x**2))
linear_model.fit(X_poly, y)
y_pred_poly_linear = linear_model.predict(X_poly)

# 多項式回帰
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(x.reshape(-1, 1))
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

# 非線形回帰（例として指数関数を使用）
def func(x, a, b, c):
    return a * np.exp(b * x) + c

params, _ = curve_fit(func, x, y)
y_pred_non_linear = func(x, *params)

# プロット
plt.figure(figsize=(14, 10))

plt.scatter(x, y, label='Data', color='black')

# 単回帰
plt.plot(x, y_pred_linear, label='Linear Regression', color='blue')

# 重回帰
plt.plot(x, y_pred_poly_linear, label='Multiple Linear Regression', color='green')

# 多項式回帰
plt.plot(x, y_pred_poly, label='Polynomial Regression (degree=3)', color='red')

# 非線形回帰
plt.plot(x, y_pred_non_linear, label='Non-linear Regression', color='purple')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression Models')
plt.legend()
plt.show()
