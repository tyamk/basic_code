import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

# データ生成
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.randn(100)

# 単回帰（2次元）
model_linear = LinearRegression()
model_linear.fit(X.reshape(-1, 1), y)
y_pred_linear = model_linear.predict(X.reshape(-1, 1))

# 重回帰（3次元）
X3d = np.vstack((X, np.random.randn(100))).T
y3d = 3 * X3d[:, 0] + 2 * X3d[:, 1] + 1 + np.random.randn(100)
model_multi = LinearRegression()
model_multi.fit(X3d, y3d)
X1, X2 = np.meshgrid(X, X)
y_pred_multi = model_multi.intercept_ + model_multi.coef_[0] * X1 + model_multi.coef_[1] * X2

# 多項式回帰（2次元）
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X.reshape(-1, 1))
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)

# 非線形回帰（2次元）
def func(x, a, b, c):
    return a * np.sin(b * x) + c

params, _ = curve_fit(func, X, y)
y_pred_nonlinear = func(X, *params)

# グラフの作成
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), tight_layout=True, squeeze=False)

# 単回帰
axes[0, 0].scatter(X, y, color='blue', label='Data')
axes[0, 0].plot(X, y_pred_linear, color='red', label='Linear Fit: $y = {:.2f}x + {:.2f}$'.format(model_linear.coef_[0], model_linear.intercept_))
axes[0, 0].set_title('単回帰 (Linear Regression)')
axes[0, 0].legend()

# 重回帰
ax = fig.add_subplot(222, projection='3d')
ax.scatter(X3d[:, 0], X3d[:, 1], y3d, color='blue', label='Data')
ax.plot_surface(X1, X2, y_pred_multi, color='yellow', alpha=0.5, label='Multiple Regression')
ax.set_title('重回帰 (Multiple Regression)')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.legend()

# 多項式回帰
axes[1, 0].scatter(X, y, color='blue', label='Data')
axes[1, 0].plot(X, y_pred_poly, color='green', label='Polynomial Fit: $y = {:.2f}x^2 + {:.2f}x + {:.2f}$'.format(model_poly.coef_[2], model_poly.coef_[1], model_poly.coef_[0]))
axes[1, 0].set_title('多項式回帰 (Polynomial Regression)')
axes[1, 0].legend()

# 非線形回帰
axes[1, 1].scatter(X, y, color='blue', label='Data')
axes[1, 1].plot(X, y_pred_nonlinear, color='purple', label='Nonlinear Fit: $y = {:.2f}\sin({:.2f}x) + {:.2f}$'.format(*params))
axes[1, 1].set_title('非線形回帰 (Nonlinear Regression)')
axes[1, 1].legend()

plt.show()
