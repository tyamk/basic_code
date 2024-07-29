import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 例として平均二乗誤差（MSE）を定義します
def mse(theta1, theta2):
    return (theta1 - 2)**2 + (theta2 - 3)**2 + 1

# theta1とtheta2の範囲を設定します
theta1_range = np.linspace(-10, 10, 100)
theta2_range = np.linspace(-10, 10, 100)

# グリッドを作成します
theta1, theta2 = np.meshgrid(theta1_range, theta2_range)
error = mse(theta1, theta2)

# 3次元プロットを作成します
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3次元グラフの描画
surf = ax.plot_surface(theta1, theta2, error, cmap='viridis')

# カラーバーを追加
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# ラベルを追加
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('Error')

# タイトルを追加
ax.set_title('Error Function Surface')

# プロットを表示
plt.show()
