import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

# Irisデータを読み込む
data = load_iris()
X = data.data[:, :2]  # 最初の2つの特徴量を使用

# Gaussian Mixture Modelを適用する
n_components = 5
model = GaussianMixture(n_components=n_components, random_state=0)
model.fit(X)
labels = model.predict(X)

# カラーマップを設定
colors = ['navy', 'turquoise', 'darkorange', 'green', 'purple']

# データのプロット
plt.figure(figsize=(10, 6))
for i, color in enumerate(colors):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], s=5, color=color, label=f'Component {i}', alpha=0.5)

# 各ガウス分布の平均をプロット
plt.scatter(model.means_[:, 0], model.means_[:, 1], marker='o', s=100, label='Means', color='red', edgecolors='black')

# 各ガウス分布の等高線をプロット
x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
X_grid, Y_grid = np.meshgrid(x, y)
Z = -model.score_samples(np.array([X_grid.ravel(), Y_grid.ravel()]).T)
Z = Z.reshape(X_grid.shape)

plt.contour(X_grid, Y_grid, Z, levels=10, linewidths=1, colors='green', linestyles='dashed', alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Gaussian Mixture Model with 5 components')
plt.legend()
plt.grid(True)
plt.show()

"""
参考
https://qiita.com/ymgc3/items/e4440981be19c6b781eb
楕円形の広がりをガウス分布ごとに表現できるため、データ点をうまく表現できる。
"""