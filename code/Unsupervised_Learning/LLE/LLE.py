import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

# Swiss Rollデータの生成
n_samples = 1500
noise = 0.05
X, color = make_swiss_roll(n_samples=n_samples, noise=noise)

# 次元削減のためのLLEの設定
n_neighbors = 12
n_components = 2
lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')

# データの次元削減
X_r = lle.fit_transform(X)

# 結果のプロット
fig = plt.figure(figsize=(12, 6))

# 元の3次元データのプロット
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Original Swiss Roll")

# 次元削減後の2次元データのプロット
ax = fig.add_subplot(122)
sc = ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
ax.set_title("LLE Reduced Swiss Roll")
plt.colorbar(sc)

plt.show()
"""
近傍点を求める→xiを再構成できるように重みWijを求める→低次元でのyiを計算する
"""
