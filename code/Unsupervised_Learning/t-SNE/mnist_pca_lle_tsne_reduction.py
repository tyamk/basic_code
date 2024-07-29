import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.preprocessing import StandardScaler

# MNISTデータの読み込みと8x8ピクセルへのリサイズ
digits = load_digits()
X, y = digits.data, digits.target

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 次元削減の設定
n_components = 2

# PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# LLE
n_neighbors = 10
lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')
X_lle = lle.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=n_components, init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# プロット
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# PCAプロット
axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
axs[0].set_title("PCA")

# LLEプロット
axs[1].scatter(X_lle[:, 0], X_lle[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
axs[1].set_title("LLE")

# t-SNEプロット
axs[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
axs[2].set_title("t-SNE")

plt.colorbar(axs[2].collections[0], ax=axs, orientation='horizontal', fraction=.1)
plt.show()

"""
t-SNEはに次元空間で数値ごとにデータをうまくまとめて、構造を分類している。
"""