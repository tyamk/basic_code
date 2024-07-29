import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# サンプルデータの生成
centers = [[2, 2], [4, 4], [6, 6]]
X, y = make_blobs(n_samples=100, n_features=2, centers=centers, cluster_std=0.8, random_state=42)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ハードマージンSVM（Cを大きく設定）
hard_margin_svm = SVC(kernel='linear', C=1e10)
hard_margin_svm.fit(X_train, y_train)

# ソフトマージンSVM（Cを小さく設定）
soft_margin_svm = SVC(kernel='linear', C=1.0)
soft_margin_svm.fit(X_train, y_train)

# テストデータで予測
y_pred_hard = hard_margin_svm.predict(X_test)
y_pred_soft = soft_margin_svm.predict(X_test)

# 正解率の計算
accuracy_hard = accuracy_score(y_test, y_pred_hard)
accuracy_soft = accuracy_score(y_test, y_pred_soft)
print('ハードマージンSVMの正解率:', accuracy_hard)
print('ソフトマージンSVMの正解率:', accuracy_soft)

# メッシュグリッドを作成
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# メッシュグリッド上の各点の予測
Z_hard = hard_margin_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_hard = Z_hard.reshape(xx.shape)
Z_soft = soft_margin_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_soft = Z_soft.reshape(xx.shape)

# ハードマージンSVMの決定境界のプロット
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_hard, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hard Margin SVM')

# ソフトマージンSVMの決定境界のプロット
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_soft, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Soft Margin SVM')

plt.show()
"""
マージンの外側のデータ
マージン上のデータ
マージンの内側のデータ
"""
