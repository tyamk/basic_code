from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# データの生成
centers = [(-1, -0.125), (0.5, 0.5)]
# 2つのクラスタを持つデータセットを生成、 n_samplesはサンプル数、n_featuresは特徴量の数、centersはクラスタの数、cluster_stdはクラスタの標準偏差
# 標準偏差が大きいとクラスタが広がる、小さいとクラスタが密集する
X, y = make_blobs(n_samples=50, n_features=2, centers=centers, cluster_std=0.5)

# データフレームの作成
df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
# 列名の変更（もしなければ、ilocを使用）
df.columns = ['feature1', 'feature2', 'label']
print(df.head())

# 散布図のプロット
plt.scatter(df['feature1'], df['feature2'], c=df['label'])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Blobs_01')
plt.show()

#D  データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# モデルの作成
model = LinearSVC()
# 学習
model.fit(X_train, y_train)

# 精度の評価
# y_testは正解ラベル、model.predict(X_test)は予測ラベル
print('正解率:', accuracy_score(y_test, model.predict(X_test)))

# メッシュグリッドを作成
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 予測
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # ravel()は多次元配列を1次元配列に変換する # np.c_は配列を結合する
y_pred = Z.reshape(xx.shape)
print(y_pred)

# メッシュグリッドを加えた可視化
plt.contourf(xx, yy, y_pred, alpha=0.3)
plt.scatter(df['feature1'], df['feature2'], c=df['label'], edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot with Decision Boundary_02')
plt.show()

print(y_test.shape)
print(y_pred.shape)

# メッシュグリッド上の予測結果の形状を確認、# y_predの形状が(420, 491)と非常に大きい
print("Shape of meshgrid predictions (Z):", y_pred.shape)
