import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データの生成
X, y = make_moons(noise=0.3, random_state=42) # 月型のデータを生成
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# k-NNモデルの作成
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 精度の計算と出力
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# データの図示
plt.figure(figsize=(12, 6))

# 訓練データ
plt.subplot(1, 2, 1)
plt.title("Training data")
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')

# テストデータの予測結果
plt.subplot(1, 2, 2)
plt.title("Test data predictions")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')

plt.show()
"""
k近傍法 (kNN)
kNNは、非パラメトリックな分類アルゴリズムである。新しいデータポイントのクラスを、最も近いk個の既知のデータポイントのクラスの多数決によって決定する。計算コストが高く、メモリ消費が大きいが、直感的である。
入力データと出力データの距離を計算する。
入力データに近いほうからK個の学習データを取得する。
ラベルで多数決を行い、分類結果とする。
ｋがハイパラになる。条件として、二値分類の場合、ｋは奇数になる。
"""