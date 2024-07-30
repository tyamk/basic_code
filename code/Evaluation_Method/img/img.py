from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 手書き数字データセットのロード
digits = datasets.load_digits()

# データとラベルの取得
X = digits.data  # 画像データ（各画像が8x8のピクセル値を1次元ベクトルに変換したもの）
y = digits.target  # 対応するラベル（0〜9）

# データセットのサイズを表示
print("Total samples:", len(X))

# 最初のサンプルの内容を表示
print("\nFirst sample vector:")
print(X[0])

# 最初のサンプルの画像を表示
plt.gray()
plt.matshow(digits.images[0])
plt.show()

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ロジスティック回帰モデルの作成と訓練
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

# 精度の表示
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 詳細な分類レポートを表示
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
