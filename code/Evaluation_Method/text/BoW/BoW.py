import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report

# 20 Newsgroupsデータセットのロード
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

# データセットのサイズを表示
print("Total samples:", len(newsgroups.data))

# 最初のサンプルの内容を表示
print("\nFirst sample:")
print(newsgroups.data[0])

# 最初のサンプルのターゲットラベルを表示
print("\nFirst sample target label:")
print(newsgroups.target[0])

# ターゲットラベルに対応するニュースグループの名前を表示
print("\nTarget label name:")
print(newsgroups.target_names[newsgroups.target[0]])

# 最初の5つのサンプルを表示
for i in range(5):
    print(f"\nSample {i+1}:")
    print(newsgroups.data[i])
    print("\nTarget label:", newsgroups.target[i])
    print("Target label name:", newsgroups.target_names[newsgroups.target[i]])
    print("="*80)

# カテゴリごとのサンプル数を表示
unique, counts = np.unique(newsgroups.target, return_counts=True)
for category, count in zip(newsgroups.target_names, counts):
    print(f"{category}: {count}")

# 特定のカテゴリを選択してデータセットを分割
categories = ['alt.atheism', 'comp.graphics', 'soc.religion.christian', 'sci.med']
remove = ('headers', 'footers', 'quotes')
twenty_train = fetch_20newsgroups(subset='train', remove=remove, categories=categories)
twenty_test = fetch_20newsgroups(subset='test', remove=remove, categories=categories)

# CountVectorizerを使用してBoW特徴量を作成
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(twenty_train.data)
X_test_counts = count_vectorizer.transform(twenty_test.data)

# LinearSVCモデルを訓練
model = LinearSVC()
model.fit(X_train_counts, twenty_train.target)

# テストデータで予測
predicted = model.predict(X_test_counts)

# 精度を表示
accuracy = accuracy_score(twenty_test.target, predicted)
print("Accuracy:", accuracy)

# 詳細な分類レポートを表示
print("\nClassification Report:")
print(classification_report(twenty_test.target, predicted, target_names=categories))
