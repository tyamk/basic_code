import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups

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

categories = ['alt.atheism', 'comp.graphics','soc.religion.christian', 'sci.med']
remove  = ('headers', 'footers', 'quotes')
twenty_train = fetch_20newsgroups(subset='train', remove=remove, categories=categories)
twenty_test = fetch_20newsgroups(subset='test', remove=remove, categories=categories)
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(twenty_train.data)
X_test_counts = count_vectorizer.transform(twenty_test.data)

model = LinearSVC()
model.fit(X_train_counts, twenty_train.target)
predicted = model.predict(X_test_counts)
print("Accuracy:", np.mean(predicted == twenty_test.target))
"""
Target label: 4
Target label name: comp.sys.mac.hardware
================================================================================
alt.atheism: 799
comp.graphics: 973
comp.os.ms-windows.misc: 985
comp.sys.ibm.pc.hardware: 982
comp.sys.mac.hardware: 963
comp.windows.x: 988
misc.forsale: 975
rec.autos: 990
rec.motorcycles: 996
rec.sport.baseball: 994
rec.sport.hockey: 999
sci.crypt: 991
sci.electronics: 984
sci.med: 990
sci.space: 987
soc.religion.christian: 997
talk.politics.guns: 910
talk.politics.mideast: 940
talk.politics.misc: 775
talk.religion.misc: 628
"""
"""
tf: 単語の出現頻度
idf: 逆文書頻度、文書が多いと少なくなる
"""