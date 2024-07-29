import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# データの読み込み
data = load_iris()
X = data.data

# WCSSのリストを初期化
wcss = []

# クラスタ数を1から10まで変えてKMeansを適用
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# WCSSをプロット
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 最適なクラスタ数を3に設定してKMeansクラスタリングを再実行
n_clusters = 3
model = KMeans(n_clusters=n_clusters, random_state=42)
model.fit(X)

# クラスタのラベルと重心を表示
print("Cluster labels:")
print(model.labels_)

print("\nCluster centers:")
print(model.cluster_centers_)


"""
似た者同士のデータをクラスタとして組み合わせる。
適当な点をクラスタの数だけ選ぶ→重心をデータ点の所属するクラスタとする→平均値を計算し、重心とする→計算ステップ数の上限に達するまで続ける.
WCSSは小さいほうがいい。
"""
