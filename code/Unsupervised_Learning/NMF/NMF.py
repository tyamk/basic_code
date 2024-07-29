from sklearn.decomposition import NMF
from sklearn.datasets._samples_generator import make_blobs

centers = [[5, 10, 5], [10, 4, 10], [6, 8, 8]]
# データセットとラベルを生成
V, labels = make_blobs(centers=centers)

# データポイントの行列を出力
print("Data points (V):")
print(V)

# クラスタラベルを出力
print("Cluster labels:")
print(labels)
n_components = 2
model = NMF(n_components=n_components)
model.fit(V)

W = model.transform(V)
H = model.components_
print(W)
print(H)

"""
NMFは行列分解手法の1つであり、元の行列の成分がすべて非負の場合にのみ適用できる。
元の行列が非負である→行列の要素が非負である→洗剤意味空間の各次元が直交するという制約を持たない。
２つの行列を分解し、次元削減を行う。たとえば、V(n行d列)を分解して、W(n行r列)とH(r行d列)にする。
"""