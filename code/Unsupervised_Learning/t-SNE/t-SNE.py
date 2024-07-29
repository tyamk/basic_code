from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
data = load_digits()
n_components = 2
model = TSNE(n_components=n_components)
print(model.fit_transform(data.data))

"""
高次元のデータを２次元や３次元に次元削減する。
xi, xjの類似度のガウス分布を利用した類似度で表す→xiと同じ数のyiをランダム配置→なるべく類似度の分布が同じになるようにyiを更新→繰り返す
"""