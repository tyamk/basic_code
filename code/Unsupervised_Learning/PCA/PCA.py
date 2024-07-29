from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
# print(data)
n_components  = 2
model = PCA(n_components=n_components)
model = model.fit(data.data)

print(model.transform(data.data))
# 相関のないデータに対して、主成分の寄与率が変わらないときはPCA以外のアルゴリズムを見る必要がある


"""
PCA（主成分分析）は、次元削減を行い、たくさんの変数を持つデータを、特徴を保ちながら少数の変数で表現する。
寄与率とは、固有値を固有値の総和でわり。主成分の重要度を割合で示したもの。
重要な変数のみを選択できる。このとき、方向と重要度によって選択する。固有ベクトルと固有値に関連する。
分散共分散行列→固有値ベクトルと固有値を求める→主成分方向にデータを表現する
"""