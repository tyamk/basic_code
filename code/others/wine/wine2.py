import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd
# k-means法で３つのクラスタに分類する
from sklearn.cluster import KMeans

data = load_wine()
X = data.data
X = pd.DataFrame(X, columns=data.feature_names)
info = pd.DataFrame.info(X)
# print(info) # 178行13列
# print(X.head()) 
# 列名を表示して確認
# print(X.columns)
# 'alcohol'と'color_intensity'の列番号を取得
# alcohol_index = X.columns.get_loc('alcohol')
# color_intensity_index = X.columns.get_loc('color_intensity')
# print(f"'alcohol'列のインデックス: {alcohol_index}")
# print(f"'color_intensity'列のインデックス: {color_intensity_index}")
"""Index(['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue',
       'od280/od315_of_diluted_wines', 'proline'],
      dtype='object')
'alcohol'列のインデックス: 0
'color_intensity'列のインデックス: 9"""
# ---------------------------------------------------------
# alcohol[0]とcolor_intensity[9]をXに代入する
X = X.iloc[:, [0, 9]] # alcohol, color_intensityのみを抽出
# print(X.head())

n_clusters = 3
model = KMeans(n_clusters=n_clusters, random_state=42)
# クラスタリングを実行
pred = model.fit_predict(X)
# 結果を２軸のグラフで表示する
colors = ['blue', 'red', 'green']
for i in range(n_clusters):
    cluster = X[pred == i]
    plt.scatter(cluster.iloc[:, 0], cluster.iloc[:, 1], c=colors[i], label=f'Label {i+1}')

centroids = model.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='x', c='black', label='Centroids')  

# グラフの描画
plt.xlabel('alcohol')
plt.ylabel('color_intensity')
plt.title("K-means Clustering of Wine Data")
plt.legend() 
plt.show()