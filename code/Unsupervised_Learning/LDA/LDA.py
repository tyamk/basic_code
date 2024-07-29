from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# removeで本文以外の情報を取り除く
data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

# データフレームに変換（必要なデータだけ取り出して変換）
df = pd.DataFrame({'text': data.data, 'target': data.target})
print(df.head())

max_features = 1000

# 文書データをベクトルに変換
tf_vectorizer = CountVectorizer(max_features=max_features, stop_words = 'english')
tf = tf_vectorizer.fit_transform(data.data)

n_topics = 20
model = LatentDirichletAllocation(n_components=n_topics)
model.fit(tf)

print(model.components_)
print(model.transform(tf))

# トピックごとに上位の単語を表示する関数
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# トピック数と上位単語数を設定
n_topics = 20
no_top_words = 10

# トピックごとに上位の単語を表示
tf_feature_names = tf_vectorizer.get_feature_names_out()
display_topics(model, tf_feature_names, no_top_words)

# トピック分布をグラフにする
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(5, 4, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx + 1}',
                     fontdict={'fontsize': 15})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=12)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=20)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

# トピックごとに上位の単語をグラフに表示
plot_top_words(model, tf_feature_names, no_top_words, 'Top words per topic in LDA model')


"""
次元削減の手法。文書のモデル化に適している。
単語についてランダムにトピックを割り当てる→トピック確率を計算する→単語確率を計算する→確率をもとに割り当てる→収束条件まで繰り返す
"""