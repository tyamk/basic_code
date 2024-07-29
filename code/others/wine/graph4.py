import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from pandas.plotting import scatter_matrix

# ---------------------------------------------------------
# データ
data = load_wine()
# データフレーム生成
df_X = pd.DataFrame(data.data, columns=data.feature_names)
# データの中身を確認
# print(df_X.head())
"""
[5 rows x 13 columns]
   alcohol  malic_acid   ash  ...   hue  od280/od315_of_diluted_wines  proline
0    14.23        1.71  2.43  ...  1.04                          3.92   1065.0
1    13.20        1.78  2.14  ...  1.05                          3.40   1050.0
2    13.16        2.36  2.67  ...  1.03                          3.17   1185.0
3    14.37        1.95  2.50  ...  0.86                          3.45   1480.0
4    13.24        2.59  2.87  ...  1.04                          2.93    735.0
"""
# 目的変数をデータフレームに変換
df_y = pd.DataFrame(data.target, columns=['target'])
# データの中身を確認
# print(df_y.head())
""""
   target
0             0
1             0
2             0
3             0
4             0

"""
# データフレームを横に結合
df = pd.concat([df_X, df_y], axis=1)
# データの中身を確認
# print(df.head())
"""
[5 rows x 14 columns]
   alcohol  malic_acid   ash  ...  od280/od315_of_diluted_wines  proline  target
0    14.23        1.71  2.43  ...                          3.92   1065.0       0
1    13.20        1.78  2.14  ...                          3.40   1050.0       0
2    13.16        2.36  2.67  ...                          3.17   1185.0       0
3    14.37        1.95  2.50  ...                          3.45   1480.0       0
4    13.24        2.59  2.87  ...                          2.93    735.0       0
"""
# ---------------------------------------------------------
# グラフ描画 
# 2行1列のsubplotを作成
fig, axes = plt.subplots(nrows = 1, ncols = 2, tight_layout = True, squeeze = False)
# 'alcohol'のデータを可視化
# ヒストグラム
df.plot.hist(y='alcohol', ax=axes[0, 0], color='b', bins=10)
# 箱ひげ図
df[["alcohol"]].plot.box(ax=axes[0, 1], boxprops=dict(color='c')) # 全ての行の'alcohol'列を指定
plt.show()
# ---------------------------------------------------------
# 集計機能
# 相関関係を確認
print(df.corr()) # 1に近いほど正の相関、-1に近いほど負の相関
"""
                               alcohol  malic_acid  ...   proline    target
alcohol                       1.000000    0.094397  ...  0.643720 -0.328222
malic_acid                    0.094397    1.000000  ... -0.192011  0.437776
ash                           0.211545    0.164045  ...  0.223626 -0.049643
alcalinity_of_ash            -0.310235    0.288500  ... -0.440597  0.517859
magnesium                     0.270798   -0.054575  ...  0.393351 -0.209179
total_phenols                 0.289101   -0.335167  ...  0.498115 -0.719163
flavanoids                    0.236815   -0.411007  ...  0.494193 -0.847498
nonflavanoid_phenols         -0.155929    0.292977  ... -0.311385  0.489109
proanthocyanins               0.136698   -0.220746  ...  0.330417 -0.499130
color_intensity               0.546364    0.248985  ...  0.316100  0.265668
hue                          -0.071747   -0.561296  ...  0.236183 -0.617369
od280/od315_of_diluted_wines  0.072343   -0.368710  ...  0.312761 -0.788230
proline                       0.643720   -0.192011  ...  1.000000 -0.633717
target                       -0.328222    0.437776  ... -0.633717  1.000000
"""
# 統計情報
print(df.describe())
"""
[14 rows x 14 columns]
          alcohol  malic_acid  ...      proline      target
count  178.000000  178.000000  ...   178.000000  178.000000
mean    13.000618    2.336348  ...   746.893258    0.938202
std      0.811827    1.117146  ...   314.907474    0.775035
min     11.030000    0.740000  ...   278.000000    0.000000
25%     12.362500    1.602500  ...   500.500000    0.000000
50%     13.050000    1.865000  ...   673.500000    1.000000
75%     13.677500    3.082500  ...   985.000000    2.000000
max     14.830000    5.800000  ...  1680.000000    2.000000

count: 有効な（欠損値ではない）エントリの数。
mean: 平均値。
std: 標準偏差（データの散らばり具合）。
min: 最小値。
25%: 第1四分位数（25パーセンタイル）。
50%: 中央値（50パーセンタイル、メディアン）。
75%: 第3四分位数（75パーセンタイル）。
max: 最大値。
"""
# 散布図の行列
_ = scatter_matrix(df, figsize=(15, 15))
# 各軸のフォントサイズを設定
for ax in _.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize=6)
    ax.set_ylabel(ax.get_ylabel(), fontsize=6)
    ax.set_title(ax.get_title(), fontsize=6)
plt.show()
# 限定された特徴量で散布図行列を作成
_ = scatter_matrix(df.iloc[:, [0, 9, -1]])
for ax in _.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize=6)
    ax.set_ylabel(ax.get_ylabel(), fontsize=6)
    ax.set_title(ax.get_title(), fontsize=6)
plt.show()