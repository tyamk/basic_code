from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# データ
data = load_wine()
# 'alcohol'と'color_intensity'の列番号を取得
x3 = data.data[:, [0]] # 'alcohol'列のインデックス: 0
y3 = data.data[:, [9]] # 'color_intensity'列のインデックス: 9
# データフレームに変換（変換後にデータの中身を確認できる）
x3_df = pd.DataFrame(x3, columns=['alcohol'])
y3_df = pd.DataFrame(y3, columns=['color_intensity'])
# 2つのDataFrameを横に結合
df = pd.concat([x3_df, y3_df], axis=1)
# df = pd.DataFrame(x3_df, y3_df) # これはエラー
# データの中身を確認
"""print(x3_df.shape)
x3_info = pd.DataFrame.info(x3_df)
y3_info = pd.DataFrame.info(y3_df)
print(x3_info)
print(y3_info)
print(x3_df.head())
print(y3_df.head())
print(df.head())"""
# ---------------------------------------------------------
# 3行1列のsubplotを作成
fig, axes = plt.subplots(nrows = 1, ncols = 3, tight_layout = True, squeeze = False)
# 散布図
df.plot.scatter(x='alcohol', y='color_intensity', ax=axes[0, 0], color='b')
# ヒストグラム
df.plot.hist(y='color_intensity', ax=axes[0, 1], color='g', bins=5)
# ヒストグラム2
df.plot.hist(y='alcohol', ax=axes[0, 2], color='r', bins=5)
plt.show()
