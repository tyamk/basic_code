import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
# データ生成   
np.random.seed(42)
data = {
    'x2': np.arange(10),
    'y2': np.arange(10) * np.random.rand(10)
} 
df = pd.DataFrame(data)
# 2行3列のsubplotを作成
fig, axes = plt.subplots(nrows = 2, ncols = 3, tight_layout = True, squeeze = False)
# 散布図
df.plot.scatter(x = 'x2', y = 'y2', ax = axes[0, 0], color = 'r') # df.plot.~ でプロット .plt.~ でプロット
# ヒストグラム
df.plot.hist(y = 'y2', ax = axes[0, 1], color = 'g', bins = 5)
# 棒グラフ
df.plot.bar(x = 'x2', y = 'y2', ax = axes[0, 2], color = 'b')
# 折れ線グラフ
df.plot(x = 'x2', y = 'y2', ax = axes[1, 0], color = 'y')
# 箱ひげ図
df[['y2']].plot.box(ax=axes[1, 1], boxprops=dict(color='c'))
#  円グラフ
df.plot.pie(y = 'y2', ax = axes[1, 2], colors = ['m', 'y', 'c', 'r', 'g'], legend=False)
plt.show()


