import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# アンスコムのデータセットをロード
anscombe = sns.load_dataset("anscombe")

# プロットの設定
fig, axes = plt.subplots(nrows=2, ncols=2, tight_layout=True, squeeze=False, figsize=(10, 10))

# データセットのリスト
datasets = ['I', 'II', 'III', 'IV']
colors = ['red', 'blue', 'green', 'purple']

for ax, dataset, color in zip(axes.flatten(), datasets, colors):
    data = anscombe[anscombe['dataset'] == dataset]
    ax.plot(data['x'], data['y'], 'o', color=color, label=f'Dataset {dataset}')
    
    # 線形回帰モデルの作成
    model = LinearRegression()
    model.fit(data[['x']], data['y'])
    
    # 回帰直線の計算
    x_vals = np.array(ax.get_xlim()).reshape(-1, 1)
    y_vals = model.predict(x_vals)
    
    # 回帰直線のプロット
    ax.plot(x_vals, y_vals, '--', color='black', label='Linear Regression')
    
    ax.set_title(f'Dataset {dataset}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

# グラフの表示
plt.show()

"""
(I): 一直線上に並んでいる。線形回帰は適している。
(Ⅱ): 曲線上に並んでいるため、線形回帰は適していない。
(Ⅲ): 外れ値があるため、前処理で外れ値を除去する必要がある。もしくは、外れ値に対して強いモデルを選択する必要がある。
(Ⅳ): y軸方向に一直線上に並んでいるため、線形回帰は適していない。また、外れ値がある。
"""