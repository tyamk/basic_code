import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names) # feature_names:特徴量の名前を指定
y = pd.DataFrame(data.target, columns=['species'])
print(X.head())
print(y.head())
df = pd.concat([X, y], axis=1) # axis=1は列方向に結合 axis=0は行方向に結合
print(df.head())