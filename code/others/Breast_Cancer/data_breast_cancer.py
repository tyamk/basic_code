from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer()

X = data.data
X = pd.DataFrame(data.data, columns=data.feature_names)
info = pd.DataFrame.info(X)
# print(info) # 569行30列
# print(X.head()) # 0: 悪性(M) 1: 良性(B)
y = data.target
y = pd.DataFrame(data.target, columns=['species'])
y_info = pd.DataFrame.info(y)
# print(y_info) # 569行1列
# print(y.head())

# １０項目の特徴量を選択
X = X.iloc[:, :10] # 平均値のみを抽出
# print(X.head(10))

