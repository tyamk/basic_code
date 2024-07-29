import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix # 適合率、再現率、F1値、サポート、TP, FP, FN, TN

# データの生成
np.random.seed(42)

# 100日のデータ
dates = pd.date_range(start='20210101', periods=100)
# 気温のデータ
temps = np.random.uniform(low=-10, high = 10, size = 100)
# 雪が降ったかどうか、「雪の日（0）」か「雪の日でない日（1）」かを表すデータ
snow_days = (temps > 1.0).astype(int)
print(snow_days)
#データフレームの作成(名前変更)
data = pd.DataFrame({
    'Date':dates,
    'Temperature':temps,
    'Snow':snow_days
})
print(data.head())
print(data.info())
"""
[0 1 1 1 0 0 0 1 1 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 1 0 1 0 0 1 1 1 0
 0 1 0 0 0 0 1 0 1 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 1 0 1 1 0 0 1
 1 1 1 0 0 0 1 1 0 0 0 0 1 1 1 0 0 1 1 1 1 0 0 0 0 0]
        Date  Temperature  Snow
0 2021-01-01    -2.509198     0
1 2021-01-02     9.014286     1
2 2021-01-03     4.639879     1
3 2021-01-04     1.973170     1
4 2021-01-05    -6.879627     0
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 3 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   Date         100 non-null    datetime64[ns]
 1   Temperature  100 non-null    float64
 2   Snow         100 non-null    int32
dtypes: datetime64[ns](1), float64(1), int32(1)
memory usage: 2.1 KB
None
"""
# 訓練データとテストデータに分割(7:3)
X = data[['Temperature']]
y = data['Snow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ロジスティック回帰モデルの作成
model = LogisticRegression()
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

# モデルの評価
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("classification_report:\n",classification_report(y_test, y_pred))

#-10度から10度の範囲で400個の等間隔のデータポイントを生成し、1列の2次元配列に変形
x_fit = np.linspace(-10, 10, 400).reshape(-1, 1)
# print(x_fit)
# 生成したデータポイントに対するクラス1（雪の日）の予測確率を計算
y_fit_prob = model.predict_proba(x_fit)[:, 1]
# print(y_fit_prob)

# 可視化
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='red', label='Test data')
plt.plot(x_fit, y_fit_prob, color='green', linestyle='--', label='Prediction Probability')
plt.axhline(0.5, color='gray', linestyle='--')  # 0.5の決定境界を追加
plt.xlabel('Temperature')
plt.ylabel('Probability of Snow (0: Snow, 1: No Snow)')
plt.title('Logistic Regression: Temperature vs Snow (Threshold: 1.0°C)')
plt.legend()
plt.grid(True)
plt.show()

data.head()

# 確率
while True:
    temp_input = input("気温を入れてください。(qで停止)： ")
    if temp_input == 'q':
        break
    try:
        specific_temp = float(temp_input)
        specific_temp_array = np.array([[specific_temp]])
        show_probability = model.predict_proba(specific_temp_array)[:, 0][0]
        print(f"気温が{specific_temp}°Cのとき、雪が降る確率は {show_probability* 100:.2f} %です。")
    except ValueError:
        print("数値を入力してください。")

"""
ロジスティック回帰
ロジスティック回帰は、カテゴリカルデータの分類に用いられる回帰分析手法である。シグモイド関数を用いて、確率を出力し、閾値を設定してクラスを予測する。バイナリ分類とマルチクラス分類に対応している。
データが各クラスに所属する確率を計算する。教師あり学習の回帰（分類）問題に適用される。二値分類だと、ある事象が起こる/ある事象が起こらないという２つに分類できる。
ex. 今の気温は２度。明日は雪が降るのか、降らないのか？
"""