import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import MeCab

# ニュース記事のタイトルとカテゴリのデータ
data = {
    'Title':[
        '新しいSF映画が公開', '宇宙飛行士がISSから帰還',
        '大ヒット映画が記録を更新', 'NASAが新しい惑星を発見',
        '受賞歴のある監督が新作を公開', '火星探査車が新しい写真を送信',
        '批評家絶賛の映画が初公開', 'SpaceXが新しいロケットを打ち上げ',
        '映画祭のハイライト', 'ブラックホールに関する新しい研究'
    ],
    'Category':[
        '映画', '宇宙', '映画', '宇宙', '映画', '宇宙', '映画', '宇宙', '映画', '宇宙'
    ]
}

# データフレームの作成
df = pd.DataFrame(data)

# MeCabのセットアップ
mecab = MeCab.Tagger("-Owakati")

# 名詞と動詞のみを抽出する関数
def extract_nouns_verbs(text):
    parsed = mecab.parse(text)
    words = []
    node = mecab.parseToNode(text)
    while node:
        if node.feature.startswith('名詞') or node.feature.startswith('動詞'):
            words.append(node.surface)
        node = node.next
    return ' '.join(words)

# 名詞と動詞のみを抽出して新しい列を作成
df['Filtered_Title'] = df['Title'].apply(extract_nouns_verbs)

# デバッグのために各タイトルから抽出された名詞と動詞を表示
print(df[['Title', 'Filtered_Title']])

# Filtered_Titleが空でないことを確認
if df['Filtered_Title'].str.strip().str.len().sum() == 0:
    raise ValueError("Filtered_Title列が空です。形態素解析が正しく行われているか確認してください。")

# テキストデータをベクトル化
vectorizer = TfidfVectorizer()
X_counts = vectorizer.fit_transform(df['Filtered_Title'])

# BoWの特徴量名を取得
feature_names = vectorizer.get_feature_names_out()

# BoWのベクトルを表示
print("BoWのベクトル:")
print(X_counts.toarray())

# 各単語とそのインデックスを表示
print("\n各単語とそのインデックス:")
for word, index in zip(feature_names, range(len(feature_names))):
    print(f"{index}: {word}")

# タイトルとカテゴリを分ける
X = df['Filtered_Title']
y = df['Category']

# データを訓練データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# テキストデータをベクトル化
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# ナイーブベイズ分類器を訓練
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# テストデータで予測
y_pred = clf.predict(X_test_counts)

# 精度を評価
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# 新しいタイトルのカテゴリを予測
new_title = ["復活した名作アクションに感動"]
new_title_counts = vectorizer.transform(new_title)
new_pred = clf.predict(new_title_counts)
print('新しいタイトルのカテゴリ:', new_pred[0])
"""
                 Title    Filtered_Title
0           新しいSF映画が公開          SF 映画 公開
1        宇宙飛行士がISSから帰還      宇宙 飛行 ISS 帰還
2         大ヒット映画が記録を更新      ヒット 映画 記録 更新
3        NASAが新しい惑星を発見        NASA 惑星 発見
4       受賞歴のある監督が新作を公開    受賞 ある 監督 新作 公開
5       火星探査車が新しい写真を送信       火星 探査 写真 送信
6         批評家絶賛の映画が初公開     批評 絶賛 映画 初 公開
7  SpaceXが新しいロケットを打ち上げ  SpaceX ロケット 打ち上げ
8            映画祭のハイライト          映画 ハイライト
9     ブラックホールに関する新しい研究   ブラック ホール 関する 研究
BoWのベクトル:
[[0.         0.         0.70881761 0.         0.         0.
  0.         0.         0.         0.         0.52716856 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.46869063 0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.5        0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.5        0.5        0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.5       ]
 [0.         0.         0.         0.         0.         0.
  0.53938158 0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.35665464 0.53938158 0.         0.
  0.         0.         0.         0.53938158 0.         0.
  0.        ]
 [0.         0.57735027 0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.57735027 0.         0.
  0.         0.         0.         0.         0.         0.57735027
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.         0.46864588 0.
  0.         0.         0.         0.         0.34854576 0.
  0.46864588 0.         0.         0.         0.         0.
  0.         0.46864588 0.         0.         0.         0.
  0.46864588 0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.5
  0.         0.         0.         0.         0.         0.
  0.5        0.         0.         0.         0.5        0.
  0.         0.         0.         0.         0.5        0.
  0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.43008419 0.
  0.         0.         0.         0.         0.         0.5782804
  0.         0.         0.38237566 0.         0.         0.
  0.         0.         0.5782804  0.         0.         0.
  0.        ]
 [0.         0.         0.         0.57735027 0.         0.
  0.         0.         0.         0.57735027 0.         0.
  0.         0.         0.         0.         0.57735027 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.         0.         0.83413787
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.55155599 0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.5        0.5        0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.5        0.         0.         0.         0.5
  0.        ]]

各単語とそのインデックス:
0: iss
1: nasa
2: sf
3: spacex
4: ある
5: ハイライト
6: ヒット
7: ブラック
8: ホール
9: ロケット
10: 公開
11: 写真
12: 受賞
13: 宇宙
14: 帰還
15: 惑星
16: 打ち上げ
17: 批評
18: 探査
19: 新作
20: 映画
21: 更新
22: 火星
23: 発見
24: 監督
25: 研究
26: 絶賛
27: 記録
28: 送信
29: 関する
30: 飛行
Accuracy: 0.3333333333333333
              precision    recall  f1-score   support

          宇宙       0.00      0.00      0.00         2
          映画       0.33      1.00      0.50         1

    accuracy                           0.33         3
   macro avg       0.17      0.50      0.25         3
weighted avg       0.11      0.33      0.17         3

新しいタイトルのカテゴリ: 映画
"""