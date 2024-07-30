-- 外部キーの参照整合性制約を有効にする
PRAGMA foreign_keys = ON;
-- ここにテーブ各カラムのデータ型を指定する。
--id カラムを主キーに設定し、AUTOINCREMENT を設定する。
--posts テーブルの user_id カラムを、users テーブルの id カラムを参照する外部キーに設定する。
--このとき、外部キーの参照整合性制約が設定されるようにする。
--次の CHECK 制約を設定する。
--posts テーブルの body カラムの文字列の長さは 1 以上 140 以下
create table 
    posts(
        id integer /*not null*/ primary key autoincrement,
        user_id integer,
        body text,
        --posts(foreign key(id) references users(id)) postsはすでに記入済み
        --外部参照
        foreign key(user_id) references users(id),
        --check(a <= length(body) <= 140)
        check(1 <= length(body) and length(body) <= 140)
    );
-- データの追加
INSERT INTO
    posts(user_id, body)
VALUES
    (1, 'おはようございます'),
    (1, '今日は暑いですね'),
    (3, 'あ');

-- テーブルを結合してデータを表示
SELECT
    users.name AS 'ユーザ名',
    posts.body AS '投稿内容'
FROM
    users
    INNER JOIN posts
        ON posts.user_id = users.id;
"""
-- 外部キーの参照整合性制約を有効にする
PRAGMA foreign_keys = ON;

-- ここにテーブルを作成するクエリを記述してください
CREATE TABLE
    posts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        body TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id),
        CHECK(1 <= LENGTH(body) AND LENGTH(body) <= 140)
    );
    
-- データの追加
INSERT INTO
    posts(user_id, body)
VALUES
    (1, 'おはようございます'),
    (1, '今日は暑いですね'),
    (3, 'あ');

-- テーブルを結合してデータを表示
SELECT
    users.name AS 'ユーザ名',
    posts.body AS '投稿内容'
FROM
    users
    INNER JOIN posts
        ON posts.user_id = users.id;
"""