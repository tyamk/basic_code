-- ここにテーブルを作成するクエリを記述してください
--各カラムのデータ型を指定する。
--id カラムを主キーに設定し、AUTOINCREMENT を設定する。
--autoincrement 番号が1から自動的に始まる
create table 
    posts(
        id integer not null primary key /*autoincrement*/ autoincrement,
        name text,
        body text
    );
-- データの追加
INSERT INTO
    posts(name, body)
VALUES
    ('アルル', 'こんにちは'),
    ('アルル', '今日は暑いですね'),
    ('イルル', 'アルルさん、こんにちは');

-- データの表示
SELECT * FROM posts;

"""
-- ここにテーブルを作成するクエリを記述してください
CREATE TABLE
    posts(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, body TEXT);

-- データの追加
INSERT INTO
    posts(name, body)
VALUES
    ('アルル', 'こんにちは'),
    ('アルル', '今日は暑いですね'),
    ('イルル', 'アルルさん、こんにちは');

-- データの表示
SELECT * FROM posts;
"""