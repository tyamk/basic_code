-- ここにテーブルを作成するクエリを記述してください
create table 
    posts(
        id integer not null primary key autoincrement,
        name text default '名無し',--nameカラムの初期値を名無し
        body text
    );
-- データの追加
INSERT INTO
    posts(body) 
VALUES
    ('こんにちは'),
    ('今日は暑いですね');
INSERT INTO
    posts(name, body)
VALUES
    ('アルル', 'こんにちは');

-- データの表示
SELECT * FROM posts;

"""
-- ここにテーブルを作成するクエリを記述してください
CREATE TABLE
    posts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT DEFAULT '名無し',
        body TEXT
    );

-- データの追加
INSERT INTO
    posts(body) 
VALUES
    ('こんにちは'),
    ('今日は暑いですね');
INSERT INTO
    posts(name, body)
VALUES
    ('アルル', 'こんにちは');

-- データの表示
SELECT * FROM posts;
"""