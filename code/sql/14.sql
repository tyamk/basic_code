-- 外部キーの参照整合性制約を有効にする
-- region_id カラムを、region テーブルの id カラムを参照する外部キーに設定
--外部キーの参照整合性制約が設定
PRAGMA foreign_keys = ON;

-- ここにテーブルを作成するクエリを記述してください
--create table テーブル1(カラムの設定, foreigin key(カラム名) preferences テーブル2(カラム名));
--create table prefectures(id, foreign key(id) references regions(regions.id))
--外部キーの設定は、CREATE TABLE 文においてカラムを設定した後
create table regions
    (id integer not null primary key, name text);
create table prefectures
    (id integer not null primary key, name text, region_id int, foreign key(region_id) references regions(id));


-- データの追加
INSERT INTO regions     VALUES (1, '北海道'), (2, '東北'), (3, '関東');
INSERT INTO prefectures VALUES (1, '北海道', 1), (2, '青森県', 2), (3, '岩手県', 2);

-- 次のクエリを実行するとエラーが出ることを確認してみましょう
-- (提出時にはコメントアウトしてください)
-- DELETE FROM regions WHERE id = 1;
-- INSERT INTO prefectures VALUES (100, 'アルゴ県', 100);

-- テーブルを結合してデータを表示
SELECT
    prefectures.name AS '都道府県名',
    regions.name AS '地方名'
FROM
    prefectures
    LEFT OUTER JOIN regions ON prefectures.region_id = regions.id;

"""
-- 外部キーの参照整合性制約を有効にする
PRAGMA foreign_keys = ON;

-- ここにテーブルを作成するクエリを記述してください
CREATE TABLE
    regions(id INTEGER NOT NULL PRIMARY KEY, name TEXT);

CREATE TABLE
    prefectures(
        id INTEGER NOT NULL PRIMARY KEY, name TEXT, region_id INT,
        FOREIGN KEY (region_id) REFERENCES regions(id)
    );

-- データの追加
INSERT INTO regions     VALUES (1, '北海道'), (2, '東北'), (3, '関東');
INSERT INTO prefectures VALUES (1, '北海道', 1), (2, '青森県', 2), (3, '岩手県', 2);

-- 次のクエリを実行するとエラーが出ることを確認してみましょう
-- (提出時にはコメントアウトしてください)
-- DELETE FROM regions WHERE id = 1;
-- INSERT INTO prefectures VALUES (100, 'アルゴ県', 100);

-- テーブルを結合してデータを表示
SELECT
    prefectures.name AS '都道府県名',
    regions.name AS '地方名'
FROM
    prefectures
    LEFT OUTER JOIN regions ON prefectures.region_id = regions.id;
"""