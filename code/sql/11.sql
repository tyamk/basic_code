-- ここにテーブルを作成するクエリを記述してください
--text型、integer型、real型(浮動小数点)
--データ型を指定している
create table prefectures(id integer, name text, area integer, population integer);
-- データの追加と表示
INSERT INTO
    prefectures
VALUES
    (1, '北海道', 83423, 5225000),
    (2, '青森県', 9645, 1238000),
    (3, '岩手県', 15275, 1211000);
SELECT * FROM prefectures;

/*
-- ここにテーブルを作成するクエリを記述してください
CREATE TABLE
    prefectures(id INTEGER, name TEXT, area INTEGER, population INTEGER);
    
-- データの追加と表示
INSERT INTO
    prefectures
VALUES
    (1, '北海道', 83423, 5225000),
    (2, '青森県', 9645, 1238000),
    (3, '岩手県', 15275, 1211000);
SELECT * FROM prefectures;
*/