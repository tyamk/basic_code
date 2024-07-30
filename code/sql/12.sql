-- ここにテーブルを作成するクエリを記述してください
--not nullを書くとnot null制約
create table prefectures (id integer not null, name /*string x 文字列はtext*/ text, area integer, population integer);
-- データの追加
INSERT INTO
    prefectures
VALUES
    (1, '北海道', 83423, 5225000),
    (2, '青森県', 9645, 1238000),
    (3, '岩手県', 15275, 1211000);

-- 次のクエリを実行するとエラーが出ることを確認してみましょう
-- (提出時にはコメントアウトしてください)
--INSERT INTO prefectures VALUES (NULL, '宮城県', 7282, 2302000);

-- データの表示
SELECT * FROM prefectures;

/*
-- ここにテーブルを作成するクエリを記述してください
CREATE TABLE
    prefectures(id INTEGER NOT NULL, name TEXT, area INTEGER, population INTEGER);
-- データの追加
INSERT INTO
    prefectures
VALUES
    (1, '北海道', 83423, 5225000),
    (2, '青森県', 9645, 1238000),
    (3, '岩手県', 15275, 1211000);

-- 次のクエリを実行するとエラーが出ることを確認してみましょう
-- (提出時にはコメントアウトしてください)
-- INSERT INTO prefectures VALUES (NULL, '宮城県', 7282, 2302000);

-- データの表示
SELECT * FROM prefectures;
*/