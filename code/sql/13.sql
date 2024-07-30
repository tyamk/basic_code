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

/*
-- ここにテーブルを作成するクエリを記述してください
CREATE TABLE
    users(
        id TEXT NOT NULL PRIMARY KEY,
        name TEXT UNIQUE,
        email TEXT UNIQUE,
        age INTEGER
    );
-- データの追加
INSERT INTO
    users
VALUES
    ('00001', 'arara234', 'arara.qa@algo-method.com', 38),
    ('00002', 'ir1ri_kame', 'iriri.yi@algo-method.com', 14),
    ('00003', 'oruru', 'oruru.td@algo-method.com', 25);

-- 次のクエリを実行するとそれぞれエラーが出ることを確認してみましょう
-- (提出時にはコメントアウトしてください)
-- INSERT INTO users VALUES (NULL, 'oruru_', 'oruru.pq@algo-method.com', 16);
-- INSERT INTO users VALUES ('00001', 'oruru_', 'oruru.pq@algo-method.com', 16);
-- INSERT INTO users VALUES ('00004', 'oruru', 'oruru.pq@algo-method.com', 16);
-- INSERT INTO users VALUES ('00004', 'oruru_', 'oruru.td@algo-method.com', 16);

-- データの表示
SELECT * FROM users;
*/