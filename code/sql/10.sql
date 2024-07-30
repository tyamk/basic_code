--3 つのデータをもつ prefectures テーブルを作成し、prefectures 
--テーブルのすべてのデータを表示するクエリを作成
create table 
    prefectures(id, name, area, population);
insert into
    prefectures 
values
        (1, '北海道', 83423, 5225000),
        (2, '青森県', 9645, 1238000),
        (3, '岩手県', 15275, 1211000);
select
    *
from 
    prefectures;

/*
-- テーブルの作成
CREATE TABLE
    prefectures(id, name, area, population);

-- データの追加
INSERT INTO
    prefectures
VALUES
    (1, '北海道', 83423, 5225000),
    (2, '青森県', 9645, 1238000),
    (3, '岩手県', 15275, 1211000);

-- データの表示
SELECT
    *
FROM
    prefectures;
*/