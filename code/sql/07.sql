--prefectures テーブル (都道府県の情報)
--cities テーブル (市町村+23区の情報)
--wards テーブル (23区以外の区の情報)
--wards テーブルに含まれるすべての区について、
--都道府県名、市町村名、区名を表示するクエリを作成
--カラム名は、それぞれ「都道府県名」「市町村名」「区名」
select
    prefectures.name as '都道府県名',
    cities.name as '市町村名',
    wards.name as '区名'
from 
    prefectures --大きいところから指定するということ？
    --wards
    inner join cities
        on prefecture_id = prefectures.id
        inner join wards
            on wards.city_id = cities.id;

/*
SELECT
    prefectures.name AS '都道府県名',
    cities.name AS '市町村名',
    wards.name AS '区名'
FROM
    prefectures
    INNER JOIN cities
        ON prefectures.id = cities.prefecture_id
        INNER JOIN wards
            ON cities.id = wards.city_id;
*/