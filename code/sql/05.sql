--prefectures_id, cities_id
--「府中市」という名前の市が存在する都道府県の名前をすべて、
--「都道府県名」というカラム名で表示するクエリを作成
--→citiesにある府中市
select
    prefectures.name as '都道府県名'
from 
    prefectures
    inner join cities
        on prefectures.id = cities.prefecture_id
where
    cities.name = '府中市';

/*
SELECT
    prefectures.name AS '都道府県名'
FROM
    prefectures
    INNER JOIN cities
        ON prefectures.id = cities.prefecture_id
WHERE
    cities.name = '府中市';
*/
