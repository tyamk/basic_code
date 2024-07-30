/*select 
    name as '都道府県名',
    region_id as '地方ID',
from prefecture*/
select 
    prefectures.name as '都道府県名',
    regions.id as '地方ID',
    regions.name as '地方名'
from 
    prefectures--xxx
--共通するカラムを通じて、prefecturesとregionsの対応関係を作り、
--情報を取得する
inner join regions--yyy
on prefectures.region_id = regions.id ;
--xxx.aaa = yyy.bbb

/*
SELECT
    prefectures.name AS '都道府県名',
    regions.id AS '地方ID',
    regions.name AS '地方名'
FROM
    prefectures
    INNER JOIN regions
        ON prefectures.region_id = regions.id;
*/


