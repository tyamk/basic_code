--students テーブルと faculties テーブル
--各学部の学部 ID、学部名、所属している学生の人数を、
--学部 ID の昇順に表示するクエリを作成
--カラム名は「学部ID」「学部名」「学生数」
select
   faculties.id as '学部ID',
   faculties.name as '学部名',
   count(*) as '学生数'
from
    students
    inner join faculties
        on students.faculty_id = faculties.id
group by
    --'学部名'
    `学部名`
order by 
    --'学部ID';
    `学部ID`;

/*
SELECT
    faculties.id AS '学部ID',
    faculties.name AS '学部名',
    COUNT(*) AS '学生数'
FROM
    students
    INNER JOIN faculties
        ON students.faculty_id = faculties.id
GROUP BY
    `学部名`
ORDER BY
    `学部ID`;
*/