--facultiesテーブルには学生が一人も所属していない
--氏名と所属学部というカラム名にする
--氏名は該当なし
select
--nullの書き換え
    ifnull(students.name, '該当なし') as '氏名',
    faculties.name as '所属学部'
from 
    faculties
    left outer join students
    on students.faculty_id = faculties.id;

/*
SELECT
    IFNULL(students.name, '該当なし') AS '氏名',
    faculties.name AS '所属学部'
FROM
    faculties
    LEFT OUTER JOIN students
        ON students.faculty_id = faculties.id;
*/