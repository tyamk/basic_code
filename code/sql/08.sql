--grades テーブル (期末試験の成績情報)
--students テーブル (学生の情報)
--subjects テーブル (開講科目の情報)
--grades テーブルに格納されているすべての成績の情報について、
--学生の氏名、科目名、点数を表示するクエリを作成
select
    students.name as '氏名',
    subjects.name as '科目名',
    grades.score as '点数'
from 
    grades
        inner join students
            on grades.student_id = students.id
                inner join subjects
                    on grades.subject_id = subjects.id ;

/*
SELECT
    students.name AS '氏名',
    subjects.name AS '科目名',
    grades.score AS '点数'
FROM
    grades
    INNER JOIN students, subjects
        ON grades.student_id = students.id
        AND grades.subject_id = subjects.id;
*/