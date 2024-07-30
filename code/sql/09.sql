--registrations テーブル (履修登録情報)
--students テーブル (学生の情報)
--subjects テーブル (開講科目の情報)
--科目名 (カラム名は「科目名」とする)
--その科目を履修している、入学年度が 2023 年度の学生の人数 (カラム名は「1年生履修者数」とする)
--その科目を履修している、入学年度が 2022 年度の学生の人数 (カラム名は「2年生履修者数」とする)
select
    subjects.name as '科目名',
    --count(*) as '1年生履修者数' ()内に条件を書く
    count(students.enrollment_year = 2023 or null) as '1年生履修者数',
    count(students.enrollment_year = 2022 or null) as '2年生履修者数'
from    
    registrations
    inner join students 
        on registrations.student_id = students.id 
            inner join subjects
                on registrations.subject_id = subjects.id
group by 
    `科目名`;

/*
SELECT
    subjects.name AS '科目名',
    COUNT(students.enrollment_year = 2023 OR NULL) AS '1年生履修者数',
    COUNT(students.enrollment_year = 2022 OR NULL) AS '2年生履修者数'
FROM
    registrations
    INNER JOIN students, subjects
        ON registrations.student_id = students.id
        AND registrations.subject_id = subjects.id
GROUP BY
    `科目名`;
*/
