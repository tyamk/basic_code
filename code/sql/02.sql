--氏名と所属学部
select
    students.name as '氏名',
    faculties.name as '所属学部'
from 
    students
inner join faculties
on students.faculty_id = faculties.id;

/*
SELECT
    students.name AS '氏名',
    faculties.name AS '所属学部'
FROM
    students
    INNER JOIN faculties
        ON students.faculty_id = faculties.id;
*/