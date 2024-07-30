--氏名と所属学部を表示するクエリを作成
--誰も所属していない学部については、所属学部と表示。
--この時、氏名はNULLとする
--inner joinはデータの欠如がない
--outer joinはデータの欠如があっても使える
--SELECT カラム名 FROM xxx LEFT OUTER JOIN yyy ON xxx.aaa = yyy.bbb;
SELECT
    students.name as '氏名',
    faculties.name as '所属学部'
FROM
    faculties--欠落が生じるRDB
    left outer join students 
    on  students.faculty_id = faculties.id;


/*
SELECT
    students.name AS '氏名',
    faculties.name AS '所属学部'
FROM
    faculties
    LEFT OUTER JOIN students
        ON students.faculty_id = faculties.id;
*/