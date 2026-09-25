# ASP + DB 연동

> 2024년 2학기 · 웹 프로그래밍 수업 과제

**ASP(Active Server Pages)** 로 Access 데이터베이스(`.mdb`)를 붙여 본 과제입니다.
서버에서 도는 스크립트가 DB를 조회하고, 넣고, 고치고, 지우는 기본 흐름을 익혔습니다.

## 무엇을 했나

| 파일 | DB · 테이블 | 하는 일 |
|---|---|---|
| `insert.asp`, `insert.htm` → `insert2.asp` | `nwind.mdb` · `tblCategories`, `tblProducts` | 분류와 상품을 **추가** |
| `read.asp` · `read2.asp` · `read3.asp` | `nwind.mdb` · 분류 · 상품 · 직원 | 테이블마다 목록 **조회** |
| `update1.htm` → `update2.asp` → `update3.asp` | `nwind.mdb` · `tblEmployees` | 직원 정보 **수정** |
| `delete.htm` → `delete2.asp` | `nwind.mdb` · `tblCategories` | 항목 **삭제** |
| `GetEmpInfo.asp` | `nwind.mdb` · `tblEmployees` | 직원 정보 조회 |
| `cwrite` · `cread` · `cupdate` · `cdelete` | `onchat.mdb` · `IdeaTime` | 글을 쓰고 읽고 고치고 지우는 **간단한 게시판** |
| `test1` ~ `test7` | — | ASP 문법 기초 실습 |

`nwind.mdb`는 Microsoft의 Northwind 예제 데이터베이스입니다.

## 구성

| 폴더 | 내용 |
|---|---|
| `root/` | ASP·HTML 소스와 Access DB 파일 (IIS 사이트 루트) |

여기서 익힌 **웹 서버 ↔ DB** 구조를 나중에
[2026년 「웹서버 DB 연동 기초」](../../2026/06.%20웹서버%20DB%20연동%20기초/)에서
Flask·FastAPI·MongoDB까지 넓혀 **15챕터 교재로 정리**했습니다.
