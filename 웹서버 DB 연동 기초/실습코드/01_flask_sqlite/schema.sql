-- 방명록 테이블 스키마
--
-- 스키마를 .sql 파일로 빼 두는 이유:
--   1) DB 구조가 한눈에 보인다 (코드 안에 흩어져 있으면 안 보인다)
--   2) 팀원이 SQL만 읽어도 구조를 안다
--   3) 나중에 MySQL/PostgreSQL 로 옮길 때 이 파일만 고치면 된다
--
-- 실행:  py db.py init

DROP TABLE IF EXISTS guestbook;

CREATE TABLE guestbook (
    -- INTEGER PRIMARY KEY 는 SQLite 에서 자동 증가(rowid 별칭)가 된다.
    -- MySQL 이면 INT AUTO_INCREMENT, PostgreSQL 이면 SERIAL 로 바꿔야 한다.
    id        INTEGER PRIMARY KEY AUTOINCREMENT,

    -- NOT NULL: 값이 반드시 있어야 한다 (DB가 강제하는 검증)
    name      TEXT    NOT NULL,
    message   TEXT    NOT NULL,

    -- 기본값을 DB가 채워 준다. 애플리케이션이 시각을 안 넘겨도 된다.
    -- SQLite 의 CURRENT_TIMESTAMP 는 UTC 기준이다.
    created_at TEXT   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스: 목록을 항상 "최신순"으로 뽑으므로 created_at 에 인덱스를 건다.
-- 행이 몇 개 없으면 체감이 없지만, 수만 건이 되면 이게 있고 없고가 크게 갈린다.
CREATE INDEX idx_guestbook_created_at ON guestbook (created_at DESC);
