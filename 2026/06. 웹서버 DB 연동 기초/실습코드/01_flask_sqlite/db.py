"""DB 접근 계층 — SQLite 연결과 쿼리를 여기 한 곳에 모은다.

app.py 에는 SQL 이 한 줄도 없다. 이렇게 나눠 두면:
  - DB 를 MySQL 로 바꿀 때 이 파일만 고치면 된다
  - app.py 를 읽는 사람은 "무엇을 하는지"만 보면 된다
  - 쿼리를 테스트하기 쉽다 (웹 서버 없이 db.py 만 불러서)

실행:  py db.py init     ← DB 파일 생성 + 스키마 적용 + 샘플 데이터
"""
import os
import sqlite3
import sys

# 이 파일이 있는 폴더 기준으로 경로를 잡는다.
# 어느 폴더에서 실행하든 같은 DB 파일을 찾게 하려는 것 —
# "py app.py" 든 "py 실습코드/01_flask_sqlite/app.py" 든 동작한다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "guestbook.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")


def get_connection():
    """SQLite 연결을 만든다.

    row_factory 를 sqlite3.Row 로 두면 결과를 인덱스가 아니라 컬럼 이름으로
    꺼낼 수 있다:
        row[0]        -> row["name"]
    컬럼 순서가 바뀌어도 코드가 안 깨지고, 읽기도 훨씬 낫다.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # 외래키 제약을 켠다. SQLite 는 기본이 꺼짐이라 연결마다 켜 줘야 한다.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """schema.sql 을 읽어 테이블을 만든다. 기존 데이터는 지워진다."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = f.read()

    conn = get_connection()
    try:
        conn.executescript(schema)   # 여러 문장을 한 번에 실행
        conn.commit()
    finally:
        conn.close()
    print(f"[OK] DB 초기화 완료: {DB_PATH}")


def list_entries(limit=100):
    """방명록 목록을 최신순으로 돌려준다."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, name, message, created_at"
            " FROM guestbook"
            " ORDER BY created_at DESC, id DESC"
            " LIMIT ?",
            (limit,),          # ← 파라미터 바인딩 (아래 add_entry 주석 참고)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def add_entry(name, message):
    """방명록 글 하나를 추가하고, 새로 만들어진 id 를 돌려준다.

    ★ 여기가 이 실습에서 가장 중요한 줄이다.

    절대 이렇게 쓰면 안 된다:
        sql = "INSERT INTO guestbook (name, message) VALUES ('" + name + "', ...)"

    name 에 다음을 넣으면 테이블이 통째로 날아간다:
        '); DROP TABLE guestbook; --

    아래처럼 ? 를 쓰면 DB가 값을 "데이터"로만 취급하고 절대 SQL 로 해석하지
    않는다. 무슨 문자를 넣든 그냥 이름일 뿐이다.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO guestbook (name, message) VALUES (?, ?)",
            (name, message),        # ← 값은 튜플로 따로 넘긴다
        )
        conn.commit()               # ← 커밋해야 실제로 저장된다
        return cur.lastrowid
    finally:
        conn.close()


def delete_entry(entry_id):
    """id 로 글 하나를 지운다. 지워진 행 수를 돌려준다(0이면 없던 id)."""
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM guestbook WHERE id = ?", (entry_id,))
        conn.commit()
        return cur.rowcount         # ← 몇 건 지워졌는지 확인할 수 있다
    finally:
        conn.close()


def count_entries():
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) AS c FROM guestbook").fetchone()["c"]
    finally:
        conn.close()


def _seed():
    """맛보기 샘플 데이터."""
    add_entry("이지용", "첫 글입니다. 웹서버 + DB 연동 성공!")
    add_entry("방문자", "SQLite 는 설치가 필요 없어서 편하네요.")
    add_entry("테스터", "?, ? 파라미터 바인딩으로 SQL 인젝션을 막습니다.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init_db()
        _seed()
        print(f"[OK] 샘플 {count_entries()}건 입력됨")
    else:
        print("사용법: py db.py init")
