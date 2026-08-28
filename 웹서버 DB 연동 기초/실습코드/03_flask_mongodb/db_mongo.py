"""DB 접근 계층 — MongoDB 판.

01_flask_sqlite/db.py 와 **함수 이름·인자·반환값이 완전히 같다.**
그래서 app.py 는 `import db` 를 `import db_mongo as db` 로 바꾸기만 하면 된다.
이게 계층을 나눠 둔 값어치다 — DB를 통째로 갈아도 웹 코드는 그대로다.

SQLite 판과 나란히 놓고 읽으면 "관계형 vs 문서형"의 차이가 한눈에 보인다.

실행:  py db_mongo.py init
필요:  MongoDB 서버가 localhost:27017 에 떠 있어야 한다
"""
import os
import sys
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# 연결 문자열. SQLite 는 "파일 경로"였는데 MongoDB 는 "서버 주소"다.
#   SQLite  : guestbook.db          ← 파일
#   MongoDB : mongodb://호스트:포트   ← 서버
# 이게 파일DB와 서버DB의 가장 큰 차이다.
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://127.0.0.1:27017")
DB_NAME = "guestbook_db"
COLLECTION = "guestbook"

_client = None


def get_client():
    """MongoClient 를 만든다 (한 번만 만들어 재사용).

    ★ SQLite 와 다른 점: MongoClient 는 연결 풀을 내부에 갖고 있어서
    요청마다 새로 만들면 안 된다. 전역에 하나 두고 재사용하는 게 정석이다.
    (SQLite 는 반대로 요청마다 열고 닫는 게 자연스럽다 — 그냥 파일이라서)
    """
    global _client
    if _client is None:
        _client = MongoClient(
            MONGO_URL,
            serverSelectionTimeoutMS=3000,   # 서버가 없으면 3초 만에 포기 (기본 30초는 너무 길다)
        )
    return _client


def get_collection():
    return get_client()[DB_NAME][COLLECTION]


def init_db():
    """컬렉션을 비우고 인덱스를 만든다.

    ★ SQLite 의 CREATE TABLE 에 해당하는 게 없다.
    MongoDB 는 스키마가 없어서(schema-less) 테이블을 미리 만들 필요가 없다 —
    문서를 넣는 순간 컬렉션이 생긴다.

    대신 인덱스는 직접 만들어야 한다. 이건 SQLite 와 같다.
    """
    col = get_collection()
    col.drop()                                        # 기존 데이터 삭제
    col.create_index([("created_at", DESCENDING)])    # 최신순 조회용
    col.create_index([("seq", ASCENDING)], unique=True)
    print(f"[OK] MongoDB 초기화 완료: {MONGO_URL}/{DB_NAME}.{COLLECTION}")


def _next_seq():
    """자동 증가 id 를 흉내 낸다.

    ★ MongoDB 에는 AUTO_INCREMENT 가 없다.
    기본 키는 _id 인데 ObjectId('507f1f77bcf86cd799439011') 같은 12바이트 값이라
    URL 에 넣기엔 길고 사람이 읽기도 어렵다.

    그래서 1, 2, 3... 짜리 seq 를 직접 만든다. 이게 문서형 DB 를 쓸 때
    자주 겪는 불편이고, "관계형이 편한 지점"의 대표 사례다.
    """
    col = get_collection()
    last = col.find_one(sort=[("seq", DESCENDING)])
    return (last["seq"] + 1) if last else 1


def list_entries(limit=100):
    """방명록 목록을 최신순으로. SQLite 판과 반환 형태가 똑같다."""
    col = get_collection()
    cursor = col.find(
        {},                                  # 조건 없음 = SELECT * (SQL 의 WHERE 자리)
        {"_id": 0},                          # _id 는 빼고 (SQL 의 SELECT 컬럼 목록 자리)
    ).sort([("created_at", DESCENDING), ("seq", DESCENDING)]).limit(limit)

    out = []
    for d in cursor:
        out.append({
            "id": d["seq"],
            "name": d["name"],
            "message": d["message"],
            # SQLite 는 TEXT 로 저장했지만 Mongo 는 진짜 datetime 타입을 저장한다.
            # 화면에 뿌리려면 문자열로 바꿔 준다 (SQLite 판과 형태를 맞추려고).
            "created_at": d["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
        })
    return out


def add_entry(name, message):
    """글 추가.

    ★ SQL 인젝션이 원리적으로 없다.
    SQL 은 "문자열"이라 값이 명령으로 해석될 수 있지만, Mongo 는 쿼리가
    "딕셔너리(BSON)"라서 값이 문법이 될 수 없다. name 에 무슨 문자를 넣든
    그냥 name 필드의 값일 뿐이다.

    (단 NoSQL 인젝션이 아예 없는 건 아니다 — 사용자 입력을 쿼리 "구조"에
     그대로 넣으면 {"$ne": null} 같은 연산자를 주입당할 수 있다.
     값 자리에 넣는 지금 코드는 안전하다.)
    """
    col = get_collection()
    seq = _next_seq()
    col.insert_one({
        "seq": seq,
        "name": name,
        "message": message,
        "created_at": datetime.now(timezone.utc),
    })
    return seq


def delete_entry(entry_id):
    """id 로 삭제. 지워진 개수를 돌려준다 (SQLite 판과 동일)."""
    col = get_collection()
    result = col.delete_one({"seq": int(entry_id)})
    return result.deleted_count


def count_entries():
    return get_collection().count_documents({})


def _seed():
    add_entry("이지용", "MongoDB 판입니다. app.py 는 한 줄만 바꿨습니다.")
    add_entry("방문자", "스키마가 없어서 CREATE TABLE 이 필요 없네요.")
    add_entry("테스터", "대신 자동증가 id 를 직접 만들어야 합니다.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        try:
            init_db()
            _seed()
            print(f"[OK] 샘플 {count_entries()}건 입력됨")
        except ServerSelectionTimeoutError:
            print(f"[에러] MongoDB 서버에 연결할 수 없습니다: {MONGO_URL}")
            print("       MongoDB 를 먼저 설치·실행하세요. README 참고.")
            sys.exit(1)
    else:
        print("사용법: py db_mongo.py init")
