# 웹 서버 + DB 연동 기초 — 교과서

> **이 폴더만 남의 코드 설명이 아니다.** 나머지 폴더는 이미 만든 프로젝트를 해설한
> 것이고, 여기는 **새로 만든 학습용 실습**이다.

**한 줄 요약**: 방명록 하나를 **웹 서버 3종 × DB 2종** 조합으로 만들어 보면서
"웹 서버가 뭐고 DB가 뭔지, 종류가 왜 나뉘고 뭘 골라야 하는지"를 익힌다.

**핵심 장치**: **똑같은 앱을 서버만 바꾸고 / DB만 바꿔서** 3벌 만들었다.
같은 것을 만들어 놓고 한 축만 바꾸면 **그 축이 무엇을 담당하는지가 드러난다.**

```
01_flask_sqlite     Flask   + SQLite     ← 기준
02_fastapi_sqlite   FastAPI + SQLite     ← 웹 서버만 교체
03_flask_mongodb    Flask   + MongoDB    ← DB만 교체
```

---

## 📚 챕터 목차

| 챕터 | 주제 |
|---|---|
| **[1챕터](1챕터.md)** | 이 폴더가 무엇인가 — 그리고 이미 해본 것들 |
| **[2챕터](2챕터.md)** | 전체 그림 — 브라우저에서 DB까지 |
| **[3챕터](3챕터.md)** | 웹 서버 종류 ① 정적 웹 서버 (Apache · Nginx · IIS) |
| **[4챕터](4챕터.md)** | 웹 서버 종류 ② 애플리케이션 프레임워크 (Flask · FastAPI · Express …) |
| **[5챕터](5챕터.md)** | 웹 서버 종류 ③ 실서비스는 둘을 겹쳐 쓴다 (WSGI/ASGI + Nginx) |
| **[6챕터](6챕터.md)** | DB 종류 ① 왜 파일로 안 하나 |
| **[7챕터](7챕터.md)** | DB 종류 ② 관계형 (SQLite · MySQL · PostgreSQL · MSSQL · Access) |
| **[8챕터](8챕터.md)** | DB 종류 ③ NoSQL (MongoDB · Redis · 기타) |
| **[9챕터](9챕터.md)** | 무엇을 골라야 하나 — 선택 가이드 |
| **[10챕터](10챕터.md)** | 실습 ① Flask + SQLite — 3계층으로 나누기 |
| **[11챕터](11챕터.md)** | 실습 ② 스키마와 SQL 기초 |
| **[12챕터](12챕터.md)** | 실습 ③ ★ 파라미터 바인딩 — SQL 인젝션 직접 막아보기 |
| **[13챕터](13챕터.md)** | 실습 ④ REST API와 HTTP 상태 코드 |
| **[14챕터](14챕터.md)** | 실습 ⑤ 서버 갈아끼우기 / DB 갈아끼우기 |
| **[15챕터](15챕터.md)** | 실행법 · 배포 · 다음 단계 |

---

## 💻 실습 코드

| 폴더 | 조합 | 포트 | 상태 |
|---|---|---|---|
| [`실습코드/01_flask_sqlite/`](실습코드/01_flask_sqlite/) | **Flask + SQLite** | 5000 | ✅ 전 기능 검증됨 |
| [`실습코드/02_fastapi_sqlite/`](실습코드/02_fastapi_sqlite/) | **FastAPI + SQLite** | 8000 | ✅ 전 기능 검증됨 |
| [`실습코드/03_flask_mongodb/`](실습코드/03_flask_mongodb/) | **Flask + MongoDB** | 5001 | ⚠ MongoDB 설치 필요 |

### 바로 실행

```bash
cd 실습코드/01_flask_sqlite
pip install -r requirements.txt      # Flask 하나뿐 (SQLite 는 파이썬 내장)
py db.py init                        # 처음 한 번 — DB 생성 + 샘플 3건
py app.py                            # → http://localhost:5000
```

**FastAPI 판은 자동 API 문서가 공짜로 딸려온다:**

```bash
cd 실습코드/02_fastapi_sqlite
py -m uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs     ★ 꼭 열어볼 것
```

> `python`이 아니라 **`py`**를 쓴다 — 이 PC의 `python`은 Microsoft Store 스텁이라
> 아무것도 하지 않는다.

---

[전체 프로젝트 목차로 →](../README.md)
