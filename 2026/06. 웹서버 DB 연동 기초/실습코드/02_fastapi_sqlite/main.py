"""똑같은 방명록을 FastAPI 로 만든 것 — 웹 서버만 바꿨다.

01_flask_sqlite 와 비교해서 읽으면 "웹 서버 종류가 뭐가 다른가"가 바로 보인다.
DB 계층(db.py)은 01 의 것을 그대로 가져다 쓴다 — 웹 서버를 바꿔도 DB 코드는
손댈 필요가 없다는 게 계층을 나눈 이유다.

실행:
    py -m uvicorn main:app --reload --port 8000
    → http://localhost:8000
    → http://localhost:8000/docs      ★ 자동 생성된 API 문서 (Flask 에는 없다)
"""
import os
import sys

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# ★ 01 폴더의 db.py 를 한 글자도 안 고치고 그대로 가져다 쓴다.
#   웹 서버를 Flask -> FastAPI 로 바꿔도 DB 계층은 손댈 필요가 없다는 증명이다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLASK_DIR = os.path.join(os.path.dirname(BASE_DIR), "01_flask_sqlite")
sys.path.insert(0, FLASK_DIR)
import db  # noqa: E402

app = FastAPI(
    title="방명록 API",
    description="FastAPI + SQLite 방명록. /docs 에서 바로 테스트할 수 있습니다.",
    version="1.0.0",
)

# 템플릿은 이 폴더 것을 쓴다 — Flask 판과 url_for 문법이 달라서 공유가 안 된다.
#   Flask    : url_for('static', filename='style.css')
#   Starlette: url_for('static', path='style.css')
# templates/index.html 의 주석 참고.
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# CSS 는 프레임워크와 무관한 순수 파일이라 01 것을 그대로 재사용한다.
app.mount("/static", StaticFiles(directory=os.path.join(FLASK_DIR, "static")), name="static")


# ──────────────────────────────────────────────────────────────
# ★ Flask 와 가장 다른 점: Pydantic 모델로 입력을 "선언"한다
#
# Flask:    data = request.get_json(); name = data.get("name")   ← 직접 꺼내고 직접 검증
# FastAPI:  아래 모델만 선언하면 검증·에러응답·문서가 자동으로 생긴다
# ──────────────────────────────────────────────────────────────

class EntryIn(BaseModel):
    name: str = Field(min_length=1, max_length=50, description="작성자 이름")
    message: str = Field(min_length=1, max_length=500, description="남길 메시지")


class EntryOut(BaseModel):
    id: int
    name: str
    message: str
    created_at: str


# ──────────────────────────────────────────────────────────────
# 화면 (HTML)
# ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    entries = db.list_entries()
    # ★ 인자 순서 주의: (request, 템플릿이름, context)
    #
    # 예전 Starlette(<0.29) 은 TemplateResponse("index.html", {"request": request, ...})
    # 였는데, 지금은 request 가 첫 번째 인자로 바뀌었다. 옛 방식으로 쓰면
    #     TypeError: unhashable type: 'dict'
    # 라는, 원인을 짐작하기 어려운 에러가 난다 (dict 가 템플릿 이름 자리로 들어가서).
    # 인터넷 예제 상당수가 아직 옛 방식이라 자주 걸리는 함정이다.
    return templates.TemplateResponse(
        request,
        "index.html",
        {"entries": entries, "count": len(entries)},
    )


@app.post("/add", include_in_schema=False)
def add(name: str = Form(""), message: str = Form("")):
    """폼 전송 처리. Flask 판과 동일하게 PRG 패턴."""
    name, message = name.strip(), message.strip()
    if name and message and len(name) <= 50 and len(message) <= 500:
        db.add_entry(name, message)
    # 303 See Other — POST 후 GET 으로 넘길 때의 정확한 코드다.
    # (Flask 의 redirect() 기본값은 302 인데, 엄밀히는 303 이 맞다)
    return RedirectResponse(url="/", status_code=303)


@app.post("/delete/{entry_id}", include_in_schema=False)
def delete(entry_id: int):
    db.delete_entry(entry_id)
    return RedirectResponse(url="/", status_code=303)


# ──────────────────────────────────────────────────────────────
# API (JSON) — /docs 에 자동으로 문서가 만들어진다
# ──────────────────────────────────────────────────────────────

@app.get("/api/entries", response_model=list[EntryOut], summary="방명록 목록")
def api_list():
    return db.list_entries()


@app.post("/api/entries", response_model=EntryOut, status_code=201, summary="방명록 작성")
def api_add(entry: EntryIn):
    """entry 가 EntryIn 타입이라고 선언만 하면:
      - 빈 문자열/길이 초과 → FastAPI 가 자동으로 422 를 돌려준다
      - /docs 에 입력 양식이 자동으로 생긴다
      - 우리 코드는 검증을 한 줄도 안 쓴다
    """
    new_id = db.add_entry(entry.name.strip(), entry.message.strip())
    rows = [e for e in db.list_entries() if e["id"] == new_id]
    return rows[0]


@app.delete("/api/entries/{entry_id}", summary="방명록 삭제")
def api_delete(entry_id: int):
    deleted = db.delete_entry(entry_id)
    if deleted == 0:
        # Flask: return jsonify({"error": ...}), 404
        # FastAPI: 예외를 던지면 알아서 JSON 에러 응답이 된다
        raise HTTPException(status_code=404, detail="없는 id 입니다")
    return {"deleted": entry_id}


@app.get("/health", summary="상태 확인")
def health():
    return {"status": "ok", "entries": db.count_entries()}


if __name__ == "__main__":
    import uvicorn

    print("http://localhost:8000        (화면)")
    print("http://localhost:8000/docs   (자동 API 문서)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
