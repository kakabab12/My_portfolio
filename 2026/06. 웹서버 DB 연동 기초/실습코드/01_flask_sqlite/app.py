"""웹 서버 계층 — Flask.

이 파일에는 SQL 이 한 줄도 없다. DB 일은 전부 db.py 가 한다.
   app.py  = "HTTP 요청을 받아서 무엇을 할지" (웹의 일)
   db.py   = "데이터를 어떻게 저장/조회할지" (DB의 일)

실행:
    py db.py init      ← 처음 한 번만 (DB 파일 생성)
    py app.py          ← http://localhost:5000
"""
from flask import Flask, jsonify, redirect, render_template, request, url_for

import db

app = Flask(__name__)


# ──────────────────────────────────────────────────────────────
# 화면 (HTML)
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """방명록 목록 페이지."""
    entries = db.list_entries()
    return render_template("index.html", entries=entries, count=len(entries))


@app.route("/add", methods=["POST"])
def add():
    """폼 전송을 받아 글을 추가한다.

    끝에서 redirect 하는 게 핵심이다 (PRG 패턴: Post → Redirect → Get).
    redirect 없이 HTML 을 그냥 돌려주면, 사용자가 F5 를 눌렀을 때
    브라우저가 "양식을 다시 제출하시겠습니까?" 를 띄우고 → 같은 글이 또 등록된다.
    redirect 하면 F5 는 그냥 GET 이라 안전하다.
    """
    name = (request.form.get("name") or "").strip()
    message = (request.form.get("message") or "").strip()

    # 서버 쪽 검증 — HTML 의 required 속성은 브라우저에서 얼마든지 우회된다.
    # 검증은 반드시 서버에서 한 번 더 해야 한다.
    if not name or not message:
        return redirect(url_for("index"))
    if len(name) > 50 or len(message) > 500:
        return redirect(url_for("index"))

    db.add_entry(name, message)
    return redirect(url_for("index"))       # ← PRG


@app.route("/delete/<int:entry_id>", methods=["POST"])
def delete(entry_id):
    """글 삭제.

    <int:entry_id> 로 타입을 못박아 두면 /delete/abc 같은 요청은
    Flask 가 알아서 404 를 낸다 — 우리 코드까지 오지 않는다.

    GET 이 아니라 POST 인 이유: GET 은 "읽기"여야 한다.
    /delete/1 을 GET 으로 만들면 검색엔진 크롤러가 링크를 따라가며
    글을 전부 지워 버린다. 실제로 자주 일어나는 사고다.
    """
    db.delete_entry(entry_id)
    return redirect(url_for("index"))


# ──────────────────────────────────────────────────────────────
# API (JSON) — 같은 데이터를 기계가 읽을 수 있게
# ──────────────────────────────────────────────────────────────

@app.route("/api/entries", methods=["GET"])
def api_list():
    return jsonify(db.list_entries())


@app.route("/api/entries", methods=["POST"])
def api_add():
    """JSON 으로 글 추가.

    테스트:
      curl -X POST http://localhost:5000/api/entries ^
           -H "Content-Type: application/json" ^
           -d "{\"name\":\"api\",\"message\":\"hello\"}"
    """
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    message = (data.get("message") or "").strip()

    if not name or not message:
        # 400 = 클라이언트 잘못. 500(서버 잘못)과 구분해서 돌려주는 게 중요하다.
        return jsonify({"error": "name 과 message 는 필수입니다"}), 400

    new_id = db.add_entry(name, message)
    # 201 = Created. 새 자원을 만들었으면 200이 아니라 201이 맞다.
    return jsonify({"id": new_id, "name": name, "message": message}), 201


@app.route("/api/entries/<int:entry_id>", methods=["DELETE"])
def api_delete(entry_id):
    deleted = db.delete_entry(entry_id)
    if deleted == 0:
        return jsonify({"error": "없는 id 입니다"}), 404
    return jsonify({"deleted": entry_id})


@app.route("/health")
def health():
    """서버·DB 가 살아 있는지 확인하는 엔드포인트.

    배포하면 이런 게 꼭 필요해진다 — 로드밸런서나 모니터링이 주기적으로 찔러 본다.
    """
    try:
        return jsonify({"status": "ok", "entries": db.count_entries()})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    # host="127.0.0.1" 이면 내 PC 에서만 접속된다.
    # 다른 기기(폰 등)에서도 보려면 host="0.0.0.0" 으로 바꾼다.
    #   (라즈베리파이 프로젝트가 0.0.0.0 을 쓴 이유가 이것이다)
    #
    # debug=True 는 개발용이다. 코드를 고치면 자동 재시작되고 에러 화면이 자세히 나온다.
    # ★ 실서비스에서는 반드시 끈다 — 에러 화면으로 소스가 노출되고,
    #   디버거 콘솔로 원격 코드 실행까지 가능하다.
    print("http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
