"""웹 서버 계층 — Flask (MongoDB 판).

★ 01_flask_sqlite/app.py 와 딱 한 줄만 다르다:

    01:  import db
    03:  import db_mongo as db          ← 이 줄

    나머지 라우트·검증·PRG·에러처리는 글자 하나 안 바뀌었다.
    DB 를 관계형에서 문서형으로 통째로 갈아엎었는데도.

이게 "계층을 나눈다"는 말의 실제 의미다.

실행:
    py db_mongo.py init      ← 처음 한 번만
    py app.py                ← http://localhost:5001
필요:
    MongoDB 서버가 localhost:27017 에 떠 있어야 한다
"""
import os

from flask import Flask, jsonify, redirect, render_template, request, url_for
from pymongo.errors import ServerSelectionTimeoutError

import db_mongo as db          # ★ 여기가 01 과 다른 유일한 줄

# CSS 는 프레임워크·DB 와 무관한 순수 파일이라 01 것을 그대로 가리킨다.
# (템플릿은 Flask 판이라 url_for 문법이 같으므로 복사본을 쓴다)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(os.path.dirname(BASE_DIR), "01_flask_sqlite", "static")

app = Flask(__name__, static_folder=STATIC_DIR)


# ──────────────────────────────────────────────────────────────
# 화면 (HTML) — 01 과 동일
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    entries = db.list_entries()
    return render_template("index.html", entries=entries, count=len(entries))


@app.route("/add", methods=["POST"])
def add():
    name = (request.form.get("name") or "").strip()
    message = (request.form.get("message") or "").strip()
    if not name or not message:
        return redirect(url_for("index"))
    if len(name) > 50 or len(message) > 500:
        return redirect(url_for("index"))
    db.add_entry(name, message)
    return redirect(url_for("index"))


@app.route("/delete/<int:entry_id>", methods=["POST"])
def delete(entry_id):
    db.delete_entry(entry_id)
    return redirect(url_for("index"))


# ──────────────────────────────────────────────────────────────
# API (JSON) — 01 과 동일
# ──────────────────────────────────────────────────────────────

@app.route("/api/entries", methods=["GET"])
def api_list():
    return jsonify(db.list_entries())


@app.route("/api/entries", methods=["POST"])
def api_add():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    message = (data.get("message") or "").strip()
    if not name or not message:
        return jsonify({"error": "name 과 message 는 필수입니다"}), 400
    new_id = db.add_entry(name, message)
    return jsonify({"id": new_id, "name": name, "message": message}), 201


@app.route("/api/entries/<int:entry_id>", methods=["DELETE"])
def api_delete(entry_id):
    deleted = db.delete_entry(entry_id)
    if deleted == 0:
        return jsonify({"error": "없는 id 입니다"}), 404
    return jsonify({"deleted": entry_id})


@app.route("/health")
def health():
    """01 과 다른 점: MongoDB 는 서버라서 "연결 실패"가 일어날 수 있다.

    SQLite 는 파일이라 서버가 죽을 일이 없다. 서버DB 를 쓰면
    이런 연결 에러 처리가 반드시 필요해진다.
    """
    try:
        return jsonify({"status": "ok", "entries": db.count_entries()})
    except ServerSelectionTimeoutError:
        return jsonify({"status": "error", "detail": "MongoDB 에 연결할 수 없습니다"}), 503


@app.errorhandler(ServerSelectionTimeoutError)
def handle_mongo_down(e):
    """MongoDB 가 꺼져 있을 때 500 대신 친절한 안내를 준다."""
    return (
        "<h1>MongoDB 에 연결할 수 없습니다</h1>"
        "<p>MongoDB 서버가 localhost:27017 에 떠 있는지 확인하세요.</p>"
        "<pre>net start MongoDB       (Windows 서비스)\n"
        "mongod --dbpath C:\\data\\db  (직접 실행)</pre>",
        503,
    )


if __name__ == "__main__":
    # 01(Flask+SQLite)이 5000 번을 쓰므로 5001 번을 쓴다.
    # 같은 포트를 두 프로그램이 못 쓴다 — "Address already in use" 에러의 원인.
    print("http://localhost:5001")
    app.run(host="127.0.0.1", port=5001, debug=True)
