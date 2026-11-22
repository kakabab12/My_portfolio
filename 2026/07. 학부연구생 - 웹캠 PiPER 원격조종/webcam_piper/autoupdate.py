"""켤 때 GitHub 에서 새 코드를 받아 온다 (2026-11-17). teleop.py 와 check_setup.py 가 맨 처음 부른다.

- git 으로 받은 폴더면 `git pull --ff-only`, git 이 없거나 압축 파일로 받은 폴더면 GitHub 에서 바뀐 파일만 내려받는다
  (바뀌기 전 파일은 이름.bak 으로 남김).
- 받은 것이 있으면 같은 명령으로 저절로 다시 켠다. requirements.txt 가 바뀌었으면 pip 로 먼저 설치한다.
- 인터넷이 안 되거나, 이 노트북에서 고친 파일 때문에 받을 수 없으면 건너뛰고 지금 코드로 켠다.
- 끄려면 명령 끝에 --no-update 를 붙이거나 환경변수 WEBCAM_PIPER_NO_UPDATE=1.
- 폴더에 .no-autoupdate 파일이 있으면 하지 않는다 (연구실 PC 의 작업 사본이 GitHub 것으로 덮이지 않게).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

REPO, BRANCH = "kakabab12/LAB-1309", "main"
PREFIX = "11_실물_PiPER/webcam_piper/"
HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "webcam-piper-updater"}


def _git(*args, timeout=30):
    return subprocess.run(["git", "-C", HERE, *args], capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def _via_git():
    """git 폴더면 (받았나, 바뀐 파일들), 아니면 None."""
    try:
        r = _git("rev-parse", "HEAD")
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    before = r.stdout.strip()
    try:
        p = _git("pull", "--ff-only", "--quiet", timeout=60)
    except subprocess.TimeoutExpired:
        print("[업데이트] GitHub 이 답하지 않음 — 지금 코드로 켭니다")
        return False, []
    if p.returncode != 0:
        last = ((p.stderr or "") + (p.stdout or "")).strip().splitlines()
        print("[업데이트] 못 받음:", last[-1] if last else "?")
        print("           지금 코드로 켭니다. 이 노트북에서 고친 파일이 있으면 git status 로 확인하세요")
        return False, []
    after = _git("rev-parse", "HEAD").stdout.strip()
    if after == before:
        return False, []
    changed = [os.path.basename(f) for f in _git("diff", "--name-only", before, after, "--", ".").stdout.split()]
    print(f"[업데이트] 새 코드 받음 ({before[:7]} → {after[:7]}, 이 폴더 파일 {len(changed)}개)")
    return True, changed


def _blob_sha(path):
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _via_api():
    """git 없이: GitHub 의 파일 목록과 비교해 다른 파일만 받는다. 돌려줌: 바뀐 파일들."""
    url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15) as r:
        tree = json.load(r)["tree"]
    changed, stamp = [], time.strftime("%Y%m%d_%H%M%S")
    for t in tree:
        if t["type"] != "blob" or not t["path"].startswith(PREFIX):
            continue
        rel = t["path"][len(PREFIX):]
        dst = os.path.join(HERE, *rel.split("/"))
        if os.path.exists(dst) and _blob_sha(dst) == t["sha"]:
            continue
        raw = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/" + urllib.parse.quote(t["path"])
        with urllib.request.urlopen(urllib.request.Request(raw, headers=UA), timeout=30) as r:
            data = r.read()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.replace(dst, f"{dst}.{stamp}.bak")                 # 바뀌기 전 파일은 남겨 둔다
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        changed.append(rel)
    return changed


def update_and_restart():
    """맨 처음 부른다. 새 코드를 받았으면 같은 명령으로 다시 켜고 이 프로세스는 끝낸다."""
    if "--no-update" in sys.argv:
        sys.argv.remove("--no-update")
        return
    if os.environ.get("WEBCAM_PIPER_NO_UPDATE") or os.path.exists(os.path.join(HERE, ".no-autoupdate")):
        return
    res = _via_git()
    if res is None:
        try:
            changed = _via_api()
        except Exception as e:                                       # 인터넷 없음, GitHub 제한 등
            print(f"[업데이트] 새 코드를 확인하지 못함 ({type(e).__name__}) — 지금 코드로 켭니다")
            return
        if changed:
            print(f"[업데이트] 새 파일 {len(changed)}개 받음: {', '.join(changed[:8])}" + (" …" if len(changed) > 8 else ""))
        updated = bool(changed)
    else:
        updated, changed = res
    if not updated:
        return
    if any(os.path.basename(c) == "requirements.txt" for c in changed):
        print("[업데이트] 필요한 프로그램 목록이 바뀜 — pip 로 설치")
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(HERE, "requirements.txt")])
    print("[업데이트] 새 코드로 다시 켭니다")
    sys.stdout.flush()
    env = dict(os.environ, WEBCAM_PIPER_NO_UPDATE="1")
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))


if __name__ == "__main__":                                          # python autoupdate.py : 받기만
    if not os.environ.get("WEBCAM_PIPER_NO_UPDATE"):
        update_and_restart()
        print("[업데이트] 받을 새 코드 없음 (또는 확인 못 함)")
