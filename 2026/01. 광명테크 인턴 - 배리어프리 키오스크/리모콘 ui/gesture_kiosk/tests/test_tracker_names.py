# -*- coding: utf-8 -*-
"""세 트래커에서 "정의 없이 쓰이는 이름"을 잡는다 (2026-09-10 신설).

왜 필요한가
-----------
입 판정을 모듈로 꺼내면서 `mouth_gesture_state = {` 부터
`def _release_mouth_hold_if_stuck():` 앞까지를 통째로 갈아 끼웠는데, 그
사이에 **상관없는 코드 두 덩이가 끼여 있었다.**

    feedback = CursorFeedback()      # 커서 색(클릭 노랑 / 드래그 파랑)
    def _reset_recenter_timer(): ... # 조작 중 자동 재정렬 막기

둘 다 함께 지워졌다. `py_compile`도 통과하고 `--check`도 "정상"이라고 답한다
— 파이썬은 이름을 **쓸 때** 찾기 때문이다. 그래서 실행하고 나서야, 그것도
입을 벌려 클릭한 뒤에야 NameError로 죽었다. 사용자가 "실행이 안 된다"고
알려 줘서 알았다.

이 시험은 그걸 정적으로 잡는다. 완벽한 검사는 아니지만(파이썬은 이름이
언제 묶이는지 실행해 봐야 정확히 안다), **통째로 지워진 정의**라는 이 실수의
모양은 확실히 잡는다.

세 파일은 main() 안쪽이 클로저 덩어리라 import만으로는 아무것도 확인되지
않는다. 그래서 소스를 읽어서 본다.
"""
import ast
import builtins
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRACKERS = ("head.py", "eyebrow.py", "forehead.py")
BUILTINS = set(dir(builtins))


def _bound_names(tree):
    """이 파일 어딘가에서 이름이 묶이는 것 전부.

    범위를 따지지 않고 모은다 — 범위까지 보려면 사실상 인터프리터를 다시
    만들어야 하고, 여기서 잡으려는 것은 "아예 사라진 정의"라서 이걸로 충분하다.
    """
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                               ast.Lambda)):
            # lambda는 이름이 없고 매개변수만 묶는다 (2026-09-24 — `lambda item: ...`의
            # item을 정의 없는 이름으로 잘못 세던 빈틈. test_script_names.py 참고)
            if not isinstance(node, ast.Lambda):
                bound.add(node.name)
            args = getattr(node, "args", None)
            if args is not None:
                for group in (args.posonlyargs, args.args, args.kwonlyargs):
                    bound.update(a.arg for a in group)
                for extra in (args.vararg, args.kwarg):
                    if extra is not None:
                        bound.add(extra.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


def _loaded_names(tree):
    return {node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


@pytest.mark.parametrize("filename", TRACKERS)
def test_no_undefined_names(filename):
    """정의 없이 쓰이는 이름이 있으면 실행 중에 NameError로 죽는다."""
    tree = ast.parse(open(os.path.join(ROOT, filename), encoding="utf-8").read())
    unknown = sorted(_loaded_names(tree) - _bound_names(tree) - BUILTINS
                     - {"__file__", "__name__", "__doc__"})
    assert not unknown, (
        "%s에서 정의 없이 쓰이는 이름: %s — 리팩터하다 정의를 지웠을 수 있다"
        % (filename, ", ".join(unknown)))


# 위 검사는 이름이 "어딘가에" 있기만 하면 통과한다. 실행에 반드시 있어야
# 하는 몇 개는 이름을 박아서 따로 확인한다 — 실제로 지워졌던 것들이다.
MUST_EXIST = (
    "feedback = CursorFeedback()",       # 없으면 클릭 색이 NameError
    "def _reset_recenter_timer():",      # 없으면 클릭할 때 NameError
    "mouth = MouthGesture(",             # 입 판정
    "click_freeze = ClickFreeze(",       # 클릭 중 커서 붙잡기
)


@pytest.mark.parametrize("filename", TRACKERS)
@pytest.mark.parametrize("snippet", MUST_EXIST)
def test_required_wiring_is_present(filename, snippet):
    source = open(os.path.join(ROOT, filename), encoding="utf-8").read()
    assert snippet in source, "%s에 '%s'가 없다" % (filename, snippet)
