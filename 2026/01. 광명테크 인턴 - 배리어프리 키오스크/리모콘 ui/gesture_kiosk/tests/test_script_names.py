# -*- coding: utf-8 -*-
"""실행 스크립트 전부에서 "정의 없이 쓰이는 이름"과 "import보다 먼저 쓰는 이름"을 잡는다
(2026-09-24 신설).

왜 필요한가
-----------
test_tracker_names.py는 세 트래커(head/eyebrow/forehead.py)만 본다. 출시 전 점검에서
그 밖의 스크립트에 두 가지가 숨어 있었다.

1. 빌드 도우미 6개와 pipe_listen.py — 파일 맨 위(docstring 바로 뒤)에

       sys.path.insert(0, os.path.dirname(...))
       import os
       import sys

   순서로 적혀 있었다. sys·os를 import하기 **전에** 쓰니 실행하자마자 NameError다.
   make_head_exe.bat 등으로 exe를 새로 빌드할 수 없는 상태였다. 이름이 파일 안에
   있기는 하므로 test_tracker_names.py 방식("어딘가에 정의가 있나")으로는 안 잡힌다.

2. scripts/check_direction.py — 좌우를 따로 재도록 고치면서 옛 코드 두 줄이 남아
   정의된 적 없는 face_x_at_peak·base_face_x를 썼다. 사용자가 안내대로 **정확히**
   움직였을 때만 가는 자리라, 제대로 쓴 사람만 NameError를 봤을 것이다.

둘 다 컴파일은 통과하고 테스트 817개도 통과했다. 파이썬은 이름을 쓸 때 찾기
때문이다(2026-09-10 일지 8절과 같은 모양의 실수).
"""
import ast
import glob
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.test_tracker_names import BUILTINS, _bound_names, _loaded_names  # noqa: E402

MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__spec__", "__package__",
                  "__loader__", "__builtins__", "__path__"}


def _python_files():
    """출시물에 들어가는 실행 코드 — 진입점, scripts, src."""
    files = ["head.py", "eyebrow.py", "forehead.py",
             "main.py", "main_dpad.py", "main_dpad_overlay_fullscreen.py"]
    files += sorted(os.path.relpath(p, ROOT) for p in
                    glob.glob(os.path.join(ROOT, "scripts", "**", "*.py"), recursive=True))
    files += sorted(os.path.relpath(p, ROOT) for p in
                    glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True))
    return [f for f in files if os.path.exists(os.path.join(ROOT, f))]


FILES = _python_files()


def _parse(relpath):
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as fp:
        return ast.parse(fp.read(), relpath)


@pytest.mark.parametrize("relpath", FILES)
def test_no_undefined_names(relpath):
    """정의가 아예 없는 이름 — 실행 중 그 줄에 닿는 순간 NameError."""
    tree = _parse(relpath)
    unknown = sorted(_loaded_names(tree) - _bound_names(tree) - BUILTINS - MODULE_DUNDERS)
    assert not unknown, "%s에서 정의 없이 쓰이는 이름: %s" % (relpath, ", ".join(unknown))


# ── import보다 먼저 쓰는 이름 ─────────────────────────────────────────────────

def _names_in(nodes):
    """지금 바로 실행되는 식에서 읽는 이름. lambda 몸통은 나중에 도므로 뺀다."""
    found = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if node is None:
            continue
        if isinstance(node, ast.Lambda):
            stack.extend(node.args.defaults + [d for d in node.args.kw_defaults if d])
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            found.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _binds_of(stmt):
    """이 문장이 모듈 범위에 묶는 이름(몸통 안 함수·클래스 내부는 뺀다)."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in stmt.names}
    bound = set()
    for node in ast.walk(stmt):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def _early_uses(body, bound, module_names, problems):
    """모듈 최상위 문장을 차례로 따라가며, 나중에야 묶이는 이름을 먼저 읽는 곳을 모은다."""
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            now = _names_in(stmt.decorator_list + stmt.args.defaults
                            + [d for d in stmt.args.kw_defaults if d])
        elif isinstance(stmt, ast.ClassDef):
            now = _names_in(stmt.decorator_list + stmt.bases + [k.value for k in stmt.keywords])
        elif isinstance(stmt, (ast.If, ast.While)):
            now = _names_in([stmt.test])
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            now = _names_in([stmt.iter])
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            now = _names_in([i.context_expr for i in stmt.items])
        elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            now = set()
        else:
            now = _names_in([stmt])
        for name in sorted(now - bound):
            if name in module_names and name not in BUILTINS:
                problems.append("%d행에서 %s를 쓰는데 정의(import)는 그 뒤에 있다"
                                % (stmt.lineno, name))
        # 복합문은 몸통을 순서대로 따라간다
        for field in ("body", "orelse", "finalbody"):
            block = getattr(stmt, field, None)
            if block and not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                ast.ClassDef)):
                _early_uses(block, bound, module_names, problems)
        for handler in getattr(stmt, "handlers", []) or []:
            if handler.name:
                bound.add(handler.name)
            _early_uses(handler.body, bound, module_names, problems)
        bound.update(_binds_of(stmt))


@pytest.mark.parametrize("relpath", FILES)
def test_nothing_is_used_before_it_is_imported(relpath):
    """sys.path.insert(...)가 import sys보다 위에 있던 빌드 도우미 같은 경우."""
    tree = _parse(relpath)
    module_names = set()
    for stmt in tree.body:
        module_names |= _binds_of(stmt)
    problems = []
    _early_uses(tree.body, set(MODULE_DUNDERS), module_names, problems)
    assert not problems, "%s: %s" % (relpath, "; ".join(problems))


def test_the_check_really_catches_the_bug_we_had():
    """검사가 있다는 착각만 남지 않도록 — 실제로 있었던 모양을 넣으면 빨간불이어야 한다."""
    buggy = ('"""doc"""\n'
             'sys.path.insert(0, os.path.dirname(__file__))\n'
             'import os\nimport sys\n')
    tree = ast.parse(buggy)
    module_names = set()
    for stmt in tree.body:
        module_names |= _binds_of(stmt)
    problems = []
    _early_uses(tree.body, set(MODULE_DUNDERS), module_names, problems)
    assert any("sys" in p for p in problems) and any("os" in p for p in problems)

    leftover = "def f():\n    return face_x_at_peak - base_face_x\n"
    tree = ast.parse(leftover)
    unknown = _loaded_names(tree) - _bound_names(tree) - BUILTINS - MODULE_DUNDERS
    assert unknown == {"face_x_at_peak", "base_face_x"}
