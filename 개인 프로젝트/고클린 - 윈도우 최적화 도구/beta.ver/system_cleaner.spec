# -*- mode: python ; coding: utf-8 -*-
#
# 빌드:  pyinstaller system_cleaner.spec
#
# alpha 버전 spec 대비 변경점
#  - uac_admin=True : 런타임 ShellExecuteW 재실행보다 UAC 승격이 훨씬 안정적이다.
#                     (경로에 공백/한글이 있어도 안 깨짐)
#  - psutil hiddenimport 추가
#  - upx=False      : UPX 로 압축하면 백신 오탐이 크게 늘어난다. 최적화 도구는
#                     가뜩이나 오탐이 잦으므로 끄는 쪽이 배포에 유리하다.

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries = [], []
hiddenimports = ["psutil", "tkinter", "tkinter.ttk", "tkinter.filedialog"]
hiddenimports += collect_submodules("system_cleaner")

for pkg in ("customtkinter",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h


a = Analysis(
    ['run.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'numpy', 'pandas', 'PIL.ImageQt', 'pytest', 'IPython'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SystemCleaner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=True,
)
