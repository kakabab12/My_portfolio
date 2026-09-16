# -*- mode: python ; coding: utf-8 -*-
#
# 빌드:  pyinstaller --noconfirm system_cleaner.spec
# 검증:  dist\SystemCleaner.exe --selftest
#
# 결정 사항
#  - uac_admin=False : 매니페스트로 관리자 권한을 강제하면 UAC 를 거절했을 때 앱이
#                      아예 안 켜지고, 자동 검증으로 exe 를 띄울 수도 없다.
#                      앱이 시작할 때 직접 승격을 요청하고, 거절하면 제한 모드로 뜬다.
#  - upx=False       : UPX 압축은 백신 오탐을 크게 늘린다.
#  - version         : 파일 속성에 제품 정보가 보이게 한다. 버전 정보가 없는
#                      서명 안 된 exe 는 휴리스틱 검사에서 더 의심받는다.

from PyInstaller.utils.hooks import collect_all, collect_submodules
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable,
    VarFileInfo, VarStruct, VSVersionInfo,
)

datas = [("system_cleaner/assets", "system_cleaner/assets")]
binaries = []
hiddenimports = ["psutil", "tkinter", "tkinter.ttk", "tkinter.filedialog"]
hiddenimports += collect_submodules("system_cleaner")

for pkg in ("customtkinter",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=(1, 0, 0, 0), prodvers=(1, 0, 0, 0), mask=0x3F,
                      flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
    kids=[
        StringFileInfo([StringTable("041204B0", [
            StringStruct("CompanyName", "이지용"),
            StringStruct("FileDescription", "SYSTEM CLEANER - 윈도우 최적화 도구"),
            StringStruct("FileVersion", "1.0.0-beta"),
            StringStruct("InternalName", "SystemCleaner"),
            StringStruct("LegalCopyright", "(c) 2026 이지용"),
            StringStruct("OriginalFilename", "SystemCleaner.exe"),
            StringStruct("ProductName", "SYSTEM CLEANER"),
            StringStruct("ProductVersion", "1.0.0-beta"),
        ])]),
        VarFileInfo([VarStruct("Translation", [0x0412, 1200])]),
    ],
)

a = Analysis(
    ['run.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'numpy', 'pandas', 'PIL', 'pytest', 'IPython', 'jedi'],
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
    icon='system_cleaner/assets/icon.ico',
    version=version_info,
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
    uac_admin=False,
)
