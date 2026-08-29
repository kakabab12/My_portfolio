<div align="center">

# 개인 프로젝트

시기를 특정하기 어렵거나, 연도와 무관하게 이어 온 것들

</div>

---

| 프로젝트 | 무엇인가 | 분류 |
|---|---|:-:|
| [**고클린 — 윈도우 최적화 도구**](고클린%20-%20윈도우%20최적화%20도구/) | PC 청소·게임모드·핑테스트·PC정보를 한 화면에 모았습니다. **exe로 빌드해 배포** | 개인 도구 |
| [**마인크래프트 falling-pickaxe**](%5B외부코드%5D%20마인크래프트%20falling-pickaxe/) | ⚠ **타인 오픈소스** — 학습·개조 목적으로 받아둔 코드 | 외부 |

> **연도가 확인된 것들은 연도 폴더로 옮겼습니다.**
> · Express 블로그 - 게시판 → [`2021/01.`](../2021/01.%20Express%20블로그%20-%20게시판/) (2021-06)
> · XAMPP 웹 자기소개서 → [`2025/03.`](../2025/03.%20XAMPP%20웹%20자기소개서/) (2025-07)
> · 임베디드 리눅스 · ASP DB 연동 → [`2024/`](../2024/) (2024-2학기)
> · 웹캠 로봇팔 원격조종 → [`2026/05.`](../2026/05.%20웹캠%20로봇팔%20원격조종/) (2026)

---

## 고클린 — 윈도우 최적화 도구

> [![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](../기술_용어집.md#python)
> [![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat-square&logo=windows&logoColor=white)](../기술_용어집.md#python)
> [![PyInstaller](https://img.shields.io/badge/PyInstaller-306998?style=flat-square)](../기술_용어집.md#pyinstaller)

시스템 관리 기능을 한 화면에 모은 개인 유틸리티입니다. 내가 쓰려고 만들었습니다.

| 기능 | 내용 |
|---|---|
| **시스템 업데이트** | `winget`·`pip`를 업데이트하고 윈도우 업데이트/MS Store를 자동으로 띄웁니다. `wmic`으로 **GPU 제조사(NVIDIA/AMD/Intel)를 감지해 해당 드라이버 페이지를 열어준다** |
| **청소 & 부스트** | DNS 캐시 초기화, Temp 삭제, `dism` 이미지 정리, `netsh winsock reset`, 디스크 조각모음 |
| **게임 모드** | 무거운 백그라운드 프로그램을 일괄 종료하고 탐색기를 재시작해 메모리를 회수합니다 |
| **핑 테스트** | 8.8.8.8과 1.1.1.1 핑 결과를 **정규식으로 파싱**해 지연시간을 등급으로 평가합니다 |
| **PC 정보** | CPU·메인보드·GPU·RAM과 내부/외부 IP를 조회합니다 |

**기술 포인트**

- **관리자 권한 자동 획득** — 권한을 확인하고 없으면 UAC 창을 띄워 스스로 재실행합니다.
  시스템 파일 삭제나 네트워크 설정 변경에는 관리자 권한이 필수이기 때문입니다
- **GUI 멈춤 방지** — 모든 작업을 `threading.Thread(daemon=True)`로 분리했습니다.
  `subprocess` 명령이 끝날 때까지 기다리면 창이 "응답 없음"이 됩니다
- 터미널 인코딩을 UTF-8(`chcp 65001`)로 바꿔 한글 깨짐을 막았습니다

| 폴더 | 버전 |
|---|---|
| `alpha.ver/` | **GUI 버전**과 exe 빌드 결과물 |
| `초기.ver/` | 초기 배치파일(.bat) 버전 — `fltmc` 명령으로 권한을 확인하는 방식이었습니다 |

**실행**

```bash
python alpha.ver/alpha_test1.py     # 또는 alpha.ver/alpha_test1.exe
```

---

## ⚠ 마인크래프트 falling-pickaxe (외부 코드)

**직접 만든 것이 아닙니다.** [vycdev/falling-pickaxe](https://github.com/vycdev/falling-pickaxe)
(GPL-3.0) 프로젝트를 **학습·개조 목적으로** 받아둔 것입니다.
Pygame 기반 게임 루프 구조와 YouTube 실시간 연동 방식을 참고하려고 보관했습니다.

폴더명에 `[외부코드]` 를 붙여 직접 만든 프로젝트와 구분했습니다.
