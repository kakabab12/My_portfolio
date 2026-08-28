<div align="center">

# 💡 개인 프로젝트

### 연도와 무관하게, 필요해서 만들거나 궁금해서 파본 것들

</div>

---

| 프로젝트 | 무엇인가 | 분류 |
|---|---|:-:|
| [**🧹 고클린 — 윈도우 최적화 도구**](고클린%20-%20윈도우%20최적화%20도구/) | PC 청소·게임모드·핑테스트·PC정보를 한 화면에. **exe로 빌드 배포** | 💡 개인 도구 |
| [**📚 Express 도서관 웹앱**](Express%20도서관%20웹앱/) | Node.js + Express + MongoDB 도서관 관리 웹앱 | 🎓 수업 실습 |
| [**📄 XAMPP 웹 자기소개서**](XAMPP%20웹%20자기소개서/) | XAMPP(Apache+PHP+MySQL) 기반 웹 자기소개서 | 🎓 수업 과제 |
| [**⛏️ 마인크래프트 falling-pickaxe**](%5B외부코드%5D%20마인크래프트%20falling-pickaxe/) | ⚠️ **타인 오픈소스** — 학습·개조 목적으로 받아둔 코드 | 📦 외부 |

> 💡 웹캠 로봇팔 원격조종은 개인 프로젝트지만 2026년 작업이라
> **[`2026/05. 웹캠 로봇팔 원격조종/`](../2026/05.%20웹캠%20로봇팔%20원격조종/)** 에 있습니다.

---

## 🧹 고클린 — 윈도우 최적화 도구

> ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
> ![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat-square&logo=windows&logoColor=white)
> **PyInstaller로 단일 exe 빌드**

시스템 관리 기능을 한 화면에 모은 개인 유틸리티입니다.

| 기능 | 내용 |
|---|---|
| 🔄 **시스템 업데이트** | `winget`·`pip` 업데이트, 윈도우 업데이트/MS Store 자동 실행. `wmic`으로 **GPU 제조사(NVIDIA/AMD/Intel)를 감지해 해당 드라이버 페이지를 자동으로 열어줌** |
| 🧹 **청소 & 부스트** | DNS 캐시 초기화, Temp 삭제, `dism` 이미지 정리, `netsh winsock reset`, 디스크 조각모음 |
| 🎮 **게임 모드** | 무거운 백그라운드 프로그램 일괄 종료 후 탐색기 재시작으로 메모리 회수 |
| 📡 **핑 테스트** | 8.8.8.8 / 1.1.1.1 핑 결과를 **정규식으로 파싱**해 지연시간을 등급으로 평가 |
| 💻 **PC 정보** | CPU·메인보드·GPU·RAM + 내부/외부 IP 조회 |

**기술 포인트**

- **관리자 권한 자동 획득** — 권한을 확인하고 없으면 UAC 창을 띄워 스스로 재실행
- **GUI 멈춤 방지** — 모든 작업을 `threading.Thread(daemon=True)`로 분리해 "응답 없음" 방지
- 터미널 인코딩을 UTF-8(`chcp 65001`)로 바꿔 한글 깨짐 방지

| 폴더 | 버전 |
|---|---|
| `alpha.ver/` | **GUI 버전** + exe 빌드 결과물 |
| `초기.ver/` | 초기 배치파일(.bat) 버전 |

---

## 📚 Express 도서관 웹앱

Node.js **Express + MongoDB** 기반 도서관 관리 웹앱입니다.
도서·저자·장르·대출 상태를 관리하는 CRUD 구조를 익혔습니다.
실행화면 캡처 4장이 함께 있습니다.

---

## 📄 XAMPP 웹 자기소개서

**XAMPP**(Apache + PHP + MySQL) 환경에서 만든 웹 자기소개서 페이지입니다.

---

## ⚠️ 마인크래프트 falling-pickaxe (외부 코드)

**직접 만든 것이 아닙니다.**
[vycdev/falling-pickaxe](https://github.com/vycdev/falling-pickaxe) (GPL-3.0) 프로젝트를
**학습·개조 목적으로** 받아둔 것입니다.
Pygame 기반 게임 루프 구조와 YouTube 실시간 연동 방식을 참고하려고 보관했습니다.

> 폴더명에 `[외부코드]` 를 붙여 직접 만든 프로젝트와 구분했습니다.
