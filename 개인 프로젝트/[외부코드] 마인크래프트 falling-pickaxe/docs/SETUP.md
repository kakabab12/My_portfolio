# Falling Pickaxe 방송 셋업 가이드

이 프로젝트는 [vycdev/falling-pickaxe](https://github.com/vycdev/falling-pickaxe) (GPL-3.0)를 기반으로,
후원 금액별 이펙트(티어 시스템)와 화면 알림/커맨드 피드를 추가한 버전입니다.

## 0. 꼭 지켜야 할 것 — 원작자 크레딧

이 게임을 방송에 쓰려면 원작자가 README에서 명시적으로 요청한 대로 **방송 설명란에 아래 크레딧을 반드시 넣어야 합니다.**

```
Falling Pickaxe game made by Vycdev
YT: https://www.youtube.com/@vycdev
GH: https://github.com/vycdev/falling-pickaxe
```

## 1. 실행 환경 준비

Python 3.x가 이미 설치되어 있다고 하셨으니 바로 실행 스크립트를 쓰면 됩니다.

```bash
powershell -ExecutionPolicy Bypass -File scripts/run.ps1
```

이 스크립트가 자동으로:
- `.venv` 가상환경 생성
- `requirements.txt` 의존성 설치
- `src/main.py` 실행, 크래시 시 자동 재시작 (창을 정상적으로 닫으면 스크립트도 종료)

최초 실행 시 `config.json`이 `default.config.json`에서 자동으로 복사됩니다.

## 2. 유튜브 채팅 연동 설정 (선택이지만 강력 추천)

`config.json`을 열어 값을 채우세요.

1. `CHAT_CONTROL`을 `true`로 변경
2. https://console.cloud.google.com 에서 프로젝트 생성 → "YouTube Data API v3" 활성화 → API 키 발급 → `API_KEY`에 입력
3. `CHANNEL_ID`: 본인 채널 ID (신규 구독자 감지 + MegaTNT 자동 발동용)
4. `LIVESTREAM_ID`: 방송을 시작한 뒤 방송 URL 전체를 붙여넣어도 자동 추출됩니다 (`watch?v=`, `/live/`, `youtu.be/` 형식 모두 지원)

`CHAT_CONTROL: false`로 두면 유튜브 연동 없이 게임만 자동으로 랜덤하게 진행됩니다 (테스트용).

## 3. 채팅 명령어 (댓글로 게임이 바뀌는 것들)

시청자가 채팅에 아래 단어를 포함해서 치면 게임에 바로 반영되고, 화면 하단에 "닉네임: 명령"으로 표시됩니다.

| 명령어 | 효과 |
|---|---|
| `tnt` | TNT 하나 소환 (닉네임이 TNT 위에 표시됨) |
| `fast` / `slow` | 곡괭이 낙하 속도 변경 |
| `big` | 곡괭이 일시적으로 커짐 |
| `nuke` | 초대형 MegaTNT(6배 크기) 소환 + 강한 화면 흔들림 |
| `missile` | TNT 6개가 흩어져서 동시에 떨어지는 융단폭격 |
| `meteor` | 대형 MegaTNT(3배) + 화염 파티클 트레일 |
| `earthquake` | 폭발 없이 강력한 화면 흔들림만 발생 |
| `shower` | 황금비 연출 (금괴가 쏟아지는 장식 이펙트, 충돌 없음) |
| `freeze` | 2초간 낙하 정지 |
| `tiny` | 곡괭이 축소 (big의 반대) |
| `confetti` | 색종이 파티클 파티 (충돌 없음) |
| `lucky` | 즉시 소량의 랜덤 광물 보너스 지급 |
| `slowmo` | 일반 slow보다 훨씬 강한 슬로우모션 |
| `wood`/`stone`/`iron`/`gold`/`diamond`/`netherite` | 곡괭이 재질 변경 |

같은 사람이 연속으로 같은 명령을 스팸해도 대기열에 한 번만 들어가도록 이미 처리되어 있습니다.

## 4. 후원(슈퍼챗) 금액별 이펙트 — 이번에 추가한 기능

원본은 슈퍼챗 금액과 상관없이 TNT 10개만 터뜨렸는데, 금액대별로 아래처럼 차등 적용되도록 만들었습니다 (`default.config.json`의 `DONATION_TIERS`에서 자유롭게 조정 가능):

| 티어 | 금액대 (KRW) | 효과 |
|---|---|---|
| 1 | 1,000 ~ 4,999 | 일반 TNT 3개 |
| 2 | 5,000 ~ 9,999 | 일반 TNT 6개 |
| 3 | 10,000 ~ 29,999 | MegaTNT 3개 + 카메라 흔들림 강화 |
| 4 | 30,000 이상 | 초대형 MegaTNT(3배 크기) 5개 + 강한 카메라 흔들림 |

후원이 들어오면 화면 상단에 "닉네임 님이 5,000원 후원!" 같은 배너가 티어별 색상(파랑→노랑→주황→빨강)으로 몇 초간 표시됩니다. 원화(KRW)가 아닌 통화로 후원이 들어오면 유튜브 자체 슈퍼챗 등급(1~수십)을 기준으로 자동으로 티어를 매칭합니다.

## 5. OBS 방송 구성

이 게임은 세로 화면(9:16, 쇼츠/세로 라이브에 최적화)으로 렌더링됩니다.

1. OBS 씬에 **창 캡처(Window Capture)** 소스 추가 → "Falling Pickaxe" 창 선택
2. 세로 방송이면 캔버스 해상도를 1080x1920으로, 가로 방송이면 게임 창을 화면 한쪽에 배치하고 나머지 공간에 다른 요소(채팅창 등) 배치
3. 데스크톱 오디오 소스 추가 (TNT 폭발음 등 게임 내장 사운드가 여기서 나옵니다)
4. 설정 > 방송에서 유튜브 선택, 유튜브 스튜디오(studio.youtube.com)에서 발급한 스트림 키 입력
5. 방송 설명란에 0번 항목의 크레딧 문구 + 참여 방법(`tnt`, `fast`, `slow`, `big`, 재질 이름 등 댓글로 참여 가능하다는 안내) 작성
6. 처음엔 **미등록(unlisted)**으로 30분~1시간 시범 운영 후 크래시/성능 문제 없는지 확인하고 공개 전환

## 6. 문제 해결

- 유튜브 연동이 안 됨: 콘솔에 출력되는 로그(`Live stream found`, `Live chat ID found` 등)를 확인. `LIVESTREAM_ID`가 정확한지, API 키에 YouTube Data API v3가 활성화되어 있는지 확인.
- 신규 구독자 MegaTNT가 안 터짐: `CHANNEL_ID`가 올바른지, `CHAT_CONTROL`이 `true`인지 확인. 구독자 수 API는 폴링 간격(`YT_POLL_INTERVAL_SECONDS`)마다만 갱신됩니다.
- 후원 이펙트가 티어대로 안 나뉨: `config.json`의 `DONATION_TIERS`가 존재하는지, 금액 통화가 KRW인지 확인 (다른 통화는 유튜브 자체 등급으로 대체 매칭됩니다).
