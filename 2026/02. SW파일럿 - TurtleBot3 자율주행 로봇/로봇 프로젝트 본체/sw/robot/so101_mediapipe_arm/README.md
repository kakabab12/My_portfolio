# SO-101 MediaPipe 팔 관절 조종

웹캠에서 사용자의 **한쪽 팔 관절**과 손 모양을 인식해 SO-101 follower arm을
상대 위치 방식으로 조종하는 독립 예제입니다. 기존 TurtleBot3 제스처 코드에는
의존하지 않습니다.

## 동작 대응

기준 자세를 한 번 잡은 뒤, 사람 팔의 변화량을 로봇팔의 현재 안전 자세에 더합니다.
그래서 로봇팔의 기계적 원점이나 사람의 체형에 맞추기 위해 팔을 처음부터 같은
자세로 맞출 필요가 없습니다.

| 사람 오른팔 동작 | SO-101 관절 |
| --- | --- |
| 어깨를 좌우/앞뒤로 움직임 | `shoulder_pan` |
| 위팔을 위·아래로 듦 | `shoulder_lift` |
| 팔꿈치를 굽힘 | `elbow_flex` |
| 손목을 굽힘 | `wrist_flex` |
| 손바닥을 비틂 | `wrist_roll` |
| 엄지 끝과 검지·중지·약지·새끼 손가락 끝의 거리 | `gripper` 열기/닫기 |

좌우 방향이 반대이거나 특정 관절이 반대로 움직이면
[`config.json`](config.json)의 해당 `direction`을 `1` ↔ `-1`로 바꿉니다.

## 먼저 안전하게 확인

1. 로봇팔을 책상 위의 충돌 없는 **안전 자세**에 놓습니다. 손으로 움직일 수 있는
   상태라면 팔을 지지합니다.
2. 아래 명령으로 카메라·관절 인식만 확인합니다. 이 모드는 USB 모터 포트를 열지
   않으며 화면에 계산된 목표값만 표시합니다.

   ```bash
   cd /home/user/sw/robot/so101_mediapipe_arm
   bash run.sh
   ```

3. 카메라 창에서 `C`를 눌러 현재 사람 팔 자세를 기준 자세로 저장합니다. 화면의
   `CALIBRATED`를 확인하고 천천히 팔을 움직여 각 목표값이 자연스럽게 변하는지
   봅니다.

## 실제 로봇팔 연결

현재 USB 연결에서는 TurtleBot3 OpenCR이 `/dev/ttyACM0`으로 확인됐습니다. 따라서
SO-101 follower controller 후보는 `/dev/ttyACM1`입니다. **아래 명령 전에는 케이블을
눈으로 한 번 확인하고, `ttyACM0`을 절대로 SO-101 포트로 쓰지 않습니다.**

```bash
cd /home/user/sw/robot/so101_mediapipe_arm
bash run.sh --enable-arm --port /dev/ttyACM1
```

`--enable-arm`을 줘야만 모터 통신 프로세스를 시작합니다. 시작할 때 6개 모터의
현재 위치를 읽어 안전 기준 위치로 사용하지만, 이 단계에서는 토크를 새로 켜지
않습니다.

카메라 창에서의 키:

| 키 | 기능 |
| --- | --- |
| `C` | 현재 사람 팔을 기준 자세로 저장. 로봇을 움직이기 전에 사용 |
| `Space` | `C`로 기준을 잡은 뒤 추종 시작/일시정지. 처음에는 일시정지 |
| `R` | 사람이 팔을 잃었거나 일시정지 뒤 새 안전 자세를 로봇 기준 위치로 다시 읽음 |
| `X` | 6개 모터 토크 해제. 팔이 중력으로 내려갈 수 있으므로 반드시 지지한 상태에서만 사용 |
| `Esc` 또는 `Q` | 추종 종료. 마지막 위치에서 유지하며 새 목표는 보내지 않음 |

처음에는 집게와 충돌물이 없는 상태에서, 팔을 아주 작게 움직이며 한 관절씩
방향을 확인합니다. 방향이 반대이면 프로그램을 종료하고 `config.json`의 그 관절
`direction`을 바꾼 뒤 다시 `C`부터 합니다.

## 준비된 환경

- 영상 처리: 이 저장소의 `.venv` (Python 3.10, MediaPipe 0.10.14)
- 모터 처리: `/home/user/miniconda3/envs/lerobot312`의 Feetech SDK
- follower 보정값 기본 경로:
  `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_follower.json`

보정 파일은 이미 존재하는 LeRobot follower 보정값을 읽기만 하며, 이 예제는
보정값이나 모터 EEPROM을 변경하지 않습니다. 파일 이름이 다르면 `--calibration`
옵션으로 지정합니다.

```bash
bash run.sh --enable-arm --port /dev/ttyACM1 \
  --calibration ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<robot-id>.json
```

## 테스트

카메라와 로봇팔 없이 관절 계산을 검증합니다.

```bash
cd /home/user/sw/robot/so101_mediapipe_arm
../.venv/bin/python -m unittest discover -s tests -v
```

