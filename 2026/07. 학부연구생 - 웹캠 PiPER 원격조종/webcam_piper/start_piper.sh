#!/bin/bash
# 바탕화면 아이콘 'PiPER 조종' 이 부르는 스크립트 (11/20).
#   start_piper.sh          → 창 두 개를 띄움: 'PiPER 서버' (팔) 와 'PiPER 카메라' (손 재기, 카메라 두 대)
#   start_piper.sh server   → 서버만 (CAN 켜기·교시 모드 확인 → robot_server, 안전 안내에서 Enter 는 사람이 누름)
#   start_piper.sh cam      → 카메라만 (서버가 켜질 때까지 기다렸다가 teleop)
# 끄기: 카메라 창 Q (또는 ESC), 서버 창 Ctrl+C. 팔은 그 자리에 멈춰 힘을 준 채 둔다.

D=/home/user/smolVLA/webcam_piper
ARM_PY=/home/user/smolVLA/chanhyuk_piper-teleop/.venv/bin/python      # agx-bus 가 깔린 찬혁님 venv (팔)
CAM_PY=$D/.venv/bin/python                                             # MediaPipe venv (카메라)
PORT=5005
# 11/19 실물 따라가기 설정. LIFT = 집게가 손보다 이만큼 위에서 따라감 (11/19 0.10).
# 11/20 0 으로: 서버가 종이 + 를 처음 자세 집게 끝에 맞추므로, 손을 + 에 대고 SPACE 를 누르면 팔이 제자리에 있음
LIFT=0
SERVER_ARGS="--robot agxbus --speed 30 --lift $LIFT --max-speed 0.3 --joint-rate 1.5 --rot-rate 90"
CAM_ARGS="--cam 0 --cam2 2 --send 127.0.0.1"

server_up() { ss -Hlun "sport = :$PORT" 2>/dev/null | grep -q .; }

run_server() {
    cd "$D" || exit 1
    echo "== PiPER 서버 (팔) =="
    if server_up; then
        echo "서버가 이미 켜져 있습니다 (포트 $PORT). 두 개를 켜면 한 팔에 명령이 겹치므로 켜지 않습니다."
        read -r -p "Enter 로 창 닫기 " _; exit 0
    fi
    if ! ip link show can0 2>/dev/null | grep -q "state UP"; then
        echo "CAN(can0) 이 꺼져 있어 켭니다. 비밀번호를 물으면 입력하세요."
        sudo ip link set can0 down 2>/dev/null
        sudo ip link set can0 up type can bitrate 1000000 || { echo "CAN 을 못 켬: USB-CAN 어댑터 연결을 확인하세요"; read -r -p "Enter 로 닫기 " _; exit 1; }
    fi
    while true; do
        out=$("$CAM_PY" save_home.py --show 2>&1)
        echo "$out"
        if echo "$out" | grep -q "못 받음"; then
            read -r -p "팔 전원을 켜고 Enter (그만두려면 Ctrl+C) " _
        elif echo "$out" | grep -q "교시"; then
            echo "팔이 교시 모드(손으로 끌기)라 명령을 받지 않습니다. 팔을 한 손으로 받친 채 전원을 껐다 켜세요."
            read -r -p "껐다 켰으면 Enter (그만두려면 Ctrl+C) " _
        else
            break
        fi
    done
    "$ARM_PY" -u robot_server.py $SERVER_ARGS
    echo
    read -r -p "서버가 끝났습니다. Enter 로 창 닫기 " _
}

run_cam() {
    cd "$D" || exit 1
    echo "== PiPER 카메라 (손 재기) =="
    if ! server_up; then
        echo "서버 창의 안전 안내에서 Enter 를 누르면 카메라가 켜집니다 (기다리는 중)..."
        for _ in $(seq 600); do server_up && break; sleep 0.5; done
    fi
    if ! server_up; then
        echo "5분 동안 서버가 켜지지 않아 그만둡니다."; read -r -p "Enter 로 닫기 " _; exit 1
    fi
    "$CAM_PY" -u teleop.py $CAM_ARGS 2>&1 | grep --line-buffered -v -E "^(INFO|W0000|WARNING: Logging|Corrupt JPEG)"
    echo
    read -r -p "카메라가 끝났습니다. Enter 로 창 닫기 " _
}

case "$1" in
    server) run_server ;;
    cam) run_cam ;;
    *)
        gnome-terminal --title="PiPER 서버" --geometry=110x30 -- bash "$D/start_piper.sh" server
        sleep 1
        gnome-terminal --title="PiPER 카메라" --geometry=110x20 -- bash "$D/start_piper.sh" cam
        ;;
esac
