import cv2

print("카메라 테스트를 시작합니다...")
cap = cv2.VideoCapture(0) # 다이소 웹캠 기본 인덱스

if not cap.isOpened():
    print("[에러] 카메라를 열 수 없습니다. /dev/video0 번호를 확인하거나 권한을 체크하세요.")
else:
    print("[성공] 카메라가 정상적으로 열렸습니다!")
    ret, frame = cap.read()
    if ret:
        print(f"영상 해상도: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("[에러] 카메라는 열렸으나 프레임(영상)을 받아오지 못했습니다.")

cap.release()