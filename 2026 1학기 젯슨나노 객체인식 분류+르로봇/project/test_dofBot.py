import cv2

cap = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라 프레임 읽기 실패")
        break

    data, points, _ = detector.detectAndDecode(frame)

    if data:
        print("QR 인식 성공!")
        print("QR 내용:", data)
        break
    else:
        print("QR 인식 대기 중...")

    cv2.imshow("QR Test", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()