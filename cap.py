import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try CAP_DSHOW for Windows
if not cap.isOpened():
    print("Error: Couldn't open webcam.")
else:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Couldn't capture a frame.")
    else:
        cv2.imshow("Webcam Test", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
cap.release()
