import cv2
import datetime

cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()
    if not ret:
        print("프레임 X")
        break

    cv2.imshow("Video",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):

        filename = datetime.datetime.now().strftime("./checkerboards/capture_%Y%m%d_%H%M%S.png")
        cv2.imwrite(filename,frame)
        print(f"{filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()