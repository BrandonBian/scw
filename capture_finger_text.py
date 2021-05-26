import time
import cv2

if __name__ == "__main__":
    print("Start Capturing...")
    cap = cv2.VideoCapture(2)
    i = 0;
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            print(f"Recorded frame {i}")

            cv2.imwrite(f"journal_finger/camera{i}.jpg", frame)
            i += 1
            time.sleep(0.2)

        else:
            print(f"Failed at iteration {i}")

