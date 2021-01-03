import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from mss import base, mss
from PIL import Image


def main():
    # last_time = time.time()
    # pyautogui.PAUSE = 0
    # sct = mss()
    # # codec = cv2.VideoWriter_fourcc(*'MPEG')
    # # codec = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
    # # codec = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    # codec = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter("video_test1.avi", codec, 30, (640, 480))
    # if out.isOpened():
    #     print("video_writer: OPENED")

    cap = cv2.VideoCapture("testvid.mp4")

    filename = "outvid.avi"
    codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter("video_test1.avi", codec, 30, (600, 200))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (600, 200))
            cv2.imshow('frame', frame)
            out.write(frame)

        # screen_np = np.array(sct.grab({'top': 0, 'left': 0, 'width': 600, 'height': 200}))
        # screen_np = cv2.resize(screen_np, (640, 480))

        # print("shape: {}".format(screen_np.shape))
        # frame = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print("shape: {}".format(frame.shape))

        #
        # cv2.imshow("Screencapture:", frame)
        # print('FPS: {} '.format((1.0 / (time.time() - last_time))))
        # last_time = time.time()
        # out.write(screen_np)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            out.release()
            cap.release()
            cv2.destroyAllWindows()
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
