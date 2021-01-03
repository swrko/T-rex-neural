import numpy as np
import cv2
import time
from mss import mss



def main():

    # # codec = cv2.VideoWriter_fourcc(*'MPEG')
    # codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("video_test1.avi", codec, 30, (600, 200))

    sct = mss()
    monitor = {'top': 0, 'left': 0, 'width': 600, 'height': 200}
    last_time = time.time()

    while True:
        frame = np.array(sct.grab(monitor))
        frame = frame[:, :, :3]
        frame = cv2.resize(frame, (600, 200))

        # print("shape: {}".format(frame.shape))
        cv2.imshow('frame', frame)
        out.write(frame)

        print('FPS: {} '.format((1.0 / (time.time() - last_time))))
        last_time = time.time()
        out.write(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            out.release()
            cv2.destroyAllWindows()
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
