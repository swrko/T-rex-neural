import numpy as np
import cv2
import time
from mss import mss
from pynput.keyboard import Key, Listener

key_buffer = []
frame_stamp = 0


def fill_empty_inputs():
    pass


def on_press(key):
    global key_buffer, frame_stamp
    if str(key) == "'q'":
        return False
    if key == Key.up or key == Key.down:
        print('{0} pressed'.format(key))
        key_buffer.append(str(key) + ",{}".format(frame_stamp))


def write_to_file(keys):
    global key_buffer
    with open("keys.txt", "a") as file:
        for key in keys:
            file.write(str(key) + "\n")
    key_buffer = []
    print("writen!")


def main():
    global key_buffer, frame_stamp
    # writer settings
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("video_test1.avi", codec, 30, (600, 200))
    # monitor recording settings
    sct = mss()
    monitor = {'top': 0, 'left': 0, 'width': 600, 'height': 200}
    last_time = time.time()
    # creating & starting listener in non-blocking mode, as separate thread
    pause = False

    while True:
        if not pause:
            frame = np.array(sct.grab(monitor))
            frame = frame[:, :, :3]
            frame = cv2.resize(frame, (600, 200))

            # print("shape: {}".format(frame.shape))
            cv2.imshow('frame', frame)
            out.write(frame)

            print('FPS: {} '.format((1.0 / (time.time() - last_time))))
            last_time = time.time()
            out.write(frame)
            frame_stamp += 1
            if frame_stamp % 100 == 0:
                write_to_file(key_buffer)

            if (cv2.waitKey(10) & 0xFF == ord('q')) or (frame_stamp == 50000):
                write_to_file(key_buffer)
                out.release()
                cv2.destroyAllWindows()
                break
            if (cv2.waitKey(15) & 0xFF == ord('p')):
                print("pausing at frame: {}".format(frame_stamp))
                pause = True
        else:
            if (cv2.waitKey(15) & 0xFF == ord('p')):
                pause = False
            if (cv2.waitKey(10) & 0xFF == ord('q')):
                out.release()
                cv2.destroyAllWindows()
                break


listener = Listener(
    on_press=on_press)
listener.start()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
