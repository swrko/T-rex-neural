import numpy as np
import cv2
import time
from mss import mss
from pynput.keyboard import Key, Listener
import re

key_buffer = []
frame_stamp = 0


def on_press(key):
    global key_buffer, frame_stamp
    if str(key) == "'q'":
        return False
    if key == Key.up or key == Key.down:
        print('{0} pressed'.format(key))
        key_buffer.append(str(key) + ",{}".format(frame_stamp))


listener = Listener(
    on_press=on_press)


class DatasetCreator():
    def __init__(self):
        global key_buffer, frame_stamp

        self.list_of_outputs = []
        self.max_stamp = 0

    def write_to_file(self, keys):
        global key_buffer
        with open("keys.txt", "a") as file:
            for key in keys:
                file.write(str(key) + "\n")
        key_buffer = []
        print("writen!")

    def fill_empty_inputs(self, name="keys.txt"):
        with open(name, "r") as file:
            f_lines = file.readlines()

        regex = []
        lines = []
        for line in f_lines:
            regex.append(int(re.findall(r"\d+", line)[0]))
            lines.append(line.split(","))
        max_stamp = max(regex)
        new_lines = []

        for stamp in range(max_stamp):
            if stamp != regex[0]:
                new_lines.append([0, 0])
            else:
                if lines[0][0] == str("Key.up"):
                    new_lines.append([1, 0])
                elif lines[0][0] == str("Key.down"):
                    new_lines.append([0, 1])
                else:
                    new_lines.append([0, 0])
                    print("i should not be here!")
                regex.pop(0)
                lines.pop(0)

        # print(new_lines)
        # print(len(new_lines))
        self.list_of_outputs = new_lines
        self.max_stamp = max_stamp

    def create_dataset(self, name="test.avi"):
        global key_buffer, frame_stamp, listener
        self.empty_dataset()
        # start keyboard listener
        listener.start()
        # writer settings
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(name, codec, 30, (600, 200))
        # monitor recording settings
        sct = mss()
        monitor = {'top': 0, 'left': 0, 'width': 600, 'height': 200}
        last_time = time.time()

        while True:
            frame = np.array(sct.grab(monitor))
            frame = frame[:, :, :3]
            frame = cv2.resize(frame, (600, 200))

            cv2.imshow('frame', frame)
            out.write(frame)

            print('FPS: {} '.format((1.0 / (time.time() - last_time))))
            last_time = time.time()

            if frame_stamp % 100 == 0:
                print("frame stamp: {}".format(frame_stamp))
                self.write_to_file(key_buffer)

            if (cv2.waitKey(10) & 0xFF == ord('q')) or (frame_stamp == 50000):
                key_buffer.append(frame_stamp)
                self.write_to_file(key_buffer)
                out.release()
                cv2.destroyAllWindows()
                break
            # next frame
            frame_stamp += 1

    def get_list_of_outputs(self):
        return self.list_of_outputs

    def get_max_stamp(self):
        return self.max_stamp

    def empty_dataset(self):
        global key_buffer, frame_stamp
        key_buffer = []
        frame_stamp = 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DsC = DatasetCreator()
    DsC.create_dataset("neuro_test2.avi")
    pass

# regex cheat sheet:  https://www.rexegg.com/regex-quickstart.html
# [-+]?\d*\.\d+|  -> [aspon jedno zo znamienka], ak je viac ber prve ?,
# ziadne alebo viac digitov,
# oddelenie desatinneho bodkou,a za nim jeden alebo viac digitov, or statement
