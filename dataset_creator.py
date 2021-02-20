import numpy as np
import cv2
import time
from mss import mss
from pynput.keyboard import Key, Listener
import re
import os

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
        self.list_of_inputs = []
        self.max_stamp = 0

    def write_keys_to_file(self, keys):
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
            # check for double key press
            r = int(re.findall(r"\d+", line)[0])
            if len(regex) > 1:
                if regex[-1] != r:
                    regex.append(r)
                    lines.append(line.split(","))
            else:
                regex.append(r)
                lines.append(line.split(","))


        max_stamp = max(regex)
        new_lines = []

        for stamp in range(max_stamp):
            if stamp != regex[0]:
                new_lines.append([0, 0, 1])
            else:
                if lines[0][0] == str("Key.up"):
                    new_lines.append([1, 0, 0])
                elif lines[0][0] == str("Key.down"):
                    new_lines.append([0, 1, 0])
                else:
                    new_lines.append([0, 0, 1])
                    print("i should not be here!")
                regex.pop(0)
                lines.pop(0)

        self.list_of_outputs = new_lines
        self.max_stamp = max_stamp

    def create_dataset(self, name="test.avi"):
        global key_buffer, frame_stamp, listener
        self.dump_dataset()
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
                self.write_keys_to_file(key_buffer)

            if (cv2.waitKey(10) & 0xFF == ord('q')) or (frame_stamp == 50000):
                key_buffer.append(frame_stamp)
                self.write_keys_to_file(key_buffer)
                out.release()
                cv2.destroyAllWindows()
                break
            # next frame
            frame_stamp += 1

    def create_dataset2(self):
        global key_buffer, frame_stamp, listener
        self.dump_dataset()
        # start keyboard listener
        listener.start()
        # writer settings
        # monitor recording settings
        sct = mss()
        monitor = {'top': 0, 'left': 0, 'width': 600, 'height': 200}
        last_time = time.time()

        while True:
            frame = np.array(sct.grab(monitor))
            frame = frame[:, :, :3]
            frame = cv2.resize(frame, (600, 200))
            frame_folder_path = r'C:\Users\Herny\Documents\shady skola\DP\T_rex\frames'
            frame_name = str(frame_stamp) + '.png'
            cv2.imwrite(os.path.join(frame_folder_path, frame_name), frame)
            cv2.imshow('frame', frame)

            print('FPS: {} '.format((1.0 / (time.time() - last_time))))
            last_time = time.time()

            if frame_stamp % 500 == 0:
                print("frame stamp: {}".format(frame_stamp))
                self.write_keys_to_file(key_buffer)

            if (cv2.waitKey(10) & 0xFF == ord('q')) or (frame_stamp == 50000 - 1):
                key_buffer.append(frame_stamp)
                self.write_keys_to_file(key_buffer)
                cv2.destroyAllWindows()
                break
            # next frame
            frame_stamp += 1

    def get_list_of_outputs(self):
        return self.list_of_outputs

    def get_list_of_inputs(self):
        return self.list_of_inputs

    def get_max_stamp(self):
        return self.max_stamp

    def dump_dataset(self):
        global key_buffer, frame_stamp
        key_buffer = []
        frame_stamp = 0

    def write_to_file(self, name, data):
        name = name + ".txt"
        with open(name, "w") as file:
            file.write(str(data) + ";\n")
        print("file was written succesfully!")

    def inputs_from_img(self, image):
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, processed_img) = cv2.threshold(processed_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        roiM = processed_img[150, 85:400]
        roiD = processed_img[175, 85:400]

        # find all black pixels at roi
        firstM = np.where(roiM == 0)
        firstD = np.where(roiD == 0)

        input = [400, 400]

        # if is not empty -> rewrite
        if len(firstD[0]) != 0:
            input[0] = firstD[0][0]
        if len(firstM[0]) != 0:
            input[1] = firstM[0][0]

        # cv2.line(processed_img, (90, 155), (400, 155), (0, 0, 0), 1)
        # cv2.line(processed_img, (90, 175), (400, 175), (0, 0, 0), 1)

        return input

    def video_to_files(self):
        self.fill_empty_inputs("keys.txt")
        cap = cv2.VideoCapture("neuro_test2.avi")
        outputs = self.get_list_of_outputs()
        current_stamp = 0.0
        inputs = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret and outputs:
                inputs.append(self.inputs_from_img(frame))
                current_stamp += 1

            progress = int((current_stamp / self.get_max_stamp()) * 100.0)
            if (progress % 10) == 0:
                print("progress: {}".format(progress))

            if progress == 100:
                print("inputs: {} \n labeled_outputs: {}".format(inputs, outputs))
                break
        self.write_to_file("inputs", inputs)
        self.write_to_file("outputs", outputs)

    def read_dataset_from_file(self, name):
        read_data = []
        name = name + ".txt"
        with open(name, "r") as file:
            r = file.read()
            regex = re.findall(r"\d+, \d+, \d+|\d+, \d+", r)
            for item in regex:
                read_data.append(list(np.fromstring(item, dtype=int, sep=',')))
        return read_data

    def get_indices_of_separates(self):
        inputs = self.read_dataset_from_file("inputs")
        outputs = self.read_dataset_from_file("outputs")

        # indices for jump -> inputs, outputs
        jump_indices = [i for i, x in enumerate(outputs) if x == [1, 0, 0]]

        # indices for duck
        duck_indices = [i for i, x in enumerate(outputs) if x == [0, 1, 0]]

        # indices for no_action
        noa_indices = [i for i, x in enumerate(outputs) if x == [0, 0, 1]]

        return jump_indices, duck_indices, noa_indices

    def get_rand_dataset_indices(self, volume=30000):
        res = volume % 3
        if res > 0:
            volume -= res

        jump_i, duck_i, noa_i = self.get_indices_of_separates()
        r_jump_i = np.random.choice(jump_i, int(volume / 3))
        r_duck_i = np.random.choice(duck_i, int(volume / 3))
        r_noa_i = np.random.choice(noa_i, int(volume / 3))
        return r_jump_i, r_duck_i, r_noa_i

    def get_separated_dataset(self):
        inputs = self.read_dataset_from_file("inputs")
        outputs = self.read_dataset_from_file("outputs")

        jump_i, duck_i, noa_i = self.get_rand_dataset_indices()

        # separation for jump -> inputs, outputs
        jump = [[inputs[i] for i in jump_i], [outputs[i] for i in jump_i]]

        # separation for duck
        duck = [[inputs[i] for i in duck_i], [outputs[i] for i in duck_i]]

        # separation for no_action
        no_action = [[inputs[i] for i in noa_i], [outputs[i] for i in noa_i]]

        return jump, duck, no_action

    def get_rand_dataset(self):
        j, d, noa = self.get_separated_dataset()

        inp = np.concatenate((j[0], d[0], noa[0]), axis=0)
        tar = np.concatenate((j[1], d[1], noa[1]), axis=0)

        indices = np.arange(len(inp))
        np.random.shuffle(indices)

        inp = [list(inp[i]) for i in indices]
        tar = [list(tar[i]) for i in indices]

        return inp, tar

    def get_list_of_dir(self, dir_name):
        list = os.listdir(dir_name)
        # li_index = list.index(str(self.get_max_stamp()) + '.png')
        # os.remove(dir_name + "\\" + list[li_index])
        # list.pop(li_index)
        return list

    def rename_files_at_dir(self, dir):
        dir_list = self.get_list_of_dir(dir)
        self.fill_empty_inputs()
        list_of_outputs = self.get_list_of_outputs()
        for frame in dir_list:
            fstamp_index = int(frame.split('.')[0])
            action = str(list_of_outputs[fstamp_index])
            src = dir + "\\" + frame
            dst = dir + "\\" + str(fstamp_index) + '_' + action + '.png'
            os.rename(src, dst)

    def frames_to_files(self):
        dir = r'C:\Users\Herny\Documents\shady skola\DP\T_rex\frames'
        self.fill_empty_inputs("keys.txt")
        list_of_outputs = self.get_list_of_outputs()
        dir_list = self.get_list_of_dir(dir)

        inputs = []
        outputs = []
        while dir_list:
            fstamp_index = int(dir_list[0].split('.')[0])
            frame_name = dir + "\\" + dir_list[0]
            frame = cv2.imread(frame_name)
            inputs.append(self.inputs_from_img(frame))
            outputs.append(list_of_outputs[fstamp_index])
            dir_list.pop(0)

        self.write_to_file("inputs", inputs)
        self.write_to_file("outputs", outputs)
    #
    # def check_dataset(self,dir):
    #     dir_list = self.get_list_of_dir(dir)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dsc = DatasetCreator()
    # DsC.create_dataset("neuro_test2.avi")
    # print(dsc.get_rand_dataset())
    # inp, tar = dsc.get_rand_dataset()
    # print(dsc.read_dataset_from_file("outputs"))
    # dsc.frames_to_files()
    # print(dsc.get_list_of_dir(r'C:\Users\Herny\Documents\shady skola\DP\T_rex\frames'))

    # dsc.rename_files_at_dir(r'C:\Users\Herny\Documents\shady skola\DP\T_rex\frames')

    dsc.fill_empty_inputs()
    list = dsc.get_list_of_outputs()
    print(list[48231])
    print(list[48841])

    # dsc.create_dataset("neuro_test2.avi")
    # print("inputs: {} \n outputs: {}".format(DsC.get_list_of_inputs(),DsC.get_list_of_outputs()))
    pass

# regex cheat sheet:  https://www.rexegg.com/regex-quickstart.html
# [-+]?\d*\.\d+|  -> [aspon jedno zo znamienka], ak je viac ber prve ?,
# ziadne alebo viac digitov,
# oddelenie desatinneho bodkou,a za nim jeden alebo viac digitov, or statement
