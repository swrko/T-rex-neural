import numpy as np
import neuroMTX as NN
import cv2
from mss import mss
import os
import neurolab as nl
import dataset_creator as DSC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyautogui
from time import sleep
import _thread as th
from tqdm import tqdm


def decide(outputs):
    index_of_max = np.argmax(outputs[0])
    value_of_max = outputs[0][index_of_max]
    if index_of_max == 0 and value_of_max >= 0.78:
        pyautogui.keyDown(' ')
        sleep(0.02)
        pyautogui.keyUp(' ')
    elif index_of_max == 1 and value_of_max > 0.5:
        pyautogui.keyDown('down')
        sleep(0.4)
        pyautogui.keyUp('down')
    else:
        pass

def decide2(outputs):
    index_of_max = np.argmax(outputs)
    value_of_max = outputs[index_of_max]
    if index_of_max == 0 and value_of_max >= 0.78:
        pyautogui.keyDown(' ')
        sleep(0.02)
        pyautogui.keyUp(' ')
    elif index_of_max == 1 and value_of_max > 0.65:
        pyautogui.keyDown('down')
        sleep(0.4)
        pyautogui.keyUp('down')
    else:
        pass


def inputs_from_process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, processed_img) = cv2.threshold(processed_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # povodne
    # roiM = processed_img[150, 85:400]
    # roiD = processed_img[175, 85:400]

    roiM = processed_img[100, 85:500]
    roiD = processed_img[120, 85:500]

    # find all black pixels at roi
    firstM = np.where(roiM == 0)
    firstD = np.where(roiD == 0)

    input = np.array([400, 400])

    # if is not empty -> rewrite
    if len(firstD[0]) != 0:
        input[0] = firstD[0][0]
    if len(firstM[0]) != 0:
        input[1] = firstM[0][0]
    # original values
    # cv2.line(processed_img, (90, 155), (400, 155), (0, 0, 0), 1)
    # cv2.line(processed_img, (90, 175), (400, 175), (0, 0, 0), 1)

    # great but comennted for compute power
    # cv2.line(processed_img, (85, 100), (500, 100), (0, 0, 0), 1)
    # cv2.line(processed_img, (85, 120), (500, 120), (0, 0, 0), 1)
    # cv2.imshow('dsad', processed_img)
    return list([list(input)])


def train_NN(network, dsc):
    cap = cv2.VideoCapture("neuro_test2.avi")
    outputs = dsc.get_list_of_outputs()

    current_stamp = 0.0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret and outputs:
            inputs = inputs_from_process_img(frame)
            # train
            network.train_gradient_descent(inputs, outputs[0])

            outputs.pop(0)
            cv2.imshow("cap", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                break
            current_stamp += 1
        progress = int((current_stamp / dsc.get_max_stamp()) * 100.0)
        if (progress % 10) == 0:
            print("progress: {}".format(progress))

        if progress == 100:
            break


def train_NN2(network, dsc):
    inp, tar = dsc.get_rand_dataset()
    # print("inp: {}\ntar: {}".format(len(inp), len(tar)))
    # print("inp: {}\ntar: {}".format(inp, tar))
    t_error = 0
    for i in tqdm(range(len(inp))):
        t_error += network.train_gradient_descent(inp[i], tar[i])
    print(t_error)
    network.write_weights_to_file("my_neuro")
    # net.trainf = nl.net.train.train_gd  # train function set to gradient descent
    # error = net.train(inp, tar, epochs=200, show=1, goal=0.02)


def testNN(network):
    network.read_weights_from_file("my_neuro")
    # trained Network
    sct = mss()

    while (True):
        screen_np = np.array(sct.grab({'top': 100, 'left': 0, 'width': 600, 'height': 200}))
        # screen_np = np.array(sct.grab({'top': 0, 'left':0, 'width': 600, 'height': 200}))
        inputs = inputs_from_process_img(screen_np)
        outputs = network.feed_forward_propagation(inputs)
        print("outputs: {}".format(outputs))
        th.start_new_thread(decide2, (outputs,))

        if cv2.waitKey(15) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def trainNL():
    pass


def testNL():
    net = nl.load('test_tretia.net')
    sct = mss()

    while (True):
        screen_np = np.array(sct.grab({'top': 0, 'left': 0, 'width': 600, 'height': 200}))
        inputs = inputs_from_process_img(screen_np)

        # # drawing of line ROI
        # cv2.line(screen_np, (90, 155), (400, 155), (0, 0, 0), 1)
        # cv2.line(screen_np, (90, 175), (400, 175), (0, 0, 0), 1)
        # cv2.imshow("Screencapture:", screen_np)

        # print("inputs: {}".format(inputs))
        outputs = net.sim(inputs)
        # print(inputs)
        # print("outputs: {}".format(outputs))
        th.start_new_thread(decide, (outputs,))

        if cv2.waitKey(15) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def main():
    network = NN.NNetwork(2, 10, 3)
    dsc = DSC.DatasetCreator()
    # network.write_weights_to_file()
    # for i in range(5):
    #     print("repeat {}/50".format(i))
    #     os.system("trainNN.py")
    # train_NN2(network, dsc)
    testNN(network)

if __name__ == '__main__':
    main()
