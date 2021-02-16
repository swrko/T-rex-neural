import numpy as np
import neuroMTX as NN
import cv2
from mss import mss
import os


def inputs_from_process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, processed_img) = cv2.threshold(processed_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    roiM = processed_img[150, 85:400]
    roiD = processed_img[175, 85:400]

    # find all black pixels at roi
    firstM = np.where(roiM == 0)
    firstD = np.where(roiD == 0)

    input = np.array([400, 400])

    # if is not empty -> rewrite
    if len(firstD[0]) != 0:
        input[0] = firstD[0][0]
    if len(firstM[0]) != 0:
        input[1] = firstM[0][0]

    cv2.line(processed_img, (90, 155), (400, 155), (0, 0, 0), 1)
    cv2.line(processed_img, (90, 175), (400, 175), (0, 0, 0), 1)

    return input


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


def testNN(network):
    network.read_weights_from_file()
    # trained Network
    sct = mss()

    while (True):
        screen_np = np.array(sct.grab({'top': 0, 'left': 0, 'width': 600, 'height': 200}))
        cv2.imshow("Screencapture:", screen_np)
        inputs = inputs_from_process_img(screen_np)
        outputs = network.feed_forward_propagation(inputs)
        print("outputs: {}".format(outputs))

        # TODO: apply outputs to game -> decide_function()

        if cv2.waitKey(15) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def main():
    network = NN.NNetwork(2, 10, 3)

    # network.write_weights_to_file()
    # for i in range(5):
    #     print("repeat {}/50".format(i))
    #     os.system("trainNN.py")
    testNN(network)

if __name__ == '__main__':
    main()
