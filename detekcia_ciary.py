import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from mss import mss


# from directkeys import PressKey,ReleaseKey,Up,Down


# DBM   x 67   y 147
# DBH   x 67   y 120

# DCD   x 67   y 172 (175)
# DCM   x 67   y 167
# DCH   x 67   y 150

##
# H   x 67   y 120
# M   x 67   y 150
# D   x 67   y 172

##DD  x1 67 x2 142

##  720 775

## x1 85   x3 136


# Full screen
# 155  - 1.P   H
#
#
def decide(D, M):
    if D:
        pyautogui.keyDown(' ')
        time.sleep(0.02)
        pyautogui.keyUp(' ')
        # pyautogui.press(" ")
    if M and not D:
        pyautogui.keyDown('down')
        time.sleep(0.25)
        pyautogui.keyUp('down')


def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, processed_img) = cv2.threshold(processed_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    roiM = processed_img[150, 85:500]
    roiD = processed_img[175, 85:500]

    firstM = np.where(roiM == 0)
    firstD = np.where(roiD == 0)

    treshold = 45
    roibD = 0
    roibM = 0
    if len(firstD[0]) != 0 and len(firstM[0]) != 0 and firstM[0][0] <= treshold and firstD[0][0] < firstM[0][0] - 10:
        roibD = 1
        roibM = 1
    else:
        if len(firstD[0]) != 0 and firstD[0][0] <= treshold:
            roibD = 1
        if len(firstM[0]) != 0 and firstM[0][0] <= treshold:
            roibM = 1


    if roibD or roibM:
        decide(roibD, roibM)

    # print(firstM[0][0])  # distance

    # len(firstM[0]) == 0 DAS IST GUT

    cv2.line(processed_img, (90, 155), (500, 155), (0, 0, 0), 1)
    cv2.line(processed_img, (90, 175), (500, 175), (0, 0, 0), 1)

    cv2.imshow('Line ROI', processed_img)
    # np.take(arr,indices)
    # return processed_img


def main():
    last_time = time.time()
    pyautogui.PAUSE = 0
    sct = mss()

    while True:
        # screen = sct.grab({'top': 0, 'left': 0, 'width': 600, 'height': 200})
        # screen_np = np.array(np.float32(screen))
        screen_np = np.array(sct.grab({'top': 0, 'left': 0, 'width': 600, 'height': 200}))
        # cv2.imshow("Screencapture:", screen_np)
        process_img(screen_np)
        print('FPS: {} '.format((1.0 / (time.time() - last_time))))
        last_time = time.time()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # while True:
    #     screen = np.array(ImageGrab.grab(bbox=(0, 0, 600, 200)))
    #     width, height, _ = screen.shape
    #     print('FPS: {} '.format((1 / (time.time() - last_time))))
    #     # print('Frame took {} seconds'.format(time.time()-last_time))
    #     last_time = time.time()
    #     process_img(screen)
    #     # new_screen = process_img(screen)
    #     # cv2.imshow('window', new_screen)
    #     # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
