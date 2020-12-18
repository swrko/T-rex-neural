import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui

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


def decide(H, M, D, DD):
    if D and DD:
        time.sleep(0.05)
        pyautogui.keyDown(' ')
        pyautogui.keyUp(' ')
    if D:
        pyautogui.keyDown(' ')
        pyautogui.keyUp(' ')
        # PressKey(Up)
    if M and D:
        pyautogui.keyDown(' ')
        pyautogui.keyUp(' ')
        # PressKey(Up)
    if M and not D:
        pyautogui.keyDown('down')
        time.sleep(0.2)
        pyautogui.keyUp('down')
        # PressKey(Down)
    if H and not M and not D:
        pyautogui.keyDown('down')
        time.sleep(0.2)
        pyautogui.keyUp('down')


def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, processed_img) = cv2.threshold(processed_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # edge detection
    # processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    # processed_img[85, 145] = 1  # H  80:90
    # processed_img[110, 145] = 1  # M     +-10
    # processed_img[137, 145] = 1  # D   132:142
    cv2.rectangle(processed_img, (154, 79), (181, 92), 0, 1)  #H
    cv2.rectangle(processed_img, (154, 104), (181, 116), 0, 1)  #M
    cv2.rectangle(processed_img, (154, 131), (181, 143), 0, 1)  #D
    cv2.rectangle(processed_img, (184, 131), (211, 143), 0, 1)  #DD


    roiH = [i[155:180] for i in processed_img[80:91]]
    roiM = [i[155:180] for i in processed_img[105:115]]
    roiD = [i[155:180] for i in processed_img[132:142]]
    roiDD = [i[205:230] for i in processed_img[132:142]]
    decide(np.any(np.invert(roiH)), np.any(np.invert(roiM)), np.any(np.invert(roiD)), np.any(np.invert(roiDD)))
    # cv2.imshow('found object', processed_img)
    # np.take(arr,indices)
    return processed_img


def main():
    last_time = time.time()
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 600, 200)))
        print('FPS: {} '.format((1 / (time.time() - last_time))))
        # print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        # process_img(screen)
        new_screen = process_img(screen)
        cv2.imshow('window', new_screen)
        # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
