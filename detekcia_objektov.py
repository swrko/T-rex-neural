# This is a sample Python script.

import numpy as np
from PIL import ImageGrab
import cv2
import time


def process_img(image):
    im_temp_CD = cv2.imread('CDP.png', cv2.CV_8U)
    im_temp_CH = cv2.imread('CHP.png', cv2.CV_8U)
    im_temp_BKU = cv2.imread('BKUP.png', cv2.CV_8U)
    im_temp_BKD = cv2.imread('BKDP.png', cv2.CV_8U)

    w1, h1 = im_temp_CD.shape[::-1]
    w2, h2 = im_temp_CH.shape[::-1]
    w3, h3 = im_temp_CH.shape[::-1]
    w4, h4 = im_temp_CH.shape[::-1]

    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # (thresh, processed_img) = cv2.threshold(processed_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # edge detection
    # processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)

    # cv2.match template
    res1 = cv2.matchTemplate(processed_img.astype(np.uint8), im_temp_CD, cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(processed_img.astype(np.uint8), im_temp_CH, cv2.TM_CCOEFF_NORMED)
    res3 = cv2.matchTemplate(processed_img.astype(np.uint8), im_temp_BKU, cv2.TM_CCOEFF_NORMED)
    res4 = cv2.matchTemplate(processed_img.astype(np.uint8), im_temp_BKD, cv2.TM_CCOEFF_NORMED)

    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(image, top_left, bottom_right, color=(255, 0, 0), thickness=2)

    loc1 = np.where(res1 >= 0.7)
    loc2 = np.where(res2 >= 0.7)
    loc3 = np.where(res3 >= 0.7)
    loc4 = np.where(res4 >= 0.7)
    for pt in zip(*loc1[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w1, pt[1] + h1), (255, 0, 0), 2)
    for pt in zip(*loc2[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w2, pt[1] + h2), (0, 0, 255), 2)
    for pt in zip(*loc3[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w3, pt[1] + h3), (0, 255, 0), 2)
    for pt in zip(*loc4[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w4, pt[1] + h4), (0, 255, 0), 2)


    cv2.imshow('found object', image)
    return processed_img

def main():
    last_time = time.time()
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 600, 200)))
        print('FPS: {} '.format((1 / (time.time() - last_time))))
        # print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
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
