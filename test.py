from mss import mss
import cv2
import numpy as np
import time
from PIL import Image

def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, processed_img) = cv2.threshold(processed_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow('Line ROI', processed_img)
    # np.take(arr,indices)
    # return processed_img



def main():
    sct = mss()
    x = 0
    y = 0
    h = 640
    w = 800
    last_time = time.time()

    while (True):

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
    # with mss.mss() as mss_instance:  # Create a new mss.mss instance
    #
    #     monitor_1 = mss_instance.monitors[1]  # Identify the display to capture
    #     screenshot = mss_instance.grab(monitor_1)
    #     screenshot = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    #     screenshot.show()
    #


if __name__ == '__main__':
    main()
