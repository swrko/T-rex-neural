import cv2


def RWProcesObject(path="C:\\Users\\invii\\Documents\DP\\Trex\\objekty", rfn="\\CD.png",
                 wfn="\\CDP.png"):
    processed_img = cv2.imread(path + rfn, cv2.IMREAD_GRAYSCALE)
    #(thresh, processed_img) = cv2.threshold(processed_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(path + wfn, processed_img)


if __name__ == '__main__':
    RWProcesObject()
