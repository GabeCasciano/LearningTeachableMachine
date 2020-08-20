import cv2
import os


def save(folderName: str, imgName: str, img):
    if not os.path.exists(folderName):
        os.mkdir(folderName)

    cv2.imwrite(f'{folderName}/{imgName}.png', img)
    print("saved im")


class Recognizer:
    def __init__(self):
        self.cropped = None

    def getcropped(self):
        return self.cropped

    def findandcrop(self, img):
        cascade = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = cascade.detectMultiScale(tmp, scaleFactor=1.1, minNeighbors=5)
        for (x, y, h, w) in rect:
            self.cropped = img[y:y + h, x:x + w]

        return self.cropped
