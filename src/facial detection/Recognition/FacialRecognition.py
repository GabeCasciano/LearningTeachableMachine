import cv2
import os


def save(folderName: str, imgName: str, img):
    if not os.path.exists(folderName):
        os.mkdir(folderName)

    cv2.imwrite(f"{folderName}/{imgName}.png", img)

    print("saved im")


class Recognizer:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.img = None
        self.cropped = None

    def getim(self):
        return self.img

    def getcropped(self):
        return self.cropped

    def capture(self):
        self.img = self.vs.read()[1]
        return self.img

    def findandcrop(self):
        img = self.capture()
        cascade = cv2.CascadeClassifier('C:\\Users\\Gabe Casciano\\Documents\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = cascade.detectMultiScale(tmp, scaleFactor=1.1, minNeighbors=5)

        for (x, y, h, w) in rect:
            self.cropped = self.img[y:y + h, x:x + w]

        return self.cropped
