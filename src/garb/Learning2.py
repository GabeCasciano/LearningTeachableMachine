import cv2
import imutils
from tkinter import Tk, Button
from PIL import ImageOps, Image
import numpy as np
import tensorflow.keras

class GUI:
    def __init__(self, tk, size_x, size_y):
        self.tk = tk
        self.x = size_x
        self.y = size_y
        tk.title("AI Remote")
        self.vs = cv2.VideoCapture(0)

        self.captureButton = Button(tk, text="Capture Frame", command=self.captureImg)
        self.captureButton.pack()

        self.initButton = Button(tk, text="Init Model", command=self.initModel)
        self.initButton.pack()

        self.exitButton = Button(tk, text="Exit", command=self.exitCommand)
        self.exitButton.pack()

        self.running = True
        self.normImg = None

        self.model = None
        self.predictions = None

    def exitCommand(self):
        self.tk.destroy()
        self.running = False
        cv2.destroyAllWindows()

    def updateVideo(self): # update the cv imshow window
        frame = self.vs.read()[1]
        im = imutils.resize(frame, width=self.x, height=self.y)
        cv2.imshow("Video Stream", im)

    def captureImg(self): # return the img
        frame = self.vs.read()[1]
        fram = Image.fromarray(frame)
        self.normImg = (np.asarray(ImageOps.fit(fram, (224, 224), Image.ANTIALIAS)).astype(np.float32) / 127) - 1
        self.classify()
        print("Captured img")

    def initModel(self):
        self.model = tensorflow.keras.models.load_model('../facial detection/Recognition/ModelData/keras_model.h5')

    def classify(self):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = self.normImg
        self.predictions = self.model.predict(data)
        print(self.predictions)


np.set_printoptions(suppress=True)
root = Tk()
gui = GUI(root, 400, 400)

while gui.running:
    gui.updateVideo()
    root.update_idletasks()
    root.update()
exit(0)