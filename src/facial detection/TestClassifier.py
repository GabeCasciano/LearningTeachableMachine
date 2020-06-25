import cv2
from tkinter import *
import sys

sys.path.append('.')

from Recognition.Classifier import Classifier


class gui:
    def __init__(self, tk: Tk):
        self.tk = tk
        self.vs = cv2.VideoCapture(0)

        self.classifier = Classifier()
        self.classifier.initModel()

        tk.title("Training data collection")
        self.tk.geometry("300x200")
        self.tk.resizable(0, 0)

        self.classifyButton = Button(tk, text="Classify", command=self.classify)
        self.cancelButton = Button(tk, text="Exit", command=self.cancel)

        self.classifyButton.pack(side=LEFT)
        self.cancelButton.pack(side=LEFT)

    def updateVideo(self):
        cv2.imshow("Current", self.vs.read()[1])

    def classify(self):
        self.classifier.classify(self.vs.read()[1])
        print(self.classifier.determine())
        pass

    def cancel(self):
        print("Exittting")
        sys.exit(0)
        pass


root = Tk()
GUI = gui(root)

while 1:
    GUI.updateVideo()
    root.update_idletasks()
    root.update()
