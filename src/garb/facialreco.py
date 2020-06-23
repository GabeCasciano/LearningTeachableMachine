import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import *
import os

class gui:
    def __init__(self, tk):
        self.tk = tk
        tk.title("Training data collection")
        self.tk.geometry("300x100")
        self.tk.resizable(0,0)

        self.imgs = []
        self.folderName = ""
        self.counter = 0

        self.vs = cv2.VideoCapture(0)

        self.infoLabel = Label(tk, text="Please enter the name of the person you wish"
                                        "to classify")

        self.enterButton = Button(tk, text="Enter", command=self.captureData)
        self.cancelButton = Button(tk, text="Cancel/Clear", command=self.clear)

        self.subjectNameEntry = Entry(tk)

        self.infoLabel.pack(side=TOP)
        self.subjectNameEntry.pack(side=LEFT)
        self.enterButton.pack(side=LEFT)
        self.cancelButton.pack(side=LEFT)

    def clear(self):
        self.subjectNameEntry.delete(0, "end")
        self.outputImages()
        self.imgs = []

    def captureData(self):
        print(self.subjectNameEntry.get())
        self.imgs.append(self.vs.read()[1])

        print(self.imgs.__len__())

    def outputImages(self):
        # this function does not work fully yet, it could probably work in pieces but its
        # does not work as a whole
        cascade = cv2.CascadeClassifier('Recognition/face.xml')
        for ims in self.imgs:
            tmp = cv2.cvtColor(ims, cv2.COLOR_BGR2GRAY)
            faces_rect = cascade.detectMultiScale(tmp, scaleFactor=1.1, minNeighbors=5)
            for (x, y, h, w) in faces_rect:
                if not os.path.exists(self.subjectNameEntry.get()):
                    os.mkdir(self.subjectNameEntry.get())
                temp = ims[y:y + h, x:x + w]  # crop the image to the rectangle
                cv2.imwrite(f"{self.subjectNameEntry.get()}/out{self.counter}", temp)
                self.counter += 1

root = Tk()
GUI = gui(root)
root.mainloop()
exit(0)