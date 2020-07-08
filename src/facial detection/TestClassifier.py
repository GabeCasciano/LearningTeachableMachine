import cv2
from tkinter import *
import sys

sys.path.append('.')

from Recognition.Classifier import Classifier

def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=640, display_height=480,
                                framerate=30, flip_method=2):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){capture_width}, height=(int){capture_height}, ' +
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            f'nvvidconv flip-method={flip_method} ! ' +
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink')



class gui:
    def __init__(self, tk: Tk):
        self.tk = tk
        self.vs = cv2.VideoCapture(get_jetson_gstreamer_source())

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
