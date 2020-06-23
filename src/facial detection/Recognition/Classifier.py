import cv2
import tensorflow.keras
import os
from .FacialRecognition import *
from PIL import ImageOps, Image
import numpy as np
import tensorflow.keras

class Classifier:
    def __init__(self):
        self.recognizer = Recognizer()
        self.img = None
        self.model = None
        self.predictions = None
        self.labels = []

    def initModel(self):
        self.model = tensorflow.keras.models.load_model('ModelData/keras_model.h5')

        lines = None
        with open('ModelData/labels.txt') as file:
            lines = file.readlines()

        for l in lines:
            empt = ""
            tmp = l.split(" ")
            self.labels.append(empt.join(tmp[1:tmp.__len__()]))

    def classify(self):
        if self.labels.__len__() < 1:
            return False

        self.recognizer.capture()
        self.img = self.recognizer.findandcrop()

        frame = Image.fromarray(self.img)
        normImg = (np.asarray(ImageOps.fit(frame, (224, 224), Image.ANTIALIAS)).astype(np.float32) / 127) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normImg
        self.predictions = self.model.predict(data)

        return self.predictions

    def determine(self):
        if self.predictions.__len__() != None:
            indx = self.predictions.index(max(self.predictions))
            return f"{self.labels[indx]} is {max(self.predictions) * 100} %"

        return "Classify first"
