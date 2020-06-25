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

    def classify(self, img):
        if self.labels.__len__() < 1:
            return False

        self.img = self.recognizer.findandcrop(img)

        if self.img is not None:
            frame = Image.fromarray(self.img)
            normImg = (np.asarray(ImageOps.fit(frame, (224, 224), Image.ANTIALIAS)).astype(np.float32) / 127) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normImg
            self.predictions = self.model.predict(data).tolist()

        return self.predictions

    def determine(self):
        if self.predictions != None:
            max = 0
            indx = 0
            counter = 0

            for p in self.predictions[0]:
                if max < p:
                    max = p
                    indx = counter
                counter += 1

            return f"{self.predictions[0][indx]}, {self.labels[indx]}"

        return "Classify first"
