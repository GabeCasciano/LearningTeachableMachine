import tensorflow.keras
import cv2
import imutils
import numpy as np
from PIL import ImageOps, Image


vc = cv2.VideoCapture(0)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
# Load an image
ret, frame = vc.read()
im_pil = Image.fromarray(frame)
img = ImageOps.fit(im_pil, (224,224), Image.ANTIALIAS)
norm = (np.asarray(img).astype(np.float32)/127) - 1

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = norm
# predict
pred = model.predict(data)
print(pred)

