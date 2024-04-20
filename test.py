import cv2 
import os
from keras.models import load_model
import numpy as np
from PIL import Image


model = load_model("categorical_model.h5")
model = load_model("binary_model.h5")

image = cv2.imread("datasets/pred/pred0.jpg")
img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img) 


input_img = np.expand_dims(img, axis=0)

result = model.predict(input_img)
print("Brain tumor detected." if result[0][0] ==1 else "No brain tumor detected.")
