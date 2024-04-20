import cv2
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tensorflow.keras.utils import normalize 
from sklearn.model_selection import train_test_split


image_directory = "datasets/"


def process_images(image_list, image_path, label_value):
    for i, image_name in enumerate(image_list):
        if image_name.split(".")[1] == 'jpg':
            image = cv2.imread(image_path + image_name)
            image = Image.fromarray(image, "RGB")
            image = image.resize((input_size, input_size))
            dataset.append(np.array(image))
            label.append(label_value)


no_tumor_images = os.listdir(image_directory + "no/")
yes_tumor_images = os.listdir(image_directory + "yes/")
input_size=64
dataset = []
label = []

process_images(no_tumor_images, image_directory + "no/", 0)
process_images(yes_tumor_images, image_directory + "yes/", 1)



dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)


print("dataset length:", len(dataset))
print("label length:", len(label))
print(x_train.shape)


x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)
