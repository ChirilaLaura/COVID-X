from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

class CovidAI:
    def __init__(self):
        self.base = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        self.head = base.output
        self.model = Model(inputs=base.input, outputs=head)
        
    def forward(self):
        base = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        head = base.output
        head = AveragePooling2D(pool_size=(4, 4))(head)
        head = Flatten(name="flatten")(head)
        head = Dense(64, activation="relu")(head)
        head = Dropout(0.5)(head)
        head = Dense(2, activation="softmax")(head)

        model = Model(inputs=base.input, outputs=head)
        return type(model)

obj = CovidAI()
print(obj.forward())
