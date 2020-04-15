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

parsing_arg = argparse.ArgumentParser()
parsing_arg.add_argument("-d", "--data", required=True)
parsing_arg.add_argument("-m", "--name_trained", default="hackthevirus.pt")
args = vars(parsing_arg.parse_args())

learn_rate = 1e-3
epochs = 100
batch_size = 64

imagini_paths = list(paths.list_images(args["data"]))
imagini_after_for = []
list_labels = []

for imagini_path in imagini_paths:
    label = imagini_path.split(os.path.sep)[-2]

    image = cv2.imread(imagini_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    imagini_after_for.append(image)
    list_labels.append(label)

#print(list_labels)
data = np.array(imagini_after_for) / 255.0
list_labels = np.array(list_labels)

lb = LabelBinarizer()
list_labels = lb.fit_transform(list_labels)
list_images = to_categorical(list_labels)

(trainX, testX, trainY, testY) = train_test_split(imagini_after_for, list_labels, test_size=0.20, stratify=list_labels, random_state=42)

trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
base = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

head = base.output
head = AveragePooling2D(pool_size=(4, 4))(head)
head = Flatten(name="flatten")(head)
head = Dense(64, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(2, activation="softmax")(head)

model = Model(inputs=base.input, outputs=head)

for layer in base.layers:
    layer.trainable = False

#compilam reteaua neurala
opt = Adam(lr=learn_rate, decay=learn_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#head-ul retelei
head_trained = model.fit_generator(trainAug.flow(trainX, trainY, batch_size=batch_size),
                                    steps_per_epoch = len(trainX) // batch_size,
                                    validation_data = (testX, testY),
                                    validation_steps = len(testX) // batch_size,
                                    epochs=epochs)

#evaluam reteaua
prediction = model.predict(textX, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

#raport clasificatie
#print(classification_report(testY.argmax(axis=1), prediction, target_names = lb.classes_))

matrix = confusion_matrix(testY.argmax(axis=1), prediction)
total = sum(sum(matrix))
acc = (matrix[0, 0] + matrix[1, 1]) / total
sensitivity = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
specificity = matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])

print("Acuratete: {}".format(acc))
