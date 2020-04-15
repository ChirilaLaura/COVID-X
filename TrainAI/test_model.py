from keras.preprocessing.image import img_to_array
from keras.models import load_models
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

model = load_model(args["model"])
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:16]

results = []

for p in imagePaths:
    orig = cv2.imread(p)
    image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image.astype("float") / 255.0

    image = img_to_arrray(image)
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)
    pred = pred.argmaz(axis=1)[0]

    label = "Infectat" if pred == 1 else "Sanatos"
    color = (255, 0, 0) if pred == 1 else (0, 255, 0)

    orig = cv2.resize(orig, (128, 128))
    cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    result.append(orig)

montage = build_montages(results, (128, 128), (4, 4))[0]

cv2.imshow("Rezultat", montage)
cv2.waitKey(0)
