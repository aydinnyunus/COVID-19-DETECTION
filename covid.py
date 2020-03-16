import argparse
import os
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def probability_to_percentage(p):
    return round(p * 100, 2)


def parse_prediction_array(prediction):
    max_index = prediction[0].argmax()
    if max_index == 0:
        return {"label": "Corona", "p": probability_to_percentage(prediction[0][max_index])}
    else:
        return {"label": "Normal", "p": probability_to_percentage(prediction[0][max_index])}


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
                help="Path of your Dataset")
args = vars(ap.parse_args())

INIT_LR = 1e-3
EPOCHS = 25
BS = 8

imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    data.append(image)
    labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

trainAug = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest")

baseModel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

predIdxs = model.predict(testX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total

print("Accuracy Score : {:.4f}".format(accuracy))
model.save("Corona.h5", save_format="h5")

img_src = 'ENTER YOUR TEST IMAGE'
img = cv2.imread(img_src)

target_width = 224
target_height = 224
target_size = (target_width, target_height)

img = cv2.resize(img, target_size)
img = img.reshape(1, target_width, target_height, 3)
loaded_model = load_model('Corona.h5')

prediction = loaded_model.predict(img)

parsed_res = parse_prediction_array(prediction)

fig, ax = plt.subplots()
title = str(parsed_res["p"]) + "% possibility " + str(parsed_res["label"])
fig.suptitle(title, fontsize=12)

np_img = mpimg.imread(img_src)
plt.imshow(np_img)
plt.show()
