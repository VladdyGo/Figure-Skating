import matplotlib

matplotlib.use("Agg")
from dataset_creation import CreateTrainData
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import imutils
from PIL import Image
import imageio
import os
from random import randrange

NUM_OF_EPOCHS = 10

class CNNModel:
    def __init__(self) -> None:
        self.model = None
        self.H = None

    def runModel(self):
        baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(len(CreateTrainData.lb.classes_), activation="softmax")(headModel)
        self.model = Model(inputs=baseModel.input, outputs=headModel)
        for layer in baseModel.layers:
            layer.trainable = False
        # compile our model (this needs to be done after our setting our layers to being non-trainable)
        print("[INFO] compiling model...")
        opt = SGD(lr=0.01, momentum=0.9, decay=0.001)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        print("[INFO] training head...")
        self.H = self.model.fit_generator(
            CreateTrainData.trainAug.flow(CreateTrainData.trainX, CreateTrainData.trainY, batch_size=32),
            steps_per_epoch=len(CreateTrainData.trainX) // 32,
            validation_data=CreateTrainData.valAug.flow(CreateTrainData.testX, CreateTrainData.testY),
            validation_steps=len(CreateTrainData.testX) // 32,
            epochs=NUM_OF_EPOCHS)
        print("Evaluating network...")

    def ModelResultsPlot(self, plot_style):
        predictions = self.model.predict(CreateTrainData.testX, batch_size=32)
        plt.style.use(plot_style)
        plt.figure()
        plt.plot(np.arange(0, NUM_OF_EPOCHS), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, NUM_OF_EPOCHS), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, NUM_OF_EPOCHS), self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, NUM_OF_EPOCHS), self.H.history["val_accuracy"], label="val_accuracy")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #" + str(NUM_OF_EPOCHS))
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("pyimagesearch2.png")

        print("[INFO] serializing network...")
        self.model.save('classification.model')
        f = open('lb.pickle', "wb")
        f.write(pickle.dumps(CreateTrainData.lb))
        f.close()