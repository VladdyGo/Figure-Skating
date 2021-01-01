import matplotlib

matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os


class CreateTrainData:
    def __init__(self) -> None:
        self.data = []
        self.labels = []
        self.lb = None
        self.tags = ["Axel", "Euler", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]
        self.trainAug = None
        self.valAug = None

    def createFrames(self, frame_folder_path):
        parent = os.listdir(frame_folder_path)
        tag = 0
        count = 0
        for video_class in parent[0:]:  # it also contains DS.store file
            child = os.listdir(frame_folder_path + "/" + video_class)
            for class_i in child[0:]:
                sub_child = os.listdir(frame_folder_path + "/" + video_class + "/" + class_i)
                for image_fol in sub_child[0:]:
                    if count % 1 == 0:  # (selected images at gap of 4)
                        image = cv2.imread(frame_folder_path + "/" + video_class + "/" + class_i + "/" + image_fol)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self.data.append(image)
                        self.labels.append(self.tags[tag])
                    count += 1
            tag += 1

    def dataToTrainAndTest(self):
        self.labels = np.array(self.labels)
        data = np.array(self.data)
        self.lb = LabelBinarizer()
        self.labels = self.lb.fit_transform(self.labels)
        (trainX, testX, trainY, testY) = train_test_split(data, self.labels, test_size=0.25,
                                                          stratify=self.labels, random_state=42)
        self.trainAug = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
        self.valAug = ImageDataGenerator()
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        self.trainAug.mean = mean
        self.valAug.mean = mean
