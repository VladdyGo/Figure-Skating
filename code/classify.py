from CNN_model import CNNModel
from keras.models import load_model
import pickle
from dataset_creation import CreateTrainData
import numpy as np
import cv2
from PIL import Image
from collections import deque

class Classify:
    def __init__(self):
        self.mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        self.writer = None
        self.model = load_model('classification.model')
        self.lb = pickle.loads(open('lb.pickle', "rb").read())

    def classifyVideo(self, input):
        (W, H) = (None, None)
        Q = deque(maxlen=10)
        while True:
            (grabbed, frame) = input.read()
            if not grabbed:
                break
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            output = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224)).astype("float32")
            frame -= self.mean
            preds = self.model.predict(np.expand_dims(frame, axis=0))[0]
            Q.append(preds)
            results = np.array(Q).mean(axis=0)
            i = np.argmax(results)
            label = self.lb.classes_[i]
            text = "activity: {}".format(label)
            cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.25, (0, 255, 0), 5)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"DIVX")
                writer = cv2.VideoWriter('result_of_video_classification.avi',
                                         fourcc, 30, (W, H), True)

            writer.write(output)
            cv2.imshow(output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        writer.release()
        input.release()