import cv2
from classify import Classify
from CNN_model import CNNModel
from dataset_creation import CreateTrainData


CreateTrainData.createFrames("frames")  #frames is a path where frames will be saved
CreateTrainData.dataToTrainAndTest()

CNNModel.runModel()
CNNModel.ModelResultsPlot("ggplot")

vs = cv2.VideoCapture('331.mp4')
Classify.classifyVideo(vs)
