import csv
import cv2
import numpy as np
import tensorflow as tf
import os
from hand_tracker import HandTracker

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "./palm_detection_without_custom_op.tflite"
#LANDMARK_MODEL_PATH = "./hand_landmark.tflite"
LANDMARK_MODEL_PATH = "./hand_landmark_3d.tflite"
ANCHORS_PATH = "./anchors.csv"
PATH = "./new_lables/"

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

lables = {}
lables_file = open("lables.csv", "w")
for line in lables:
    lables[line.split("\t")[0]] = line.split("\t")[1]

output_file = open("output.txt", "w")
for name_dir in os.listdir(PATH):

    output_file.write(name_dir + "\t" + lables[name_dir] + "\t" + str(len(os.listdir(PATH + name_dir))) + "\t")

    for name_file in os.listdir(PATH + name_dir):
        image = cv2.imread(PATH + name_dir + "/" + name_file, flags=cv2.IMREAD_COLOR)
        points, bbox = detector(image)

        if points is not None:
            for p in points:
                output_file.write(str(p[0]) + " " + str(p[1]) + " ")

    output_file.write("\n")
