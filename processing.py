import cv2
import os
from hand_tracker import HandTracker
import time
from multiprocessing import Pool

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "./models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "./models/hand_landmark.tflite"
#LANDMARK_MODEL_PATH = "./models/hand_landmark_3d.tflite"
ANCHORS_PATH = "./anchors.csv"
PATH = "./data/"

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

labeles = {}
labeles_file = open("labeles.csv", "r")
for line in labeles_file:
    labeles[line.split("\t")[0]] = line.split("\t")[1].strip()


def process_dir(args):
    t1 = time.time()
    num_dir, name_dir = args
    with open("./outputs/output%d.txt" % num_dir, "w") as output_file:
        output_file.write(name_dir + "\t" + labeles[name_dir] + "\t" + str(len(os.listdir(PATH + name_dir))) + "\t")

        for name_file in list(sorted(os.listdir(PATH + name_dir))):
            image = cv2.imread(PATH + name_dir + "/" + name_file, flags=cv2.IMREAD_COLOR)
            points, bbox = detector(image)

            if points is not None:
                for point in points:
                    output_file.write(str(point[0]) + " " + str(point[1]) + " ")

        output_file.write("\n")
    print('%d (%d)' % (num_dir, time.time() - t1))


dirs = list(enumerate(os.listdir(PATH)))
p = Pool(36)
p.map(process_dir, dirs)
