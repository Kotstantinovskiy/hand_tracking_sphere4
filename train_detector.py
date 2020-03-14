from catboost import CatBoostRegressor
import numpy as np
from ./lib/Gesture import Gesture

class Detector:

    def __init__(self, iterations=800, lr=0.1, depth=8):
        detector = CatBoostRegressor(iterations=iterations,
                                     learning_rate=lr,
                                     depth=depth)

    def train(self, gestures, window=8):

        X_train = list()
        y_list = list()

        for gesture in gestures:

            for g in gesture.slice(i, i+window):


        with open(path_train, "r") as train_file:
            for line in train_file:
                parts = line.split("\t").strip()





    def predict(self):
