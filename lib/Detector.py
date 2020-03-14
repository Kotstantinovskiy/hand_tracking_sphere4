from catboost import CatBoostClassifier
import numpy as np
from ./lib/Gesture import Gesture

class Detector:

    def __init__(self, window=8, iterations=800, lr=0.1, depth=8, type_model="catboost"):
        if type_model == "catboost":
            self.detector = CatBoostClassifier(iterations=iterations,
                                         learning_rate=lr,
                                        depth=depth)
        self.queue = list()
        self.window = window

    def train(self, gestures):

        ## processing data
        print("Start processing data")
        X_train = list()
        y_train = list()

        for i, gesture in enumerate(gestures):
            if len(gesture) < self.window:
                continue

            for g in gesture.slice(i, i+self.window):
                X_train.append(g)
                if i+window == len(gestures)-1:
                    y_train.append(1)
                else:
                    y_train.append(0)


        X_train = numpy.concatenate(X_train, axis=0)
        y_train = np.array(y_train)

        ##train Catboost
        print("Start train catboost")
        self.detector.fit(X_train, y_train, verbose=True)

    def predict(self, x):
        return self.detector.predict(x)

    def predict_proba(self, x):
        return self.detector.predict_proba(x)
