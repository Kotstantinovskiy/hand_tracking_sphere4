from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class Detector:

    def __init__(self, window=8, iterations=800, lr=0.1, depth=8, type_model="catboost"):
        self.type_model = type_model
        if type_model == "catboost":
            self.detector = CatBoostClassifier(iterations=iterations,
                                               learning_rate=lr,
                                               depth=depth)
        elif type_model == "logistic_reg":
            self.detector = LogisticRegression(max_iter=iterations,
                                               verbose=1)

        self.window = window

    def train(self, gestures):

        ## processing data
        print("Start processing data")
        X_train = list()
        y_train = list()

        for i, gesture in enumerate(gestures):
            if len(gesture) < self.window:
                continue

            for g in gesture.data(i, i+self.window):
                X_train.append(g)
                if i+window == len(gestures)-1:
                    y_train.append(1)
                else:
                    y_train.append(0)

        X_train = numpy.concatenate(X_train, axis=0)
        y_train = np.array(y_train)

        ##train Catboost
        print("Start train model")
        if self.type_model == "catboost":
            self.detector.fit(X_train, y_train)
        elif self.type_model == "logistic_regression":
            self.detector.fit(X_train, y_train)

    def predict(self, x):
        return self.detector.predict(x)

    def predict_proba(self, x):
        return self.detector.predict_proba(x)
