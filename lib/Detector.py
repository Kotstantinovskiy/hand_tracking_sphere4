from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
        X = list()
        y = list()

        for j, gesture in enumerate(gestures):

            if len(gesture) < self.window:
                continue

            for i in range(len(gesture)):
                if i + self.window > len(gesture):
                    break

                g = gesture.data(i, i+self.window)
                X.append(g.reshape(1, -1))

                if i + self.window == len(gesture) - 1:
                    if gesture.label == "No gesture":
                        y.append(0)
                    else:
                        y.append(1)
                else:
                    y.append(0)
            print('%d/%d' % (j, len(gestures)), end='\r')

        X = np.concatenate(X, axis=0)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        ##train Catboost
        print("Start train model on %d samples" % X.shape[0])
        if self.type_model == "catboost":
            self.detector.fit(X_train, y_train, eval_set=Pool(X_test, y_test))
        elif self.type_model == "logistic_regression":
            self.detector.fit(X_train, y_train)

        return X_train, X_test, y_train, y_test

    def predict(self, x):
        return self.detector.predict(x)

    def predict_proba(self, x):
        return self.detector.predict_proba(x)
