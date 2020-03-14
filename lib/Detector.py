from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

class Detector:

    def __init__(self, window=8, iterations=800, lr=0.1, depth=8, type_model="catboost"):
        self.type_model = type_model
        if type_model == "catboost":
            self.detector = None
            #self.detector = CatBoostClassifier(iterations=iterations,
            #                                   learning_rate=lr,
            #                                   depth=depth)
        elif type_model == "logistic_reg":
            self.detector = LogisticRegression(max_iter=iterations,
                                               verbose=1)

        self.window = window

    def train(self, gestures, valid=False):

        ## processing data
        print("Start processing data")
        X = list()
        y = list()

        for gesture in gestures:

            if len(gesture) < self.window:
                continue

            for i in range(len(gesture)):
                if i+self.window > len(gesture):
                    break

                for g in gesture.data(i, i+self.window):

                    X.append(g.reshape(1, -1))

                    if i+self.window == len(gesture)-1:
                        y.append(1)
                    else:
                        y.append(0)

        X = np.concatenate(X, axis=0)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        X_train_lgb = lgb.Dataset(X_train, label=y_train)
        X_test_lgb = lgb.Dataset(X_test, label=y_test)

        ##train Catboost
        print("Start train model")
        if self.type_model == "catboost":
            #elf.detector.fit(X_train, y_train, eval_set=(X_test, y_test))
            params = {'num_leaves': 16, 'objective': 'binary', 'metric': 'auc'}
            self.detector = lgb.train(params, X_train_lgb, valid_sets=[X_test_lgb])
        elif self.type_model == "logistic_regression":
            pass
            #self.detector.fit(X_train, y_train)

    def predict(self, x):
        return self.detector.predict(x)

    def predict_proba(self, x):
        return self.detector.predict_proba(x)
