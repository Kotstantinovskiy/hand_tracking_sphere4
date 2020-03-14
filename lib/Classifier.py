from catboost import CatBoostRegressor

import numpy as np

class Classifier:
    def __init__(self):
        self.model = CatBoostRegressor()

    def train(self, data):
        X = np.empty((len(data), 32 * 42))
        for i, gest in enumerate(data):
            if len(gest) < 17:
                continue
            if len(gest) < 32:
                gest.pretty(32)

            X[i, :] = gest.data(-32)
            y[i] = gest.label

        print ("INFO: Made %s samples." % X.shape[0])

        self.model.fit(X, y)

    def predict(queue):
        X = np.array(queue).reshape(1, -1)
        return self.model.predict(queue)

    def load():
        self.model.load_model("classifier.ctbst")
        print("INFO: Model was loaded.")

    def dumpt():
        self.model.save_model("classifier.ctbst")
        print("INFO: Model was dumped.")

