from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

import numpy as np

class Classifier:
    def __init__(self):
        self.model = CatBoostClassifier()

    def train(self, data):
        sample_size = len(data)
        X = np.empty((sample_size, 32 * 42))
        y = np.empty((1, sample_size))
        for i, gest in enumerate(data):

            if len(gest) < 17:
                continue

            if len(gest) < 32:
                gest.pretty(32)

            X[i, :] = gest.data(-32)
            y[i] = gest.label
            print('%d/%d' % (i, len(data)), end='\r')

        print ("INFO: Made %s samples." % X.shape[0])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

        self.model.fit(X_train, y_train, eval_set=Pool(X_val, y_val))

    def predict(self, queue, *args, **kwargs):
        return self.model.predict(np.array(queue).reshape(1, -1),  *args, **kwargs)

    def load(self):
        self.model.load_model("classifier.ctbst")
        print("INFO: Model was loaded.")

    def dump(self):
        self.model.save_model("classifier.ctbst")
        print("INFO: Model was dumped.")
