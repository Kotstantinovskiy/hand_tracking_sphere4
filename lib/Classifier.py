import lightgbm as lgb
from sklearn.model_selection import train_test_split

import numpy as np

class Classifier:
    def __init__(self):
        self.model = CatBoostRegressor()

    def train(self, data):
        sample_size = len(data)
        X = np.empty((sample_size, 32 * 42))
        y = np.empty((1, sample_size)
        for i, gest in enumerate(data):
            if len(gest) < 17:
                continue
            if len(gest) < 32:
                gest.pretty(32)

            X[i, :] = gest.data(-32)
            y[i] = gest.label
            print('%d/%d' % (i, len(data)), end='\r')

        print ("INFO: Made %s samples." % X.shape[0])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        train_data = lgb.Dataset(X_train, label=y_train)
        validation_data = lgb.Dataset(X_val, y_val, reference=train_data)
        param = {'num_leaves': 31, 'objective': 'binary'}
        param['metric'] = 'auc'
        self.model = bst = lgb.train(params, train_data, valid_sets=[validation_data])

    def predict(queue):
        X = np.array(queue).reshape(1, -1)
        return self.model.predict(queue)

    def load():
        self.model.load_model("classifier.ctbst")
        print("INFO: Model was loaded.")

    def dump():
        self.model.save_model("classifier.ctbst")
        print("INFO: Model was dumped.")
