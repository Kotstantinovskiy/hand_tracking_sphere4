from catboost import CatBoostRegressor
from uril
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
                gest = pretty_gest(gest, 32)

