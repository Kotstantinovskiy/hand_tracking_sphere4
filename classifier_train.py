from catboost import CatBoostRegressor
import numpy as np

def train_detector(path_train, window=8, iterations=800, learning_rate=0.1, depth=8):

