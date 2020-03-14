from catboost import CatBoostRegressor
import numpy as np


def train_detector(path_train, window=8, iterations=800, learning_rate=0.1, depth=8):

    detector = CatBoostRegressor(iterations=iterations,
                                 learning_rate=learning_rate,
                                 depth=depth)

    X_train = list()
    y_train_class = list()


    with open(path_train, "r") as train_file:
        for line in train_file:
            parts = line.split("\t").strip()

        name_dir = int(parts[0])
        lable = parts[1]
        len_ = int(parts[2])
        vecs = np.array(list(map(float, parts[3].split(" "))))

        index_sliding = np.arange(window*63)[None, :] + 63*np.arange(int(vecs/63)-window)[:, None] ## ?????

        vecs = vecs[index_sliding].reshape((-1, 63, -1))
        X_train.append(vecs)
        y_train_class.append(lable)


