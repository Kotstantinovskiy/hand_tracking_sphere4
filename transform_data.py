import numpy as np

def get_mean(norm_value=255):
    return np.array([
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]).reshape([3,1,1])


def get_std(norm_value=255):
    return np.array([
        38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value
    ]).reshape([3,1,1])

def Normalize(img):
    return (img - get_mean()) / get_std()
