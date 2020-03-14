import numpy as np

class Gesture:
    def __init__(self, line):
        splited = line.split("\t")
        self.label = splited[1]
        self._data = list()

        landmarks = list(map(float, splited[3].split(" ")))

        assert len(landmarks) % 63 == 0, "Bad gesture shape"
        for l in range(63, len(landmarks)+1, 63):
            self.data.append(np.array(landmarks[l-63:l]))

    def data(i, j):
        assert i >= 0 and i < len(self._data), "Bad i"
        assert j > i and j < len(self._data), "Bad j"
        subdata = self.data[i:j]
        zero_min = np.min(subdata[0])
        zero_std = np.max(subdata[0]) - zero_min
        subdata[0] -= zero_min
        subdata[0] /= zero_std
        for i, vec in enumerate(subdata):
            if i == 0:
                continue
            subdata[i] = (vec - subdata[i - 1]) / zero_std
        return subdata