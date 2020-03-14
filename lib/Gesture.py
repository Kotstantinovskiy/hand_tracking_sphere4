import numpy as np

class Gesture:
    def __init__(self, frames=list()):
        self._data = frames

    def parse_line(self, line):
        splited = line.split("\t")
        self.label = splited[1]
        self._data = list()

        landmarks = list(map(float, splited[3].split(" ")))

        assert len(landmarks) % 42 == 0, "Bad gesture shape"
        for l in range(42, len(landmarks)+1, 42):
            self._data.append(np.array(landmarks[l-42:l]))

    def __len__(self):
        return len(self._data)

    def data(self, i=None, j=None):
        assert i != None or j != None, "Bad slice"
        if i != None and j != None:
            subdata = self._data[i:j]
        elif i == None:
            subdata = self._data[:j]
        else:
            subdata = self._data[i:]
        zero_min = np.min(subdata[0])
        zero_std = np.max(subdata[0]) - zero_min
        subdata[0] -= zero_min
        subdata[0] /= zero_std
        for i, vec in enumerate(subdata):
            if i == 0:
                continue
            subdata[i] = (vec - subdata[i - 1]) / zero_std
        return np.array(subdata).flatten()

    def push(self, frame):
        assert len(frame) == 42, "Bad frame shape"
        self._data += [np.array(frame)]

    def drop_first(self):
        self._data = self._data[1:]
