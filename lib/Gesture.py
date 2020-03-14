import numpy as np

class Gesture:
    def __init__(self, frames=list()):
        self._data = frames

    def parse_line(self, line):
        splited = line.split("\t")
        self.label = splited[1]
        self._data = list()

        landmarks = list(map(float, splited[3].split(" ")[:-1]))

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
        first_frame = subdata[0]
        for i, vec in enumerate(subdata):
            if i == 0:
                continue
            subdata[i] -= first_frame
        subdata.pop(0)
        return np.array(subdata).flatten()

    def push(self, frame):
        assert len(frame) == 42, "Bad frame shape"
        self._data += [np.array(frame)]

    def drop_first(self):
        self._data = self._data[1:]

    def pretty(self, asking_size):
        deter = asking_size - len(self)
        original_mask = [2 * i for i in range(len(self))]
        fake_mask = list(np.random.choice([2 * i + 1 for i in range(len(self) - 1)], deter, False))
        answer = []
        n = 0
        for i in range(2 * len(self)):
            if i in original_mask:
                answer.append(self._data[i / 2])
                n += 1
            if i in fake_mask:
                answer.append((self._data[(i - 1) / 2] - self._data[(i + 1) / 2]) / 2)
                n += 1
        if n != asking_size:
            print("ERROR: bug in pretty function")

        self._data = answer

