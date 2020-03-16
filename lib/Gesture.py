import numpy as np


def norm_0(subdata):
    normed = []
    for vec in subdata:
        normed.append(vec.copy())
    return normed


def norm_bug(subdata):
    zero_min = np.min(subdata[0])
    zero_std = np.max(subdata[0]) - zero_min
    subdata[0] -= zero_min
    subdata[0] /= zero_std
    normed = []
    for i, vec in enumerate(subdata):
        if i == 0:
            continue
        subdata[i] = (vec - subdata[i - 1]) / zero_std
    return subdata


def norm_1(subdata):
    first_vec = subdata[0]
    for i, vec in enumerate(subdata):
        if i == 0:
            continue
        subdata[i] = subdata[i] - first_vec

    subdata.pop(0)

    return subdata


def norm_2(subdata):
    first_vec = subdata[0]
    for i, vec in enumerate(subdata):
        if i == 0:
            continue
        subdata[i] = subdata[i] - first_vec
    subdata.pop(0)

    for i, vec in enumerate(subdata):
        if i == 0:
            max_x = np.max(vec[::2])
            max_y = np.max(vec[1::2])
            min_x = np.min(vec[::2])
            min_y = np.min(vec[1::2])
        else:
            if max_x > np.max(vec[::2]):
                max_x = np.max(vec[::2])
            if max_y > np.max(vec[1::2]):
                max_y = np.max(vec[1::2])
            if min_x < np.min(vec[::2]):
                min_x = np.min(vec[::2])
            if min_y < np.min(vec[1::2]):
                min_y = np.min(vec[1::2])

    std_x = max_x - min_x
    std_y = max_y - min_y

    for i in range(len(subdata)):
        subdata[i][::2] /= std_x
        subdata[i][1::2] /= std_y

    return subdata

def norm_delta(subdata):
    zero_min = np.min(subdata[0])
    zero_std = np.max(subdata[0]) - zero_min
    subdata[0] -= zero_min
    subdata[0] /= zero_std
    tmp = subdata[0].copy()
    for i, vec in enumerate(subdata):
        if i == 0:
            continue
        _ = subdata[i].copy()
        subdata[i] = (vec - tmp) / zero_std
        tmp = _
    return subdata

def norm_split_delta(subdata):
    zero_minx = np.min(subdata[0][::2])
    zero_miny = np.min(subdata[0][1::2])
    zero_stdx = np.max(subdata[0][::2]) - zero_minx
    zero_stdy = np.max(subdata[0][1::2]) - zero_miny
    subdata[0][::2] -= zero_minx
    subdata[0][1::2] -= zero_miny
    subdata[0][::2] /= zero_stdx
    subdata[0][1::2] /= zero_stdy
    tmp = subdata[0].copy()
    for i, vec in enumerate(subdata):
        if i == 0:
            continue
        _ = subdata[i].copy()
        subdata[i][::2] = (vec[::2] - tmp[::2]) / zero_stdx
        subdata[i][1::2] = (vec[1::2] - tmp[1::2]) / zero_stdy
        tmp = _
    return subdata

def norm_split(subdata):
    zero_minx = np.min(subdata[0][::2])
    zero_miny = np.min(subdata[0][1::2])
    zero_stdx = np.max(subdata[0][::2]) - zero_minx
    zero_stdy = np.max(subdata[0][1::2]) - zero_miny
    subdata[0][::2] -= zero_minx
    subdata[0][1::2] -= zero_miny
    subdata[0][::2] /= zero_stdx
    subdata[0][1::2] /= zero_stdy
    tmp = subdata[0].copy()
    for i, vec in enumerate(subdata):
        if i == 0:
            continue
        subdata[i][::2] = (vec[::2] - zero_minx) / zero_stdx
        subdata[i][1::2] = (vec[1::2] - zero_miny) / zero_stdy
    return subdata

class Gesture:
    def __init__(self, frames=None):
        if frames is None:
            frames = list()
        self._data = frames

        self.norm_dict = {
        "norm_0": norm_0,
        "norm_1": norm_1,
        "norm_2": norm_2,
        "bug": norm_bug,
        "delta": norm_delta,
        "split_delta": norm_split_delta,
        "split": norm_split
        }

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

    def data(self, i=None, j=None, norm_name="split"):
        norm = self.norm_dict.get(norm_name)

        if norm is None:
            print("ERROR: bad normalization name.")

        assert i != None or j != None, "Bad slice"

        if i != None and j != None:
            subdata = self._data[i:j]
        elif i == None:
            subdata = self._data[:j]
        else:
            subdata = self._data[i:]
        subdata = norm(subdata)

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
