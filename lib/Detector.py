from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

from torch import nn
from torch import optim

class NeuroDetector(nn.Module):
    def __init__(self, window=8, frame_size=42, in_channels=1):
        super(NeuroDetector, self).__init__()
        self.window = window
        self.in_channels = in_channels
        self.frame_size = frame_size
        self.layer1 = nn.Conv1d(self.in_channels, self.in_channels*2, self.frame_size, stride=self.frame_size)
        self.act1 = nn.PReLU(self.in_channels*2)
        self.layer2 = nn.Conv1d(self.in_channels*2, self.in_channels*4, 3)
        self.act2 = nn.PReLU(self.in_channels*4)
        self.layer3 = nn.Conv1d(self.in_channels*4, self.in_channels*8, 3)
        self.act3 = nn.PReLU(self.in_channels*8)
        self.layer4 = nn.Conv1d(self.in_channels*8, self.in_channels*16, 3)
        self.act4 = nn.PReLU(self.in_channels*16)
        self.layer_final = nn.Conv1d(self.in_channels*16, 1, self.window - 7)
        self.act_final = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.act1(self.layer1(x))
        x2 = self.act2(self.layer2(x1))
        x3 = self.act3(self.layer3(x2))
        x4 = self.act4(self.layer4(x3))
        x5 = self.layer_final(x4)
        x5 = x5.view(-1, 2)
        return self.act_final(x5)


class Detector:
    def __init__(self, window=8, iterations=800, lr=0.1, depth=8, type_model="catboost"):
        self.type_model = type_model
        if type_model == "catboost":
            self.detector = CatBoostClassifier(iterations=iterations,
                                               learning_rate=lr,
                                               depth=depth)
        elif type_model == "logistic_reg":
            self.detector = LogisticRegression(max_iter=iterations,
                                               verbose=1)
        elif type_model == 'neuro':
            self.detector = NeuroDetector(window=window)
            self.optimizer = optim.SGD(self.detector.parameters(),
                                       lr=lr)

        self.window = window
        self.iterations = iterations

    def train(self, gestures):
        ## processing data
        print("Start processing data")
        X = list()
        y = list()

        for j, gesture in enumerate(gestures):

            if len(gesture) < self.window:
                continue

            for i in range(len(gesture)):
                if i + self.window > len(gesture):
                    break

                g = gesture.data(i, i+self.window)
                X.append(g.reshape(1, -1))

                if i + self.window == len(gesture):
                    if gesture.label == "No gesture":
                        y.append(0)
                    else:
                        y.append(1)
                else:
                    y.append(0)
            print('%d/%d' % (j, len(gestures)), end='\r')

        X = np.concatenate(X, axis=0)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        ##train Catboost
        print("Start train model on %d samples" % X.shape[0])
        if self.type_model == "catboost":
            self.detector.fit(X_train, y_train, eval_set=Pool(X_test, y_test))
        elif self.type_model == "logistic_reg":
            self.detector.fit(X_train, y_train)
        elif self.type_model == 'neuro':
            loss_fn = nn.CrossEntropyLoss()
            test_tensor = torch.from_numpy(X_test)
            test_target = torch.from_numpy(y_test, dtype=torch.long).view(-1)
            for iter in range(self.iterations):
                for sample, target in zip(X_train, y_train):
                    self.optimizer.zero_grad()
                    sample_tensor = torch.from_numpy(sample).view(1, 1, self.window*42)
                    target_tensor = torch.tensor([0], dtype=torch.long)
                    if target:
                        target_tensor[0] = 1
                    predict_tensor = self.detector(sample_tensor)
                    loss = loss_fn(predict_tensor, target_tensor)
                    train_loss = loss.item()
                    loss.backward()
                    self.optimizer.step()
                    with torch.no_grad():
                        test_predict = self.detector(test_tensor)
                        test_loss = loss_fn(test_predict, test_target).item()
                        print('Iteration: %d, Train loss: %f, Test loss: %f' % (iter, train_loss, test_loss))

        return X_train, X_test, y_train, y_test

    def predict(self, x):
        return self.detector.predict(x)

    def predict_proba(self, x):
        return self.detector.predict_proba(x)
