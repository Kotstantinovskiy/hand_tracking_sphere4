from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.random as npr
from tqdm import tqdm

import torch
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
        self.layer2 = nn.Conv1d(self.in_channels*2, self.in_channels*3, 3, padding=1)
        self.short2 = nn.Conv1d(self.in_channels*2, self.in_channels*3, 1, bias=False)
        self.act2 = nn.PReLU(self.in_channels*3)
        self.layer3 = nn.Conv1d(self.in_channels*3, self.in_channels*5, 3, padding=1)
        self.short3 = nn.Conv1d(self.in_channels*3, self.in_channels*5, 1, bias=False)
        self.act3 = nn.PReLU(self.in_channels*5)
        self.layer4 = nn.Conv1d(self.in_channels*5, self.in_channels*6, 3, padding=1)
        self.short4 = nn.Conv1d(self.in_channels*5, self.in_channels*6, 1, bias=False)
        self.act4 = nn.PReLU(self.in_channels*6)
        self.layer5 = nn.Conv1d(self.in_channels*6, self.in_channels*8, 3, padding=1)
        self.short5 = nn.Conv1d(self.in_channels*6, self.in_channels*8, 1, bias=False)
        self.act5 = nn.PReLU(self.in_channels*8)
        self.layer6 = nn.Conv1d(self.in_channels*8, self.in_channels*9, 3, padding=1)
        self.short6 = nn.Conv1d(self.in_channels*8, self.in_channels*9, 1, bias=False)
        self.act6 = nn.PReLU(self.in_channels*9)
        self.layer7 = nn.Conv1d(self.in_channels*9, self.in_channels*11, 3, padding=1)
        self.short7 = nn.Conv1d(self.in_channels*9, self.in_channels*11, 1, bias=False)
        self.act7 = nn.PReLU(self.in_channels*11)
        self.layer8 = nn.Conv1d(self.in_channels*11, self.in_channels*12, 3, padding=1)
        self.short8 = nn.Conv1d(self.in_channels*11, self.in_channels*12, 1, bias=False)
        self.act8 = nn.PReLU(self.in_channels*12)
        self.layer9 = nn.Conv1d(self.in_channels*12, self.in_channels*14, 3, padding=1)
        self.short9 = nn.Conv1d(self.in_channels*12, self.in_channels*14, 1, bias=False)
        self.act9 = nn.PReLU(self.in_channels*14)
        self.layer10 = nn.Conv1d(self.in_channels*14, self.in_channels*15, 3, padding=1)
        self.short10 = nn.Conv1d(self.in_channels*14, self.in_channels*15, 1, bias=False)
        self.act10 = nn.PReLU(self.in_channels*15)
        self.layer_final = nn.Conv1d(self.in_channels*15, 1, self.window - 1)
        self.act_final = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.act1(self.layer1(x))
        x2 = self.act2(self.layer2(x1) + self.short2(x1))
        x3 = self.act3(self.layer3(x2) + self.short3(x2))
        x4 = self.act4(self.layer4(x3) + self.short4(x3))
        x5 = self.act5(self.layer5(x4) + self.short5(x4))
        x6 = self.act6(self.layer6(x5) + self.short6(x5))
        x7 = self.act7(self.layer7(x6) + self.short7(x6))
        x8 = self.act8(self.layer8(x7) + self.short8(x7))
        x9 = self.act9(self.layer9(x8) + self.short9(x8))
        x10 = self.act10(self.layer10(x9) + self.short10(x9))
        x11 = self.layer_final(x10)
        x11 = x11.view(-1, 2)
        return self.act_final(x11)

    def predict(self, x):
        x_tensor = torch.from_numpy(x).view(-1, 1, self.frame_size*self.window).float()
        with torch.no_grad():
            y = self(x_tensor).numpy()
        logits = np.argmax(y, axis=1)
        return logits


class Detector:
    def __init__(self, window=8, iterations=800, lr=0.1, batch_size=64, depth=8, type_model="catboost"):
        self.type_model = type_model
        if type_model == "catboost":
            self.detector = CatBoostClassifier(iterations=iterations,
                                               learning_rate=lr,
                                               depth=depth)
        elif type_model == "logistic_reg":
            self.detector = LogisticRegression(max_iter=iterations,
                                               verbose=1)
        elif type_model == 'neuro':
            self.detector = NeuroDetector(window=window, in_channels=3)
            self.optimizer = optim.SGD(self.detector.parameters(),
                                       lr=lr)

        self.window = window
        self.iterations = iterations
        self.batch_size = batch_size

    def train(self, gestures):
        ## processing data
        print("Start processing data")
        X = list()
        y = list()

        for j, gesture in tqdm(enumerate(gestures), desc='Loading dataset'):

            if len(gesture) < self.window:
                continue

            for i in range(len(gesture)):
                if i + self.window > len(gesture):
                    break

                g1 = gesture.data(i, i+self.window, norm_name='split')
                g2 = gesture.data(i, i+self.window, norm_name='split_delta')
                g3 = gesture.data(i, i+self.window, norm_name='norm_1')
                if self.type_model == 'neuro':
                    g = np.vstack([g1, g2, g3])
                    X.append(g.reshape(1, 3, -1))
                else:
                    X.append(g1.reshape(1, -1))

                if i + self.window == len(gesture):
                    if gesture.label == "No gesture":
                        y.append(0)
                    else:
                        y.append(1)
                else:
                    y.append(0)

        X = np.concatenate(X, axis=0)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        print("Start train model on %d samples" % X.shape[0])
        if self.type_model == "catboost":
            self.detector.fit(X_train, y_train, eval_set=Pool(X_test, y_test))
        elif self.type_model == "logistic_reg":
            self.detector.fit(X_train, y_train)
        elif self.type_model == 'neuro':
            self.detector.cuda()
            loss_fn = nn.CrossEntropyLoss()
            test_tensor = torch.from_numpy(X_test).view(-1, 3, self.window*42).float().cuda()
            test_target = torch.from_numpy(y_test).view(-1).long().cuda()
            train_tensor = torch.from_numpy(X_train).view(-1, 3, self.window*42).float().cuda()
            train_target = torch.from_numpy(y_train).view(-1).long().cuda()
            order = np.arange(len(train_target))
            for iter in range(self.iterations):
                avg_train_loss = 0
                avg_test_loss = 0
                npr.shuffle(order)
                for batch_id in tqdm(range(int(len(order)/self.batch_size)), desc='Iteration %d' % (iter + 1)):
                    sample_ids = order[batch_id*self.batch_size:(batch_id + 1)*self.batch_size]
                    batch = train_tensor[sample_ids, :, :]
                    target = train_target[sample_ids]
                    self.optimizer.zero_grad()
                    predict_tensor = self.detector(sample.view(-1, 3, self.window*42))
                    loss = loss_fn(predict_tensor, target.view(-1))
                    train_loss = loss.item()
                    loss.backward()
                    self.optimizer.step()
                    with torch.no_grad():
                        test_predict = self.detector(test_tensor)
                        test_loss = loss_fn(test_predict, test_target).item()
                    avg_train_loss += train_loss
                    avg_test_loss += test_loss
                print('\nIteration: %d, Train loss: %f, Test loss: %f' % (iter + 1, avg_train_loss/len(X_train), avg_test_loss/len(X_train)))
            self.detector.cpu()

        return X_train, X_test, y_train, y_test

    def predict(self, x):
        return self.detector.predict(x)

    def predict_proba(self, x):
        return self.detector.predict_proba(x)
