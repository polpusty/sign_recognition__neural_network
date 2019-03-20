from torch import nn
from torch import optim

from tr.layers import Flatten
from tr.neural_network import Network


class LeNet:
    @classmethod
    def get_network(cls, classes, size):
        return Network([
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(16 * 5 * 5),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, len(classes)),
        ], classes, size, optim.Adam)


class VGGNet:
    @classmethod
    def get_network(cls, classes, size):
        return Network([
            nn.Conv2d(32, 3, 3, 1),
            nn.Conv2d(32, 3, 32, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 3, 32, 1),
            nn.Conv2d(64, 3, 64, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 3, 64, 1),
            nn.Conv2d(128, 3, 128, 1),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(128, 128 * 4 * 4),
            nn.Linear(128, 128),
            nn.Linear(len(classes), 128, 'softmax')
        ], classes, size, optim.Adam)
