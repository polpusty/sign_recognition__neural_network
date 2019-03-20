from nn.functions import *
from nn.layers import *
from nn.networks import MultiProcessingNetwork, AsyncNetwork
from nn.optimizers import AdamOptimizer


class VGGNet:
    @classmethod
    def get_network(cls, classes, size):
        return MultiProcessingNetwork([
            Convolution2d(32, 3, 3, 1),
            Activation(relu, relu_prime),
            Convolution2d(32, 3, 32, 1),
            Activation(relu, relu_prime),
            MaxPooling2d(2),
            Convolution2d(64, 3, 32, 1),
            Activation(relu, relu_prime),
            Convolution2d(64, 3, 64, 1),
            Activation(relu, relu_prime),
            MaxPooling2d(2),
            Convolution2d(128, 3, 64, 1),
            Activation(relu, relu_prime),
            Convolution2d(128, 3, 128, 1),
            Activation(relu, relu_prime),
            MaxPooling2d(2),
            Flatten(),
            FullConnected(128, 128 * 4 * 4),
            Activation(relu, relu_prime),
            FullConnected(128, 128),
            Activation(relu, relu_prime),
            Dropout(0.7, (1, 128)),
            FullConnected(len(classes), 128),
            Activation(softmax, softmax_prime)
        ], classes, size, AdamOptimizer(0.001, 0.9, 0.999, 10 ** -8))


class LeNet:
    @classmethod
    def get_network(cls, classes, size):
        net2 = AsyncNetwork([
            Convolution2d(6, 5, 3, 0),
            Activation(relu, relu_prime),
            MaxPooling2d(2),
            Convolution2d(16, 5, 6, 0),
            Activation(relu, relu_prime),
            MaxPooling2d(2),
            Flatten(),
            FullConnected(120, 400),
            Activation(relu, relu_prime),
            FullConnected(84, 120),
            Activation(relu, relu_prime),
            FullConnected(2, 84),
        ], classes, size, AdamOptimizer(0.001, 0.9, 0.999, 10 ** -8))

        return net2
