from nn.layers import *
from nn.neural_network import Network
from nn.optimizers import AdamOptimizer


class VGGNet:

    @classmethod
    def get_network(cls, classes, size):
        return Network([
            ConvolutionLayer(32, 3, 3, 1),
            ConvolutionLayer(32, 3, 32, 1),
            MaxPoolingLayer(2),
            ConvolutionLayer(64, 3, 32, 1),
            ConvolutionLayer(64, 3, 64, 1),
            MaxPoolingLayer(2),
            ConvolutionLayer(128, 3, 64, 1),
            ConvolutionLayer(128, 3, 128, 1),
            MaxPoolingLayer(2),
            FlattenLayer(),
            FullConnectedLayer(128, 128 * 4 * 4),
            FullConnectedLayer(128, 128),
            DropoutLayer(0.7, (1, 128)),
            FullConnectedLayer(len(classes), 128, 'softmax')
        ], classes, size, AdamOptimizer(0.001, 0.9, 0.999, 10 ** -8))
