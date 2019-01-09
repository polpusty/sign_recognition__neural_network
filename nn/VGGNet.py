from nn.neural_network import Network
from nn.layers import *


class VGGNet:

    @classmethod
    def get_network(cls, number_classes):
        return Network([
            ConvolutionLayer(32, 3, 3, 32 * 32 * 3, 1),
            ConvolutionLayer(32, 3, 32, 32 ** 3, 1),
            MaxPoolingLayer(2),
            ConvolutionLayer(64, 3, 32, 32 * 16 * 16, 1),
            ConvolutionLayer(64, 3, 64, 16 * 16 * 64, 1),
            MaxPoolingLayer(2),
            ConvolutionLayer(128, 3, 64, 8 * 8 * 64, 1),
            ConvolutionLayer(128, 3, 128, 8 * 8 * 128, 1),
            MaxPoolingLayer(2),
            FlattenLayer(),
            FullConnectedLayer(128, 128 * 4 * 4),
            FullConnectedLayer(128, 128),
            DropoutLayer(0.7, (128, 1)),
            FullConnectedLayer(number_classes, 128)
        ])
