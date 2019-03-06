import abc

import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce

from nn.math_functions import relu, relu_prime, softmax, softmax_prime

rnd = np.random.RandomState(2)


class Layer(abc.ABC):
    def __init__(self):
        self.result = None
        self.name = None

    def has_weights(self):
        return hasattr(self, 'weights')

    @abc.abstractmethod
    def forward(self, input_data):
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, delta, input_data):
        raise NotImplementedError


class ConvolutionLayer(Layer):
    def __init__(self, number_kernels, kernel_size, depth, padding=0):
        super(Layer, self).__init__()
        self.number_kernels = number_kernels
        self.kernel_size = kernel_size
        self.depth = depth
        self.padding = padding

        self.weights = rnd.randn(number_kernels, depth, kernel_size, kernel_size) / np.sqrt(number_kernels / 2.)
        self.biases = np.ones((number_kernels, 1)) * 0.01

    def __str__(self):
        return f'Convolution {self.weights.shape}'

    def forward(self, input_data):
        self.check_input_data(input_data)
        self.result = relu(self.convolution(input_data, self.weights, self.biases, self.padding))

    @staticmethod
    def convolution(input_data, weights, biases, padding):
        depth, height, width = input_data.shape
        number_kernels, depth_kernel, kernel_size, kernel_size = weights.shape

        result = np.zeros((number_kernels,
                           height - kernel_size + 2 * padding + 1,
                           width - kernel_size + 2 * padding + 1))
        input_with_padding = np.zeros((depth, height + 2 * padding, width + 2 * padding))
        input_with_padding[:, padding:height + padding, padding:width + padding] = input_data

        for number_kernel, kernel in enumerate(weights):
            for d in range(depth):
                result[number_kernel] += convolve2d(input_with_padding[d], kernel[d][::-1, ::-1], 'valid')
            result[number_kernel] += biases[number_kernel]

        return result

    def backward(self, delta, input_data):
        return self.backward_convolution_layer(delta, input_data)

    def backward_convolution_layer(self, delta, input_data):
        delta_biases = np.zeros(self.biases.shape)
        delta_weights = np.zeros(self.weights.shape)
        weights_rotated = np.zeros((self.depth, self.number_kernels, self.kernel_size, self.kernel_size))
        delta = delta * relu_prime(delta)
        depth, height, width = input_data.shape
        input_with_padding = np.zeros((depth, height + 2 * self.padding, width + 2 * self.padding))
        input_with_padding[:, self.padding:height + self.padding, self.padding:width + self.padding] = input_data

        for number_kernel in range(len(self.weights)):
            for d in range(self.depth):
                delta_weights[number_kernel][d] = convolve2d(input_with_padding[d], delta[number_kernel][::-1, ::-1],
                                                             'valid')
                weights_rotated[d][number_kernel] = self.weights[number_kernel][d][::-1, ::-1]

            delta_biases[number_kernel] = np.sum(delta[number_kernel])

        delta_result = self.convolution(delta, weights_rotated, np.zeros_like(delta_biases), self.padding)

        return delta_result, delta_biases, delta_weights

    def check_input_data(self, input_data):
        depth, width, height = input_data.shape
        assert depth == self.depth
        assert width > self.kernel_size
        assert height > self.kernel_size


class MaxPoolingLayer(Layer):
    def __init__(self, stride):
        super(Layer, self).__init__()
        self.stride = stride

    def __str__(self):
        return f'MaxPooling {self.stride}x{self.stride}'

    def forward(self, input_data):
        self.check_input_data(input_data)
        self.result = self.pooling(input_data, self.stride)

    def backward(self, delta, input_data):
        return self.backward_pooling_layer(delta, input_data)

    @staticmethod
    def pooling(input_data, stride):
        depth, height, width = input_data.shape

        result = np.zeros((depth, int(height / stride), int(width / stride)))

        for i in range(depth):
            result[i] = block_reduce(input_data[i], (stride, stride), np.max)

        return result

    def backward_pooling_layer(self, delta, input_data):
        delta_result = np.zeros_like(input_data)
        depth, height, width = delta.shape
        s, shape, unravel_index = self.stride, (self.stride, self.stride), np.unravel_index
        for i in range(depth):
            for y in range(height):
                y_s = y * s
                for x in range(width):
                    x_s = x * s
                    input_data_piece = input_data[i, y_s:y_s + s, x_s:x_s + s]
                    y_max, x_max = unravel_index(input_data_piece.argmax(), shape)
                    delta_result[i, y_s + y_max, x_s + x_max] = delta[i, y, x]
        return delta_result

    def check_input_data(self, input_data):
        depth, height, width = input_data.shape
        assert (height / self.stride) % 1 == 0
        assert (width / self.stride) % 1 == 0


class BatchNormalizationLayer(Layer):
    def __init__(self, number_inputs, eps=10 ** -8):
        super(BatchNormalizationLayer, self).__init__()

        self.gamma = self.weights = np.ones((1, number_inputs))
        self.beta = self.biases = np.zeros((1, number_inputs))
        self.eps = eps

        self.input_flat = None
        self.mean = None

    def forward(self, input_data):
        self.input_flat = input_data.ravel().reshape(input_data.shape[0], -1)
        self.mean = np.mean(self.input_flat, axis=0)

        return input_data

    def backward(self, delta, input_data):
        return delta


class DropoutLayer(Layer):
    def __init__(self, effort, shape):
        super(Layer, self).__init__()
        self.effort = effort
        self.shape = shape

        self.mask = (np.random.rand(*shape) < effort) / effort

    def __str__(self):
        return f'Dropout {self.effort} {self.shape}'

    def forward(self, input_data):
        self.result = self.dropout(input_data, self.mask)

    @staticmethod
    def dropout(input_data, mask):
        return np.multiply(input_data, mask)

    def backward(self, delta, _):
        return self.dropout(delta, self.mask)


class FlattenLayer(Layer):
    def __init__(self):
        super(Layer, self).__init__()

    def __str__(self):
        return f'Flatten'

    def forward(self, input_data):
        self.result = self.flatten(input_data)

    @staticmethod
    def flatten(input_data):
        return input_data.reshape(1, np.prod(input_data.shape))

    def backward(self, delta, input_data):
        delta = delta.reshape(input_data.shape)
        return delta


class FullConnectedLayer(Layer):
    def __init__(self, number_outputs, number_inputs, activation='relu'):
        super(Layer, self).__init__()
        if activation == 'relu':
            self.activation, self.activation_prime = relu, relu_prime
        elif activation == 'softmax':
            self.activation, self.activation_prime = softmax, softmax_prime
        self.weights = rnd.randn(number_inputs, number_outputs) * np.sqrt(number_outputs / 2.)
        self.biases = np.ones((1, number_outputs)) * 0.01
        self.f = None

    def __str__(self):
        return f'FullConnected {self.weights.shape}'

    def forward(self, input_data):
        self.f = np.dot(input_data, self.weights) + self.biases
        self.result = self.activation(self.f)

    def backward(self, delta, input_data):
        delta_result = delta * self.activation_prime(delta)
        delta_biases = delta_result
        delta_weights = np.dot(input_data.T, delta_result)
        delta_result = np.dot(delta_result, self.weights.T)
        return delta_result, delta_biases, delta_weights
