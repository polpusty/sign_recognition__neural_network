import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce
from functools import reduce

from nn.math_functions import relu, relu_prime

rnd = np.random.RandomState(2)


class Layer:
    def __init__(self):
        self.result = None

    def has_weights(self):
        return hasattr(self, 'weights')


class ConvolutionLayer(Layer):
    def __init__(self, number_kernels, kernel_size, depth, input_size, padding=0):
        super(Layer, self).__init__()
        self.number_kernels = number_kernels
        self.kernel_size = kernel_size
        self.depth = depth
        self.padding = padding

        self.weights = rnd.randn(number_kernels, depth, kernel_size, kernel_size) * np.sqrt(2.0 / input_size)
        self.biases = np.ones((number_kernels, 1)) * 0.01

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
                result[number_kernel] += convolve2d(input_with_padding[d], np.rot90(kernel[d], 2), mode='valid') + \
                                         biases[number_kernel]

        return result

    def backward(self, delta, input_data):
        return self.backward_convolutional_layer(delta, input_data)

    def backward_convolutional_layer(self, delta, input_data):
        delta_biases = np.zeros_like(self.biases)
        delta_weights = np.zeros_like(self.weights)
        weights_rotated = np.zeros((self.depth, self.number_kernels, self.kernel_size, self.kernel_size))
        delta = delta * relu_prime(delta)
        depth, height, width = input_data.shape
        input_with_padding = np.zeros((depth, height + 2 * self.padding, width + 2 * self.padding))
        input_with_padding[:, self.padding:height + self.padding, self.padding:width + self.padding] = input_data

        for number_kernel in range(len(self.weights)):
            for y in range(self.depth):
                delta_weights[number_kernel][y] = convolve2d(input_with_padding[y], np.rot90(delta[number_kernel], 2),
                                                             mode='valid')
                weights_rotated[y][number_kernel] = np.rot90(self.weights[number_kernel][y], 2)

            delta_biases[number_kernel] = np.sum(delta[number_kernel])

        delta_result = self.convolution(delta, weights_rotated, np.zeros_like(delta_biases), self.padding)

        return delta_biases, delta_weights, delta_result

    def check_input_data(self, input_data):
        depth, width, height = input_data.shape
        assert depth == self.depth
        assert width > self.kernel_size
        assert height > self.kernel_size


class MaxPoolingLayer(Layer):
    def __init__(self, stride):
        super(Layer, self).__init__()
        self.stride = stride

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


class DropoutLayer(Layer):
    def __init__(self, effort, shape):
        super(Layer, self).__init__()
        self.effort = effort
        self.shape = shape

        self.mask = (np.random.rand(*shape) < effort) / effort

    def forward(self, input_data):
        self.result = self.dropout(input_data, self.mask)

    def dropout(self, input_data, mask):
        return np.multiply(input_data, mask)

    def backward(self, delta, input_data):
        return self.dropout(delta, self.mask)


class FlattenLayer(Layer):
    def __init__(self):
        super(Layer, self).__init__()

    def forward(self, input_data):
        self.result = self.flatten(input_data)

    @staticmethod
    def flatten(input_data):
        return input_data.reshape(1, (reduce(lambda a, b: a * b, input_data.shape)))

    @staticmethod
    def backward(delta, input_data):
        delta = delta.reshape(input_data.shape)
        return delta


class FullConnectedLayer(Layer):
    def __init__(self, number_inputs, number_outputs):
        super(Layer, self).__init__()
        self.number_outputs = number_outputs
        self.number_inputs = number_inputs

        self.weights = rnd.randn(number_outputs, number_inputs) * np.sqrt(2.0 / number_outputs)
        self.biases = np.ones((number_outputs, 1)) * 0.01
        self.f = None

    def forward(self, input_data):
        self.f = np.dot(input_data.T, self.weights) + self.biases
        self.result = relu(self.f)

    def backward(self, error, input_data):
        gradient = error * relu_prime(self.f)
        delta_biases = gradient
        delta_weights = np.dot(gradient, input_data)
        gradient = np.dot(self.weights.T, gradient)
        return gradient, delta_weights, delta_biases
