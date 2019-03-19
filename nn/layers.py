import abc

import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce

rnd = np.random


class BaseLayer(abc.ABC):
    def __init__(self):
        self.result = None
        self.name = None

    def has_weights(self):
        return hasattr(self, 'weights')

    @abc.abstractmethod
    def forward(self, input_data):
        """
        :type input_data: numpy.ndarray
        """
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, delta, input_data):
        """
        :type delta: numpy.ndarray
        :type input_data: numpy.ndarray
        :rtype: numpy.ndarray
        """
        raise NotImplementedError


class Convolution2d(BaseLayer):
    def __init__(self, number_kernels, kernel_size, depth, padding=0):
        super(BaseLayer, self).__init__()
        self.number_kernels = number_kernels
        self.kernel_size = kernel_size
        self.depth = depth
        self.padding = padding

        k = np.sqrt(1. / (depth * kernel_size * kernel_size))
        self.weights = rnd.uniform(-k, k, (number_kernels, depth, kernel_size, kernel_size))
        self.biases = rnd.uniform(-k, k, (number_kernels, 1))

    def __str__(self):
        return f'{self.__class__.__name__}Convolution {self.weights.shape}'

    def __call__(self, input_data):
        self.forward(input_data)
        return self.result

    def forward(self, input_data):
        self.check_input_data(input_data)
        self.result = self.convolution(input_data, self.weights, self.biases, self.padding)

    @staticmethod
    def convolution(input_data, weights, biases, padding, mode='valid'):
        depth, height, width = input_data.shape
        number_kernels, depth_kernel, kernel_size, kernel_size = weights.shape

        result = np.zeros((number_kernels,
                           height - kernel_size + 2 * padding + 1,
                           width - kernel_size + 2 * padding + 1))
        input_with_padding = np.zeros((depth, height + 2 * padding, width + 2 * padding))
        input_with_padding[:, padding:height + padding, padding:width + padding] = input_data

        for number_kernel, kernel in enumerate(weights):
            for d in range(depth):
                result[number_kernel] += convolve2d(input_with_padding[d], kernel[d][::-1, ::-1], mode)
            result[number_kernel] += biases[number_kernel]

        return result

    def backward(self, delta, input_data):
        return self.backward_convolution_layer(delta, input_data)

    def backward_convolution_layer(self, delta, input_data):
        delta_biases = np.zeros(self.biases.shape)
        delta_weights = np.zeros(self.weights.shape)
        depth, height, width = input_data.shape
        input_with_padding = np.zeros((depth, height + 2 * self.padding, width + 2 * self.padding))
        input_with_padding[:, self.padding:height + self.padding, self.padding:width + self.padding] = input_data
        delta_result = np.zeros(input_with_padding.shape)

        for number_kernel in range(len(self.weights)):
            for d in range(self.depth):
                delta_weights[number_kernel][d] = convolve2d(input_with_padding[d], delta[number_kernel][::-1, ::-1],
                                                             'valid')
                # delta_result[d] += convolve2d(self.weights[number_kernel][d][::-1, ::-1], delta[number_kernel], 'full')

            delta_biases[number_kernel] = np.sum(delta[number_kernel])

        delta_result = self.convolutionBackward(delta, input_with_padding, self.weights, 1)

        return delta_result, delta_biases, delta_weights

    def convolutionBackward(self, dconv_prev, conv_in, filt, s):
        '''
        Backpropagation through a convolutional layer.
        '''
        (n_f, n_c, f, _) = filt.shape
        (_, orig_dim, _) = conv_in.shape
        ## initialize derivatives
        dout = np.zeros(conv_in.shape)
        for curr_f in range(n_f):
            # loop through all filters
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    # loss gradient of filter (used to update the filter)
                    # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                    dout[:, curr_y:curr_y + f, curr_x:curr_x + f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
            # loss gradient of the bias

        return dout

    def check_input_data(self, input_data):
        depth, width, height = input_data.shape
        assert depth == self.depth
        assert width > self.kernel_size
        assert height > self.kernel_size


class MaxPooling2d(BaseLayer):
    def __init__(self, stride):
        super(BaseLayer, self).__init__()
        self.stride = stride

    def __str__(self):
        return f'{self.__class__.__name__} {self.stride}x{self.stride}'

    def __call__(self, input_data):
        self.forward(input_data)
        return self.result

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


class Dropout(BaseLayer):
    def __init__(self, effort, shape):
        super(BaseLayer, self).__init__()
        self.effort = effort
        self.shape = shape

        self.mask = (np.random.rand(*shape) < effort) / effort

    def __str__(self):
        return f'{self.__class__.__name__}Dropout {self.effort} {self.shape}'

    def __call__(self, input_data):
        self.forward(input_data)
        return self.result

    def forward(self, input_data):
        self.result = self.dropout(input_data, self.mask)

    @staticmethod
    def dropout(input_data, mask):
        return np.multiply(input_data, mask)

    def backward(self, delta, _):
        return self.dropout(delta, self.mask)


class Flatten(BaseLayer):
    def __str__(self):
        return f'{self.__class__.__name__}Flatten'

    def __call__(self, input_data):
        self.forward(input_data)
        return self.result

    def forward(self, input_data):
        self.result = self.flatten(input_data)

    @staticmethod
    def flatten(input_data):
        return input_data.reshape(1, np.prod(input_data.shape))

    def backward(self, delta, input_data):
        delta = delta.reshape(input_data.shape)
        return delta


class FullConnected(BaseLayer):
    def __init__(self, number_outputs, number_inputs):
        super(BaseLayer, self).__init__()
        k = np.sqrt(1. / number_inputs)
        self.weights = rnd.uniform(-k, k, (number_outputs, number_inputs))
        self.biases = rnd.uniform(-k, k, (number_outputs,))

    def __str__(self):
        return f'{self.__class__.__name__} {self.weights.shape}'

    def __call__(self, input_data):
        self.forward(input_data)
        return self.result

    def forward(self, input_data):
        self.result = input_data @ self.weights.T + self.biases

    def backward(self, delta, input_data):
        delta_biases = delta.reshape(self.biases.shape)
        delta_weights = delta.T @ input_data
        delta_result = delta @ self.weights
        return delta_result, delta_biases, delta_weights


class Activation(BaseLayer):
    def __init__(self, activation, activation_prime):
        super(Activation, self).__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __call__(self, input_data):
        self.forward(input_data)
        return self.result

    def forward(self, input_data):
        self.result = self.activation(input_data)

    def backward(self, delta, input_data):
        return delta * self.activation_prime(input_data)
