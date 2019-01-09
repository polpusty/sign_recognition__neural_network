import random

from nn.math_functions import *


class Network:
    def __init__(self, layers_list):
        self.layers = layers_list

    def forward(self, input_data):
        result = input_data
        for layer in self.layers:
            layer.forward(result)
            result = layer.result
        return result

    def backward_propagation(self, input_data, label, result):
        correction_weights = dict((l, np.zeros_like(l.weights)) for l in self.layers if l.has_weights())
        correction_biases = dict((l, np.zeros_like(l.biases)) for l in self.layers if l.has_weights())

        error = (result - label)

        for index, layer in reversed(list(enumerate(self.layers))):
            input_layer = self.layers[index - 1].result if index != 0 else input_data
            if not hasattr(layer, 'weights'):
                error = layer.backward(error, input_layer)
            else:
                error, delta_biases, delta_weights = layer.backward(error, input_layer)
                correction_biases[layer] = delta_biases
                correction_weights[layer] = delta_weights

        return correction_biases, correction_weights

    def train(self, training_data, number_epochs, batch_len, eta):
        for epoch in range(number_epochs):
            random.shuffle(training_data)
            batches = [training_data[i:i + batch_len] for i in range(0, len(training_data), batch_len)]
            error = 0
            for batch in batches:
                error += self.train_batch(batch, eta)
            print((error / len(training_data) / batch_len, epoch / number_epochs))

    def train_batch(self, batch, eta):
        gradients_weights = dict((l, np.zeros_like(l.weights)) for l in self.layers if l.has_weights())
        gradients_biases = dict((l, np.zeros_like(l.biases)) for l in self.layers if l.has_weights())
        length_batch = len(batch)

        for image, label in batch:
            result = self.forward(image)
            delta_biases, delta_weights = self.backward_propagation(image, label, result)
            gradients_weights = dict((l, gradients_weights[l] + delta_weights[l])
                                     for l in self.layers if l.has_weights())
            gradients_biases = dict((l, gradients_biases[l] + delta_biases[l])
                                    for l in self.layers if l.has_weights())

        for layer in self.layers:
            if layer.has_weights():
                layer.biases += eta * gradients_biases[layer] / length_batch
                layer.weights += eta * gradients_weights[layer] / length_batch

        return loss(label, result)
