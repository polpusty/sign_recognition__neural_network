import random

from numpy import zeros_like, zeros
from tornado import gen

from nn.math_functions import loss
from nn.preprocessing import transform_image_to_array


class Network:
    """
    :type layers_list: list[nn.layers.*]
    :type classes_list: list[dict]
    :type size_input_image: (int, int)
    :type optimizer: nn.optimizers.Optimizer
    """

    def __init__(self, layers_list, classes_list, size_input_image, optimizer):
        self.layers = layers_list
        self.classes_list = classes_list
        self.size_input_image = size_input_image
        self.optimizer = optimizer

    def forward(self, input_data):
        result = input_data
        for layer in self.layers:
            layer.forward(result)
            result = layer.result
        return result

    async def backward(self, image, label):
        result = self.forward(image)
        delta_biases, delta_weights = self.backward_propagation(image, label, result)
        return delta_biases, delta_weights, result, label

    def backward_propagation(self, input_data, label, result):
        correction_weights = dict((l, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        correction_biases = dict((l, zeros_like(l.biases)) for l in self.layers if l.has_weights())

        error = (result - label)

        for index, layer in reversed(list(enumerate(self.layers))):
            input_layer = self.layers[index - 1].result if index != 0 else input_data
            if not layer.has_weights():
                error = layer.backward(error, input_layer)
            else:
                error, delta_biases, delta_weights = layer.backward(error, input_layer)
                correction_biases[layer] = delta_biases
                correction_weights[layer] = delta_weights

        return correction_biases, correction_weights

    def get_perfect_output_for_class(self, number_class):
        result = zeros(len(self.classes_list))
        result[number_class] = 1.0
        return result

    def prepare_data_for_training(self, data):
        result = []
        for (image, class_code) in data:
            output = self.get_perfect_output_for_class(self.classes_list.index(class_code))
            image_array = transform_image_to_array(self.size_input_image, image)
            result.append((image_array, output))
        return result

    async def train(self, data, number_epochs, batch_len):
        training_data = self.prepare_data_for_training(data)
        for epoch in range(number_epochs):
            random.shuffle(training_data)
            batches = [training_data[i:i + batch_len] for i in range(0, len(training_data), batch_len)]
            error = 0
            for index, batch in enumerate(batches):
                error += await self.train_batch(batch)
                print(f"Epoch: {epoch} Batch {index}/{len(batches)}")
            print(f"ERROR: {error / len(batches)}")

    async def train_batch(self, batch):
        gradients_weights = dict((l, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        gradients_biases = dict((l, zeros_like(l.biases)) for l in self.layers if l.has_weights())
        length_batch = len(batch)
        label, result = None, None
        backward_futures = [self.backward(image, label) for image, label in batch]

        for delta_biases, delta_weights, result, label in await gen.multi(backward_futures):
            gradients_weights = dict((l, gradients_weights[l] + delta_weights[l])
                                     for l in self.layers if l.has_weights())
            gradients_biases = dict((l, gradients_biases[l] + delta_biases[l])
                                    for l in self.layers if l.has_weights())

        self.optimizer.optimize(self.layers, gradients_weights, gradients_biases, length_batch)

        return loss(label, result)
