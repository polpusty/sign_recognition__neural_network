<<<<<<< HEAD
from multiprocessing import Pool

from numpy import zeros_like, zeros, random
from tornado import gen

from nn.functions import cross_entropy, cross_entropy_prime, softmax
from nn.preprocessing import transform_image_to_array, normalize_image


class Network:
    def __init__(self, layers_list, classes_list, size_input_image, optimizer, operation_data, api_url):
        self.layers = layers_list
        self.classes_list = classes_list
        self.size_input_image = size_input_image
        self.optimizer = optimizer
        self.operation_data = operation_data

    def forward(self, input_data):
        """
        :type input_data: numpy.ndarray
        :rtype: numpy.ndarray
        """
        result = input_data
        for layer in self.layers:
            layer.forward(result)
            result = layer.result
        return result

    def backward(self, image, label):
        result = self.forward(image)
        error = cross_entropy_prime(result, label)
        delta_biases, delta_weights = self.backward_propagation(image, error)
        return delta_biases, delta_weights, result, label

    def backward_propagation(self, input_data, error):
        correction_weights = dict((l, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        correction_biases = dict((l, zeros_like(l.biases)) for l in self.layers if l.has_weights())

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
        result = zeros((len(self.classes_list),))
        result[number_class] = 1.0
        return result

    def prepare_data_for_training(self, data):
        result = []
        for (image, class_code) in data:
            output = self.get_perfect_output_for_class(self.classes_list.index(class_code))
            result.append((self.prepare_image(image), output))
        return result

    def prepare_image(self, image):
        return normalize_image(transform_image_to_array(self.size_input_image, image).swapaxes(2, 0))

    def fit(self, data, number_epochs, batch_len, client=None):
        training_data = self.prepare_data_for_training(data)

        for epoch in range(number_epochs):
            random.shuffle(training_data)
            batches = [training_data[i:i + batch_len] for i in range(0, len(training_data), batch_len)]
            error = 0
            for index, batch in enumerate(batches):
                error += self.train_batch(batch)
            print(f"ERROR: {error / len(batches)}")

    def predict(self, image):
        image_array = self.prepare_image(image)
        return softmax(self.forward(image_array))

    def get_backwards(self, batch):
        return [self.backward(image, label) for image, label in batch]

    def train_batch(self, batch):
        gradients_weights = dict((l, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        gradients_biases = dict((l, zeros_like(l.biases)) for l in self.layers if l.has_weights())
        length_batch = len(batch)
        label, result = None, None
        backwards = self.get_backwards(batch)

        for delta_biases, delta_weights, result, label in backwards:
            gradients_weights = dict((l, gradients_weights[l] + delta_weights[l])
                                     for l in self.layers if l.has_weights())
            gradients_biases = dict((l, gradients_biases[l] + delta_biases[l])
                                    for l in self.layers if l.has_weights())

        self.optimizer.optimize(self.layers, gradients_weights, gradients_biases, length_batch)

        return cross_entropy(result, label)


class AsyncNetwork(Network):
    async def backward(self, image, label):
        return super(AsyncNetwork, self).backward(image, label)

    async def get_backwards(self, batch):
        return await gen.multi([self.backward(image, label) for image, label in batch])

    async def update_operations(self, client, training_progress, status="pending"):
        self.operation_data = await client.update_operation(self.operation_data, training_progress, status)

    async def fit(self, data, number_epochs, batch_len, client=None):
        training_data = self.prepare_data_for_training(data)
        training_progress = []
        for epoch in range(number_epochs):
            random.shuffle(training_data)
            batches = [training_data[i:i + batch_len] for i in range(0, len(training_data), batch_len)]
            error = 0
            for index, batch in enumerate(batches):
                error += await self.train_batch(batch)
                if index % 10 == 0:
                    training_progress.append({"name": f'Epoch {epoch}', "predict": error / (index + 1)})
                    await self.update_operations(client, training_progress)
                    print(f"Epoch: {epoch} Batch {index}/{len(batches)} Error {error / index}")
            print(f"ERROR: {error / len(batches)}")
        await self.update_operations(client, training_progress, status="finished")

    async def train_batch(self, batch):
        gradients_weights = dict((l, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        gradients_biases = dict((l, zeros_like(l.biases)) for l in self.layers if l.has_weights())
        length_batch = len(batch)
        label, result = None, None
        backwards = await self.get_backwards(batch)

        for delta_biases, delta_weights, result, label in backwards:
            gradients_weights = dict((l, gradients_weights[l] + delta_weights[l])
                                     for l in self.layers if l.has_weights())
            gradients_biases = dict((l, gradients_biases[l] + delta_biases[l])
                                    for l in self.layers if l.has_weights())

        self.optimizer.optimize(self.layers, gradients_weights, gradients_biases, length_batch)

        return cross_entropy(result, label)


class MultiProcessingNetwork(Network):
    def __init__(self, *args, **kwargs):
        super(MultiProcessingNetwork, self).__init__(*args, **kwargs)
        self.init_layers_name()
        self.layers_dict = dict((layer.name, layer) for layer in self.layers)

    def init_layers_name(self):
        for index, layer in enumerate(self.layers):
            layer.name = f'{layer} NR {index}'

    def item_backward(self, batch_item):
        image, label = batch_item
        return self.backward(image, label)

    def get_backwards(self, batch):
        with Pool(6) as pool:
            return list(pool.imap_unordered(self.item_backward, batch))

    def backward_propagation(self, input_data, error):
        correction_weights = dict((l.name, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        correction_biases = dict((l.name, zeros_like(l.biases)) for l in self.layers if l.has_weights())

        for index, layer in reversed(list(enumerate(self.layers))):
            input_layer = self.layers[index - 1].result if index != 0 else input_data
            if not layer.has_weights():
                error = layer.backward(error, input_layer)
            else:
                error, delta_biases, delta_weights = layer.backward(error, input_layer)
                correction_biases[layer.name] = delta_biases
                correction_weights[layer.name] = delta_weights

        return correction_biases, correction_weights

    def train_batch(self, batch):
        gradients_weights = dict((l, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        gradients_biases = dict((l, zeros_like(l.biases)) for l in self.layers if l.has_weights())
        length_batch = len(batch)
        label, result = None, None
        backwards = self.get_backwards(batch)

        for delta_biases, delta_weights, result, label in backwards:
            gradients_weights = dict((l, gradients_weights[l] + delta_weights[l.name])
                                     for l in self.layers if l.has_weights())
            gradients_biases = dict((l, gradients_biases[l] + delta_biases[l.name])
                                    for l in self.layers if l.has_weights())

        self.optimizer.optimize(self.layers, gradients_weights, gradients_biases, length_batch)

        return cross_entropy(label, result)
=======
from multiprocessing import Pool

from numpy import zeros_like, zeros, random
from tornado import gen

from api import ApiClient
from nn.functions import cross_entropy, cross_entropy_prime, softmax
from nn.preprocessing import transform_image_to_array, normalize_image


class Network:
    def __init__(self, layers_list, classes_list, size_input_image, optimizer, operation_data, api_url):
        self.layers = layers_list
        self.classes_list = classes_list
        self.size_input_image = size_input_image
        self.optimizer = optimizer
        self.operation_data = operation_data

    def forward(self, input_data):
        """
        :type input_data: numpy.ndarray
        :rtype: numpy.ndarray
        """
        result = input_data
        for layer in self.layers:
            layer.forward(result)
            result = layer.result
        return result

    def backward(self, image, label):
        result = self.forward(image)
        error = cross_entropy_prime(result, label)
        delta_biases, delta_weights = self.backward_propagation(image, error)
        return delta_biases, delta_weights, result, label

    def backward_propagation(self, input_data, error):
        correction_weights = dict((l, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        correction_biases = dict((l, zeros_like(l.biases)) for l in self.layers if l.has_weights())

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
        result = zeros((len(self.classes_list),))
        result[number_class] = 1.0
        return result

    def prepare_data_for_training(self, data):
        result = []
        for (image, class_code) in data:
            output = self.get_perfect_output_for_class(self.classes_list.index(class_code))
            result.append((self.prepare_image(image), output))
        return result

    def prepare_image(self, image):
        return normalize_image(transform_image_to_array(self.size_input_image, image).swapaxes(2, 0))

    def fit(self, data, number_epochs, batch_len, client=None):
        training_data = self.prepare_data_for_training(data)

        for epoch in range(number_epochs):
            random.shuffle(training_data)
            batches = [training_data[i:i + batch_len] for i in range(0, len(training_data), batch_len)]
            error = 0
            for index, batch in enumerate(batches):
                error += self.train_batch(batch)
            print(f"ERROR: {error / len(batches)}")

    def predict(self, image):
        image_array = self.prepare_image(image)
        return softmax(self.forward(image_array))

    def get_backwards(self, batch):
        return [self.backward(image, label) for image, label in batch]

    def train_batch(self, batch):
        gradients_weights = dict((l, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        gradients_biases = dict((l, zeros_like(l.biases)) for l in self.layers if l.has_weights())
        length_batch = len(batch)
        label, result = None, None
        backwards = self.get_backwards(batch)

        for delta_biases, delta_weights, result, label in backwards:
            gradients_weights = dict((l, gradients_weights[l] + delta_weights[l])
                                     for l in self.layers if l.has_weights())
            gradients_biases = dict((l, gradients_biases[l] + delta_biases[l])
                                    for l in self.layers if l.has_weights())

        self.optimizer.optimize(self.layers, gradients_weights, gradients_biases, length_batch)

        return cross_entropy(result, label)


class AsyncNetwork(Network):
    async def backward(self, image, label):
        return super(AsyncNetwork, self).backward(image, label)

    async def get_backwards(self, batch):
        return await gen.multi([self.backward(image, label) for image, label in batch])

    async def update_operations(self, client, training_progress, status="pending"):
        self.operation_data = await client.update_operation(self.operation_data, training_progress, status)

    async def fit(self, data, number_epochs, batch_len, client=None):
        training_data = self.prepare_data_for_training(data)
        training_progress = []
        for epoch in range(number_epochs):
            random.shuffle(training_data)
            batches = [training_data[i:i + batch_len] for i in range(0, len(training_data), batch_len)]
            error = 0
            for index, batch in enumerate(batches):
                error += await self.train_batch(batch)
                if index % 10 == 0:
                    training_progress.append({"name": f'Epoch {epoch}', "predict": error / (index + 1)})
                    await self.update_operations(client, training_progress)
                    print(f"Epoch: {epoch} Batch {index}/{len(batches)} Error {error / index}")
            print(f"ERROR: {error / len(batches)}")
        await self.update_operations(client, training_progress, status="finished")

    async def train_batch(self, batch):
        gradients_weights = dict((l, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        gradients_biases = dict((l, zeros_like(l.biases)) for l in self.layers if l.has_weights())
        length_batch = len(batch)
        label, result = None, None
        backwards = await self.get_backwards(batch)

        for delta_biases, delta_weights, result, label in backwards:
            gradients_weights = dict((l, gradients_weights[l] + delta_weights[l])
                                     for l in self.layers if l.has_weights())
            gradients_biases = dict((l, gradients_biases[l] + delta_biases[l])
                                    for l in self.layers if l.has_weights())

        self.optimizer.optimize(self.layers, gradients_weights, gradients_biases, length_batch)

        return cross_entropy(result, label)


class MultiProcessingNetwork(Network):
    def __init__(self, *args, **kwargs):
        super(MultiProcessingNetwork, self).__init__(*args, **kwargs)
        self.init_layers_name()
        self.layers_dict = dict((layer.name, layer) for layer in self.layers)

    def init_layers_name(self):
        for index, layer in enumerate(self.layers):
            layer.name = f'{layer} NR {index}'

    def item_backward(self, batch_item):
        image, label = batch_item
        return self.backward(image, label)

    def get_backwards(self, batch):
        with Pool(6) as pool:
            return list(pool.imap_unordered(self.item_backward, batch))

    def backward_propagation(self, input_data, error):
        correction_weights = dict((l.name, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        correction_biases = dict((l.name, zeros_like(l.biases)) for l in self.layers if l.has_weights())

        for index, layer in reversed(list(enumerate(self.layers))):
            input_layer = self.layers[index - 1].result if index != 0 else input_data
            if not layer.has_weights():
                error = layer.backward(error, input_layer)
            else:
                error, delta_biases, delta_weights = layer.backward(error, input_layer)
                correction_biases[layer.name] = delta_biases
                correction_weights[layer.name] = delta_weights

        return correction_biases, correction_weights

    def train_batch(self, batch):
        gradients_weights = dict((l, zeros_like(l.weights)) for l in self.layers if l.has_weights())
        gradients_biases = dict((l, zeros_like(l.biases)) for l in self.layers if l.has_weights())
        length_batch = len(batch)
        label, result = None, None
        backwards = self.get_backwards(batch)

        for delta_biases, delta_weights, result, label in backwards:
            gradients_weights = dict((l, gradients_weights[l] + delta_weights[l.name])
                                     for l in self.layers if l.has_weights())
            gradients_biases = dict((l, gradients_biases[l] + delta_biases[l.name])
                                    for l in self.layers if l.has_weights())

        self.optimizer.optimize(self.layers, gradients_weights, gradients_biases, length_batch)

        return cross_entropy(label, result)
>>>>>>> be1166b6555c110b5013a51645a58c3c880c2925
