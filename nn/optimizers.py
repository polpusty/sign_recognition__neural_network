import abc

import numpy as np


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def optimize(self, layers, gradient_weights, gradient_biases, batch_length):
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    def __init__(self, eta=0.01):
        self.eta = eta

    def optimize(self, layers, gradient_weights, gradient_biases, batch_length):
        for layer in layers:
            if layer.has_weights():
                layer.biases += self.eta * gradient_biases[layer] / batch_length
                layer.weights += self.eta * gradient_weights[layer] / batch_length


class AdamOptimizer(Optimizer):
    def __init__(self, step, fade_1, fade_2, constant):
        self.step = step
        self.fade_1 = fade_1
        self.fade_2 = fade_2
        self.constant = constant

        self.momentum_1 = dict()
        self.momentum_2 = dict()
        self.iteration = 0

    def init_momentum(self, layers):
        for layer in layers:
            if layer.has_weights():
                self.momentum_1.update(
                    {layer: {"biases": np.zeros_like(layer.biases), "weights": np.zeros_like(layer.weights)}}
                )
                self.momentum_2.update(
                    {layer: {"biases": np.zeros_like(layer.biases), "weights": np.zeros_like(layer.weights)}}
                )

    def optimize(self, layers, gradient_weights, gradient_biases, batch_length):
        if self.iteration == 0:
            self.init_momentum(layers)

        self.iteration += 1

        p1, p2, m1, m2 = self.fade_1, self.fade_2, self.momentum_1, self.momentum_2

        for layer in layers:
            if layer.has_weights():
                grad = {"biases": gradient_biases[layer] / batch_length,
                        "weights": gradient_weights[layer] / batch_length}
                m1[layer]["biases"] = p1 * m1[layer]["biases"] + (1 - p1) * grad["biases"]
                m1[layer]["weights"] = p1 * m1[layer]["weights"] + (1 - p1) * grad["weights"]
                m2[layer]["biases"] = p2 * m2[layer]["biases"] + (1 - p2) * np.square(grad["biases"])
                m2[layer]["weights"] = p2 * m2[layer]["weights"] + (1 - p2) * np.square(grad["weights"])

                m1_cor = {'biases': m1[layer]["biases"] / (1 - p1 ** self.iteration),
                          'weights': m1[layer]["weights"] / (1 - p1 ** self.iteration)}
                m2_cor = {'biases': m2[layer]["biases"] / (1 - p2 ** self.iteration),
                          'weights': m2[layer]["weights"] / (1 - p2 ** self.iteration)}

                layer.weights -= self.step * m1_cor["weights"] / (np.sqrt(m2_cor["weights"]) + self.constant)
                layer.biases -= self.step * m1_cor["biases"] / (np.sqrt(m2_cor["biases"]) + self.constant)
