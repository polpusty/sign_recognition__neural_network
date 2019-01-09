import numpy as np


def relu(x):
    return np.where(x > 0, x, 0.01 * x)


def relu_prime(x):
    return np.where(x > 0, 1.0, 0.01)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def loss(label, result):
    return 0.5 * np.sum(result - label) ** 2
