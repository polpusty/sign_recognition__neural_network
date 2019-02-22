import numpy as np


def relu(x):
    return np.where(x > 0, x, 0)


def relu_prime(x):
    return np.where(x > 0, 1.0, 0)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / exps.sum(axis=1)


def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))


def loss(label, result):
    return 0.5 * np.sum((result - label) ** 2)
