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


def cross_entropy(result, label, eps=10 ** -9):
    prediction = softmax(result)
    return -np.log(prediction[0, label.argmax()] + eps)


def cross_entropy_prime(result, label):
    prediction = softmax(result)
    prediction[0, label.argmax()] -= 1
    return prediction
