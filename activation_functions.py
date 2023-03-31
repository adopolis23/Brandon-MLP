import numpy as np

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    if x >= 0:
        return 0
    else:
        return x