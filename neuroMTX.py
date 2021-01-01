from math import exp
from random import random, seed
import numpy as np


# Calculate neuron activation for an input
def activate(weights, inputs):
    output = list()
    activation = inputs.dot(weights)
    for iterator in activation:
        output.append(1.0 / (1.0 + exp(-iterator)))

    return np.array(output)


def feed_forward_propagation(network, inputs):
    inputs = normalize_inputs(inputs)

    for layer in network:
        outputs = activate(layer, inputs)
        inputs = outputs

    return outputs


def back_propagation():
    pass


def cost_function(network, inputs, y):
    outputs = feed_forward_propagation(network, inputs)
    d3 = np.multiply(-(y - outputs), d_sigmoid())
    pass


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# s stands for output of sigmoid function
# def derSigmoid(s):
#     return s * (1.0 - s)

def d_sigmoid(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


def main():
    inputs = np.array([100, 120])  # testovaci input
    network = initialise_network(2, 5, 2)
    feed_forward_propagation(network, inputs)


def normalize_inputs(inputs):
    min = 0
    max = 400
    new_inputs = []
    for input in inputs:
        new_input = (input - min) / (max - min)
        new_inputs.append(new_input)
    return np.array(new_inputs)


# Initialize a network
def initialise_network(n_inputs, n_hidden, n_outputs):
    seed(1)
    w_hdl1 = np.array([[random() for i in range(n_hidden)] for i in range(n_inputs)])
    w_ol = np.array([[random() for i in range(n_outputs)] for i in range(n_hidden)])
    print(w_hdl1)
    print("\n")

    print(w_ol)

    weights = (w_hdl1, w_ol)
    return weights


if __name__ == '__main__':
    main()

# https://medium.com/@hindsellouk13/matrix-based-back-propagation-fe143ce2b2df
