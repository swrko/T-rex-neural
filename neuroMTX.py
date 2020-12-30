from math import exp
from random import random, seed
import numpy as np


# Calculate neuron activation for an input
def activate(weights, inputs):
    output = list()
    activation = inputs.dot(weights)
    print(activation)
    for iterator in activation:
        output.append(1.0 / (1.0 + exp(-iterator)))
    print(output)
    return np.array(output)


def forwardPropagation(network, inputs):
    inputs = normalizeInputs(inputs)
    # print(np.array(network[0]))
    # print("\n")
    # print(inputs)
    # print("\n")
    # inputs = activate(network[0], inputs)
    # # print("\n")
    # # print(np.array(network[1]))
    # # print("\n")
    # outputs = activate(network[1], inputs)

    for layer in network:
        outputs = activate(layer, inputs)
        inputs = outputs

    return outputs


def main():
    inputs = np.array([100, 120])  # testovaci input
    network = initialiseNetwork(2, 5, 2)
    forwardPropagation(network, inputs)


def normalizeInputs(inputs):
    min = 0
    max = 400
    new_inputs = []
    for input in inputs:
        new_input = (input - min) / (max - min)
        new_inputs.append(new_input)
    return np.array(new_inputs)


# Initialize a network
def initialiseNetwork(n_inputs, n_hidden, n_outputs):
    seed(1)
    w_hdl1 = np.array([[random() for i in range(n_hidden)] for i in range(n_inputs)])
    w_ol = np.array([[random() for i in range(n_outputs)] for i in range(n_hidden)])
    # a_hdl1 = np.zeros((n_hidden, 1))
    # a_ol = np.zeros((n_outputs, 1))
    # w_ = (w_hdl1, w_ol)

    # print(w_)

    network = (w_hdl1, w_ol)
    return network
    # # matrix = np.ndarray()
    # print(weights[1])
    # print(weights[2])
    #
    # # a.dot(b)   ->    a*b'
    # print(weights[1].dot(weights[2]))


if __name__ == '__main__':
    main()

# https://medium.com/@hindsellouk13/matrix-based-back-propagation-fe143ce2b2df
