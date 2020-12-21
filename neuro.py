from math import exp
from random import seed
from random import random


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = 0
    # activation = weights[-1] # pre bias ak by bol posledny
    for i in range(len(weights)):
        activation += weights[i] * inputs[i]
    return activation


# Forward propagate input to a network output
def forward_propagation(network, row):
    inputs = row
    new_inputs = []
    for layer in network:
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = 1.0 / (1.0 + exp(-activation))
            new_inputs.append(neuron['output'])

        inputs = new_inputs
    return inputs


def main():
    # seed(1)

    # test forward propagation
    # network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
    #            [{'weights': [0.2550690257394217, 0.49543508709194095]},
    #             {'weights': [0.4494910647887381, 0.651592972722763]}]]
    network = init_network(2, 5, 2)

    # row = [1, 0, None]
    # output = forward_propagation(network)
    # print(output)

    for layer in network:
        print(layer)
        for neuron in layer:
            print(neuron)
            for weights in neuron['weights']:
                print(weights)
    # print(normalize_inputs([0, 250, 400]))


# Initialize a network
def init_network(n_inputs, n_hidden, n_outputs):
    network = list()
    # pre bias treba dat do vnutorneho foru +1 vahu
    hidden_layer = [{'weights': [random() for i in range(n_inputs)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def normalize_inputs(inputs):
    min = 0
    max = 400
    new_inputs = []
    for input in inputs:
        new_input = (input - min) / (max - min)
        new_inputs.append(new_input)
    return new_inputs


if __name__ == '__main__':
    main()
