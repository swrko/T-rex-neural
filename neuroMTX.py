from math import exp
from random import random, seed
import numpy as np


class NNetwork(object):
    # Initialize a network
    def __init__(self, n_inputs, n_hidden, n_outputs):
        # seed for same generation - due develop and testing
        seed(1)
        # weights hidden layer
        self.w_2 = np.array([[random() for i in range(n_hidden)] for i in range(n_inputs)])
        # weights of output layer
        self.w_3 = np.array([[random() for i in range(n_outputs)] for i in range(n_hidden)])

        # print of generated weights
        print("w2: ")
        print(self.w_2)
        print("w3: ")
        print(self.w_3)

    def feed_forward_propagation(self, inputs):
        # name conventions:
        # w - weights between layers
        # z - weighted input of activation function
        # a - activation value of the neuron / output of the activation function
        # var_L - L stands for number of Layer
        self.x = self.normalize_inputs(inputs, 0, 400)
        # feed_forward through 1st layer
        self.z_2 = np.dot(self.x, self.w_2)
        self.a_2 = self.sigmoid(self.z_2)
        # feed_forward through 2nd layer
        self.z_3 = np.dot(self.a_2, self.w_3)
        self.outputs = self.sigmoid(self.z_3)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def d_sigmoid(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    # s stands for output of sigmoid function
    # def derSigmoid(s):
    #     return s * (1.0 - s)

    def normalize_inputs(self, inputs, min, max):
        # min = 0
        # max = 400
        new_inputs = []
        for input in inputs:
            new_input = (input - min) / (max - min)
            new_inputs.append(new_input)
        return np.array(new_inputs)

    def cost_function(self, outputs, desired_output):
        return sum(((outputs - desired_output) ** 2) / 2)
    def d_cost_function(self,outputs):
        # name convetions:
        pass

def back_propagation():
    pass


# def cost_function(network, inputs, y):
#     outputs = feed_forward_propagation(network, inputs)
#     d3 = np.multiply(-(y - outputs), d_sigmoid())
#     pass


def main():
    inputs = np.array([100, 120])  # testovaci input
    print("inputs: ")
    print(inputs)
    Network = NNetwork(2, 5, 2)
    Network.feed_forward_propagation(inputs)
    print("outputs: ")
    print(Network.outputs)


if __name__ == '__main__':
    main()

# https://medium.com/@hindsellouk13/matrix-based-back-propagation-fe143ce2b2df
