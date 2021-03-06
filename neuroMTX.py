from math import exp
from random import random, seed
import numpy as np
import re


class NNetwork(object):
    # Initialize a network
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        # seed for same generation - due develop and testing
        seed(1)
        # weights hidden layer
        self.w_1 = np.array([[random() for i in range(n_hidden)] for i in range(n_inputs)])
        # weights of output layer
        self.w_2 = np.array([[random() for i in range(n_outputs)] for i in range(n_hidden)])

        # default learning rate
        self.set_learning_rate(0.2)

    def forward_propagation(self):
        # name conventions:
        # w - weights between layers
        # z - weighted input of activation function
        # a - activation value of the neuron / output of the activation function
        # var_L - L stands for number of Layer
        # self.set_inputs(inputs)

        # feed_forward through 1st layer
        self.z_2 = np.dot(self.inputs, self.w_1)
        self.a_2 = self.sigmoid(self.z_2)
        # feed_forward through 2nd layer
        self.z_3 = np.dot(self.a_2, self.w_2)
        self.outputs = self.sigmoid(self.z_3)
        return self.outputs

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def d_sigmoid(self, z):
        # print("Z: {}".format(z))
        return np.exp(-z) / ((1.0 + np.exp(-z)) ** 2.0)

    # s stands for output of sigmoid function
    # def derSigmoid(s):
    #     return s * (1.0 - s)

    def normalize_inputs(self, inputs, max):
        # min = 0
        # max = 400
        return [i / max for i in inputs[0]]
        # return inputs / max
        # new_inputs = []
        # for input in inputs:
        #     new_input = (input - min) / (max - min)
        #     new_inputs.append(new_input)
        # return np.array(new_inputs)

    def cost_function(self):
        return sum(((self.desired_outputs - self.outputs) ** 2.0) / 2.0)

    def d_cost_function(self):
        # name convetions:
        # delta_3 - backprop error from output to hidden layer
        # dw_2 - derivative of cost function with respect to w_2 - aplication of bperor to weights
        # delta_2 - backprop error from hidden layer to input layer
        # dw_1 - derivative of cost function with respect to w_1 - aplication of bperor to weights

        delta_3 = np.multiply(-(self.desired_outputs - self.outputs), self.d_sigmoid(self.z_3))

        dw_2 = np.dot(np.reshape(self.a_2, (-1, 1)), np.reshape(delta_3, (1, -1)))

        delta_2 = np.dot(delta_3, self.w_2.T) * self.d_sigmoid(self.z_2)
        dw_1 = np.dot(np.reshape(self.inputs, (-1, 1)), np.reshape(delta_2, (1, -1)))

        return dw_1, dw_2

    def set_inputs(self, inputs):
        self.inputs = inputs

    def get_inputs(self):
        return self.inputs

    def set_desired_outputs(self, desired_outputs):
        self.desired_outputs = desired_outputs

    def get_desired_outputs(self):
        return self.desired_outputs

    def get_outputs(self):
        return self.outputs

    def set_learning_rate(self, lr):
        self.lr = lr

    def get_learning_rate(self):
        return self.lr

    def set_weights1(self, w1):
        self.w_1 = w1

    def get_weights1(self):
        return self.w_1

    def set_weights2(self, w2):
        self.w_2 = w2

    def get_weights2(self):
        return self.w_2

    def flaten_weights(self):
        # print("Flatened weights: {}".format(np.concatenate([self.get_weights1().ravel(),self.get_weights2().ravel()])))
        return np.concatenate([self.get_weights1().ravel(), self.get_weights2().ravel()])

    def parse_weights(self, w):
        # unused metod
        w1 = w[0:self.n_inputs * self.n_hidden]
        w2 = w[self.n_inputs * self.n_hidden:(len(w))]

        w1 = w1.reshape((self.n_inputs, self.n_hidden))
        w2 = w2.reshape((self.n_hidden, self.n_outputs))
        return w1, w2
        # print("w1: {}".format(w1))
        # print("w2: {}".format(w2))

    def back_propagation(self, desired_outputs):
        # set desired output
        self.set_desired_outputs(desired_outputs)

        # FFD
        self.forward_propagation()
        # error = desired_outputs - output
        error = self.cost_function()
        # calculate error
        dw_1, dw_2 = self.d_cost_function()

        # apply error to weights
        self.w_1 = self.w_1 - self.lr * dw_1
        self.w_2 = self.w_2 - self.lr * dw_2
        return error

    def train_gradient_descent(self, inputs, desired_outputs):
        # for minibatch -> input has to be matrice, backprop error wil be summed for each input by matrix multiplication
        # normalize input and set it
        # print("inputs: {} \n outputs: {}".format(inputs,desired_outputs))
        self.set_inputs(self.normalize_inputs(inputs, 0, 400))
        # backprop - set_desired_output, FFD, gradient error, apply error
        error = self.back_propagation(desired_outputs)
        m_error = 0
        # TODO: add calling an error function for futher comparison, method will need argument with err function
        # MSE
        for i in error:
            m_error += i
        return abs(m_error / len(error))

    def feed_forward_propagation(self, inputs):
        self.set_inputs(self.normalize_inputs(inputs, 0, 400))
        self.forward_propagation()
        return self.get_outputs()

    def write_weights_to_file(self, name):
        with open(name + ".txt", "w") as file:
            file.write(str(self.get_weights1()) + ";\n")
            file.write(str(self.get_weights2()) + ";\n")
        print("Weights was written succesfully!")

    def read_weights_from_file(self, name):
        w1 = []
        w2 = []
        with open(name + ".txt", "r") as file:
            r = file.read()
            r = r.split(";")

            r[0] = r[0].replace("[", "")
            r[0] = r[0].replace("]", "")
            r[0] = r[0].splitlines()

            r[1] = r[1].replace("[", "")
            r[1] = r[1].replace("]", "")
            r[1] = r[1].splitlines()

            for line in r[0]:
                regex = re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+|[-+]?\d*\.\d+|\d+", line)
                for number in regex:
                    w1.append(float(number))
            for line in r[1]:
                regex = re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+|[-+]?\d*\.\d+|\d+", line)
                for number in regex:
                    w2.append(float(number))

        self.set_weights1(np.array(w1).reshape((self.n_inputs, self.n_hidden)))
        self.set_weights2(np.array(w2).reshape((self.n_hidden, self.n_outputs)))
        print("Weights was read succesfully!")


def get_params(N):
    params = np.concatenate((N.get_weights1().ravel(), N.get_weights2().ravel()))
    return params


def set_params(N, params):
    w1_start = 0
    w1_end = N.n_hidden * N.n_inputs
    w1 = np.reshape(params[w1_start:w1_end],
                    (N.n_inputs, N.n_hidden))
    N.set_weights1(w1)
    w2_end = w1_end + (N.n_hidden * N.n_outputs)
    w2 = np.reshape(params[w1_end:w2_end],
                    (N.n_hidden, N.n_outputs))
    N.set_weights2(w2)


def compute_gradients(N):
    N.forward_propagation()
    dw1, dw2 = N.d_cost_function()
    return np.concatenate((dw1.ravel(), dw2.ravel()))


def compute_numerical_gradient(N):
    params_initial = get_params(N)
    numgrad = np.zeros(params_initial.shape)
    perturb = np.zeros(params_initial.shape)
    e = 1e-4

    for i in range(len(params_initial)):
        perturb[i] = e
        set_params(N, params_initial + perturb)
        N.forward_propagation()
        loss2 = N.cost_function()

        set_params(N, params_initial - perturb)
        N.forward_propagation()
        loss1 = N.cost_function()

        numgrad[i] = (loss2 - loss1) / (2 * e)

        perturb[i] = 0

    set_params(N, params_initial)
    return numgrad


def main():
    inputs = np.array([100, 120])  # testovaci input
    print("inputs: {}".format(inputs))
    Network = NNetwork(2, 5, 2)

    # print of generated weights
    print("w1: {}".format(Network.get_weights1()))
    print("w2: {}".format(Network.get_weights2()))

    Network.set_inputs(Network.normalize_inputs(inputs, 0, 400))
    print("inputs: {}".format(Network.get_inputs()))

    Network.forward_propagation()
    print("outputs: {}".format(Network.get_outputs()))
    print("inputs: {}".format(Network.get_inputs()))

    desired_outputs = np.array([1, 0])
    Network.back_propagation(desired_outputs)
    print("desired_output: {}".format(Network.get_desired_outputs()))
    print("inputs: {}".format(Network.get_inputs()))

    num_grad = compute_numerical_gradient(Network)
    print("inputs: {}".format(Network.get_inputs()))
    grad = compute_gradients(Network)
    print("inputs: {}".format(Network.get_inputs()))

    print("Numerical gradients: {}".format(num_grad))
    print("Gradients of network: {}".format(grad))

    print("norm: {}".format(np.linalg.norm(grad - num_grad) / np.linalg.norm(grad + num_grad)))

    print("inputs: {}".format(Network.get_inputs()))
    print("inputs: {}".format(Network.get_inputs()))


if __name__ == '__main__':
    main()

# https://medium.com/@hindsellouk13/matrix-based-back-propagation-fe143ce2b2df
