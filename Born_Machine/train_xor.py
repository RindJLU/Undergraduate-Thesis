
import numpy as np
import projectq
from projectq.ops import Rx, Rz, CNOT, Measure
import matplotlib.pyplot as plt
import copy

class Trainer(object):
    def __init__(self):
        self.theta = np.random.rand(2*3*3).reshape(3, 2, 3)
        self.data = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.label = [0, 1, 1, 0]
        self.loss = []

        self.eng = projectq.MainEngine()
        self.qureg = self.eng.allocate_qureg(2)
        self.eng.flush()

    def cal_loss(self, y_model):
        loss = 0
        for i in range(len(y_model)):
            loss -= self.label[i] * np.log(y_model[i]) + (1 - self.label[i]) * np.log(1 - y_model[i])
        return loss/4

    def run(self):
        for l_index in range(len(self.theta)):
            for q_index in range(len(self.theta[l_index])):
                Rz(self.theta[l_index, q_index, 0]) | self.qureg[q_index]
                Rx(self.theta[l_index, q_index, 1]) | self.qureg[q_index]
                Rz(self.theta[l_index, q_index, 2]) | self.qureg[q_index]
            if l_index != len(self.qureg):
                CNOT | (self.qureg[0], self.qureg[1])
                CNOT | (self.qureg[1], self.qureg[0])

        self.eng.flush()
        prob = self.eng.backend.get_probability('0', [self.qureg[1]])
        return prob

    def measure(self):
        Measure | self.qureg

    def train(self):
        y_model = []
        for d in self.data:
            (a, b) = d
            a_vec = a * np.array([0, 1]) + (1 - a) * np.array([1, 0])
            b_vec = b * np.array([0, 1]) + (1 - b) * np.array([1, 0])
            wave_fun = np.kron(a_vec, b_vec)
            self.eng.backend.set_wavefunction(wave_fun, self.qureg)
            y_model.append(self.run())
            self.measure()
        c = self.cal_loss(y_model)
        return c


if __name__ == '__main__':
    # cl = Trainer()
    # theta = cl.theta.reshape(18)
    # del_theta = 0.00001
    # for iter in range(200):
    #     for i in range(len(theta)):
    #         theta[i] += del_theta
    #         cl.theta = theta.reshape(3, 2, 3)
    #         loss_plus = cl.train()
    #
    #         theta[i] += -2 * del_theta
    #         cl.theta = theta.reshape(3, 2, 3)
    #         loss_min = cl.train()
    #
    #         theta[i] += del_theta - 0.1 * (loss_plus - loss_min) / (2 * del_theta)
    #     cl.loss.append((loss_plus + loss_min)/2)
    #     print(loss_plus)
    #
    # plt.plot(cl.loss)
    # plt.xlabel('iterations')
    # plt.ylabel('loss')
    # plt.yscale('log')
    # plt.show()
    # print(cl.theta)

    x = np.arange(-5, 5, 0.01)

    plt.subplot(2, 2, 1)
    y_linear = x
    plt.plot(x, y_linear, linewidth=0.8, c='black')
    h_1 = max(y_linear)*0.8
    h_2 = max(y_linear)*0.6
    plt.text(-5, h_1, '$linear: cost: f(x) = 1 × (x > 0) - 1 × (x < 0) $', fontsize=10)
    plt.text(-4.1, h_2, '$  grad: f^{,}(x) = 1 $', fontsize=10)

    plt.subplot(2, 2, 2)
    y_relu = copy.deepcopy(x)
    y_relu[0:len(x)//2] = 0
    h_1 = max(y_relu)*0.9
    h_2 = max(y_relu)*0.8
    plt.plot(x, y_relu, linewidth=0.8, c='black')
    plt.text(-5, h_1, '$relu: cost: f(x) =  max(0, x)$', fontsize=10)
    plt.text(-4.2, h_2, '$  grad: f^{,}(x) = 0 + 1 × (x > 0) $', fontsize=10)

    plt.subplot(2, 2, 3)
    y_sigmoid = 1/(1 + np.e**(-x))
    h_1 = max(y_sigmoid)*0.9
    h_2 = max(y_sigmoid)*0.8
    plt.plot(x, y_sigmoid, linewidth=0.8, c='black')
    plt.text(-5, h_1, '$sigmoid: cost: f(x) = \dfrac{1}{1 + e^{-x}}$', fontsize=10)
    plt.text(-3.7, h_2, '$  grad: f^{,}(x) = f(x)(1-f(x)) $', fontsize=10)

    plt.subplot(2, 2, 4)
    y_tanh  = 2/(1 + np.e**(-2*x)) - 1
    h_1 = max(y_tanh)*0.8
    h_2 = max(y_tanh)*0.6
    plt.plot(x, y_tanh, linewidth=0.8, c='black')
    plt.text(-5, h_1, '$sigmoid: cost: f(x) = \dfrac{2}{1 + e^{-x}} - 1$', fontsize=10)
    plt.text(-3.7, h_2, '$  grad: f^{,}(x) = 1 - f(x)^{2} $', fontsize=10)

    plt.show()
