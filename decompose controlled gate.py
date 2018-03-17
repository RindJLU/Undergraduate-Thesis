# coding: utf-8

import numpy as np
import projectq
from projectq.ops import H, Measure, Rz, Ry
import copy
import matplotlib.pyplot as plt


class Decompose(object):
    def __init__(self, v_input, v_output):
        self.v_in = v_input
        self.v_tgt = v_output
        self.theta = np.pi*(np.random.rand(3))
        self.eng = projectq.MainEngine()
        self.qureg = self.eng.allocate_qureg(1)
        self.eng.flush()

    def main(self):
        self.eng.backend.set_wavefunction(self.v_in/np.linalg.norm(self.v_in), self.qureg)
        Rz(self.theta[0]) | self.qureg[0]
        Ry(self.theta[1]) | self.qureg[0]
        Rz(self.theta[2]) | self.qureg[0]
        self.eng.flush()
        mapping, wavefunction = copy.deepcopy(self.eng.backend.cheat())
        # print("The full wavefunction is: {}".format(wavefunction))
        Measure | self.qureg
        v_out_test = np.array(wavefunction)
        # print(wavefunction)
        return np.linalg.norm(self.v_tgt/np.linalg.norm(self.v_tgt) - v_out_test)


v_in = [3, 4.0]
v_out = [5, 8.0]
a = Decompose(v_in, v_out)

delta_theta_test = 0.001
delta_theta_new = 0.05
loss = []

for i in range(10):
    a.theta = np.pi * (np.random.rand(3))
    for iters in range(200):
        a.theta += delta_theta_test
        loss_plus = a.main()
        a.theta += -2 * delta_theta_test
        loss_min = a.main()
        a.theta += delta_theta_test - ((loss_plus - loss_min)/(2 * delta_theta_test)) * delta_theta_new
        loss.append(loss_plus)

    plt.plot(loss)
    plt.show()
    print(a.theta)

#  [ 0.66714566  0.13101928 -0.66585639]
#  [-0.41836957  0.05978185  0.46541381]
