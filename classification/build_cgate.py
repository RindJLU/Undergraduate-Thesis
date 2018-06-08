# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from projectq import MainEngine
from projectq.ops import H, X, Rz, Ry, CNOT, Measure


class Build_CGate(object):

    def __init__(self, v_in, v_out):
        self.theta = np.random.random_sample(3)
        self.loss = []
        self.input = np.array(v_in)
        self.input_norm = np.linalg.norm(self.input)
        self.output = np.array(v_out)
        self.output_norm = np.linalg.norm(self.output)
        self.input = self.input/self.input_norm
        self.output = self.output/self.output_norm
        print('=====================Calculate rotated angle======================')
        print('The rotation angle for u is {}'.format(2*np.arccos(self.input[0])))
        print('The rotation angle for projective measurement is {}'.format(2*np.arccos(self.input_norm/np.sqrt\
            (self.output_norm**2 + self.input_norm**2)) - np.pi/2))

        # self.ref_wavefun = (np.kron(np.array([[1, 0]]), np.array(self.input)/self.input_norm) +
        #                    np.kron(np.array([[0, 1]]), np.array(self.output)/self.output_norm))/np.sqrt(2)
        # print(self.ref_wavefun)
        # self.eng = MainEngine()
        # self.qureg = self.eng.allocate_qureg(2)

    def cal_mat(self):
        C = np.array(Rz(self.theta[0]).matrix)
        B = np.array(Ry(self.theta[1]).matrix)
        A = np.dot(np.linalg.inv(np.array(Rz(self.theta[0]).matrix)), np.linalg.inv(np.array(Ry(self.theta[1]).matrix)))
        gates = [A, X.matrix, B, X.matrix, C]
        mat = np.array([[1, 0], [0, 1]])
        for g in gates:
            mat = np.dot(mat, g)
        # print(mat)
        return mat

    def cal_loss(self):
        mat = self.cal_mat()
        v_out_test = np.dot(mat, self.input)
        loss = np.linalg.norm(self.output - v_out_test)
        return loss

    def projectq(self):
        self.eng.flush()
        self.eng.backend.set_wavefunction(np.array([1.0, 0, 0, 0]), self.qureg)
        H | self.qureg[0]
        Ry(1.85) | self.qureg[1]

        Rz(self.theta[0]) | self.qureg[1]
        CNOT | (self.qureg[0], self.qureg[1])
        Ry(self.theta[1]) | self.qureg[1]
        CNOT | (self.qureg[0], self.qureg[1])
        # Ry(-self.theta[1]) | self.qureg[1]
        # Rz(-self.theta[0]) | self.qureg[1]
        Rz(-self.theta[2]) | self.qureg[1]

        self.eng.flush()
        b, wavefun = self.eng.backend.cheat()
        loss = np.linalg.norm(self.ref_wavefun - wavefun)

        Measure | self.qureg
        return loss

        # H | qureg[1]
        # phi = 2 * np.arcsin(self.input_norm/np.sqrt(self.input_norm**2 + self.output_norm**2)) - np.pi/2
        # Ry(phi) | qureg[1]
        # self.eng.flush()

        # a = eng.backend.get_probability('1', [qureg[1]])

        # Measure | qureg
        # print(a, wavefun)
        # return a


if __name__ == '__main__':
    v_in = [2, 8.0]
    v_out = [3, 4.0]
    exp = Build_CGate(v_in, v_out)

    for iters in range(1000):
        for i in range(len(exp.theta)):
            exp.theta[i] += 0.0001
            loss_plus = exp.cal_loss()
            exp.theta[i] += -0.0002
            loss_min = exp.cal_loss()

            exp.loss.append(loss_plus)
            exp.theta[i] += 0.0001 - ((loss_plus - loss_min)/(2 * 0.0001))*0.01

    print(exp.theta, 180*np.array(exp.theta)/np.pi)
    print(exp.loss[-1])

    # print(np.sqrt(2*a*114))
    plt.plot(exp.loss)
    plt.show()

# [-0.10789846 -0.25448352]
# [-0.12229118 -0.25317151]
# [ -7.00676875 -14.50565914]
# [ 0.04367479 -0.26573644] [  2.50238111 -15.22557654]
# [ 0.15753203 -0.2708032 ] [  9.02592066 -15.51588038]
# [-0.38516352 -0.07627915]  [5, 8]
# [-0.11944007 -0.25343458] [ -6.84341183 -14.5207318 ]