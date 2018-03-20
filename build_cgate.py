# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from projectq import MainEngine
from projectq.ops import H, X, Rz, Ry, CNOT, Measure


class Build_CGate(object):

    def __init__(self, v_in, v_out):
        self.theta = np.random.random_sample(2)
        self.loss = []
        self.input = v_in
        self.input_norm = np.linalg.norm(self.input)
        self.output = v_out
        self.output_norm = np.linalg.norm(self.output)

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
        loss = np.linalg.norm(self.output/self.output_norm - v_out_test/self.input_norm)
        return loss

    def projectq(self):
        eng = MainEngine()
        qureg = eng.allocate_qureg(2)
        H | qureg[0]
        Ry(1.85) | qureg[1]

        Rz(self.theta[0]) | qureg[1]
        CNOT | (qureg[0], qureg[1])
        Ry(self.theta[1]) | qureg[1]
        CNOT | (qureg[0], qureg[1])
        Ry(-self.theta[1]) | qureg[1]
        Rz(-self.theta[0]) | qureg[1]

        H | qureg[1]
        Ry(-0.07414198) | qureg[1]
        eng.flush()

        a = eng.backend.get_probability('1', [qureg[1]])
        Measure | qureg
        print(a)
        return a


if __name__ == '__main__':
    v_in = [3, 4.0]
    v_out = [2, 5.0]
    exp = Build_CGate(v_in, v_out)

    for iter in range(300):
        exp.theta += 0.0001
        loss_plus = exp.cal_loss()
        exp.theta += -0.0002
        loss_min = exp.cal_loss()

        exp.loss.append(loss_plus)

        exp.theta += 0.0001 - ((loss_plus - loss_min)/(2 * 0.0001))*0.02
    print(exp.theta)
    a = exp.projectq()
    print(np.sqrt(2*a*114))
    plt.plot(exp.loss)
    plt.show()


# [-0.10789846 -0.25448352]
# [-0.38516352 -0.07627915]  [5, 8]
