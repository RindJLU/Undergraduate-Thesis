# coding: utf-8

import numpy as np
import projectq
from projectq.ops import Rx, Rz, CNOT, Measure, ControlledGate, X, Z, H, Swap
import matplotlib.pyplot as plt
import copy


class BMQC(object):
    def __init__(self, depth, geom, num_coin_ctrl, targ_num, oracle):
        """
        initializing trained parameters of born machine, specifically for bar and stripe problem.
        :param depth: int, the depth of the born machine
        :param geom: 2 dimensional tuple, represent the number of col and row respectively
        :param num_coin_ctrl: int, the evidence value
        """
        self.num_qubits = geom[0]*geom[1]
        self.depth = depth
        # self.ctrl_pairs = self.get_nn_pairs(geom)
        self.ctrl_pairs = [(1, 7), (2, 8), (4, 7), (5, 3), (6, 0), (6, 3), (6, 7), (8, 5)]
        self.oracle = oracle

        self.targ_bin_rep = self._cal_bin_rep(num_coin_ctrl)[targ_num]
        self.theta = self.init_theta()

        '''initializing the quantum circuit'''
        self.eng = projectq.MainEngine()
        self.qureg = self.eng.allocate_qureg(self.num_qubits)
        self.anc_qubit = self.eng.allocate_qubit()
        self.eng.flush()

    def init_theta(self):
        theta = np.load("bs33/theta-histo-23.npy")[-1].reshape(self.depth + 1, self.num_qubits, 3)
        return theta

    def measure(self):
        """
        measure the final wave function
        :return: None
        """
        self.eng.flush()
        mapping, wavefunction = copy.deepcopy(self.eng.backend.cheat())
        # print("The full wavefunction is: {}".format(wavefunction))
        prob = np.abs(wavefunction)**2
        self.plot(prob[2**self.num_qubits:])  # with respect to the anc-qubit state |1>
        Measure | self.qureg
        Measure | self.anc_qubit

    def _cal_bin_rep(self, num_coin_ctrl):
        """
        construct the list storing binary string from 0 to 2**len(num_coin_ctrl)
        :param num_coin_ctrl: int
        :return:
        """
        index_bin_rep_list = []
        for i in range(2**num_coin_ctrl):
            bin_index = bin(i)[2:]
            while len(bin_index) < num_coin_ctrl:
                bin_index = '0' + bin_index
            index_bin_rep_list.append(bin_index)
        return index_bin_rep_list

    def bin_rep(self, num):
        """
        change the number to binary representation
        :param num: int
        :return:
        """
        for n in range(len(num)):
            num_bin_rep = bin(num[n])[2:]
            while len(num_bin_rep) < self.num_qubits:
                num_bin_rep = '0' + num_bin_rep
            num[n] = num_bin_rep
        return num

    def run(self, type):
        """
        construct the prob dist according to the parameters trained using born machine.
        :param type: str, forward or backward
        :return:
        """
        if type == 'forward':
            for dep_index in range(self.depth + 1):
                for i in range(self.num_qubits):
                    Rz(self.theta[dep_index, i, 0]) | self.qureg[i]
                    Rx(self.theta[dep_index, i, 1]) | self.qureg[i]
                    Rz(self.theta[dep_index, i, 2]) | self.qureg[i]
                # entanglement
                if dep_index == self.depth:
                    pass
                else:
                    for ctrl_pair in self.ctrl_pairs:
                        CNOT | (self.qureg[ctrl_pair[0]], self.qureg[ctrl_pair[1]])
        elif type == 'backward':
            for dep_index in range(self.depth, -1, -1):
                # entanglement
                if dep_index != self.depth:
                    for ctrl_pair in self.ctrl_pairs[::-1]:
                        CNOT | (self.qureg[ctrl_pair[0]], self.qureg[ctrl_pair[1]])

                for i in range(self.num_qubits):
                    Rz(self.theta[dep_index, i, 2]).get_inverse() | self.qureg[i]
                    Rx(self.theta[dep_index, i, 1]).get_inverse() | self.qureg[i]
                    Rz(self.theta[dep_index, i, 0]).get_inverse() | self.qureg[i]

    def _evidence_ctrl(self):
        """here reordering the control qubits"""
        order = [7, 3, 8, 1, 6, 4, 0, 2, 5]
        for i in range(len(self.targ_bin_rep)):
            if self.targ_bin_rep[len(self.targ_bin_rep) - 1 - i] == '0':
                X | self.qureg[order.index(i)]

        # ControlledGate(X, len(search_bin_rep)) | ([self.qureg[int(i)] for i in range(len(search_bin_rep))],
        #                                           self.anc_qubit)
        ControlledGate(X, 3) | (self.qureg[order.index(0)], self.qureg[order.index(1)], self.qureg[order.index(2)], self.anc_qubit)

        for i in range(len(self.targ_bin_rep)):
            if self.targ_bin_rep[len(self.targ_bin_rep) - 1 - i] == '0':
                X | self.qureg[order.index(i)]

    def oracle_block(self):
        """
        construct cin state to determing whether we could measure the target.
        Attention: we search the first several qubit, corresponding to the last several bit in binary number.
        :param search_target: decimal number to search
        :return: None
        """
        if self.oracle == 1:
            H | self.anc_qubit
            # Apply R_{\epsilon}
            self._evidence_ctrl()
            H | self.anc_qubit
        elif self.oracle == 2:
            Z | self.anc_qubit
            self._evidence_ctrl()

    def initial_block(self):
        self.run('forward')
        if self.oracle == 1:
            X | self.anc_qubit
        elif self.oracle == 2:
            self._evidence_ctrl()

    def grover_block(self, repeat_time):
        for t in range(repeat_time):
            # # R_{e}^{dag}
            self.oracle_block()
            # Oracle
            self.run('backward')

            for i in range(len(self.qureg)):
                X | self.qureg[i]

            '''exchange 2 and 8'''
            ControlledGate(Z, 8) | (self.qureg[0], self.qureg[1], self.qureg[8], self.qureg[3],
                                    self.qureg[4], self.qureg[5], self.qureg[6], self.qureg[7], self.qureg[2])

            for i in range(len(self.qureg)):
                X | self.qureg[i]

            # re-Oracle
            self.run('forward')

            if self.oracle == 2:
                self._evidence_ctrl()

    def plot(self, prob):
        plt.plot(np.arange(0, 2**self.num_qubits, 1), prob)
        plt.xlabel('computational basis')
        plt.ylabel('probability')

        max_prob = max(prob)
        max_list = []
        for i in range(len(prob)):
            if prob[i] > 2*max_prob/3:
                max_list.append(prob[i])
        max_position = []
        for e in max_list:
            max_position.append(int(np.where(prob==e)[0]))
        # if len(max_list) == 1:
        #     print('peak position: ' + str(self.bin_rep(max_position)) + ',' + 'peak_value: ' + str(max_list[0]))
        # else:
        print('peak positions: {}'.format(self.bin_rep(max_position)))
        print('peak values: {}'.format(max_list))

        plt.show()

    def reordering(self):
        """reordering using swap-gate"""
        order = [7, 3, 8, 1, 6, 4, 0, 2, 5]
        for i in range(len(order)):
            if order[i] != i:
                Swap | (self.qureg[order.index(i)], self.qureg[i])
                temp = order[i]
                order[order.index(i)] = temp
                order[i] = i


if __name__ == '__main__':
    # print('===================================C-d Grover Search===================================')
    # oracle = int(input('please select the oracle method: 1. boolean oracle. 2. phase oracle (enter the index)'))
    # if oracle == 1:
    #     print('you have selected BOOLEAN oracle')
    # elif oracle == 2:
    #     print('you have selected PHASE oracle')
    # else:
    #     print('error, please reselect')
    # targ_num = int(input('please enter the evidence value(in decimal form):'))
    # print('----------------------------------------running----------------------------------------')
    #
    # depth = 10
    # geom = (3, 3)
    # num_coin_ctrl = 3  # number of evidence qubits
    # repeat_time = 2
    # bm = BMQC(depth, geom, num_coin_ctrl, targ_num, oracle)
    # bm.initial_block()
    # bm.grover_block(repeat_time)
    #
    # print('The target is {}'.format(bm.targ_bin_rep))
    # print('After repeating grover block {} time(s)...'.format(repeat_time))
    # bm.reordering()
    # bm.measure()
    # print('==========================================End==========================================')
    x = np.arange(0, 16, 1)
    y = np.array([1, 0, 1, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         1, 0, 0, 1])
    y = y/y.sum()
    plt.bar(x, y)
    plt.ylabel('probability')
    plt.xlabel('computational basis')
    plt.show()
