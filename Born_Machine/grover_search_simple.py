# coding: utf-8

import numpy as np
from projectq.ops import Rx, Rz, CNOT, Measure, ControlledGate, X, Z, H
import matplotlib.pyplot as plt
import copy, pdb
from bs33 import load_bs33


class BMQC(object):
    def __init__(self, circuit, theta, target_bin_rep, oracle):
        """
        initializing trained parameters of born machine, specifically for bar and stripe problem.
        :param num_coin_ctrl: int, the evidence value
        """
        self.circuit = circuit
        self.theta = theta
        self.targ_bin_rep = target_bin_rep
        self.oracle = oracle

    def A(self, qureg):
        self.circuit(qureg, self.theta)

    def Adag(self, qureg):
        self.circuit.dagger()(qureg, self.theta[::-1])

    def _evidence_ctrl(self, qureg):
        """here reordering the control qubits"""
        for i in range(len(self.targ_bin_rep)):
            if self.targ_bin_rep[len(self.targ_bin_rep) - 1 - i] == '0':
                X | qureg[i]

        ControlledGate(X, 3) | (qureg[0], qureg[1], qureg[2], self.anc_qubit)

        for i in range(len(self.targ_bin_rep)):
            if self.targ_bin_rep[len(self.targ_bin_rep) - 1 - i] == '0':
                X | qureg[i]

    def oracle_block(self, qureg):
        """
        construct cin state to determing whether we could measure the target.
        Attention: we search the first several qubit, corresponding to the last several bit in binary number.
        :param search_target: decimal number to search
        :return: None
        """
        if self.oracle == 1:
            H | self.anc_qubit
            # Apply R_{\epsilon}
            self._evidence_ctrl(qureg)
            H | self.anc_qubit
        elif self.oracle == 2:
            Z | self.anc_qubit
            self._evidence_ctrl(qureg)

    def initial_block(self, qureg):
        self.anc_qubit = qureg[-1]
        self.A(qureg)
        if self.oracle == 1:
            X | self.anc_qubit
        elif self.oracle == 2:
            self._evidence_ctrl(qureg)

    def grover_op(self, qureg):
        # # R_{e}^{dag}
        self.oracle_block(qureg)
        # Oracle
        self.Adag(qureg)

        for i in range(len(qureg)):
            X | qureg[i]

        '''exchange 2 and 8'''
        ControlledGate(Z, 8) | (qureg[0], qureg[1], qureg[8], qureg[3],
                                qureg[4], qureg[5], qureg[6], qureg[7], qureg[2])

        for i in range(len(qureg)):
            X | qureg[i]

        # re-Oracle
        self.A(qureg)

        if self.oracle == 2:
            self._evidence_ctrl(qureg)


def plot(prob):
    num_bit = len(prob).bit_length() - 1
    plt.plot(np.arange(0, 2**num_bit, 1), prob)
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
    print('peak positions: {}'.format([bin(n)[2:].zfill(num_bit) for n in max_position]))
    print('peak values: {}'.format(max_list))
    plt.show()


def run(oracle, targ_bin):
    print('===================================C-d Grover Search===================================')
    repeat_time = 2
    trainer, theta = load_bs33(10, context='projectq')
    bm = BMQC(trainer.circuit, theta, targ_bin, oracle)
    with trainer.context(bm.circuit.num_bit+1, 'simulate') as cc:
        bm.initial_block(cc.qureg)
        for i in range(repeat_time):
            bm.grover_op(cc.qureg)
            prob = np.abs(cc.get_wf())**2
            print("After %d step, probability to get evidence 1 is %.4f"%(i+1, prob[len(prob)//2:].sum()))

    print('The target is {}'.format(bm.targ_bin_rep))
    print('After repeating grover block {} time(s)...'.format(repeat_time))
    prob = np.abs(cc.wf)**2
    plot(prob[len(prob)//2:])  # with respect to the anc-qubit state |1>
    print('==========================================End==========================================')


if __name__ == '__main__':
    run(2, '0001110')
