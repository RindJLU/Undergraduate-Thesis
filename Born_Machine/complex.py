#!/usr/bin/env python
'''
Learning 2 x 3 bar and stripe using Born Machine.
'''
import numpy as np
import pdb, os, fire
import scipy.sparse as sps

from qcbm import train
from qcbm.testsuit import load_complex

from program import UI

np.random.seed(2)
try:
    os.mkdir('data')
except:
    pass

class UI_COMPLEX(UI):
    # the testcase used in this program.
    def __init__(self):
        self.depth = 10
        self.geometry = (3,2)
        self.bm = load_complex(self.geometry, self.depth)
        self.tag = 'zx'
        self.optimizer = 'GradientDescent'
        self.step_rate = 1.

    def checkgrad(self):
        '''check the correctness of our gradient.'''
        bm = self.bm
        theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
        for i in range(10):
            bm.random_basis()
            g1 = bm.gradient(theta_list)
            g2 = bm.gradient_numerical(theta_list)
            error_rate = np.abs(g1-g2).sum()/np.abs(g2).sum()
            print('Error Rate = %.4e'%error_rate)

if __name__ == '__main__':
    fire.Fire(UI_COMPLEX)
