#!/usr/bin/env python
'''
Learning 2 x 3 bar and stripe using Born Machine.
'''
import numpy as np
import pdb, os, fire
import scipy.sparse as sps

from qcbm import train
from qcbm.testsuit import load_complexgan

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
        self.bm = load_complexgan(self.geometry, self.depth)
        self.tag = 'gan'
        self.optimizer = 'GradientDescent'
        self.step_rate = 1.

    def checkgrad(self):
        '''check the correctness of our gradient.'''
        bm = self.bm
        theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
        g1 = bm.gradient(theta_list)
        g2 = bm.gradient_numerical(theta_list)
        g2[-bm.circuit[-1].num_param:]*=-1
        error_rate = np.abs(g1-g2).sum()/np.abs(g1).sum()
        print('Error Rate = %.4e'%error_rate)

    def fidelity(self):
        wf = data.wf(theta_list)
        wf_data = bm._data

    def vpdf(self):
        '''visualize probability densitys'''
        bm = self.bm
        import matplotlib.pyplot as plt
        from qcbm.dataset import barstripe_pdf
        rot = bm.circuit.pop(-1)
        pl0 = bm._data
        plt.plot(pl0.real, color='r')
        plt.plot(pl0.imag, color='r', ls='--')
        try:
            theta_list, gamma_list = np.split(np.loadtxt('data/theta-cl-%s.dat'%self.tag),
                    [-2*bm.circuit.num_bit])
            wf = bm.wf(theta_list)
            print(theta_list)
            pdb.set_trace()
            plt.plot(wf.real, color='g')
            plt.plot(wf.imag, color='g', ls='--')
        except:
            print('Warning, No Born Machine Data')
        plt.legend(['Data-Real', 'Data-Imag', 'BM-Real', 'BM-Imag'])
        plt.show()

if __name__ == '__main__':
    fire.Fire(UI_COMPLEX)
