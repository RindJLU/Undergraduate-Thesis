#!/usr/bin/env python
'''
Learning 2 x 3 bar and stripe using Born Machine.
'''
import numpy as np
import pdb, os, fire
import scipy.sparse as sps

from qcbm import train
from qcbm.testsuit import load_barstripe

np.random.seed(2)
try:
    os.mkdir('data')
except:
    pass

class UI(object):
    # the testcase used in this program.
    def __init__(self):
        super(UI, self).__init__()
        self.depth = 7
        self.geometry = (2,3)
        self.bm = load_barstripe(self.geometry, self.depth, structure='chowliu-tree')
        self.tag = 'bs'
        self.optimizer = 'L-BFGS-B'
        self.step_rate = 0.1

    def checkgrad(self):
        '''check the correctness of our gradient.'''
        bm = self.bm
        theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
        g1 = bm.gradient(theta_list)
        g2 = bm.gradient_numerical(theta_list)
        error_rate = np.abs(g1-g2).sum()/np.abs(g1).sum()
        print('Error Rate = %.4e'%error_rate)

    def train(self):
        '''train this circuit.'''
        bm = self.bm
        theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
        loss, theta_list = train(bm, theta_list, self.optimizer, max_iter=200, step_rate=self.step_rate)
        # save
        np.savetxt('data/loss-cl-%s.dat'%self.tag, bm._loss_histo)
        np.savetxt('data/theta-cl-%s.dat'%self.tag, theta_list)

    def vcircuit(self):
        '''visualize circuit of Born Machine.'''
        bm = self.bm
        from qcbm import ProjectQContext
        bm.context = ProjectQContext
        bm.viz()

    def vpdf(self):
        '''visualize probability densitys'''
        bm = self.bm
        import matplotlib.pyplot as plt
        from qcbm.dataset import barstripe_pdf
        pl0 = barstripe_pdf(self.geometry)
        plt.plot(pl0)
        try:
            theta_list = np.loadtxt('data/theta-cl-%s.dat'%self.tag)
            pl = bm.pdf(theta_list)
            plt.plot(pl)
        except:
            print('Warning, No Born Machine Data')
        plt.legend(['Data', 'Born Machine'])
        plt.show()

    def generate(self):
        '''show generated samples for bar and stripes'''
        from qcbm.dataset import binary_basis
        from qcbm.utils import sample_from_prob
        import matplotlib.pyplot as plt
        # generate samples
        bm = self.bm
        size = (7,5)
        try:
            theta_list = np.loadtxt('data/theta-cl-%s.dat'%self.tag)
        except:
            print('run `./program.py train` before generating data!')
            return
        pl = bm.pdf(theta_list)
        indices = np.random.choice(np.arange(len(pl)), np.prod(size), p=pl)
        samples = binary_basis(self.geometry)[indices]

        # show
        fig = plt.figure(figsize=(5,4))
        gs = plt.GridSpec(*size)
        for i in range(size[0]):
            for j in range(size[1]):
                plt.subplot(gs[i,j]).imshow(samples[i*size[1]+j], vmin=0, vmax=1)
                plt.axis('equal')
                plt.axis('off')
        plt.show()

    def statgrad(self):
        '''layerwise gradient statistics'''
        import matplotlib.pyplot as plt
        nsample = 10
        bm = self.bm

        # calculate
        grad_stat = [[] for i in range(self.depth+1)]
        for i in range(nsample):
            print('running %s-th random parameter'%i)
            theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
            loss = bm.mmd_loss(theta_list)
            grad = bm.gradient(theta_list)
            loc = 0
            for i, layer in enumerate(bm.circuit[::2]):
                grad_stat[i] = np.append(grad_stat[i], grad[loc:loc+layer.num_param])
                loc += layer.num_param

        # get mean amplitude, expect first and last layer, they have less parameters.
        var_list = []
        for grads in grad_stat[1:-1]:
            var_list.append(np.abs(grads).mean())

        plt.figure(figsize=(5,4))
        plt.plot(range(1,self.depth), var_list)
        plt.ylabel('Gradient Std. Err.')
        plt.xlabel('Depth')
        plt.ylim(0,np.max(var_list)*1.2)
        plt.show()

if __name__ == '__main__':
    fire.Fire(UI)
