import numpy as np
import pdb

from .qcbm import QCBM, _pdf
from .blocks import Rot2Basis, CleverBlockQueue
from .qclibd import vec2polar

class CloneQCBM(QCBM):
    '''
    Args:
        wf_data (1darray): wave function as data.
    '''
    def __init__(self, circuit, mmd, wf_data, batch_size=None):
        np.testing.assert_almost_equal(_pdf(wf_data).sum(), 1)
        super(CloneQCBM, self).__init__(circuit, mmd, wf_data, batch_size=None)
        self.random_basis()

    @property
    def p_data(self):
        wf = self.rot_basis(self._data)
        return _pdf(wf)

    def wf(self, theta_list):
        wf = super(CloneQCBM, self).wf(theta_list)
        wf = self.rot_basis(wf)
        return wf

    def wf_origin(self, theta_list):
        return super(CloneQCBM, self).wf(theta_list)

    def random_basis(self):
        '''randomly set the measurement basis'''
        rotter = Rot2Basis(self.circuit.num_bit)
        if not hasattr(self, '_counter'):
            self._counter = 0
        if not hasattr(self, '_measure_basis'):
            self._measure_basis = random_basis(self.circuit.num_bit).ravel()
        #self._measure_basis[np.random.randint(0,self.circuit.num_bit, 3)*2] += np.pi/2.
        self._measure_basis = np.random.random(self.circuit.num_bit*2)*np.pi
        #self._measure_basis[::2] += np.pi
        self._rot_mats = rotter.tocsr_seq(self._measure_basis)
        self._counter += 1

    def rot_basis(self, wf):
        '''rotate basis of wave function to target basis for measurements.'''
        mats = self._rot_mats
        for mat in mats:
            wf = mat.dot(wf)
        return wf

    def _sample_or_prob(self, theta_list):
        wf = self.wf(theta_list)
        pl = _pdf(wf)
        if self.batch_size is not None:
            # introducing sampling error
            samples = sample_from_prob(np.arange(len(pl)), pl, self.batch_size)
            if self.mmd.use_prob:
                return prob_from_sample(samples, len(pl), False)
            else:
                return samples
        else:
            return pl
 

class CloneGAN(QCBM):
    '''
    Args:
        wf_data (1darray): wave function as data.
    '''
    def __init__(self, circuit, mmd, wf_data, batch_size=None):
        np.testing.assert_almost_equal(_pdf(wf_data).sum(), 1)
        super(CloneGAN, self).__init__(circuit, mmd, wf_data, batch_size=None)

        # add rotter to the circuit.
        self.circuit = self.circuit[:]
        rotter = Rot2Basis(self.circuit.num_bit)
        self.circuit.append(rotter)
        self.data_circuit = CleverBlockQueue([rotter])

    def wf_origin(self, theta_list):
        cc = self.context( self.circuit.num_bit, 'simulate')
        cc.__enter__()
        self.circuit[:-1](cc.qureg, theta_list[:-2*self.circuit.num_bit])
        cc.__exit__()
        return cc.wf

    def p_data(self, gamma_list):
        wf = self._data.copy()
        self.data_circuit(wf, gamma_list)
        return _pdf(wf)

    def mmd_loss(self, theta_list):
        '''get the loss'''
        # get probability distritbution of Born Machine
        self._prob = self.pdf(theta_list)
        # use wave function to get mmd loss
        loss = self.mmd(self._prob, self.p_data(theta_list[-2*self.circuit.num_bit:]))
        self._loss_histo.append(loss)
        return loss

    def gradient(self, theta_list):
        '''
        cheat and get gradient.
        '''
        ngamma = 2*self.circuit.num_bit
        gamma_list = theta_list[-ngamma:]
        # for stability consern, we do not use the cached probability output.
        prob = self.pdf(theta_list)
        # for performance consern in real training, prob can be reused!
        #prob = self._prob
        p_data = self.p_data(gamma_list)

        # get gradient with respect to generating circuit
        def get1(i):
            theta_list_ = theta_list.copy()
            # pi/2 phase
            theta_list_[i] += np.pi/2.
            prob_pos = self.pdf(theta_list_)
            # -pi/2 phase
            theta_list_[i] -= np.pi
            prob_neg = self.pdf(theta_list_)

            grad_pos = self.mmd.kernel_expect(prob, prob_pos) - self.mmd.kernel_expect(prob, prob_neg)
            grad_neg = self.mmd.kernel_expect(p_data, prob_pos) - self.mmd.kernel_expect(p_data, prob_neg)
            return grad_pos - grad_neg

        def get1_data(i):
            gamma_list_ = gamma_list.copy()
            # pi/2 phase
            gamma_list_[i] += np.pi/2.
            data_pos = self.p_data(gamma_list_)
            # -pi/2 phase
            gamma_list_[i] -= np.pi
            data_neg = self.p_data(gamma_list_)

            grad_pos = self.mmd.kernel_expect(p_data, data_pos) - self.mmd.kernel_expect(p_data, data_neg)
            grad_neg = self.mmd.kernel_expect(prob, data_pos) - self.mmd.kernel_expect(prob, data_neg)
            return grad_pos - grad_neg

        from .mpiutils import mpido
        grad = np.array(mpido(get1, inputlist=np.arange(len(theta_list))))
        grad_data = mpido(get1_data, inputlist=np.arange(ngamma))
        grad[-ngamma:] += grad_data

        # gamma
        grad[-ngamma:] *= -np.linalg.norm(grad[-ngamma:])/np.linalg.norm(grad[:-ngamma])*0.5
        return np.array(grad)


def random_basis(num_bit):
    '''
    random measurement basis.

    Args:
        num_bit (int): the number of bit.

    Returns:
        2darray: first column is the theta angles, second column is the phi angles.
    '''
    res = np.random.randn(num_bit, 3)
    polars = vec2polar(res)
    polars[:,2]%=np.pi
    return polars[:,1:]
