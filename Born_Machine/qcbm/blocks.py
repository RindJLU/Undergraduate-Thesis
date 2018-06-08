try:
    from projectq.ops import *
    from projectq import ops
except:
    print('warning: fail to import projectq')
from functools import reduce
import scipy.sparse as sps
import numpy as np
import pdb, copy

from . import qclibd, qclibs

class CircuitBlock(object):
    '''
    the building block of a circuit. This is an abstract class.
    '''
    def __init__(self, num_bit):
        self.num_bit = num_bit

    def __call__(self, qureg, theta_list):
        '''
        build a quantum circuit.

        Args:
            theta_list (1darray<float>, len=3*num*bit*(depth+1)): parameters in this quantum circuit, here, depth equals to the number of entanglement operations.

        Return:
            remaining theta_list
        '''
        raise

    @property
    def num_param(self):
        '''
        number of parameters it consume.
        '''
        pass

    def tocsr(self, theta_list):
        '''
        build this block into a sequence of csr_matrices.

        Args:
            theta_list (1darray): parameters,

        Returns:
            list: a list of csr_matrices, apply them on a vector to perform operation.
        '''
        raise

    def tocsr_seq(self, theta_list):
        '''build a list of csr_matrix.'''
        raise

    def dagger(self):
        res = copy.copy(self)
        res._dagger = True
        return res

class BlockQueue(list):
    '''
    BlockQueue is a sequence of CircuitBlock instances.
    '''
    @property
    def num_bit(self):
        return self[0].num_bit

    @property
    def num_param(self):
        return sum([b.num_param for b in self])

    def __call__(self, qureg, theta_list):
        for block in self:
            theta_i, theta_list = np.split(theta_list, [block.num_param])
            block(qureg, theta_i)
        #np.testing.assert_(len(theta_list)==0)

    def __str__(self):
        return '\n'.join([str(b) for b in self])

    def __getitem__(self, i):
        res = super(BlockQueue, self).__getitem__(i)
        return type(self)(res) if isinstance(i, slice) else res

    def dagger(self):
        return self.__class__([b.dagger() for b in self[::-1]])

class CleverBlockQueue(BlockQueue):
    '''
    Clever Block Queue that keep track of theta_list changing history, for fast update.
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.theta_last = None
        self.memo = None

    def __call__(self, qureg, theta_list):
        if not isinstance(qureg, np.ndarray):
            return super(CleverBlockQueue, self).__call__(qureg, theta_list)
        # cache? if theta_list change <= 1 parameters, then don't touch memory.
        remember = self.theta_last is None or (abs(self.theta_last-theta_list)>1e-12).sum() > 1

        mats = []
        theta_last = self.theta_last
        if remember:
            self.theta_last = theta_list.copy()

        qureg_ = qureg
        for iblock, block in enumerate(self):
            # generate or use a block matrix
            num_param = block.num_param
            theta_i, theta_list = np.split(theta_list, [num_param])
            if theta_last is not None:
                theta_o, theta_last = np.split(theta_last, [num_param])
            if self.memo is not None and (num_param==0 or np.abs(theta_i-theta_o).max()<1e-12):
                # use data cached in memory
                mat = self.memo[iblock]
            else:
                if self.memo is not None and not remember:
                    # update the changed gate, but not touching memory.
                    mat = block._seq_update1(self.memo[iblock], theta_o, theta_i)
                else:
                    # regenerate one
                    mat = block.tocsr_seq(theta_i)
            for mat_i in mat:
                qureg_ = mat_i.dot(qureg_)
            mats.append(mat)

        if remember:
            # cache data
            self.memo = mats
        # update register
        qureg[...] = qureg_
        #np.testing.assert_(len(theta_list)==0)


class ArbituaryRotation(CircuitBlock):
    def __init__(self, num_bit):
        super(ArbituaryRotation, self).__init__(num_bit)
        self.mask = np.array([True] * (3*num_bit), dtype='bool')
        self._dagger = False

    def __call__(self, qureg, theta_list):
        gates = [Rz, Rx, Rz]
        GL = []
        if self._dagger: theta_list = theta_list[::-1]
        theta_list_ = np.zeros(self.num_bit*3)
        theta_list_[self.mask] = theta_list
        for i, (theta, mask) in enumerate(zip(theta_list_, self.mask)):
            ibit, igate = i//3, i%3
            if mask:
                gate = gates[igate](theta)
                GL.append((gate, ibit))
        if self._dagger: GL = [(DaggeredGate(g), ibit) for g, ibit in GL[::-1]]
        for g, ibit in GL:
            g | qureg[ibit]

    def __str__(self):
        return 'Rotate[%d]'%(self.num_param)

    @property
    def num_param(self):
        return self.mask.sum()

    def tocsr_seq(self, theta_list):
        '''transform this block to csr_matrix.'''
        theta_list_ = np.zeros(3*self.num_bit)
        theta_list_[self.mask] = theta_list
        rots = [qclibs.rot(*ths) for ths in theta_list_.reshape([self.num_bit,3])]
        res = [qclibs._([rot], [i], self.num_bit) for i,rot in enumerate(rots)]
        return res

    def _seq_update1(self, old, theta_old, theta_new):
        '''
        rotation layer csr_matrix update method.
        
        Args:
            rot (ArbituaryRotation): rotatio layer.
            old (csr_matrix): old matrices.
            theta_old (1darray): old parameters.
            theta_new (1darray): new parameters.

        Returns:
            csr_matrix: new rotation matrices after the theta changed.
        '''
        idiff_param = np.where(abs(theta_old-theta_new)>1e-12)[0].item()
        idiff = np.where(self.mask)[0][idiff_param]

        # get rotation parameters
        isite = idiff//3
        theta_list_ = np.zeros(3*self.num_bit)
        theta_list_[self.mask] = theta_new

        new = old[:]
        new[isite] = qclibs._(qclibs.rot(*theta_list_[isite*3:isite*3+3]), isite, self.num_bit)
        return new

class Entangler(CircuitBlock):
    def __init__(self, num_bit, pairs, gate, num_param_per_pair):
        super(Entangler, self).__init__(num_bit)
        self.pairs = pairs
        self.gate = gate
        self.num_param_per_pair = num_param_per_pair
        self.mask = np.array([True]*(len(self.pairs)*num_param_per_pair), dtype='bool')
        self._dagger = False

    def __str__(self):
        pair_str = ','.join(['%d-%d'%(i,j) for i,j in self.pairs])
        return '%s(%s)'%(self.gate, pair_str)

    def __call__(self, qureg, theta_list):
        GL = []
        if self._dagger: theta_list = theta_list[::-1]
        for pair in self.pairs:
            gate = eval("ops.%s"%self.gate)
            if self.num_param_per_pair == 0:
                pass
            else:
                theta_i, theta_list = np.split(theta_list, self.num_param_per_pair)
                gate = gate(*theta_i)
            GL.append((gate, pair))

        if self._dagger: GL = GL[::-1]
        for g, (i, j) in GL:
            g | (qureg[i], qureg[j])

    @property
    def num_param(self):
        return self.mask.sum()

    def tocsr(self, theta_list):
        '''transform this block to csr_matrix.'''
        i, j = self.pairs[0]
        gate = eval("qclibs.%s"%self.gate)
        res = gate(i, j, self.num_bit)
        for i, j in self.pairs[1:]:
            res = gate(i,j,self.num_bit).dot(res)
        return res

    def tocsr_seq(self, theta_list):
        '''transform this block to csr_matrix.'''
        if len(theta_list) == 0:
            return [self.tocsr(theta_list)]
        res = []
        gate = eval("qclibs.%s"%self.gate)
        for theta, (i, j), mask in zip(theta_list.reshape([-1,self.num_param_per_pair]), self.pairs, self.mask):
            if not mask: continue
            res.append(gate(i,j, self.num_bit, theta))
        return res

    def _seq_update1(self, old, theta_old, theta_new):
        '''
        rotation layer csr_matrix update method.
        
        Args:
            rot (ArbituaryRotation): rotatio layer.
            old (csr_matrix): old matrices.
            theta_old (1darray): old parameters.
            theta_new (1darray): new parameters.

        Returns:
            csr_matrix: new rotation matrices after the theta changed.
        '''
        n1 = self.num_param_per_pair
        idiff_param = np.where(abs(theta_old-theta_new)>1e-12)[0].item()
        ipair = idiff_param//n1
        ipair = np.where(self.mask)[0][idiff_param]

        new = old[:]
        i, j = self.pairs[ipair]
        new[ipair] = self.gate(i, j, self.num_bit, *theta_new[ipair*n1:(ipair+1)*n1])
        return new

def cnot_entangler(num_bit, pairs):
    '''controled-not entangler, the scipy version.'''
    return Entangler(num_bit, pairs, 'CNOT', 0)

def expcz_entangler(num_bit, pairs):
    '''controled-Z entangler.'''
    return Entangler(num_bit, pairs, 'ExpCZ', 1)

class BondTimeEvolution(CircuitBlock):
    def __init__(self, num_bit, pairs, hamiltonian):
        super(BondTimeEvolution, self).__init__(num_bit)
        self.pairs = pairs
        self.hamiltonian = hamiltonian
        self.mask = np.array([True]*len(pairs),dtype='bool')

    def __call__(self, qureg, theta_list):
        npar = len(self.pairs)
        for pair, ti, mask_bi in zip(self.pairs, theta_list[:npar], self.mask):
            if mask_bi:
                hamiltonian = self.hamiltonian.replace('i', str(pair[0])).replace('j', str(pair[1]))
                gate = TimeEvolution(ti, QubitOperator(hamiltonian))
                gate | qureg
        return theta_list[npar:]

    def __str__(self):
        pair_str = ','.join(['%d-%d'%(i,j) for i,j in self.pairs])
        return '%s[%s](t)'%(self.hamiltonian, pair_str)

    @property
    def num_param(self):
        return sum(self.mask)

class Rot2Basis(CircuitBlock):
    '''
    Rotate basis, change state like Ry(-theta)Rz(-phi)|psi>.
        0 < theta < pi
        0 < phi < 2pi
    '''
    def __call__(self, qureg, theta_list):
        theta_list = theta_list.reshape([-1,2])
        for i, (theta, phi) in enumerate(theta_list):
            Rz(-phi) | qureg[i]
            Ry(-theta) | qureg[i]

    def __str__(self):
        return 'Rotate Basis'

    @property
    def num_param(self):
        return 2 * self.num_bit

    def tocsr_seq(self, theta_list):
        '''transform this block to csr_matrix.'''
        theta_list = np.reshape(theta_list, [-1,2])
        res = []
        for i, (theta, phi) in enumerate(theta_list):
            rot = qclibs._([self._rot1(theta, phi)],[i], self.num_bit)
            res.append(rot)
        return res

    def _rot1(self, theta, phi):
        return qclibs.ry(-theta).dot(qclibs.rz(-phi))

    def _seq_update1(self, old, theta_old, theta_new):
        '''
        rotation layer csr_matrix update method.
        
        Args:
            rot (ArbituaryRotation): rotatio layer.
            old (csr_matrix): old matrices.
            theta_old (1darray): old parameters.
            theta_new (1darray): new parameters.

        Returns:
            csr_matrix: new rotation matrices after the theta changed.
        '''
        idiff = np.where(abs(theta_old-theta_new)>1e-12)[0].item()
        # get rotation parameters
        isite = idiff//2

        new = old[:]
        new[isite] = qclibs._(self._rot1(*theta_new[isite*2:isite*2+2]), isite, self.num_bit)
        return new

class GroverBlock(CircuitBlock):
    def __init__(self, num_bit, oracle, psi=None):
        super(GroverBlock, self).__init__(num_bit)
        self.oracle = oracle
        if psi is None: 
            psi = np.ones(2**num_bit)/np.sqrt(2**num_bit)
        self.psi = psi

    @property
    def num_param(self):
        return 0

    def __str__(self):
        return 'Grover'

    def __call__(self, qureg, *args, **kwargs):
        if self.oracle is not None:
            qureg = self.oracle.dot(qureg)
        return 2*self.psi.conj().dot(qureg)*self.psi - qureg

    def tocsr(self, *args, **kwargs):
        n = 2**self.num_bit
        return sps.linalg.LinearOperator((n, n), matvec = self.__call__)

    def tocsr_seq(self, *args, **kwargs):
        '''transform this block to csr_matrix sequence.'''
        res = [self.oracle] if self.oracle is not None else []
        for i in range(self.num_bit):
            res.append(qclibs._(qclibs.h, i, self.num_bit))
        project0 = -np.ones(2**self.num_bit)
        project0[0] = 1
        res.append(sps.diags(project0))
        for i in range(self.num_bit):
            res.append(qclibs._(qclibs.h, i, self.num_bit))
        return res

def get_demo_circuit(num_bit, depth, pairs, trim_head=True, trim_tail=True):
    '''
    Args:
        trim_head, trim_tail (bool): delete heading and trailing Rz gates if True.
    '''
    blocks = []
    # build circuit
    for idepth in range(depth+1):
        blocks.append(ArbituaryRotation(num_bit))
        if idepth!=depth:
            blocks.append(cnot_entangler(num_bit, pairs))

    # set leading and trailing Rz to disabled
    if trim_head: blocks[0].mask[::3] = False
    if trim_tail: blocks[-1].mask[2::3] = False
    return CleverBlockQueue(blocks)

def get_demo_circuit_v2(num_bit, depth, pairs, trim_head=True, trim_tail=True):
    '''
    Args:
        trim_head, trim_tail (bool): delete heading and trailing Rz gates if True.
    '''
    blocks = []
    # build circuit
    for idepth in range(depth+1):
        blocks.append(ArbituaryRotation(num_bit))
        if idepth!=depth:
            blocks.append(expcz_entangler(num_bit, pairs))

    # set leading and trailing Rz to disabled
    if trim_head: blocks[0].mask[::3] = False
    if trim_tail: blocks[-1].mask[2::3] = False
    return CleverBlockQueue(blocks)
