'''elementary library for quantum computation.'''

from functools import reduce
import scipy as np
import pdb
import scipy.sparse as sps
from scipy.sparse import linalg

from scipy.sparse import kron
# from .flib.kron import kron_csr as kron
from . import qclibd
from .utils import take_bit

######  Pauli Matrices  ########

ss = [sps.csr_matrix(si) for si in qclibd.ss]
I2, sx, sy, sz = ss
p0, p1, h = [sps.csr_matrix(op) for op in (qclibd.p0, qclibd.p1, qclibd.h)]

def _ri(si, theta):
    return np.cos(theta/2.)*I2 - 1j*np.sin(theta/2.)*si

def rx(theta):
    return _ri(sx, theta)

def ry(theta):
    return _ri(sy, theta)

def rz(theta):
    return _ri(sz, theta)

def rot(t1, t2, t3):
    '''
    a general rotation gate rz(t3)rx(r2)rz(t1).

    Args:
        t1, t2, t3 (float): three angles.

    Returns:
        2x2 csr_matrix: rotation matrix.
    '''
    return rz(t3).dot(rx(t2)).dot(rz(t1))

def polar(s):
    '''polar angles of pauli matrix.'''

# multiple bit construction

def CNOT(ibit, jbit, n):
    '''Controled not gate'''
    res = _([p0, I2], [ibit, jbit], n)
    res = res + _([p1, sx], [ibit, jbit], n)
    return res

def CZ(ibit, jbit, n):
    '''Controled Z gate.'''
    res = _([p0, I2], [ibit, jbit], n)
    res = res + _([p1, sz], [ibit, jbit], n)
    return res

def ExpCZ(ibit, jbit, n, theta):
    '''Evolutional Controled Z gate.'''
    res = np.exp(-1j*theta/2.)*np.ones(2**n)
    basis = np.arange(2**n)
    res[(take_bit(basis, ibit)&take_bit(basis,jbit)).astype('bool')] = np.exp(1j*theta/2.)
    return sps.diags(res)

def _(ops, locs, n):
    '''
    Put operators in a circuit and compile them.

    notice the big end are high loc bits!

    Args:
        ops (list): list of single bit operators.
        locs (list): list of positions.
        n (int): total number of bits.

    Returns:
        csr_matrix: resulting matrix.

    Note:
        rotation gates may be optimized, using 1+1j*sin(theta)*sigma
    '''
    if np.ndim(locs) == 0:
        locs = [locs]
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    locs = np.asarray(locs)
    locs = n - locs -1
    order = np.argsort(locs)
    locs = np.concatenate([[-1], locs[order], [n]])
    return _wrap_identity([ops[i] for i in order], np.diff(locs) - 1)


def _wrap_identity(data_list, num_bit_list):
    if len(num_bit_list) != len(data_list) + 1:
        raise Exception()

    res = sps.eye(2**num_bit_list[0])
    for data, nbit in zip(data_list, num_bit_list[1:]):
        res = kron(res, data)
        res = kron(res, sps.eye(2**nbit, dtype='complex128'))
    return res
