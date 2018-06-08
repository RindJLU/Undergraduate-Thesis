import numpy as np
from numpy.testing import dec, assert_, assert_raises,\
    assert_almost_equal, assert_allclose
import matplotlib.pyplot as plt
import pdb, os
from profilehooks import profile
import scipy.sparse as sps

from .blocks import get_demo_circuit, Rot2Basis, GroverBlock
from .structure import nearest_neighbor
from .dataset import gaussian_pdf, barstripe_pdf, digit_basis, binary_basis
from .contexts import ProjectQContext
from .mmd import RBFMMD2
from .train import train
from .testsuit import load_gaussian, load_barstripe
from .qclibs import rot, CNOT, ry, I2
from . import qclibd, qclibs
from .visualize import BlochPlot
from .cloneqcbm import random_basis

def test_grover():
    num_bit = 12
    diag = np.zeros(2**num_bit); diag[0] = 1
    gb = GroverBlock(num_bit, oracle = sps.diags((-1)**diag))
    # prepair a random state
    #state = qclibd.random_state(num_bit)
    state = gb.psi

    # use sequence, the simulation way.
    state1 = state
    mats = gb.tocsr_seq()
    for mi in mats:
        state1 = mi.dot(state1)
    # direct, the algirithmic way.
    state2 = gb(state)
    state3 = gb.tocsr().dot(state)
    assert_allclose(state1 ,state2)
    assert_allclose(state1 ,state3)

    # solve a real search problem
    num_iter = int(np.round(np.pi/4.*np.sqrt(2**num_bit)))
    for i in range(num_iter):
        state = gb(state)
    assert_allclose(state, diag, atol=1e-3)


def test_inference():
    num_bit = 8
    # target the state where first 2 bits are 0, the space if |psi>|e>.
    np.repeat(psi, 2)

    # prepair a random state
    psi = qclibd.random_state(num_bit)   # random state
    psi_t = np.zeros(2**num_bit); psi_t[0] = 1  # the target
    alpha = psi_t.conj().dot(psi)
    e = 

    Spsi = GroverBlock(num_bit, oracle=None, psi = psi)
    # equivalent form
    #S0 = GroverBlock(num_bit, oracle=None, psi = zeros)
    #Spsi = AS0A^

    Se = GroverBlock(num_bit, oracle=None, psi = e)

    # solve a real search problem
    num_iter = int(np.round(np.pi/4.*np.sqrt(2**num_bit)))
    for i in range(num_iter):
        state = gb(state)
    assert_allclose(state, diag, atol=1e-3)


if __name__ == '__main__':
    test_inference()
