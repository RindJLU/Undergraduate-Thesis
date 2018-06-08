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

def test_dataset():
    geometry = (3,3)
    pl1 = gaussian_pdf(geometry, mu=0, sigma=255.5)
    pl2 = barstripe_pdf(geometry)
    plt.plot(pl1)
    plt.plot(pl2)
    plt.ylim(0,0.01)
    plt.show()

def test_bm():
    depth = 2
    np.random.seed(2)

    #bm = load_gaussian(6, depth)
    bm = load_barstripe((3,3), depth)
    theta_list = np.random.random(bm.circuit.num_param)*2*np.pi

    assert_(bm.depth == depth)
    print(bm.mmd_loss(theta_list))
    g1 = bm.gradient(theta_list)
    g2 = bm.gradient_numerical(theta_list)
    assert_allclose(g1, g2, atol=1e-5)

def test_wf():
    depth = 2
    geometry = (6,)

    num_bit = np.prod(geometry)
    pairs =nearest_neighbor(geometry)
    circuit = get_demo_circuit(num_bit, depth, pairs)

    with ProjectQContext('simulate', np.prod(geometry)) as cc:
        circuit(cc.qureg, theta_list)

    wf = np.zeros(2**num_bit)
    wf[0] = 1
    assert_allclose(cc.wf, wf)

def test_train_gaussian():
    depth = 6
    np.random.seed(2)

    bm = load_gaussian(6, depth, 'projectq')
    theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
    loss, theta_list = train(bm, theta_list, 'L-BFGS-B', max_iter=20)
    pl = bm.pdf(theta_list)

    # display
    plt.ion()
    plt.plot(bm.p_data)
    plt.plot(pl)
    plt.legend(['Data', 'Gradient Born Machine'])
    pdb.set_trace()

def test_train_gaussian_scipy():
    depth = 6
    np.random.seed(2)

    bm = load_gaussian(6, depth)
    theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
    loss, theta_list = train(bm, theta_list, 'L-BFGS-B', max_iter=20)
    pl = bm.pdf(theta_list)

    # display
    plt.ion()
    plt.plot(bm.p_data)
    plt.plot(pl)
    plt.legend(['Data', 'Gradient Born Machine'])
    pdb.set_trace()

def test_qclib():
    cnot = CNOT(1, 0 ,2)
    assert_(cnot.nnz==4)
    assert_allclose(cnot.toarray(), sps.coo_matrix(([1,1,1,1],([0,1,2,3],[0,1,3,2]))).toarray())
    assert_allclose(rot(-np.pi/2.,np.pi/4.,np.pi/2.).toarray(),ry(np.pi/4.).toarray())

def test_qclibd():
    cnot = qclibd.CNOT(1,0,2)
    assert_allclose(cnot, sps.coo_matrix(([1,1,1,1],([0,1,2,3],[0,1,3,2]))).toarray())
    assert_allclose(qclibd.rot(-np.pi/2.,np.pi/4.,np.pi/2.),qclibd.ry(np.pi/4.))

def grad_stat_layer(seed=2):
    '''layerwise gradient statistics'''
    np.random.seed(seed)
    folder = 'data'
    nsample = 10
    depth = 100

    bm = load_barstripe((3, 3), depth)
    num_bit = bm.circuit.num_bit

    # calculate
    grad_stat = []
    for i in range(nsample):
        theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
        loss = bm.mmd_loss(theta_list)
        grad = bm.gradient(theta_list)
        print('grad = %s'%(grad,))
        grad_stat.append(grad)
    np.savetxt(os.path.join(folder, 'grads-%d.dat'%depth), grad_stat)

def test_wittness():
    from scipy.stats import laplace, norm
    from mmd import mix_rbf_kernel, witness
    basis = np.linspace(-5,5,200)
    K = mix_rbf_kernel(basis, basis, [0.5], False)
    yl = laplace.pdf(basis, scale=0.7)
    yg = norm.pdf(basis)
    #yl=yl/yl.sum()
    #yg=yg/yg.sum()
    wit = witness(K, yl, yg)
    wit = wit/np.linalg.norm(wit)
    plt.ion()
    plt.plot(basis, yl)
    plt.plot(basis, yg)
    plt.plot(basis, wit)
    plt.legend(['laplace', 'gaussian'])
    pdb.set_trace()

def test_svec():
    # test polar vec
    s = qclibd.random_pauli()
    vec = qclibd.s2vec(s)[1:].real
    eigen_vec = qclibd.polar2u(qclibd.vec2polar(vec)[1:])
    ei = eigen_vec.conj().dot(s.dot(eigen_vec))
    assert_allclose(ei, 1)

    polar = qclibd.vec2polar(vec)
    vec2 = qclibd.polar2vec(polar)
    assert_allclose(vec, vec2)

    # test pauli operator.
    s2 = qclibd.vec2s(vec)
    assert_allclose(s, s2)
    e, v = np.linalg.eigh(s2)
    v = v[:,1]
    polar2 = qclibd.u2polar(v)
    v2 = qclibd.polar2u(polar2)
    assert_almost_equal(abs(v.conj().dot(v2)), 1)
    v = qclibd.polar2vec((1,)+tuple(polar2))
    assert_allclose(v, vec)

    with BlochPlot() as bp:
        bp.add_polar(*polar[1:])
        bp.ball.add_vectors(v)

def test_rmeasure():
    rb = Rot2Basis(1)
    angles = np.random.random(2)*2*np.pi
    # prepair a state in the angles direction.
    state = qclibd.polar2u(angles)
    # rotate to the same direction for measurements.
    mats = rb.tocsr_seq([angles[0], angles[1]])
    for mi in mats:
        state = mi.dot(state)
    assert_allclose(state, [1,0])

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

def test_random_basis():
    basis = random_basis(6)
    assert_(all(basis[:,0]>=0) and all(basis[:,0]<=np.pi))
    assert_(all(basis[:,1]>=-np.pi) and all(basis[:,1]<=np.pi))

def test_expcz():
    ibit, jbit, n = 2,0,5
    theta = np.pi/2.
    res1 = qclibs.ExpCZ(ibit, jbit, n, theta)
    res2 = sps.linalg.expm(-1j*qclibs.CZ(ibit, jbit, n)*theta/2.)
    pdb.set_trace()

if __name__ == '__main__':
    #test_dataset()
    #test_wf()
    #test_bm()
    #test_wittness()
    #test_svec()
    #test_expcz()
    #test_random_basis()
    #test_rmeasure()
    #test_grover()
    #test_qclibd()
    test_qclib()
    #test_train_gaussian_scipy()
    #grad_stat_layer()
