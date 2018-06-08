import numpy as np
import pdb

from qcbm.blocks import get_demo_circuit
from qcbm.qcbm import QCBM
from qcbm.mmd import RBFMMD2
from qcbm.contexts import ProjectQContext
from qcbm.utils import packnbits, unpacknbits

def load_bs33(depth, context='scipy'):
    '''3 x 3 bar and stripes.'''
    geometry = (3, 3)
    num_bit = np.prod(geometry)

    # bar and stripe
    p_bs = np.loadtxt("bs33/p_dataset-23.dat")

    # standard circuit
    pairs = np.atleast_2d(np.loadtxt("bs33/cnot_pairs.dat").astype('int32'))
    circuit = get_demo_circuit(num_bit, depth, pairs)

    # mmd loss
    mmd = RBFMMD2([0.5], num_bit, True)

    # Born Machine
    bm = QCBM(circuit, mmd, p_bs)
    if context == 'projectq':
        bm.context = ProjectQContext

    # load parameters
    theta = np.loadtxt("bs33/theta-cl-bs.dat")
    return bm, theta

def test_qcbm():
    num_bit = 9
    bm, theta_list = load_bs33(10, context='projectq')
    # bm, theta_list = load_bs33(10, context='scipy')
    with bm.context(bm.circuit.num_bit, 'simulate') as cc:
        bm.circuit(cc.qureg, theta_list)
        bm.circuit.dagger()(cc.qureg, theta_list[::-1])
    #wf = bm.wf(theta_list)
    wf = cc.wf
    basis = unpacknbits(np.arange(2**num_bit)[:, None], num_bit)
    samples = basis[abs(wf)>1e-2]
    print(samples)

    import matplotlib.pyplot as plt
    plt.ion()
    #plt.plot(np.abs(wf)**2)

    # show
    size = (4,4)
    fig = plt.figure(figsize=(5,4))
    gs = plt.GridSpec(*size)
    for i in range(size[0]):
        for j in range(size[1]):
            try:
                plt.subplot(gs[i,j]).imshow(samples[i*size[1]+j].reshape([3,3]), vmin=0, vmax=1)
                plt.axis('equal')
                plt.axis('off')
            except:
                pass
    plt.show()
    pdb.set_trace()


if __name__ == '__main__':
    test_qcbm()
