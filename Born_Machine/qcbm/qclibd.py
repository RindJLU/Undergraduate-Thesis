'''elementary library for quantum computation.'''

from functools import reduce
import scipy as np
import pdb

######  Pauli Matrices  ########

I2 = np.eye(2)
sx = np.array([[0,1],[1,0.]])
sy = np.array([[0,-1j],[1j,0.]])
sz = np.array([[1,0],[0,-1.]])

p0 = (sz + I2) / 2
p1 = (-sz + I2) / 2
h = (sx + sz) / np.sqrt(2.)
ss = [I2, sx, sy, sz]

# single bit rotation matrices

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
        2x2 array: rotation matrix.
    '''
    return rz(t3).dot(rx(t2)).dot(rz(t1))

def random_pauli(pure=True):
    '''random pauli matrix.'''
    vec = np.random.randn(4)
    if pure: vec[0] = 0
    vec/=np.linalg.norm(vec)
    return vec2s(vec)

def s2vec(s):
    '''
    Transform a spin to a 4 dimensional vector, corresponding to s0,sx,sy,sz component.

    Args:
        s (matrix): the 2 x 2 pauli matrix.
    '''
    res=np.array([np.trace(np.dot(si,s)) for si in ss])/2.
    return res

def vec2s(n):
    '''
    Transform a vector of length 3 or 4 to a pauli matrix.

    Args:
        n (int): a 1-D array of length 3 or 4 to specify the `direction` of spin.
    Returns:
        2 x 2 matrix.
    '''
    if len(n) > 4:
        raise Exception('length of vector %s too large.'%len(n))
    sl = ss[1:] if len(n) <= 3 else ss
    return reduce(lambda x,y:x+y, [si*ni for si,ni in zip(sl, n)])

def polar2vec(polar):
    r, theta, phi = polar
    vec = np.concatenate([(np.sin(theta) * np.cos(phi))[...,None], (np.sin(theta) * np.sin(phi))[...,None], np.cos(theta)[...,None]])*r
    return vec

def vec2polar(vec):
    '''transform a vector to polar axis.'''
    r = np.linalg.norm(vec, axis=-1, keepdims=True)
    theta = np.arccos(vec[...,2:3]/r)
    phi = np.arctan2(vec[...,1:2], vec[...,:1])
    res = np.concatenate([r, theta, phi], axis=-1)
    return res

def u2polar(vec):
    ratio = vec[1]/vec[0]
    theta = np.arctan(abs(ratio))*2
    phi = np.angle(ratio)
    return theta, phi

def polar2u(polar):
    theta, phi = polar
    return np.array([np.cos(theta/2.)*np.exp(-1j*phi/2.), np.sin(theta/2.)*np.exp(1j*phi/2.)])

# multiple bit construction

def CNOT(ibit, jbit, n):
    res = _([p0, I2], [ibit, jbit], n)
    res = res + _([p1, sx], [ibit, jbit], n)
    return res

def _(ops, locs, n):
    '''
    Put operators in a circuit and compile them.

    notice the big end are high loc bits!

    Args:
        ops (list): list of single bit operators.
        locs (list): list of positions.
        n (int): total number of bits.

    Returns:
        array: resulting matrix.
    '''
    if np.ndim(locs) == 0:
        locs = [locs]
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    locs = np.asarray(locs)
    locs = n - locs
    order = np.argsort(locs)
    locs = np.concatenate([[0], locs[order], [n + 1]])
    return _wrap_identity([ops[i] for i in order], np.diff(locs) - 1)


def _wrap_identity(data_list, num_bit_list):
    if len(num_bit_list) != len(data_list) + 1:
        raise Exception()

    res = np.eye(2**num_bit_list[0])
    for data, nbit in zip(data_list, num_bit_list[1:]):
        res = np.kron(res, data)
        res = np.kron(res, np.eye(2**nbit, dtype='complex128'))
    return res

def random_state(num_bit):
    '''generate random state.'''
    psi = np.random.randn(2**num_bit)*np.exp(2j*np.pi*np.random.random(2**num_bit))
    return psi/np.linalg.norm(psi)
