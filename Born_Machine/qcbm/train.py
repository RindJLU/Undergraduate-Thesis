import numpy as np
from scipy.optimize import minimize

from .contexts import ProjectQContext

def train(bm, theta_list, method, max_iter=1000, step_rate=0.1):
    '''
    train a Born Machine.
    
    Args:
        bm (QCBM): quantum circuit born machine training strategy.
        theta_list (1darray): initial parameters.
        method ('Adam'|'L-BFGS-B'):
            * L-BFGS-B: efficient, but not noise tolerant.
            * Adam: noise tolerant.
        max_iter (int): maximum allowed number of iterations.
        step_rate (float): learning rate for Adam optimizer.
        
    Returns:
        (float, 1darray): final loss and parameters.
    '''
    if bm.context is ProjectQContext:
        print('ProjecQ could be slow in training,\
since in scipy context, we can cache a lot gates to speed up calculation.')
    theta_list = np.array(theta_list)
    if method in ['Adam', 'GradientDescent', 'RmsProp']:
        import climin
        optimizer = eval('climin.%s'%method)(wrt=theta_list, fprime=bm.gradient,step_rate=step_rate)
        for info in optimizer:
            step = info['n_iter']
            loss = bm.mmd_loss(theta_list)
            print('step = %d, loss = %s'%(step, loss))
            if step == max_iter:
                break
            # for CloneQCBM
            if hasattr(bm, 'wf_origin'):
                wf = bm.wf_origin(theta_list)
                fidelity = np.abs(wf.conj().dot(bm._data))
                print('Fidelity = %s'%fidelity)
            if hasattr(bm, 'random_basis'):
                bm.random_basis()
        return bm.mmd_loss(theta_list), theta_list
    else:
        # for CloneQCBM
        def callback(x):
            if hasattr(bm, 'wf_origin'):
                wf = bm.wf_origin(x)
                fidelity = np.abs(wf.conj().dot(bm._data))
                print('Fidelity = %s'%fidelity)
            if hasattr(bm, 'random_basis'):
                bm.random_basis()
        res = minimize(bm.mmd_loss, x0=theta_list,
                       method=method, jac = bm.gradient, tol=1e-12,
                       options={'maxiter': max_iter, 'disp': 2, 'gtol':1e-12, 'ftol':0},
                       callback=callback,
                       )
        return res.fun, res.x
