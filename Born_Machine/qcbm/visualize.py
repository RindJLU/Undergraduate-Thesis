import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .qclibd import polar2vec

class BlochPlot():
    '''
    Dynamic plot context, intended for displaying geometries.
    like removing axes, equal axis, dynamically tune your figure and save it.

    Args:
        figsize (tuple, default=(6,4)): figure size.
        filename (filename, str): filename to store generated figure, if None, it will not save a figure.

    Attributes:
        figsize (tuple, default=(6,4)): figure size.
        filename (filename, str): filename to store generated figure, if None, it will not save a figure.
        ax (Axes): matplotlib Axes instance.

    Examples:
        with BlochPlot() as bp:
            bp.ball.add_vectors([0.4, 0.6, 0.8])
    '''

    def __init__(self, figsize=(6, 4), filename=None, dpi=300):
        self.figsize = figsize
        self.filename = filename
        self.ax = None

    def __enter__(self):
        from qutip import Bloch3d, Bloch
        plt.ion()
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = Axes3D(self.fig, azim=-30, elev=15)
        self.ball = Bloch(axes=self.ax)
        self.ball.zlabel = [r'$|N\rangle$', r'$|S\rangle$']
        return self

    def __exit__(self, *args):
        self.ax.axis('off')
        self.ax.set_aspect("equal")
        #self.ball.make_sphere()
        self.ball.make_sphere()
        plt.close()
        if self.filename is not None:
            print('Press `c` to save figure to "%s", `Ctrl+d` to break >>' %
                  self.filename)
            pdb.set_trace()
            plt.savefig(self.filename, dpi=300)
        else:
            pdb.set_trace()

    def add_polar(self, theta, phi, color=None):
        self.ball.add_vectors(polar2vec([1, theta, phi]))
        self.ball.vector_color.append(color)

    def text(self, vec, txt, fontsize=14):
        if len(vec) == 2:
            vec = polar2vec(*vec)
        self.ax.text(vec[0], vec[1], vec[2], txt, fontsize=fontsize, va='center', ha='center')

if __name__ == '__main__':
    with BlochPlot() as bp:
        bp.add_polar(np.pi, 0, color='r')
        bp.text((np.pi, 0), 'dot')
