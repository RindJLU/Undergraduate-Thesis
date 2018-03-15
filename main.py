# coding: utf-8
import projectq
from projectq.ops import H, Ry, Measure
import numpy as np
import matplotlib.pyplot as plt
import time


class EntML(object):
    """
    This class is used to solve classification problems, mainly to calculte the euclid distance in quantum way

    Parameters:
        data(Input)(ndarray): the vector representation of data
        data_dim(int): the dimension/length of the data
        data_amp(list): the amplitude of the vectors in data

        rot_theta(list): stores the rotation theta for the rotation operation

        ref_data(Input)(list): the vector representation of the reference data, usually two elements. Can be expanded.
        ref_data_amp(list): the amplitude of the vectors in ref_data
    """

    def __init__(self, data, ref_data):
        self.data = data
        self.data_dim = len(self.data[0])
        self.data_amp = []
        self.rot_theta = []

        self.ref_data = np.array(ref_data)
        self.ref_data_origin = np.array(ref_data)
        self.ref_data_amp = []

        self.distance = []
        self.col = []
        self.time = time.strftime("%y %m %d %H:%M:%S")

        self.eng = projectq.MainEngine()
        self.qureg = self.eng.allocate_qureg(1 + int(np.log2(self.data_dim)))
        self.eng.flush()
        # self.qubits = self.eng.allocate_qureg(1 + np.log2(self.data_dim))  # wrong doing, np.log2 is not integer

        '''
        transform classical data into quantum data
        '''
        for index in range(len(self.data)):
            self.data_amp.append(np.linalg.norm(self.data[index]))  # add the norm to the vector
            self.data[index] = self.data[index]/self.data_amp[-1]
            # normalize the vector so it can be stored in quantum states
        for index in range(len(self.ref_data)):
            self.ref_data_amp.append(np.linalg.norm(self.ref_data[index]))  # add the norm to the vector
            self.ref_data[index] = self.ref_data[index]/self.ref_data_amp[-1]
        for index in range(len(self.data_amp)):
            temp_theta = []
            for ref_index in range(len(self.ref_data_amp)):
                temp_theta.append(np.arcsin(abs(self.data_amp[index]/np.sqrt(self.data_amp[index]**2 + self.ref_data_amp[ref_index]**2))))
            self.rot_theta.append(temp_theta)
        # print(self.rot_theta)
        # print('the normalized reference state is '.format(self.ref_data))

    def sim(self):
        for new_data_index in range(len(self.data)):
            u_wave_fun = self.data[new_data_index]
            distance = []
            for RefData_index in range(len(self.ref_data)):
                v_wave_fun = self.ref_data[RefData_index]
                wave_fun = (np.kron(np.array([1, 0]), u_wave_fun) + np.kron(np.array([0, 1]), v_wave_fun))/np.sqrt(2)
                # print(wave_fun)  # check the wave_function

                self.eng.backend.set_wavefunction(wave_fun, self.qureg)
                H | self.qureg[-1]
                Ry(2 * self.rot_theta[new_data_index][RefData_index] - np.pi/2) | self.qureg[-1]
                self.eng.flush()
                prop = self.eng.backend.get_probability('1', [self.qureg[-1]])
                Measure | self.qureg

                distance.append(np.sqrt(2*prop*(self.data_amp[new_data_index]**2 + self.ref_data_amp[RefData_index]**2)))
            # self.col.append(distance[0] < distance[1])  # True for the vector near the first ref vector
            self.distance.append(distance)
            self.col.append(distance.index(min(distance)))
            # print('The distances are {}'.format(distance))

    def save(self):
        """
        save data: from http://blog.51cto.com/pmghong/1349978
        """
        data_file = open('/home/yufeng/Documents/undergraduate thesis/projects/quantum distance calculation/data.txt',
                         'a')
        data_file.write(' \n')
        data_file.write(self.time + "\n")
        self._tsave(self.ref_data_origin, data_file)  # save the ref data

        new_data = np.zeros([200, self.data.shape[1] + 2 + self.ref_data.shape[0]])
        new_data[:, 0:self.data.shape[1]] = Data
        new_data[:, self.data.shape[1]+1:-1] = self.distance
        new_data[:, -1] = test.col
        data_save = new_data
        self._tsave(data_save, data_file)

        data_file.close()

    def plot(self):
        col_dict = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']  # max 8 classes

        for ref_index in range(len(self.ref_data)):
            plt.scatter(self.ref_data[ref_index, 0]*self.ref_data_amp[ref_index], self.ref_data[ref_index, 1]*self.ref_data_amp[ref_index], c=col_dict[ref_index], marker='x')
        for new_index in range(len(self.data)):
            plt.scatter(self.data[new_index, 0]*self.data_amp[new_index], self.data[new_index, 1]*self.data_amp[new_index], c=col_dict[self.col[new_index]], marker='o')

        plt.xlabel('x'), plt.ylabel('y')
        plt.title('Classification')
        plt.savefig(fname=self.time)
        plt.show()

    def _tsave(self, data_save, data_file):
        """
        a function that can save list
        :param data_save: data would be saved
        :param data_file: target file
        :return: None
        """
        for i in data_save:
            k = ' '.join([str(j) for j in i])
            data_file.write(k + "\n")
        data_file.write(' \n')

def d_cal(vec1, vec2):
    """
    a function that calculate the euclid distance in a quantum way
    :param vec1: a list form vector, like [1, 2, 3, 5], totally 2^n elements
    :param vec2: a list form vector, with the same shape of vec1
    :return: distance between vec1 and vec2
    """
    # first normalize the two vectors:
    norm1 = np.linalg.norm(np.array(vec1))
    norm2 = np.linalg.norm(np.array(vec2))
    print(norm1, norm2)
    theta = np.arcsin(abs(norm1/np.sqrt(norm1**2 + norm2**2)))
    u_wavefun = vec1/norm1
    v_wavefun = vec2/norm2
    wavefun = (np.kron(np.array([1, 0]), np.array(u_wavefun)) + np.kron(np.array([0, 1]), np.array(v_wavefun)))/np.sqrt(2)
    # print(wavefun)

    # projectq setup
    eng = projectq.MainEngine()
    qureg = eng.allocate_qureg(int(np.log2(len(vec1))) + 1)
    eng.flush()

    eng.backend.set_wavefunction(wavefun, qureg)
    H | qureg[-1]
    Ry(2 * theta - np.pi / 2) | qureg[-1]
    eng.flush()
    prop = eng.backend.get_probability('1', [qureg[-1]])
    Measure | qureg

    print(prop)
    return np.sqrt(2*prop*(norm1**2 + norm2**2))


if __name__ == '__main__':
    print('please select the task you want to perform: ')
    print('1: classification, 2: calculate the distance between two vectors')
    check_index = input('your choice: ')

    if check_index == '1':
        ref_Data = [[1, 2.0],
                    [5, 3.0],
                    [3, 6.0],
                    [3, 0.0]]
        Data = 7 * np.random.rand(200, np.array(ref_Data).shape[1])
        test = EntML(Data, ref_Data)
        test.sim()
        if test.data_dim == 2:  # if the data is more than two dimension, it is not easy to visualize.
            test.save()
            test.plot()
        else:
            test.save()

    elif check_index == '2':
        v1 = [1.0, 2.0]
        v2 = [1.0, 2.0]
        print(d_cal(v1, v2))
    else:
        print('you have selected a wrong index, please recheck that!!!')
