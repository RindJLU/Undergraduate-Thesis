# coding: utf-8
import numpy as np
from PIL import Image
from projectq import MainEngine
from projectq.ops import Measure, H
import matplotlib.pyplot as plt


def read_image():
    im = Image.open('/home/yufeng/Documents/undergraduate thesis/projects/quantum distance calculation/edge_detect/download.jpeg')
    # im.show()
    width, height = im.size
    im = im.convert('L')

    data = np.zeros([256, 256])
    data_image = im.getdata()
    data_image = (np.matrix(data_image, dtype='float').reshape(225, 225))/255
    data[0:225, 0:225] = data_image
    return data


def cal_wavefun(data):
    norm = np.linalg.norm(data)
    wave_fun = (data/norm).reshape(256*256, 1)
    return wave_fun


def simu(wave_fun):
    eng = MainEngine()
    qureg = eng.allocate_qureg(16)
    eng.flush()
    eng.backend.set_wavefunction(wave_fun, qureg)

    H | qureg[0]
    eng.flush()
    a, new_wavefun = eng.backend.cheat()
    Measure | qureg

    new_wavefun = np.array(new_wavefun).reshape(256, 256)
    return new_wavefun


def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


new_wavefun = (abs(simu(cal_wavefun(read_image()))))
# new_im = MatrixToImage(new_wavefun)
#
# new_im.show()
print(new_wavefun[166, 136] > 0)
new_image_data = np.zeros([256, 256])
edge_index = new_wavefun.max()/1.7
(row_num, col_num) = new_wavefun.shape
print(row_num, col_num)
for row in range(row_num):
    for col in range(col_num):
        if new_wavefun[row][col] > edge_index:
            new_image_data[row][col] = new_wavefun[row][col]*300

new_im = MatrixToImage(new_image_data)
# plt.imshow(new_image_data, cmap=plt.cm.gray, interpolation='nearest')
new_im.show()
# new_im.save('half_detect.bmp')


# image_data = open('/home/yufeng/Documents/undergraduate thesis/projects/quantum distance calculation/imagedata.txt', 'a')
# for i in new_wavefun:
#     k = ' '.join([str(j) for j in i])
#     image_data.write(k + "\n")
# image_data.write(' \n')
