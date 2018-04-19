# coding: utf-8
import numpy as np
from PIL import Image
from projectq import MainEngine
from projectq.ops import Measure, H
import matplotlib.pyplot as plt


def read_image():
    im = Image.open('/home/yufeng/Documents/undergraduate thesis/projects/quantum distance calculation/edge_detect/download.jpeg')
    # width, height = im.size
    im = im.convert('L')

    data = np.zeros([256, 256])
    data_image = im.getdata()
    print(np.matrix(data_image, dtype='float').shape)
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


data = read_image()
origin_wavefun = cal_wavefun(data)
new_wavefun = simu(origin_wavefun)

# new_im.show()
# print(new_wavefun[166, 136] > 0)

new_image_data = np.ones([256, 256])
edge_index = origin_wavefun.max()/15

(row_num, col_num) = new_wavefun.shape

for row in range(row_num):
    for col in range(col_num):
        if col % 2 == 1:
            if new_wavefun[row][col] > edge_index:
                new_image_data[row][col] = 0
            elif new_wavefun[row][col] < -edge_index:
                new_image_data[row][col-1] = 0
        else:
            new_image_data[row][col] = 1


new_im = MatrixToImage(new_image_data[0:row_num][0:col_num])
# plt.imshow(new_image_data, cmap=plt.cm.gray, interpolation='nearest')
new_im.show()
# new_im.save('half_detect_smile.bmp')


# image_data = open('/home/yufeng/Documents/undergraduate thesis/projects/quantum distance calculation/edge_detect/imagedata.txt', 'a')
# for i in data:
#     k = ' '.join([str(j) for j in i])
#     image_data.write(k + "\n")
# image_data.write(' \n')
