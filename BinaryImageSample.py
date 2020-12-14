import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
from skimage.color import rgb2gray
from PIL import Image
import math
from sklearn.metrics import mean_squared_error


# Data = np.load('./Datasets/Goku.npy')
# c = plt.imshow(Data,cmap='gray')

# plt.show(c)

# _filename = './Datasets/LennaBinary'
# col = Image.open(filename + '.png')
# gray = col.convert('L')
# bw = gray.point(lambda x: 0 if x<128 else 255, '1')
# bw.save(filename + '2b.png')

# _filename = filename + '2b'
def cv2bi(filename):
    col = Image.open(filename + '.png')
    gray = col.convert('L')
    bw = gray.point(lambda x: 0 if x<128 else 255, '1')
    bw.save(filename + '2b.png')
    return filename + '2b'

def cv2npy(filename):
    img = Image.open(filename + '.png')
    Data = np.array(img, dtype='uint8')
    np.save(filename+'.npy', Data)
    return Data

def showImg(filename):
    img_array = np.load(filename+'.npy')
    c = plt.imshow(img_array, cmap='gray')
    plt.show(c)
