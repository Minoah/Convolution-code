import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
from skimage.color import rgb2gray
from PIL import Image
import math
from sklearn.metrics import mean_squared_error

# Convert png sang anh binary image
def cv2bi(filename):
    col = Image.open(filename + '.png')
    gray = col.convert('L')
    bw = gray.point(lambda x: 0 if x<128 else 255, '1')
    bw.save(filename + '2b.png')
    return filename + '2b'

# convert anh png sang npy
def cv2npy(filename):
    img = Image.open(filename + '.png')
    Data = np.array(img, dtype='uint8')
    np.save(filename+'.npy', Data)
    return Data

# bieu dien anh
def showImg(filename):
    img_array = np.load(filename+'.npy')
    c = plt.imshow(img_array, cmap='gray')
    plt.show(c)
