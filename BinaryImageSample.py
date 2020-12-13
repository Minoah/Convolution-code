import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
from skimage.color import rgb2gray
from PIL import Image
import math
from sklearn.metrics import mean_squared_error

Data = np.load('./Datasets/Goku.npy')
# c = plt.imshow(Data,cmap='gray')

# plt.show(c)
