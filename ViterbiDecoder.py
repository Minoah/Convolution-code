import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.color import rgb2gray
from PIL import Image
import math
from sklearn.metrics import mean_squared_error
from BinaryImageSample import Data
from convEncoder import *

def HammingDistance(a,b):
    return np.sum(np.square(a-b))

def ViterbiDecoder(DeModulatedStream,mu):
    # Look-up Table for State Machine
    '''
    State_Table = [
            [0, 0, [0,0], 0],
            [0, 0, [1,1], 1],
            [1, 0, [1,0], 2],
            [1, 0, [0,1], 3],
            [2, 1, [1,1], 0],
            [2, 1, [0,0], 1],
            [3, 1, [0,1], 2],
            [3, 1, [1,0], 3]
    ]
    '''
    
    # MIT's State Table (Type-1)
    # State_Table = [
    #         [0, 0, [0,0], 0],
    #         [0, 0, [1,0], 1],
    #         [1, 0, [1,1], 2],
    #         [1, 0, [0,1], 3],
    #         [2, 1, [1,1], 0],
    #         [2, 1, [0,1], 1],
    #         [3, 1, [0,0], 2],
    #         [3, 1, [1,0], 3]
    # ]

    State_Table = [
            [0, 0, [0,0], 0],
            [0, 0, [1,1], 1],
            [1, 0, [1,0], 2],
            [1, 0, [0,1], 3],
            [2, 1, [1,1], 0],
            [2, 1, [0,0], 1],
            [3, 1, [0,1], 2],
            [3, 1, [1,0], 3]
    ]

    m = DeModulatedStream.shape[0]
    k = pow(2,mu)

    # Branch Metric is Hamming Distance.
    Path_Metric = np.zeros((k,m+1))
    Path_Metric[1:4,0] = 1e6 * np.ones((3,))

    # Creating Path Metrics.
    for i in range(1,m+1):
        for j in range(k):
            BM1 = HammingDistance(DeModulatedStream[i-1],State_Table[2*j][2])
            BM2 = HammingDistance(DeModulatedStream[i-1],State_Table[2*j + 1][2])
            V1 = Path_Metric[State_Table[2*j][3]][i-1] + BM1
            V2 = Path_Metric[State_Table[2*j + 1][3]][i-1] + BM2
            
            Path_Metric[j][i] = min(V1,V2)
    
    Output = []
    
    ind = np.argmin(Path_Metric[:,i])    
    for i in reversed(range(1,m+1)):
        a = State_Table[2*ind][3]
        b = State_Table[2*ind + 1][3]
        if (Path_Metric[:,i-1][a] <= Path_Metric[:,i-1][b]):
            ind_prev = a
        else:
            ind_prev = b
            
        for j in range(len(State_Table)):
            if (State_Table[j][0] == ind and State_Table[j][3] == ind_prev):
                Output.append(State_Table[j][1])
        ind = ind_prev
                
    Output.reverse()
    
    Output.pop()
    Output.pop()
    return np.array(Output)