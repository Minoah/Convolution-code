import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.color import rgb2gray
from PIL import Image
import math
from sklearn.metrics import mean_squared_error
from BinaryImageSample import Data

def ConvEncoder(BitStream,mu,n):
    k = BitStream.shape[0]
    TeriBits = np.zeros(mu,)
    U = np.zeros((mu,),dtype=int)
    OutputStream = np.zeros(((k+mu),2))

    Input = np.append(BitStream,TeriBits)
    '''

    '''
    for i in range(Input.shape[0]):
        a = U[0]
        b = U[1]
        c = int(Input[i])

        v1 = c ^ a
        v2 = c ^ a ^ b
        OutputStream[i] = np.array([v2,v1])
        
        U[0] = c
        U[1] = a

    return OutputStream

def PAM2Encoder(EncodedStream):
    S1 = np.where(EncodedStream == 1,EncodedStream, -1)
    return S1

def Transmission(ModulatedStream,T,fc,N):
    S = np.sqrt(2)*np.cos(2*np.pi*fc*np.linspace(0,T,N))
    l,m = ModulatedStream.shape
    n = S.shape[0]
    Output = np.zeros((l,m,n))

    for i in range(l):
        for j in range(m):
            Output[i][j] = ModulatedStream[i][j] * S

    return Output

def ReceivingData(TransmittedStream,Variance):
    l,m,n = TransmittedStream.shape
    Output = np.zeros(TransmittedStream.shape)
    for i in range(l):
        for j in range(m):
            Output[i][j] = TransmittedStream[i][j] + np.random.normal(0,np.sqrt(Variance),size=n)

    return Output

def PAM2Waveforms(T,N,fc):
    Waveforms = np.zeros((2,N))

    Waveforms[0] = -np.sqrt(2)*np.cos(2*np.pi*fc*np.linspace(0,T,N))
    Waveforms[1] = np.sqrt(2)*np.cos(2*np.pi*fc*np.linspace(0,T,N))

    return Waveforms

def DeModulation(ReceivedStream,Waveforms):

    Output = np.zeros((ReceivedStream.shape[0],ReceivedStream.shape[1]))

    for i in range(ReceivedStream.shape[0]):
        for j in range(ReceivedStream.shape[1]):
            Output[i][j] = np.argmin(np.sum(np.multiply((ReceivedStream[i][j] - Waveforms),(ReceivedStream[i][j] - Waveforms)),axis=1))

    return Output