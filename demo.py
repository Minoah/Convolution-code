import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.color import rgb2gray
from PIL import Image
import math
from sklearn.metrics import mean_squared_error
from BinaryImageSample import *
from convEncoder import *
from ViterbiDecoder import *

EncodedStream = ConvEncoder(Data.flatten(),2,2)
# print("encode",EncodedStream)

ModulatedStream = PAM2Encoder(EncodedStream)

TransmittedStream = Transmission(ModulatedStream,1e-3,20000,50)

# print("trans",TransmittedStream)

ReceivedStream = ReceivingData(TransmittedStream,10)
# print("receive",ReceivedStream)

Waveforms = PAM2Waveforms(1e-3,50,20000)

# print('Waveform', Waveforms)

DeModulatedStream = DeModulation(ReceivedStream,Waveforms)

# print('demoduled',DeModulatedStream)

e = np.sum(np.abs(DeModulatedStream-EncodedStream))

# print ('e =', str(e))

DecodedStream = ViterbiDecoder(DeModulatedStream,2)

print('decode', str(DecodedStream))

OutputImage = np.reshape(DecodedStream,Data.shape)
d = plt.imshow(OutputImage,cmap = 'gray')

# plt.show(c)
plt.show (d)

# c = plt.imshow(Data,cmap='gray')

# plt.show(c)