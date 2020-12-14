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

filename = './Datasets/LennaBinary'
fn = cv2bi(filename)
cv2npy(fn) 

Data = np.load(fn+'.npy')

EncodedStream = ConvEncoder(Data.flatten(),2,2)

ModulatedStream = PAM2Encoder(EncodedStream)

TransmittedStream = Transmission(ModulatedStream,1e-3,20000,50)

ReceivedStream = ReceivingData(TransmittedStream,10)

Waveforms = PAM2Waveforms(1e-3,50,20000)

DeModulatedStream = DeModulation(ReceivedStream,Waveforms)

e = np.sum(np.abs(DeModulatedStream-EncodedStream))

DecodedStream = ViterbiDecoder(DeModulatedStream,2)

OutputImage = np.reshape(DecodedStream,Data.shape)

ErrorBits = np.sum(np.abs(DecodedStream - Data.flatten()))

Percentage = (1-float(ErrorBits)/float(DecodedStream.shape[0]))*100
def _print(title, value):
    print(title)
    print(value)
_print("Encoded Stream:", EncodedStream.shape[0])

# _print("Modulated Stream:", ModulatedStream)

# _print("Transmitted Stream:", TransmittedStream)

# _print("Waveform:", Waveform)
_print("Demodulated Stream:", DecodedStream.shape[0])
# _print("Error:", e)
# _print("Received Stream:", ReceivedStream)
_print("Error Bits:", ErrorBits)
_print("Percentage:", Percentage)

# dimg = plt.imshow(OutputImage,cmap='gray')
# plt.show(dimg)
# showImg(fn)
