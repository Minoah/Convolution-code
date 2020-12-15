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
# file anh mau
filename = './Datasets/LennaBinary'
# convert anh png sang binary image
fn = cv2bi(filename)
# convert anh png sang dang npy
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
'''
Ham in du lieu
'''
def _print(title, value):
    print(title)
    print(value)
'''
Show array endecoded stream
'''
# _print("Encoded Stream:", EncodedStream)
'''
Show modulated stream: du lieu sau khi them chuyen ve dang song
'''
# _print("Modulated Stream:", ModulatedStream)

'''
Show transmitted stream: du lieu voi truyen
'''
# _print("Transmitted Stream:", TransmittedStream)
'''
Show Waveforms: bieu dien song mang
'''
# _print("Waveforms:", Waveforms)
'''
Show Demoduled stream: du lieu duoc giai dieu khoi song mang
'''
# _print("Demodulated Stream:", DecodedStream)
'''
Show error: so bit khac nhau giua luong du lieu giai ma voi du lieu giai dieu
'''
# _print("Error:", e)
'''
Show reiceived stream: bieu dien du lieu cuoi cung nhan duoc sau khi giai dieu va loai bo song mang
'''
# _print("Received Stream:", ReceivedStream)
'''
Show Error bits: bieu dien so bit khac nhau giua du lieu duoc giai boi viterbi voi du lieu anh goc
'''
# _print("Error Bits:", ErrorBits)
'''
Show percentage: bieu dien ty le phan tram giai ma dung
'''
# _print("Percentage:", Percentage)
'''
Show image decoded
'''
dimg = plt.imshow(OutputImage,cmap='gray')
plt.show(dimg)
'''
Show image original
'''
# showImg(fn)
