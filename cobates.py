from pylsl import StreamInlet, resolve_stream
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.image import imread
from scipy import signal
from scipy.fft import fftshift, fftfreq
import pandas as pd
import os
import time

import tensorflow as tf
MODEL_NAME ='8chanele.h5' 
model = tf.keras.models.load_model(MODEL_NAME)
streams = resolve_stream('type', 'EEG')

inlet = StreamInlet(streams[0])

def tes():
    
    for i in range(1):
           

            data_1 = []
            data_2 = []
            data_3 = []
            data_4 = []
            data_5 = []
            data_6 = []
            data_7 = []
            data_8 = []
            
            for i in range(250):
                sample, timestamp = inlet.pull_sample()
                
                data_1.append( sample[0:1])
                data_2.append( sample[1:2])
                data_3.append( sample[2:3])
                data_4.append( sample[3:4])
                data_5.append( sample[4:5])
                data_6.append( sample[5:6])
                data_7.append( sample[6:7])
                data_8.append( sample[7:8])
                
                 
    d1= np.array(data_1)
    d2= np.array(data_2)
    d3= np.array(data_3)
    d4= np.array(data_4)
    d5= np.array(data_5)
    d6= np.array(data_6)    
    d7= np.array(data_7)
    d8= np.array(data_8)
    tx = np.dstack((d1,d2,d3,d4,d5,d6,d7,d8))
    dg = tx.reshape(2000)
    
    
    #mengambil chanel 
    a=np.transpose(d1)
    #print(a[0].shape)
    ds =np.transpose(dg)
    #print(dg[0].shape)


    fig = plt.figure()
    cek =pd.DataFrame(dg)
    N = 250
    # sample spacing
    T = 1.0 / 250.0
    yf = fftshift(dg)
    xf = fftfreq(N, T)
    xf = fftshift(xf)
    fftx =1.0/N *np.abs(fftshift(yf))    

    f, t, Sxx = signal.spectrogram(fftx,nperseg=70, noverlap=13, 
    #fs=1000,
    return_onesided=False, scaling='density', axis=- 1, mode='psd')
    plt.pcolormesh(t, f, Sxx, shading='gouraud', 
    cmap='jet')
    #plt.show()
    
    plt.savefig('hasil')
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = fig.canvas.get_width_height()
    data = data.reshape((1,h, w, 3))
    #print(data.shape)
    plt.close()

    
    pred=model.predict(([data]))
    print(pred[0]) 
    
    if pred[0]>0.9 :
            print('nofill')
    else :
        print('filter7-13')


    
    
while True:
       tes()

    
   

    









