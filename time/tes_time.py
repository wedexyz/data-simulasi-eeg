from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

import math
import os
import random
from playsound import playsound
import pandas as pd
from keras.models import Sequential,load_model


MODEL_NAME ='timeCNN.h5' 
model = tf.keras.models.load_model(MODEL_NAME)
FFT_MAX_HZ = 60
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

def tes():
    channel_datas = []
    for i in range(1):
            channel_data = []
            for i in range(250):
                sample, timestamp = inlet.pull_sample()
                channel_data.append(sample[:FFT_MAX_HZ])
                network_input = np.array(channel_data)
                channel_datas.append(channel_data)
    a = np.array(channel_datas)
    rms = np.sqrt(np.mean(np.square(a), axis = 0))
    print(a.shape)
    pred=model.predict(a.reshape(-1,250,8,1))
    #print(pred[0])
    if pred[[0]] ==[1.] :
        print("nofilter")
    else :
        print("filter 7-13")
    
while True: 
    tes ()
    

    
   

    









