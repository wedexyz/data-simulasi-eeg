from pylsl import StreamInlet, resolve_stream
import numpy as np
import time


import math
import os
import random
from playsound import playsound
import pandas as pd

streams = resolve_stream('type', 'EEG')

inlet = StreamInlet(streams[0])
def tes():
    channel_datas = []
    for i in range(1):
            channel_data = []
            for i in range(250):
                sample, timestamp = inlet.pull_sample()
                channel_data.append(sample[:8])
                network_input = np.array(channel_data)
                channel_datas.append(channel_data)
    ACTION = 'pola_2' 
    datadir = "time_series"
    actiondir = f"{datadir}/{ACTION}"
    

    if not os.path.exists(datadir):
        os.mkdir(datadir)
    if not os.path.exists(actiondir):
        os.mkdir(actiondir)

    a = np.array(channel_datas)
    #rms = np.sqrt(np.mean(np.square(a), axis = 0))
    print(a)
    print(f"saving {ACTION} data...,",a.shape)
    np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
    
for i in range(10) : 
    tes ()
    

    
   

    









