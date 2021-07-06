from pylsl import StreamInlet, resolve_stream
import numpy as np
from data import db
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt



streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

def tes():
    data_1 = []
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
    c1= np.array(data_1)    
    c2= np.array(data_2)
    c3= np.array(data_3)
    c4= np.array(data_4)
    c5= np.array(data_5)
    c6= np.array(data_6)    
    c7= np.array(data_7)
    c8= np.array(data_8)
    t = np.dstack((c1,c2,c3,c4,c5,c6,c7,c8))
    data_global = t.reshape(1,250,8)
    
    chanel1 =pd.DataFrame(c1)
    data_global_f = pd.DataFrame(data_global[0])

    print(chanel1,"data chanel 1")
    print(data_global_f, "bentuk  data global")
    print('bentuk shape',c1.shape,data_global.shape)

    plt.subplot(131)
    plt.plot(c1)
    plt.title("chanel 1")

    plt.subplot(132)
    plt.plot(c8)
    plt.title("chanel 8")

    plt.subplot(133)
    plt.plot(data_global[0])
    plt.title("total data")

    
    plt.show()
    
 

tes ()

    
   

    









