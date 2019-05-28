import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def dataToCsv(file,data1):
    data1 = list(data1)
    file_data = pd.DataFrame(data1,index=range(len(data1)),columns=['DE_time'])
    file_data.to_csv(file)

def matToCsv(filename,save):
    lists = []
    data = sio.loadmat(filename)
    key = list(data.keys())[-1]
    for j in data[key]:
        for i in j:
            lists.append(float(i))
            data = lists
    data = np.array(data).reshape(-1,1)
    dataToCsv(save, data)

matToCsv(filename='3001.mat',save='Inner_race_28.csv')

