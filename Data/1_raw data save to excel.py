import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def dataToCsv(file,data1,data2):
    data1 = list(data1)
    data2 = list(data2)
    file_data1 = pd.DataFrame(data1,index=range(len(data1)),columns=['DE_time'])
    file_data2 = pd.DataFrame(data2,index=range(len(data2)),columns=['Label'])
    file_all = file_data1.join(file_data2,how='outer')

    file_all.to_csv(file)

# matlab文件名
Normal_data = sio.loadmat('97.mat')
Inner_race_7 = sio.loadmat('105.mat')
Inner_race_14 = sio.loadmat('169.mat')
Inner_race_21 = sio.loadmat('209.mat')
#Inner_race_28 = sio.loadmat('3001.mat')

#print(type(data))
#print(data.keys())
#print(data['X097_DE_time'])
print(len(Normal_data['X097_DE_time']),len(Inner_race_7['X105_DE_time']),len(Inner_race_14['X169_DE_time']),len(Inner_race_21['X209_DE_time']))

#将5个数据集转换为array
A = []; B = []; C = []; D = []
for j in Normal_data['X097_DE_time']:
    for i in j:
        A.append(float(i))
        Normal_data = A
Normal_data = np.array(Normal_data).reshape(-1,1)

for j in Inner_race_7['X105_DE_time']:
    for i in j:
        B.append(float(i))
        Inner_race_7 = B
Inner_race_7 = np.array(Inner_race_7).reshape(-1,1)

for j in Inner_race_14['X169_DE_time']:
    for i in j:
        C.append(float(i))
        Inner_race_14 = C
Inner_race_14 = np.array(Inner_race_14).reshape(-1,1)

for j in Inner_race_21['X209_DE_time']:
    for i in j:
        D.append(float(i))
        Inner_race_21 = D
Inner_race_21 = np.array(Inner_race_21).reshape(-1,1)
'''
for j in Inner_race_21['X3001_DE_time']:
    for i in j:
        D.append(float(i))
        Inner_race_28 = E
Inner_race_28 = np.array(Inner_race_28).reshape(-1,1)
'''
#为4类数据打标签
#Random_data = np.arange(243938).reshape(-1,1)
label_data_1 = np.zeros(243938).reshape(-1,1)
dataToCsv("Normal_data.csv",Normal_data,label_data_1)
label_data_2 = np.ones(121265).reshape(-1,1)
dataToCsv("Inner_race_7.csv",Inner_race_7,label_data_2)
label_data_3 = (np.ones(121846)*2).reshape(-1,1)
dataToCsv("Inner_race_14.csv",Inner_race_14,label_data_3)
label_data_4 = (np.ones(122136)*3).reshape(-1,1)
dataToCsv("Inner_race_21.csv",Inner_race_21,label_data_4)

#print(help(np.array))
