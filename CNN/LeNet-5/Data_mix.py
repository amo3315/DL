import numpy as np
import pandas as pd

def dataToCsv(file,data1,data2):
    data1 = list(data1)
    data2 = list(data2)
    #data3 = list(data3)
    file_data1 = pd.DataFrame(data1,index=range(len(data1)),columns=['UB'])
    file_data2 = pd.DataFrame(data2,index=range(len(data2)),columns=['label'])
    #file_data3 = pd.DataFrame(data3,index=range(len(data3)),columns=['TIME'])
    file_ = file_data1.join(file_data2,how='outer')
    #file_all = file_.join(file_data3,how='outer')
    file_.to_csv(file)

vibr_normal = pd.read_csv('vibration_normal_0.csv')
X_normal = vibr_normal["UB"].reshape(6000,400)
vibr_labeled_normal = np.zeros((6000, 1), dtype=int)

vibr_fault_1 = pd.read_csv('vibration_1chip_0.csv')
X_fault_1 = vibr_fault_1["UB"].reshape(6000,400)
vibr_labeled_fault_1 = np.ones((6000, 1), dtype=int)

vibr_fault_2 = pd.read_csv('vibration_2chip_0.csv')
X_fault_2 = vibr_fault_2["UB"].reshape(6000,400)
vibr_labeled_fault_2 = np.ones((6000, 1), dtype=int)*2

vibr_fault_3 = pd.read_csv('vibration_3chip_0.csv')
X_fault_3 = vibr_fault_3["UB"].reshape(6000,400)
vibr_labeled_fault_3 = np.ones((6000, 1), dtype=int)*3

vibr_fault_4 = pd.read_csv('vibration_4chip_0.csv')
X_fault_4 = vibr_fault_4["UB"].reshape(6000,400)
vibr_labeled_fault_4 = np.ones((6000, 1), dtype=int)*4

vibr_fault_5 = pd.read_csv('vibration_5chip_0.csv')
X_fault_5 = vibr_fault_5["UB"].reshape(6000,400)
vibr_labeled_fault_5 = np.ones((6000, 1), dtype=int)*5

print (vibr_labeled_fault_5.shape)

X_labeled = np.r_[X_normal, X_fault_1, X_fault_2, X_fault_3, X_fault_4, X_fault_5]
y = np.r_[vibr_labeled_normal,vibr_labeled_fault_1,vibr_labeled_fault_2,vibr_labeled_fault_3,vibr_labeled_fault_4,vibr_labeled_fault_5]

dataToCsv('vibra_0.csv',X_labeled,y)



#train_set, test_set = train_test_split(vibr_labeled, test_size = 0.2, random_state = 42)
