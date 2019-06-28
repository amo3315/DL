import pandas as pd
import json

def datamix(Var,n):
    list = [[0 for j in range(n)]for i in range(int(len(Var)/n))]
    for i in range((int(len(Var)/n))):
        for j in range(n):
            list[i][j] = (Var[n*i+j])
    return list

def save_dict(filename,dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False)
        

vibr_0 = pd.read_csv('E:/vibration/Normal/vibration_0.csv')
cur_0 = pd.read_csv('D:/ML/data_handle_all/181112-01_Z&R_Normal/Normal_0/cur_0.csv')
qj_0 = pd.read_csv('D:/ML/data_handle_all/181112-01_Z&R_Normal/Normal_0/angleqj_0.csv')
tly_0 = pd.read_csv('D:/ML/data_handle_all/181112-01_Z&R_Normal/Normal_0/angletly_0.csv')
temp_0 = pd.read_csv('D:/ML/data_handle_all/181112-01_Z&R_Normal/Normal_0/temp_0.csv')

dict0 = {'vibr_UB':{},'vibr_LB':{},'temp1':{},'temp2':{},'temp3':{},'temp4':{},'curA':{},'curB':{},'curC':{},'qjX':{},'qjY':{},'tlyX':{},'tlyY':{}}

for i in range(600):
    dict0['vibr_UB'][i]=datamix(vibr_0["UB"],4000)[i]
    dict0['vibr_LB'][i]=datamix(vibr_0["LB"],4000)[i]
    dict0['temp1'][i]=datamix(temp_0["temp1"],5)[i]
    dict0['temp2'][i]=datamix(temp_0["temp2"],5)[i]
    dict0['temp3'][i]=datamix(temp_0["temp3"],5)[i]
    dict0['temp4'][i]=datamix(temp_0["temp4"],5)[i]
    dict0['curA'][i]=datamix(cur_0["A"],32)[i]
    dict0['curB'][i]=datamix(cur_0["B"],32)[i]
    dict0['curC'][i]=datamix(cur_0["C"],32)[i]
    dict0['qjX'][i]=datamix(qj_0["X"],20)[i]
    dict0['qjY'][i]=datamix(qj_0["Y"],20)[i]
    dict0['tlyX'][i]=datamix(tly_0["X"],20)[i]
    dict0['tlyY'][i]=datamix(tly_0["Y"],20)[i]
#print (dict)
save_dict('Normal_0.json',dict0)



