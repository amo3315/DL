import json
import sys
import pandas as pd

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
        
vibr_path = sys.argv[1]
temp_path = sys.argv[2]
cur_path = sys.argv[3]
qj_path = sys.argv[4]
tly_path=sys.argv[5]


vibr_0 = pd.read_csv(vibr_path)
cur_0 = pd.read_csv(cur_path)
qj_0 = pd.read_csv(qj_path)
tly_0 = pd.read_csv(tly_path)
temp_0 = pd.read_csv(temp_path)
dict0 = {}


for i in range(600):
    dict0[i]['vibr_UB']=datamix(vibr_0["UB"],4000)[i]
    dict0[i]['vibr_LB']=datamix(vibr_0["LB"],4000)[i]
    dict0[i]['temp1']=datamix(temp_0["temp1"],5)[i]
    dict0[i]['temp2']=datamix(temp_0["temp2"],5)[i]
    dict0[i]['temp3']=datamix(temp_0["temp3"],5)[i]
    dict0[i]['temp4']=datamix(temp_0["temp4"],5)[i]
    dict0[i]['curA']=datamix(cur_0["A"],32)[i]
    dict0[i]['curB']=datamix(cur_0["B"],32)[i]
    dict0[i]['curC']=datamix(cur_0["C"],32)[i]
    dict0[i]['qjX']=datamix(qj_0["X"],20)[i]
    dict0[i]['qjY']=datamix(qj_0["Y"],20)[i]
    dict0[i]['tlyX']=datamix(tly_0["X"],20)[i]
    dict0[i]['tlyY']=datamix(tly_0["Y"],20)[i]
#print (dict)
save_dict('Normal_0.json',dict0)



