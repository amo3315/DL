# -*- coding: utf-8 -*-
import json
import numpy as np

def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as json_file:
	    dict = json.load(json_file)
    return dict

def load_data(filename):
    dict = load_dict(filename)
    vibr_UB = []
    vibr_LB = []
    temp_1 = []
    temp_2 = []
    temp_3 = []
    temp_4 = []
    cur_A = []
    cur_B = []
    cur_C = []
    qj_X = []
    qj_Y = []
    tly_X = []
    tly_Y = []
    for j in range(600):
        vibr_UB.append(dict[str(j)]['vibr_UB'])
        vibr_LB.append(dict[str(j)]['vibr_LB'])
        temp_1.append(dict[str(j)]['temp1'])
        temp_2.append(dict[str(j)]['temp2'])
        temp_3.append(dict[str(j)]['temp3'])
        temp_4.append(dict[str(j)]['temp4'])
        cur_A.append(dict[str(j)]['curA'])
        cur_B.append(dict[str(j)]['curB'])
        cur_C.append(dict[str(j)]['curC'])
        qj_X.append(dict[str(j)]['qjX'])
        qj_Y.append(dict[str(j)]['qjY'])
        tly_X.append(dict[str(j)]['tlyX'])
        tly_Y.append(dict[str(j)]['tlyY'])
    data = np.c_[vibr_UB,vibr_LB,temp_1,temp_2,temp_3,temp_4,cur_A,cur_B,cur_C,qj_X,qj_Y,tly_X,tly_Y]

    return data
