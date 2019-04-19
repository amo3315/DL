# C:/Python36 python
# _*_ coding:utf-8 _*_
import json
def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as json_file:
	    dic = json.load(json_file)
    return dic

data = load_dict('Normal_0.json')
print (len(data['vibr_UB']['0']))
for

