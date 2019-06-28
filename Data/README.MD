该数据集的数据均为json格式，读取方法调用下面的函数即可。
```
#读取数据
def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as json_file:
	    dic = json.load(json_file)
    return dic
```