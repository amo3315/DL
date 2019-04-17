import keras
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def Data_mix(vibr,temp,cur,qj,tly):

    x_u = np.array(vibr["UB"]).reshape(600,4000)
    x_l= np.array(vibr["LB"]).reshape(600,4000)
    x_temp1 = np.array(temp["temp1"]).reshape(600,5)
    x_temp2 = np.array(temp["temp2"]).reshape(600,5)
    x_temp3 = np.array(temp["temp3"]).reshape(600,5)
    x_temp4 = np.array(temp["temp4"]).reshape(600,5)
    x_curA = np.array(cur["A"]).reshape(600,32)
    x_curB = np.array(cur["B"]).reshape(600,32)
    x_curC = np.array(cur["C"]).reshape(600,32)
    x_qjX = np.array(qj["X"]).reshape(600,20)
    x_qjY = np.array(qj["Y"]).reshape(600,20)
    x_tlyX = np.array(tly["X"]).reshape(600,20)
    x_tlyY = np.array(tly["Y"]).reshape(600,20)

    data = np.c_[x_u,x_l,x_temp1,x_temp2,x_temp3,x_temp4,x_curA,x_curB,x_curC,x_qjX,x_qjY,x_tlyX,x_tlyY]
    print (data.shape)

    return data

vibr_0_path = sys.argv[1];temp_0_path = sys.argv[2];cur_0_path = sys.argv[3];qj_0_path = sys.argv[4];tly_0_path = sys.argv[5]
vibr_1_path = sys.argv[6];temp_1_path = sys.argv[7];cur_1_path = sys.argv[8];qj_1_path = sys.argv[9];tly_1_path = sys.argv[10]
vibr_2_path = sys.argv[11];temp_2_path = sys.argv[12];cur_2_path = sys.argv[13];qj_2_path = sys.argv[14];tly_2_path = sys.argv[15]
vibr_3_path = sys.argv[16];temp_3_path = sys.argv[17];cur_3_path = sys.argv[18];qj_3_path = sys.argv[19];tly_3_path = sys.argv[20]
vibr_4_path = sys.argv[21];temp_4_path = sys.argv[22];cur_4_path = sys.argv[23];qj_4_path = sys.argv[24];tly_4_path = sys.argv[25]
vibr_5_path = sys.argv[26];temp_5_path = sys.argv[27];cur_5_path = sys.argv[28];qj_5_path = sys.argv[29];tly_5_path = sys.argv[30]

vibr_0 = pd.read_csv(vibr_0_path)
temp_0 = pd.read_csv(temp_0_path)
cur_0 = pd.read_csv(cur_0_path)
qj_0 = pd.read_csv(qj_0_path)
tly_0 = pd.read_csv(tly_0_path)

data_normal = Data_mix(vibr_0,temp_0,cur_0,qj_0,tly_0)
labeled_normal = np.c_[data_normal, np.zeros((600, 1), dtype=int)]

vibr_1 = pd.read_csv(vibr_1_path)
temp_1 = pd.read_csv(temp_1_path)
cur_1 = pd.read_csv(cur_1_path)
qj_1 = pd.read_csv(qj_1_path)
tly_1 = pd.read_csv(tly_1_path)

data_1chip = Data_mix(vibr_1,temp_1,cur_1,qj_1,tly_1)
labeled_1chip = np.c_[data_1chip, np.ones((600, 1), dtype=int)]

vibr_2 = pd.read_csv(vibr_2_path)
temp_2 = pd.read_csv(temp_2_path)
cur_2 = pd.read_csv(cur_2_path)
qj_2 = pd.read_csv(qj_2_path)
tly_2 = pd.read_csv(tly_2_path)

data_2chip = Data_mix(vibr_2,temp_2,cur_2,qj_2,tly_2)
labeled_2chip = np.c_[data_2chip, np.ones((600, 1), dtype=int)*2]

vibr_3 = pd.read_csv(vibr_3_path)
temp_3 = pd.read_csv(temp_3_path)
cur_3 = pd.read_csv(cur_3_path)
qj_3 = pd.read_csv(qj_3_path)
tly_3 = pd.read_csv(tly_3_path)

data_3chip = Data_mix(vibr_3,temp_3,cur_3,qj_3,tly_3)
labeled_3chip = np.c_[data_3chip, np.ones((600, 1), dtype=int)*3]

vibr_4 = pd.read_csv(vibr_4_path)
temp_4 = pd.read_csv(temp_4_path)
cur_4 = pd.read_csv(cur_4_path)
qj_4 = pd.read_csv(qj_4_path)
tly_4 = pd.read_csv(tly_4_path)

data_4chip = Data_mix(vibr_4,temp_4,cur_4,qj_4,tly_4)
labeled_4chip = np.c_[data_4chip, np.ones((600, 1), dtype=int)*4]

vibr_5 = pd.read_csv(vibr_5_path)
temp_5 = pd.read_csv(temp_5_path)
cur_5 = pd.read_csv(cur_5_path)
qj_5 = pd.read_csv(qj_5_path)
tly_5 = pd.read_csv(tly_5_path)

data_5chip = Data_mix(vibr_5,temp_5,cur_5,qj_5,tly_5)
labeled_5chip = np.c_[data_5chip, np.ones((600, 1), dtype=int)*5]


data_labeled = np.r_[labeled_normal, labeled_1chip,labeled_2chip,labeled_3chip,labeled_4chip,labeled_5chip]
print(data_labeled.shape)

train_set, test_set = train_test_split(data_labeled, test_size = 0.2, random_state = 42)

X_train = train_set[:,0:8196].reshape(-1,8196,1,1)
X_test = test_set[:,0:8196].reshape(-1,8196,1,1)

'''label->onehot encoding'''
y_train = keras.utils.to_categorical(train_set[:,8196])
y_test = keras.utils.to_categorical(test_set[:,8196])

from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from  keras.models import Sequential

lenet = Sequential()
lenet.add(Conv2D(6,(100,1),strides=1,padding='valid',input_shape=(8196,1,1)))
lenet.add(MaxPool2D((2,1),strides=(1,1)))
lenet.add(Conv2D(16,(100,1),strides=1,padding='valid'))
lenet.add(MaxPool2D((2,1),strides=(1,1)))
lenet.add(Flatten())
lenet.add(Dense(120))
lenet.add(Dense(84))
lenet.add(Dense(6,activation='softmax'))

lenet.compile(optimizer=keras.optimizers.Adam(lr=0.005),loss='categorical_crossentropy',metrics=['accuracy'])

batch_size = 1440
epochs = 10
history = lenet.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=[X_test,y_test])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Model_accuracy.png')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Model_loss.png')
