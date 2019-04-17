import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from  keras.layers import Conv1D,AveragePooling1D,Dense,Flatten
from  keras.models import Sequential
from sklearn.grid_search import GridSearchCV

def Data_mix(vibr_normal,vibr_fault_1,vibr_fault_2 ,vibr_fault_3,vibr_fault_4,vibr_fault_5):

    X_normal = np.array(vibr_normal["UB"]).reshape(6000,400)
    vibr_labeled_normal = np.c_[X_normal, np.zeros((6000, 1), dtype=int)]

    X_fault_1 = np.array(vibr_fault_1["UB"]).reshape(6000,400)
    vibr_labeled_fault_1 = np.c_[X_fault_1, np.ones((6000, 1), dtype=int)]

    X_fault_2 = np.array(vibr_fault_2["UB"]).reshape(6000,400)
    vibr_labeled_fault_2 = np.c_[X_fault_2, np.ones((6000, 1), dtype=int)*2]

    X_fault_3 = np.array(vibr_fault_3["UB"]).reshape(6000,400)
    vibr_labeled_fault_3 = np.c_[X_fault_3, np.ones((6000, 1), dtype=int)*3]

    X_fault_4 = np.array(vibr_fault_4["UB"]).reshape(6000,400)
    vibr_labeled_fault_4 = np.c_[X_fault_4, np.ones((6000, 1), dtype=int)*4]

    X_fault_5 = np.array(vibr_fault_5["UB"]).reshape(6000,400)
    vibr_labeled_fault_5 = np.c_[X_fault_5, np.ones((6000, 1), dtype=int)*5]

    print (vibr_labeled_fault_5.shape)
    vibr_labeled = np.r_[vibr_labeled_normal, vibr_labeled_fault_1, vibr_labeled_fault_2, vibr_labeled_fault_3, vibr_labeled_fault_4, vibr_labeled_fault_5]

    return vibr_labeled

def create_model(optimizer):
    lenet = Sequential()
    lenet.add(Conv1D(6,kernel_size=100 , strides=1, padding = 'valid', input_shape = (400,1)))
    #lenet.add(AveragePooling1D(pool_size=2,strides=2))
    lenet.add(Conv1D(16,kernel_size=100,strides=1,padding='valid'))
    #lenet.add(AveragePooling1D(pool_size=2,strides=2))
    lenet.add(Flatten())
    lenet.add(Dense(120))
    lenet.add(Dense(84))
    lenet.add(Dense(6,activation='softmax'))

    lenet.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return lenet

if __name__=='__main__':
    vibr_normal = pd.read_csv('vibration_normal_0.csv')
    vibr_fault_1 = pd.read_csv('vibration_1chip_0.csv')
    vibr_fault_2 = pd.read_csv('vibration_2chip_0.csv')
    vibr_fault_3 = pd.read_csv('vibration_3chip_0.csv')
    vibr_fault_4 = pd.read_csv('vibration_4chip_0.csv')
    vibr_fault_5 = pd.read_csv('vibration_5chip_0.csv')

    DataSet = Data_mix(vibr_normal,vibr_fault_1,vibr_fault_2 ,vibr_fault_3,vibr_fault_4,vibr_fault_5)

    train_set, test_set = train_test_split(DataSet, test_size = 0.2, random_state = 42)

    X_train = train_set[:,0:400].reshape(-1,400,1)
    X_test = test_set[:,0:400].reshape(-1,400,1)

    '''label->onehot encoding'''
    y_train = keras.utils.to_categorical(train_set[:,400])
    y_test = keras.utils.to_categorical(test_set[:,400])

    model = KerasClassifier(build_fn=create_model, verbose=0)

    batch_size = [240*6*10]
    epochs = [20]
    optimizer = ['sgd','RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs,)

    grid = GridSearchCV(estimator=model, param_grid=param_grid,)

    grid_result = grid.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    for params, mean_score, scores in grid_result.grid_scores_:

        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
#lenet.fit(X_train,y_train,batch_size=240*6,epochs=20,validation_data=[X_test,y_test])
