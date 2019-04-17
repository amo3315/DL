import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def next_batch(train_data, train_label, batch_size):
    index = [ i for i in range(0,len(train_label)) ]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0,batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_label[index[i]])
    return batch_data, batch_target

if __name__ == '__main__':
    n_steps = 400
    n_inputs = 1
    n_neurons = 150
    n_outputs = 6

    '''数据集制作'''
    vibr_normal = pd.read_csv('vibration_normal_0.csv')
    X_normal = vibr_normal["UB"].values.reshape(6000,400)
    vibr_labeled_normal = np.c_[X_normal, np.zeros((6000, 1), dtype=int)]

    vibr_fault_1 = pd.read_csv('vibration_1chip_0.csv')
    X_fault_1 = vibr_fault_1["UB"].values.reshape(6000,400)
    vibr_labeled_fault_1 = np.c_[X_fault_1, np.ones((6000, 1), dtype=int)]

    vibr_fault_2 = pd.read_csv('vibration_2chip_0.csv')
    X_fault_2 = vibr_fault_2["UB"].values.reshape(6000,400)
    vibr_labeled_fault_2 = np.c_[X_fault_2, np.ones((6000, 1), dtype=int)*2]

    vibr_fault_3 = pd.read_csv('vibration_3chip_0.csv')
    X_fault_3 = vibr_fault_3["UB"].values.reshape(6000,400)
    vibr_labeled_fault_3 = np.c_[X_fault_3, np.ones((6000, 1), dtype=int)*3]

    vibr_fault_4 = pd.read_csv('vibration_4chip_0.csv')
    X_fault_4 = vibr_fault_4["UB"].values.reshape(6000,400)
    vibr_labeled_fault_4 = np.c_[X_fault_4, np.ones((6000, 1), dtype=int)*4]

    vibr_fault_5 = pd.read_csv('vibration_5chip_0.csv')
    X_fault_5 = vibr_fault_5["UB"].values.reshape(6000,400)
    vibr_labeled_fault_5 = np.c_[X_fault_5, np.ones((6000, 1), dtype=int)*5]

    print (vibr_labeled_fault_5.shape)

    vibr_labeled = np.r_[vibr_labeled_normal, vibr_labeled_fault_1, vibr_labeled_fault_2, vibr_labeled_fault_3, vibr_labeled_fault_4, vibr_labeled_fault_5]

    train_set, test_set = train_test_split(vibr_labeled, test_size = 0.2, random_state = 42)

    X_train = train_set[:,0:400]
    X_test = test_set[:,0:400]
    y_train = train_set[:,400]
    y_test = test_set[:,400]

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X')
    y = tf.placeholder(tf.int64,[None], name = 'y')

    learning_rate = 0.001

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    logits = tf.layers.dense(states, n_outputs)

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()

    batch_size = 240*6
    n_epochs = 20

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(20):
                X_batch, y_batch = next_batch(X_train,y_train,batch_size)
                X_batch = np.array(X_batch).reshape((-1, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            X_test = X_test.reshape((-1, n_steps, n_inputs))
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

