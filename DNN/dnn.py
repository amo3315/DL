import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from functools import partial

def next_batch(train_data, train_label, batch_size):
    index = [ i for i in range(0,len(train_label)) ]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0,batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_label[index[i]])
    return batch_data, batch_target

def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

def max_norm_regularizer(threshold, axes=1, name="max_norm",collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None # there is no regularization loss term
    return max_norm


if __name__ == '__main__':
    n_inputs = 400 * 1
    n_hidden1 = 300
    n_hidden2 = 300
    n_hidden3 = 100
    n_outputs = 6

    '''数据集制作'''
    vibr_normal = pd.read_csv('vibration_normal_0.csv')
    X_normal = vibr_normal["UB"].reshape(6000,400)
    vibr_labeled_normal = np.c_[X_normal, np.zeros((6000, 1), dtype=int)]

    vibr_fault_1 = pd.read_csv('vibration_1chip_0.csv')
    X_fault_1 = vibr_fault_1["UB"].reshape(6000,400)
    vibr_labeled_fault_1 = np.c_[X_fault_1, np.ones((6000, 1), dtype=int)]

    vibr_fault_2 = pd.read_csv('vibration_2chip_0.csv')
    X_fault_2 = vibr_fault_2["UB"].reshape(6000,400)
    vibr_labeled_fault_2 = np.c_[X_fault_2, np.ones((6000, 1), dtype=int)*2]

    vibr_fault_3 = pd.read_csv('vibration_3chip_0.csv')
    X_fault_3 = vibr_fault_3["UB"].reshape(6000,400)
    vibr_labeled_fault_3 = np.c_[X_fault_3, np.ones((6000, 1), dtype=int)*3]

    vibr_fault_4 = pd.read_csv('vibration_4chip_0.csv')
    X_fault_4 = vibr_fault_4["UB"].reshape(6000,400)
    vibr_labeled_fault_4 = np.c_[X_fault_4, np.ones((6000, 1), dtype=int)*4]

    vibr_fault_5 = pd.read_csv('vibration_5chip_0.csv')
    X_fault_5 = vibr_fault_5["UB"].reshape(6000,400)
    vibr_labeled_fault_5 = np.c_[X_fault_5, np.ones((6000, 1), dtype=int)*5]

    print (vibr_labeled_fault_5.shape)

    vibr_labeled = np.r_[vibr_labeled_normal, vibr_labeled_fault_1, vibr_labeled_fault_2, vibr_labeled_fault_3, vibr_labeled_fault_4, vibr_labeled_fault_5]

    train_set, test_set = train_test_split(vibr_labeled, test_size = 0.2, random_state = 42)

    X_train = train_set[:,0:400]
    X_test = test_set[:,0:400]
    y_train = train_set[:,400]
    y_test = test_set[:,400]

    X = tf.placeholder(tf.float32, shape= (None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name = 'y')
    training = tf.placeholder_with_default(False, shape=(), name='training')

    '''Dropout:50%'''
    dropout_rate = 0 # == 1 - keep_prob
    X_drop = tf.layers.dropout(X, dropout_rate, training=training)

    '''权重矩阵w初始化:2/根号n'''
    he_init = tf.contrib.layers.variance_scaling_initializer()

    '''正则化'''
    #scale = 0.001
    max_norm_reg = max_norm_regularizer(threshold=1.0)
    my_dense_layer = partial(tf.layers.dense, activation=tf.nn.relu,kernel_initializer=he_init, kernel_regularizer=max_norm_reg)

    with tf.name_scope('dnn'):#先Drop再标准化

        hidden1 = my_dense_layer(X, n_hidden1, name= 'hidden1')
        hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
        bn1 = tf.layers.batch_normalization(hidden1_drop, training=training, momentum=0.9)#批量标准化
        bn1_act = tf.nn.relu(bn1)

        hidden2 = my_dense_layer(bn1_act, n_hidden2, name='hidden2')
        hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
        bn2 = tf.layers.batch_normalization(hidden2_drop, training=training, momentum=0.9)#批量标准化
        bn2_act = tf.nn.relu(bn2)

        hidden3 = my_dense_layer(bn2_act, n_hidden3, name='hidden3')
        hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training=training)
        bn3 = tf.layers.batch_normalization(hidden3_drop, training=training, momentum=0.9)#批量标准化
        bn3_act = tf.nn.relu(bn3)

        logits_before_bn = my_dense_layer(bn3_act, n_outputs, activation = None,name="outputs",kernel_regularizer=None)
        logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits = logits)
        loss = tf.reduce_mean(xentropy, name="loss") # not shown
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #loss = tf.add_n([base_loss] + reg_losses, name="loss")

    learning_rate = 0.01

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits ,y ,1)#是否与真值一致 返回布尔值
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #tf.cast将数据转化为0,1序列

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 20
    batch_size = 240*6

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    clip_all_weights = tf.get_collection("max_norm")

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(20):
                X_batch, y_batch = next_batch(X_train, y_train, batch_size)
                sess.run([training_op,extra_update_ops],feed_dict={X:X_batch,y: y_batch})
                sess.run(clip_all_weights)
            acc_train = accuracy.eval(feed_dict={X:X_batch,y: y_batch})
            acc_test = accuracy.eval(feed_dict={X:X_test,y: y_test})

            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)


