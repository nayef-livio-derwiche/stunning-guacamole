# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:15:57 2017

@author: Nayef
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

time_window = 6

"""
TO DO:
- neural network
- regularization
- batching and iterations
- proper train/test loop and iterations
- CNN (to do)
- CNN autoencoder (done)
- DBN (to do later)
- RF ? (to think)
- LSTM (to do asap!)
"""

"""
- un fichier de lib pour gestion des données
- un fichier par type de modèle
    - entrainement/test
    - prédiction
- un fichier pour la comparaison des modèles et des stratégies 
- un fichier démo/test
"""

def load_data():
    df = pd.read_excel("SP500.xlsx")
    #symbols = list(df.keys())
    data = df.as_matrix()
    del(df)
    for i in range(data.shape[0]-1):
        data[i,:] = np.log(data[i+1,:]/data[i,:])
    data = data[:-1,:]
    mean = data.mean()
    std = data.std()
    data = (data - mean)/std
    labels = np.zeros(data.shape)
    labels[data>0] = 1
    return data, labels, mean, std

def prepare_model_log_reg(data_shape, reg=10.):
    x = tf.placeholder(tf.float32, shape=[None, time_window * data_shape[1]])
    y_true = tf.placeholder(tf.float32, shape = [None, data_shape[1]])
    W = tf.Variable(tf.zeros([time_window * data_shape[1], data_shape[1]]))
    b = tf.Variable(tf.zeros([data_shape[1]]))
    
    z = tf.matmul(x,W) + b
    y = tf.sigmoid(z)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=z)) + reg * tf.nn.l2_loss(W)
    
    return loss, x, y, y_true, W, b
    
def prepare_model_neural(data_shape, hidden_layer_size=32, reg=10.):
    x = tf.placeholder(tf.float32, shape=[None, time_window * data_shape[1]])
    y_true = tf.placeholder(tf.float32, shape = [None, data_shape[1]])
    W1 = tf.Variable(tf.zeros([time_window * data_shape[1], hidden_layer_size]))
    b1 = tf.Variable(tf.zeros([hidden_layer_size]))

    W2 = tf.Variable(tf.zeros([hidden_layer_size, data_shape[1]]))
    b2 = tf.Variable(tf.zeros(data_shape[1]))
    
    z1 = tf.matmul(x,W1) + b1
    a1 = tf.sigmoid(z1)
    z2 = tf.matmul(a1,W2) + b2
    y = tf.sigmoid(z2)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=z2)) + reg * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
    
    return loss, x, y, y_true, [W1, W2], [b1, b2]

def data_generator(data, labels, batch_size = 1):
    i = 0
    while True:
        x = np.zeros([batch_size, time_window * data.shape[1]])
        y_true = np.zeros([batch_size, data.shape[1]])
        for k in range(batch_size):
            x[k,:] = np.reshape(data[i:i+time_window,:], [time_window * data.shape[1]])
            y_true[k,:] = labels[i+time_window,:]
            i += 1
            if i == data.shape[0] - time_window:
                i = 0
        yield x, y_true
            
def random_strategy(time, nb_stocks):
    return normalize_strategy(np.random.rand(time, nb_stocks))

def hold_all_strategy(time, nb_stocks):
    return normalize_strategy(np.ones([time, nb_stocks]))

def normalize_strategy(strategy):
    return (strategy.transpose()/strategy.sum(1)).transpose()

def evaluate_strategy(eval_data, strategy):
    strategy = normalize_strategy(strategy)
    R = np.zeros(eval_data.shape[0] + 1)
    R[0] = 1
    for i in range(eval_data.shape[0]):
        R[i+1] = R[i] * np.dot(np.exp(eval_data[i,:]), strategy[i,:])
    return R

sess = tf.Session()
data, labels, mean, std = load_data()
eval_data = data[-52:] * std + mean
train_data = data[:-52]
train_labels = labels[:-52]
test_data = data[-52:]
test_labels = labels[-52:] 
loss, X, Y, Y_true, W, b = prepare_model_log_reg(data.shape, reg=10.)

data_gen = data_generator(train_data, train_labels, batch_size=1)
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

for k in range(train_data.shape[0]*5):
    x, y = next(data_gen)
    sess.run(train, {X:x, Y_true:y})
print(sess.run(loss, {X:x, Y_true:y}))

strategy_random = random_strategy(eval_data.shape[0], eval_data.shape[1])
strategy_hold = hold_all_strategy(eval_data.shape[0], eval_data.shape[1])

strategy_logistic = np.ones(test_data.shape)
for i in range(strategy_logistic.shape[0] - time_window):
    historic_window = np.reshape(test_data[i:i+time_window,:], [time_window * test_data.shape[1]])
    strategy_logistic[i+time_window] = sess.run(Y,{X:[historic_window]})
preds = strategy_logistic
strategy_logistic = normalize_strategy(strategy_logistic)


strategy_logistic_train = np.ones(train_data.shape)
for i in range(strategy_logistic_train.shape[0] - time_window):
    historic_window = np.reshape(train_data[i:i+time_window,:], [time_window * train_data.shape[1]])
    strategy_logistic_train[i+time_window] = sess.run(Y,{X:[historic_window]})
preds2 = strategy_logistic_train
strategy_logistic_train = normalize_strategy(strategy_logistic_train)

R_rand = evaluate_strategy(eval_data, strategy_random)
R_hold = evaluate_strategy(eval_data, strategy_hold)
R_logistic = evaluate_strategy(eval_data, strategy_logistic)
R_logistic_train = evaluate_strategy(train_data * std + mean, strategy_logistic_train)
R_random_train = evaluate_strategy(train_data * std + mean, random_strategy(train_data.shape[0], train_data.shape[1]))

print(R_logistic[-1])
print(R_rand[-1])
print(R_hold[-1])
print("Cumulative excess returns:")
plt.plot(R_logistic - R_hold)

"""

def generator_dem():
    n = 0    
    while True:
        yield n
        n *= 2
"""   