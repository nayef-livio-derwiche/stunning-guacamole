# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:00:06 2017

@author: Nayef
"""
import pandas as pd
import numpy as np


def load_data():
    df = pd.read_excel("SP500.xlsx")
    prices = df.as_matrix()
    del(df)
    returns = np.diff(prices, axis=0)/prices[:-1,:]
    log_returns = np.log(prices[1:,:]/prices[:-1,:])
    
    return prices, returns, log_returns


def normalize(data):
    return (data - data.mean())/data.std(), data.mean(), data.std()


def data_generator(data, batch_size=16, shape = "flat", label = "class1", window=5):
    i = 0
    while True:
        if shape == "flat":
            data_shape = [window * data.shape[1]]
        elif shape == "tensor":
            data_shape = [window, data.shape[1]]
        x = []
        y = []
        for k in range(batch_size):
            x.append(np.reshape(data[i:i+window,:], data_shape))
            if label == "reg1":
                y.append(data[i+window,:])
            elif label == "class1":
                lab = np.zeros_like(data[i+window,:])
                lab[data[i+window,:] > 0] = 1
                y.append(lab)
            i += 1
            if i == data.shape[0] - window:
                i = 0
                if k < batch_size - 1:
                    break
        x = np.array(x)
        y = np.array(y)
        if label == "reg2":
            y = x
        yield x, y


def get_n_batches(data, batch_size, window):
    n_samples = data.shape[0] - window
    n_b = int(n_samples / batch_size)
    if n_samples % batch_size != 0:
        n_b += 1
    return n_b