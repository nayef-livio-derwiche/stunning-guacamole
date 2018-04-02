from models import LogisticRegressionModel, NeuralNetwork, ConvNet, LSTMModel
from data_utils import load_data, normalize, data_generator, get_n_batches
import numpy as np

prices, returns, log_returns = load_data()
X, mean, std = normalize(log_returns)

batch_size = 16
X_train = X[:-104]
X_val = X[-104:-52]
n_s_val = X_val.shape[0] / batch_size + 1
X_test = X[-52:]

n_stocks = X.shape[1]

#Model 1

w_1 = 6
n_s_train = get_n_batches(X_train, batch_size, w_1)
n_s_val = get_n_batches(X_train, batch_size, w_1)


gen_train_1 = data_generator(X_train, shape='flat', label='class1', window=w_1)
gen_val_1 = data_generator(X_val, shape='flat', label='class1', window=w_1)
model1 = LogisticRegressionModel((n_stocks * w_1,), (n_stocks,))
model1.train(gen_train_1, 2, steps_per_epoch=n_s_train,
             validation_data=gen_val_1, val_steps=n_s_val,
             lr=0.001, regularization=1., restart=True)

#Model 2

w_2 = 6
n_s_train = get_n_batches(X_train, batch_size, w_1)
n_s_val = get_n_batches(X_val, batch_size, w_1)


gen_train_2 = data_generator(X_train, shape='flat', label='class1', window=w_1)
gen_val_2 = data_generator(X_val, shape='flat', label='class1', window=w_1)
model1 = NeuralNetwork((n_stocks * w_1,), (n_stocks,), "classification", [32])
model1.train(gen_train_2, 2, steps_per_epoch=n_s_train,
             validation_data=gen_val_2, val_steps=n_s_val,
             lr=0.001, regularization=1., restart=True)

#Model 3

w_1 = 6
n_s_train = get_n_batches(X_train, batch_size, w_1)
n_s_val = get_n_batches(X_train, batch_size, w_1)


gen_train_1 = data_generator(X_train, shape='tensor', label='class1', window=w_1)
gen_val_1 = data_generator(X_val, shape='tensor', label='class1', window=w_1)
model1 = ConvNet((w_1, n_stocks), (n_stocks,), "classification", 32, 3, 1)
model1.train(gen_train_1, 2, steps_per_epoch=n_s_train,
             validation_data=gen_val_1, val_steps=n_s_val,
             lr=0.001, regularization=1., restart=True)

#Model 4

w_1 = 6
n_s_train = get_n_batches(X_train, batch_size, w_1)
n_s_val = get_n_batches(X_val, batch_size, w_1)


gen_train_1 = data_generator(X_train[:(n_s_train-1)*batch_size + w_1], shape='tensor', label='class1', window=w_1)
gen_val_1 = data_generator(X_val[:(n_s_val-1)*batch_size + w_1], shape='tensor', label='class1', window=w_1)
model1 = LSTMModel(batch_size, w_1, n_stocks, 32, (n_stocks,), model_type="classification", n_stacks=1)
model1.train(gen_train_1, 2, steps_per_epoch=n_s_train,
             validation_data=gen_val_1, val_steps=n_s_val,
             lr=0.001, regularization=1., restart=True)