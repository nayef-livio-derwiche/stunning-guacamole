# -*- coding: utf-8 -*-
"""

train

predict

evaluate

"""
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, LSTM, Flatten, Reshape
from keras.regularizers import l2
from keras.optimizers import adam
from keras.losses import binary_crossentropy, mean_squared_error


class BaseModel():
    def __init__(self):
        pass
    
    def train(self, data_generator, epochs, steps_per_epoch, validation_data=None, lr=0.001, regularization=0., restart=True):
        pass
    
    def predict(self, data_generator, steps_per_epoch):
        pass
    
    def save_model(self, path):
        pass
    
    def load_model(self, path):
        pass


class LogisticRegressionModel(BaseModel):
    
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        
    def train(self, data_generator, epochs, steps_per_epoch, validation_data=None, val_steps=None, lr=0.001, regularization=0., restart=True):
        if restart or self.model is None:
            model = Sequential(name="LogisticRegression")
            model.add(Dense(self.output_shape[0], input_shape=self.input_shape, activation="sigmoid", kernel_regularizer=l2(regularization)))
            self.model = model
        self.model.compile(optimizer=adam(lr=lr), loss=binary_crossentropy)
        self.model.fit_generator(data_generator, steps_per_epoch, epochs,
                                 validation_data=validation_data, validation_steps=val_steps)

    def predict(self, data_generator, steps_per_epoch):
        if self.model is not None:
            pred = self.model.predict_generator(data_generator, steps_per_epoch)
            return pred
        else:
            print("Model not trained")

    def save_model(self, path):
        if self.model:
            self.model.save(path)
        else:
            print("Model not trained")

    def load_model(self, path):
        model = load_model(path)
        if model.name == "LogisticRegression":
            self.model = model
        else:
            print(model.name + " wrong Model, expected LogisticRegression")


class NeuralNetwork(BaseModel):

    def __init__(self, input_shape, output_shape, model_type="regression", hidden_layers=[32]):
        self.input_shape = input_shape
        self.output_shape = output_shape
        if model_type in ["regression", "classification"]:
            self.model_type = model_type
        else:
            print(str(model_type),
                  " is not a valid model type, regression or classification expected, regression is default")
            self.model_type = "regression"
        if self.model_type == "regression":
            self.model_loss = binary_crossentropy
        elif self.model_type == "classification":
            self.model_loss = mean_squared_error
        self.model_name = "NeuralNetwork_" + self.model_type
        self.hidden_layers = hidden_layers
        self.model = None

    def train(self, data_generator, epochs, steps_per_epoch, validation_data=None, val_steps=None, lr=0.001,
              regularization=0., restart=True):
        if restart or self.model is None:
            model = Sequential(name=self.model_name)
            for l in self.hidden_layers:
                if len(model.layers) == 0:
                    model.add(Dense(l, activation="relu", input_shape=self.input_shape, kernel_regularizer=l2(regularization)))
                else:
                    model.add(Dense(l, activation="relu", kernel_regularizer=l2(regularization)))
            if self.model_type == "regression":
                model.add(Dense(self.output_shape[0], kernel_regularizer=l2(regularization)))
            elif self.model_type == "classification":
                model.add(Dense(self.output_shape[0], activation="sigmoid", kernel_regularizer=l2(regularization)))
            self.model = model
        self.model.compile(optimizer=adam(lr=lr), loss=self.model_loss)
        self.model.fit_generator(data_generator, steps_per_epoch, epochs,
                                 validation_data=validation_data, validation_steps=val_steps)

    def predict(self, data_generator, steps_per_epoch):
        if self.model is not None:
            pred = self.model.predict_generator(data_generator, steps_per_epoch)
            return pred
        else:
            print("Model not trained")

    def save_model(self, path):
        if self.model:
            self.model.save(path)
        else:
            print("Model not trained")

    def load_model(self, path):
        model = load_model(path)
        if model.name == self.model_name:
            self.model = model
        else:
            print(model.name + " wrong Model, expected " + self.model_name)

class ConvNet(BaseModel):

    def __init__(self, input_shape, output_shape, model_type="regression", n_filters=32, kernel_size=5, n_layers=3):
        self.input_shape = input_shape
        self.output_shape = output_shape
        if model_type in ["regression", "classification"]:
            self.model_type = model_type
        else:
            print(str(model_type),
                  " is not a valid model type, regression or classification expected, regression is default")
            self.model_type = "regression"
        if self.model_type == "regression":
            self.model_loss = binary_crossentropy
        elif self.model_type == "classification":
            self.model_loss = mean_squared_error
        self.model_name = "CNN_" + self.model_type
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.kernel_size=kernel_size
        self.model = None

    def train(self, data_generator, epochs, steps_per_epoch, validation_data=None, val_steps=None, lr=0.001,
              regularization=0., restart=True):
        if restart or self.model is None:
            model = Sequential(name=self.model_name)
            for i in range(self.n_layers):
                if len(model.layers) == 0:
                    model.add(Conv1D(self.n_filters, self.kernel_size, input_shape=self.input_shape, activation="relu",
                                     padding="same", kernel_regularizer=l2(regularization)))
                else:
                    model.add(Conv1D(self.n_filters, self.kernel_size, activation="relu", padding="same",
                                     kernel_regularizer=l2(regularization)))
                model.add(MaxPooling1D(self.kernel_size))
            model.add(Flatten())
            if self.model_type == "regression":
                model.add(Dense(self.output_shape[0], kernel_regularizer=l2(regularization)))
            elif self.model_type == "classification":
                model.add(Dense(self.output_shape[0], activation="sigmoid", kernel_regularizer=l2(regularization)))
            self.model = model
        self.model.compile(optimizer=adam(lr=lr), loss=self.model_loss)
        self.model.fit_generator(data_generator, steps_per_epoch, epochs,
                                 validation_data=validation_data, validation_steps=val_steps)

    def predict(self, data_generator, steps_per_epoch):
        if self.model is not None:
            pred = self.model.predict_generator(data_generator, steps_per_epoch)
            return pred
        else:
            print("Model not trained")

    def save_model(self, path):
        if self.model:
            self.model.save(path)
        else:
            print("Model not trained")

    def load_model(self, path):
        model = load_model(path)
        if model.name == self.model_name:
            self.model = model
        else:
            print(model.name + " wrong Model, expected " + self.model_name)


class LSTMModel(BaseModel):

    def __init__(self, batch_size, timesteps, data_dim, lstm_dim, output_shape, model_type="regression", n_stacks=1):
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.data_dim = data_dim
        self.lstm_dim = lstm_dim
        self.output_shape = output_shape
        if model_type in ["regression", "classification"]:
            self.model_type = model_type
        else:
            print(str(model_type),
                  " is not a valid model type, regression or classification expected, regression is default")
            self.model_type = "regression"
        if self.model_type == "regression":
            self.model_loss = binary_crossentropy
        elif self.model_type == "classification":
            self.model_loss = mean_squared_error
        self.model_name = "LSTM_" + self.model_type
        self.n_stacks = n_stacks
        self.model = None

    def train(self, data_generator, epochs, steps_per_epoch, validation_data=None, val_steps=None, lr=0.001,
              regularization=0., restart=True):
        if restart or self.model is None:
            model = Sequential(name=self.model_name)
            model.add(LSTM(self.lstm_dim, return_sequences=True, stateful=True,
                           batch_input_shape=(self.batch_size, self.timesteps, self.data_dim),
                           kernel_regularizer=l2(regularization)))
            for i in range(self.n_stacks):
                model.add(LSTM(self.lstm_dim, return_sequences=True, stateful=True, kernel_regularizer=l2(regularization)))
            model.add(LSTM(self.lstm_dim, stateful=True, kernel_regularizer=l2(regularization)))
            if self.model_type == "regression":
                model.add(Dense(self.output_shape[0], kernel_regularizer=l2(regularization)))
            elif self.model_type == "classification":
                model.add(Dense(self.output_shape[0], activation="sigmoid", kernel_regularizer=l2(regularization)))
            self.model = model
        self.model.compile(optimizer=adam(lr=lr), loss=self.model_loss)
        self.model.fit_generator(data_generator, steps_per_epoch, epochs,
                                 validation_data=validation_data, validation_steps=val_steps)

    def predict(self, data_generator, steps_per_epoch):
        if self.model is not None:
            pred = self.model.predict_generator(data_generator, steps_per_epoch)
            return pred
        else:
            print("Model not trained")

    def save_model(self, path):
        if self.model:
            self.model.save(path)
        else:
            print("Model not trained")

    def load_model(self, path):
        model = load_model(path)
        if model.name == self.model_name:
            self.model = model
        else:
            print(model.name + " wrong Model, expected " + self.model_name)


class Autoencoder(BaseModel):

    def __init__(self, input_shape, layers=[128], embed_dim=32):
        self.input_shape = input_shape
        self.layers = layers
        self.embed_dim = embed_dim
        self.model_name = "Autoencoder_Dense"
        self.model = None

    def train(self, data_generator, epochs, steps_per_epoch, validation_data=None, val_steps=None, lr=0.001,
              regularization=0., restart=True):
        if restart or self.model is None:
            model = Sequential(name=self.model_name)
            for l in self.layers:
                if len(model.layers) == 0:
                    model.add(Dense(l, input_shape=self.input_shape, activation="relu",
                                    kernel_regularizer=l2(regularization)))
                else:
                    model.add(Dense(l, activation="relu", kernel_regularizer=l2(regularization)))
            model.add(Dense(self.embed_dim, activation="relu", kernel_regularizer=l2(regularization)))
            for l in self.layers:
                model.add(Dense(l, activation="relu", kernel_regularizer=l2(regularization)))
            model.add(Dense(self.input_shape[0], kernel_regularizer=l2(regularization)))
            self.model = model
        self.model.compile(optimizer=adam(lr=lr), loss=self.model_loss)
        self.model.fit_generator(data_generator, steps_per_epoch, epochs,
                                 validation_data=validation_data, validation_steps=val_steps)

    def predict(self, data_generator, steps_per_epoch):
        if self.model is not None:
            pred = self.model.predict_generator(data_generator, steps_per_epoch)
            return pred
        else:
            print("Model not trained")

    def save_model(self, path):
        if self.model:
            self.model.save(path)
        else:
            print("Model not trained")

    def load_model(self, path):
        model = load_model(path)
        if model.name == self.model_name:
            self.model = model
        else:
            print(model.name + " wrong Model, expected " + self.model_name)


class CNNAutoencoder(BaseModel):

    def __init__(self, input_shape, n_layers, kernel_size=32, n_filters=32, embed_dim=32):
        self.input_shape = input_shape
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.embed_dim = embed_dim
        self.model_name = "Autoencoder_Dense"
        self.model = None

    def train(self, data_generator, epochs, steps_per_epoch, validation_data=None, val_steps=None, lr=0.001,
              regularization=0., restart=True):
        if restart or self.model is None:
            model = Sequential(name=self.model_name)
            for i in range(self.n_layers):
                if len(model.layers == 0):
                    model.add(Conv1D(self.n_filters, self.kernel_size, input_shape=self.input_shape, padding="same",
                                     kernel_regularizer=l2(regularization)))
                else:
                    model.add(Conv1D(self.n_filters, self.kernel_size, padding="same",
                                     kernel_regularizer=l2(regularization)))
                model.add(MaxPooling1D(self.kernel_size))
            model.add(Flatten())
            model.add(Dense(self.embed_dim))
            model.add(Dense(model.layers[-3].output_shape[1] * model.layers[-3].output_shape[2]))
            model.add(Reshape(model.layers[-4].output_shape[1:]))
            for i in range(self.n_layers):
                model.add(UpSampling1D())
                model.add(Conv1D(self.n_filters, self.kernel_size, padding="same",
                                 kernel_regularizer=l2(regularization)))
                model.add(Conv1D(self.input_shape[1], 1, padding="same"))
            self.model = model
        self.model.compile(optimizer=adam(lr=lr), loss=self.model_loss)
        self.model.fit_generator(data_generator, steps_per_epoch, epochs,
                                 validation_data=validation_data, validation_steps=val_steps)

    def predict(self, data_generator, steps_per_epoch):
        if self.model is not None:
            pred = self.model.predict_generator(data_generator, steps_per_epoch)
            return pred
        else:
            print("Model not trained")

    def save_model(self, path):
        if self.model:
            self.model.save(path)
        else:
            print("Model not trained")

    def load_model(self, path):
        model = load_model(path)
        if model.name == self.model_name:
            self.model = model
        else:
            print(model.name + " wrong Model, expected " + self.model_name)