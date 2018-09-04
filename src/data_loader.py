import os
import numpy as np
from keras.datasets import fashion_mnist, mnist

def get_loader(config):
    if config.dataset is 'mnist':
        (x_train,y_train),(x_test,y_test) = prepare_keras_dataset(mnist)
    elif config.dataset is 'fasion_mnist':
        (x_train,y_train),(x_test,y_test) = prepare_keras_dataset(fasion_mnist)

def prepare_keras_dataset(k_data):
    (x_train, y_train), (x_test, y_test) = k_data.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # print(x_train.shape)
    # print(x_test.shape)
    return (x_train,y_train),(x_test,y_test)

def create_shifted_dataset(num_train, num_test):
    # pre-allocate shifted inputs
    sx_train = np.empty((num_train,784*4))
    sx_test = np.empty((num_test,784*4))
