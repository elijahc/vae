import os
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm as tqdm

def get_loader(config):
    if config.dataset is 'mnist':
        from keras.datasets import mnist
        return prepare_keras_dataset(mnist)
    elif config.dataset is 'fashion_mnist':
        from keras.datasets import fashion_mnist
        return prepare_keras_dataset(fashion_mnist)

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

class Shifted_Data_Loader():
    def __init__(self,dataset,scale=2):
        self.scale=scale
        self.dataset=dataset

        print('loading {}'.format(self.dataset))
        if dataset=='mnist':
            from keras.datasets import mnist
            (self.x_train, self.y_train),(self.x_test, self.y_test) = prepare_keras_dataset(mnist)
        elif dataset=='fashion_mnist':
            from keras.datasets import fashion_mnist
            (self.x_train, self.y_train),(self.x_test, self.y_test) = prepare_keras_dataset(fashion_mnist)
    
        num_train = len(self.y_train)
        num_test =  len(self.y_test)

        self.sx_train = np.empty((num_train, 784*self.scale*self.scale))
        self.sx_test =  np.empty((num_test, 784*self.scale*self.scale))

        self.delta_train = np.empty((num_train,2))
        self.delta_test = np.empty((num_test,2))
        
        print('making training data...')
        for i in tqdm(np.arange(num_train)):
            letter = self.x_train[i].reshape(28,28)
            new_im,offsets = self.shift_image(letter)
            self.sx_train[i] = new_im.reshape(1,4*784)
            self.delta_train[i] = offsets

        print('making testing data...')
        for i in tqdm(np.arange(num_test)):
            letter = self.x_test[i].reshape(28,28)
            new_im,offsets = self.shift_image(letter)
            self.sx_test[i] = new_im.reshape(1,4*784)
            self.delta_test[i] = offsets

    def shift_image(self,X):
        bg_size = (28*self.scale,28*self.scale)

        dx = int(np.random.randint(-10,10))+14
        dy = int(np.random.randint(-10,10))+14
        
        dx = max(dx,0)
        dx = min(dx,bg_size[0]-28)
        
        dy = max(dy,0)
        dy = min(dy,bg_size[0]-28)
    #     print(dx,dy)
        new_im = np.zeros(bg_size)
        new_im[dx:dx+28,dy:dy+28] = X
        
        return new_im,np.array([dx,dy])