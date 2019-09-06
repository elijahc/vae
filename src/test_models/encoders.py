from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.regularizers import l1
import keras.backend as K

def build_dense_encoder(input_shape,layers=[3000,2000,500],kernel_regularization=None):
    k_reg = None
    if kernel_regularization is not None:
        k_reg = l1(kernel_regularization)
    
    # Shared layers between encoder and q
    model = Sequential(name='embedder')  
    model.add(Flatten(input_shape=input_shape))
        
    for l_sz in layers:
        model.add(Dense(l_sz,kernel_regularizer=k_reg))
        model.add(LeakyReLU(alpha=0.2))

                
    return model