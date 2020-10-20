from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
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

def build_conv_encoder(input_shape, layers=[16,32,64,128], k_sz=3, strides=2, drop_rate=None, bn_momentum=0.8):
    
    model = Sequential(name='embedders')
    
    for i,nk in enumerate(layers):
        if i == 0:
            # Create Conv layers with input_shape if first layer
            model.add(Conv2D(nk, kernel_size=k_sz, strides=strides, input_shape=input_shape, padding="same"))
        
        elif i == len(layers)-1:
            # Only stride 1 on last layer
            model.add(Conv2D(nk, kernel_size=k_sz, strides=2, padding="same"))
            
        else:
            model.add(Conv2D(nk, kernel_size=k_sz, strides=strides, padding="same"))
        
        model.add(LeakyReLU(alpha=0.2))
        
        if drop_rate is not None:
            model.add(Dropout(drop_rate))
            
        if bn_momentum is not None and i < len(layers)-1:
            model.add(BatchNormalization(momentum=bn_momentum))
    
    model.add(Flatten())
    
    return model
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dropout(0.25))
        
#         model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
#         model.add(ZeroPadding2D(padding=((0,1),(0,1))))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dropout(0.25))
#         model.add(BatchNormalization(momentum=0.8))
        
#         model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dropout(0.25))
#         model.add(BatchNormalization(momentum=0.8))
        
#         model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(Dropout(0.25))