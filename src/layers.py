from datetime import date

from keras.layers import Dense
from keras.models import Model,load_model
from keras.engine.topology import Layer
import keras.backend as K
import os

DEFAULT_DIR = '/home/elijahc/projects/vae'
def build_dense(inputs,layers,activations=None):
    if isinstance(activations,list):
        if len(activations) is not len(layers):
            raise Exception('activation lists and layers list must be same len')
        acts = activations
    elif isinstance(activations,str):
        acts = [activations]*len(layers)
    else:
        raise Exception('activations must be either list or str')

    x = Dense(layers[0],activation=acts[0])(inputs)
    for num_units,act in zip(layers[1:],acts[1:]):
        x = Dense(num_units,activation=act)(x)
    outputs = x
    return outputs

class FiLM(Layer):
    def __init__(self, units, conditioner, **kwargs):
        self.units = units
        self.conditioner = conditioner
        self.conditioner_shape = K.int_shape(self.conditioner)
        super(FiLM, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.gamma_k = self.add_weight(name='gamma_kernel', 
                                      shape=(self.conditioner_shape[-1], self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.gamma_b = self.add_weight(name='gamma_bias', 
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
        
                

        
        self.beta_k = self.add_weight(name='beta_kernel', 
                                      shape=(self.conditioner_shape[-1],self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.beta_b = self.add_weight(name='beta_bias', 
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
        
        self.gamma = K.dot(self.conditioner,self.gamma_k)+self.gamma_b
        self.beta = K.dot(self.conditioner,self.beta_k)+self.beta_b
        super(FiLM, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x*self.gamma + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape