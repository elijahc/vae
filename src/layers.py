from datetime import date

from keras.layers import Dense
from keras.models import Model,load_model
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