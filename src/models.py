from datetime import date

from keras.layers import Dense,Input,Lambda
from keras.models import Model,load_model
import keras.backend as K
import os

class VAE_Builder():
    def __init__(
        self,
        enc_layers=[512,64],
        latent_dim=2,
        activations='relu',
        dec_layers=None,):

        self.enc_layers = enc_layers
        if dec_layers is None:
            # assume symmetric
            self.dec_layers = enc_layers.copy()
            self.dec_layers.reverse()
        self.latent_dim = 2
        self.activations='relu'
        self.layers = []


    def build(self, input_shape):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape,name='input')

        def chain_dense(inputs,layers,activations):
            if isinstance(activations,list):
                if len(activations) is not len(layers):
                    raise Exception('activation lists and layers list must be same len')
                acts = activations
            elif isinstance(activations,str):
                acts = [activations]*len(layers)
            else:
                raise Exception('activations must be either list or str')
            x = Dense(layers[0],activation=acts[0])(inputs)
            self.layers.append(x)
            for num_units,act in zip(layers[1:],acts[1:]):
                x = Dense(num_units,activation=act)(x)
                self.layers.append(x)
            return x

        self.enc_x = chain_dense(self.input,self.enc_layers,self.activations)

        z_mean = Dense(self.latent_dim,name='z_mean')(self.enc_x)
        z_log_sigma = Dense(self.latent_dim,name='z_log_sigma')(self.enc_x)

        def sampler(args):
            mean,log_stddev = args
            std_norm = K.random_normal(shape=(K.shape(mean)[0],self.latent_dim),mean=0,stddev=1)
            
            return mean + K.exp(log_stddev) * std_norm

        self.var_z = Lambda(sampler,name='var_z')([z_mean,z_log_sigma])
        self.layers.append(self.var_z)

        self.dec_x = chain_dense(self.var_z, self.dec_layers,activations=self.activations)
        self.recon_x = Dense(self.input_shape[0],activation='sigmoid',name='reconstruction')(self.dec_x)
        self.layers.append(self.recon_x)
        self.output = self.recon_x