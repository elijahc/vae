from datetime import date

from keras.layers import Dense,Input,Lambda,Concatenate
from keras.models import Model,load_model
import keras.backend as K
import os

def build_dense(inputs,layers,activations):
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
    return x

class TandemVAEBuilder():
    def __init__(
        self,
        enc_layers=[500,500],
        y_dim=10,
        z_dim=2,
        activations='relu',
        dec_layers=None,):

        self.enc_layers = enc_layers
        if dec_layers is None:
            # assume symmetric
            self.dec_layers = enc_layers.copy()
            self.dec_layers.reverse()
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.activations='relu'
        self.layers = []

    def build(self,input_shape):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape,name='input')

        # Build encoder layers
        x = Dense(self.enc_layers[0],activation=self.activations)(self.input)
        self.layers.append(x)
        for num_units in self.enc_layers[1:]:
            x = Dense(num_units,activation=self.activations)(x)
            self.layers.append(x)

        # Build Latent representation layers
        encoded = x
        z_lat = Dense(self.z_dim,name='z_lat')(encoded)
        self.layers.append(z_lat)
        y_lat = Dense(self.y_dim,activation='softmax',name='y_lat')(encoded)
        self.layers.append(y_lat)
        lat_vec = Concatenate()([y_lat,z_lat])
        self.layers.append(lat_vec)

        # Build Decoder layers
        x = Dense(self.dec_layers[0],activation=self.activations)(lat_vec)
        self.layers.append(x)
        for num_units in self.dec_layers[1:]:
            x = Dense(num_units,activation=self.activations)(x)
            self.layers.append(x)
        recon = Dense(self.input_shape[0],activation='sigmoid',name='reconstruction')(x)

        self.output = recon

        return Model(self.input,self.output)


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

class CheungVae():
    def __init__(self,dataset='mnist'):
        if dataset=='mnist':
            mod = self.build_mnist()
            self.input = mod.inputs
            self.output = mod.outputs
            self.model = mod

    def build_mnist(self):
        inputs = Input(shape=(784,))
        x = Dense(500,activation='relu')(inputs)
        x = Dense(500,activation='relu')(x)
        y_class = Dense(10,activation='softmax')(x)
        z_mean = Dense(2,activation='linear')(x)

        lat_vec = Concatenate()([y_class,z_mean])

        dec = Dense(500,activation='relu')(lat_vec)
        dec = Dense(500,activation='relu')(dec)
        recon = Dense(784,activation='linear')(dec)

        mod = Model(inputs,recon)
        return mod
