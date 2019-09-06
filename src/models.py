from datetime import date
import numpy as np
import keras
from keras.layers import Dense,Input,Lambda,Concatenate,Flatten,Reshape
from keras.layers import Conv2D,Conv2DTranspose,UpSampling2D,BatchNormalization,Activation,Add,AveragePooling2D,MaxPooling2D
from keras.models import Model,load_model
from keras.regularizers import l2,l1
import keras.backend as K
# from ..layers import FiLM
# from tabulate import tabulate
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

class ResBlock():
    def __init__(self,units,kernel_size,activation='relu',block_id=None,cond_norm=None,first_block=False):
        self.units=units
        self.kernel_size=kernel_size
        self.activation = activation
        self.block_id = block_id
        self.cond_norm=cond_norm
        self.first_block=first_block

    def _shortcut(self,input, residual, layer_num=1):
        """Adds a shortcut between input and residual block and merges them with "sum"
        """
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3

        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
        stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
        equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                            kernel_size=(1, 1),
                            strides=(stride_width, stride_height),
                            padding="valid",
                            kernel_initializer="he_normal",
                            kernel_regularizer=l2(0.0001))(input)

        return Add(name=self.name_layer('Add_{}'.format(layer_num)))([shortcut, residual])
    
    def _bn_relu(self,layer_num=1):
        def f(x):
            x = BatchNormalization(name=self.name_layer('BN_{}'.format(layer_num)))(x)
            x = Activation(self.activation,
                           name=self.name_layer('ReLU_{}'.format(layer_num)))(x)
            return x
        return f
    
    def res_layer(self,units,kernel_size,layer_num=1,conv_first=False):
        conv = Conv2DTranspose(units,
                name=self.name_layer('deconv_{}'.format(layer_num)),
                kernel_size=kernel_size,
                data_format='channels_last',padding='same')
        def f(x):
            if conv_first:
                x = conv(x)
                x = self._bn_relu(layer_num)(x)
            else:
                x = self._bn_relu(layer_num)(x)
                x = conv(x)      
            return x
        return f
    
    def __call__(self,input,conditioner=None):
        x = self.res_layer(units=self.units,kernel_size=(1,1),layer_num=1)(input)
        x = self.res_layer(units=self.units,kernel_size=(3,3),layer_num=2)(x)
#         x = self._bn_relu_conv(units=4*self.units,kernel_size=(1,1),layer_num=3)(x)
        F = self._shortcut(input=input,residual=x,layer_num=1)
#         x = Add(name=self.name_layer('Add'))([F,input])
        x = UpSampling2D()(F)

        return x
    
            
    def name_layer(self,layer_name):
        if self.block_id is not None:
            prefix = 'block_'+str(int(self.block_id))
            return prefix+'_'+layer_name
        else:
            return None

class EncConvBlock(object):
    def __init__(self,units,kernel_size,activation='relu',block_id=None,cond_norm=None,first_block=False):
        self.units=units
        self.kernel_size=kernel_size
        self.activation = activation
        self.block_id = block_id
        self.cond_norm=cond_norm
        self.first_block=first_block
    
    def conv_layer(self,units,kernel_size,activation='relu',strides=(1,1),layer_num=1):
        conv = Conv2D(units,
                name=self.name_layer('conv_{}'.format(layer_num)),
                kernel_size=kernel_size,strides=strides,
                # kernel_regularizer='',
                activation=activation,
                data_format='channels_last',padding='same')

        return conv
    
    def name_layer(self,layer_name):
        if self.block_id is not None:
            prefix = 'block_'+str(int(self.block_id))
            return prefix+'_'+layer_name
        else:
            return None
        
    def __call__(self,x_in):
        x = self.conv_layer(units=self.units,
                            kernel_size=(3,3),strides=(2,2),layer_num=1)(x_in)

        return x

class EncResBlock(ResBlock):

    def res_layer(self,units,kernel_size,strides=(1,1),layer_num=1,conv_first=False,bn_relu=True):
        conv = Conv2D(units,
            name=self.name_layer('conv_{}'.format(layer_num)),
            kernel_size=kernel_size,strides=strides,
            # kernel_regularizer='',
            data_format='channels_last',padding='same')
        
        def f(x):
            if conv_first:
                x = conv(x)
                if bn_relu:
                    x = self._bn_relu(layer_num)(x)
            else:
                if bn_relu:
                    x = self._bn_relu(layer_num)(x)
                x = conv(x)      
            return x
        return f
    
    def __call__(self,x_in):
        
        if self.first_block:
            x = self.res_layer(self.units,kernel_size=(3,3),layer_num=0,conv_first=True)(x_in)
            x = self.res_layer(self.units,kernel_size=(1,1),strides=(1,1),layer_num=1,bn_relu=False)(x)
#             x = self.res_layer(self.units,kernel_size(3,3),strides=(1,1),layer_num=2)
        else:
            x = self.res_layer(self.units,kernel_size=(3,3),strides=(2,2),layer_num=1)(x_in)
        
        x = self.res_layer(self.units,
                               kernel_size=(3,3),layer_num=2)(x)
        F = self._shortcut(input=x_in,residual=x,layer_num=1)

        return F


class EConvNet():
    def __init__(self,
                 kernel_size=(3,3),
                 activations='relu',
                 blocks=[2,2],
                 output_size=256,
                 ch=16,
                 CN=False,
                 y_dim=10,
                 z_dim=2,
                ):
        self.kernel_size=kernel_size
        self.activations=activations
        self.output_size=output_size
        self.blocks = blocks
        self.ch = ch
        self.CN = CN
        self.y_dim=y_dim
        self.z_dim=z_dim
        
    def build(self, x):
        
        for i,num_units in enumerate(self.blocks):
            x = EncConvBlock(
                units=num_units,
                kernel_size=self.kernel_size,
                activation=self.activations,
                )(x)
        
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=4)(x)
        y = Flatten()(x)
        outputs = Dense(self.output_size,activation='linear')(y)

        return outputs
    
    def __call__(self,x):
        self.input = x
        self.output = self.build(self.input)
        
        return self.output
        
class EResNet():
    def __init__(self,
                 kernel_size=(3,3),
                 activations='relu',
                 blocks=[2,2],
                 output_size=512,
                 ch=16,
                 CN=False,
                 y_dim=10,
                 z_dim=2,
                ):
        self.kernel_size=kernel_size
        self.activations=activations
        self.output_size=output_size
        self.blocks = blocks
        self.ch = ch
        self.CN = CN
        self.y_dim=y_dim
        self.z_dim=z_dim
        
    def build(self, x):
        
        for i,num_units in enumerate(self.blocks):
            x = EncResBlock(num_units,
                self.kernel_size,
                activation=self.activations,
                block_id=i+1,
                first_block=i==0,
                )(x)
        
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=4)(x)
        y = Flatten()(x)
        outputs = Dense(self.output_size,activation='linear')(y)

        return outputs
    
    def __call__(self,x):
        self.input = x
        self.output = self.build(self.input)
        
        return self.output
        

class EDenseNet():
    def __init__(self,
                 enc_layers=[500,500],
                 activations='relu',
                 dec_layers=None,
                 y_dim=10,
                 z_dim=2,
                 regularizer_weight=None,
                ):
        
        self.enc_layers = enc_layers
        
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.activations='relu'
        self.layers = []
        self.regularizer_weight=regularizer_weight
        
    def build(self,input_shape):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape,name='input')

        # Build encoder layers
        if self.regularizer_weight is not None:
            reg = l1(0.01)
        else:
            reg = None
        x = Dense(self.enc_layers[0],activation=self.activations,kernel_regularizer=reg)(self.input)
        self.layers.append(x)
        for num_units in self.enc_layers[1:]:
            if self.regularizer_weight is not None:
                reg = l1(0.01)
            else:
                reg = None
            x = Dense(num_units,activation=self.activations,kernel_regularizer=reg)(x)
            self.layers.append(x)
            
        self.encoded = x
        return x
    
class GDenseNet():
    def __init__(self,
                 enc_layers=[500,500],
                 activations='relu',
                 dec_layers=None,
                 output_size=56**2,
                 y_dim=10,
                 z_dim=2
                ):
        self.enc_layers=enc_layers
        self.activations=activations
        self.dec_layers=dec_layers
        self.output_size=output_size
        self.y_dim=y_dim
        self.z_dim=z_dim
        
        if dec_layers is None:
            # assume symmetric
            self.dec_layers = enc_layers.copy()
            self.dec_layers.reverse()
        
    def build(self,input_shape):
        # Build Decoder layers
        self.input_shape=input_shape
        self.input = Input(shape=self.input_shape)
        x = Dense(self.dec_layers[0],activation=self.activations)(self.input)
#         self.layers.append(x)
        for num_units in self.dec_layers[1:]:
            x = Dense(num_units,activation=self.activations)(x)
#             self.layers.append(x)
        recon = Dense(self.output_size,activation='tanh',name='reconstruction')(x)
        
        self.output = recon
        
        return Model(self.input,self.output)
    
class GResNet():
    def __init__(self,
                 kernel_size=(3,3),
                 activations='relu',
                 output_size=56**2,
                 dec_blocks=[2,1],
                 ch=16,
                 CN=False,
                 y_dim=10,
                 z_dim=2,
                 flatten_out=True,
                ):
        self.kernel_size=kernel_size
        self.activations=activations
        self.output_size=output_size
        self.dec_blocks = dec_blocks
        self.ch = ch
        self.CN = CN
        self.y_dim=y_dim
        self.z_dim=z_dim
        self.flatten_out=flatten_out
        
    def build(self, latent):
        
        dim = int(np.sqrt(self.output_size)/(2**len(self.dec_blocks)))
        feat_map = (dim,dim,self.dec_blocks[0]*self.ch)
        
#         if self.CN:
#             c = Lambda(lambda x: x[:, 0:self.y_dim])(latent)
#             z1 = Lambda(lambda x: x[:, self.y_dim:self.y_dim+3])(latent)
#             z2 = Lambda(lambda x: x[:, self.y_dim+3:self.y_dim+6])(latent)
#             z3 = Lambda(lambda x: x[:, self.y_dim+6:])(latent)
#             z_splits = [z1,z2,z3]
#             x = Dense(np.prod(feat_map),activation=self.activations)(z1)
#         else:
        x = Dense(np.prod(feat_map),activation=self.activations)(latent)
        x = Reshape(feat_map)(x)
        x = UpSampling2D()(x)

        for i,num_units in enumerate(self.dec_blocks[1:]):
            
            x = ResBlock(num_units*self.ch,
                         self.kernel_size,
                         activation=self.activations,
                         block_id=i+1,
#                          cond_norm=Concatenate()([c,z_splits[i+1]]),
                        )(x) 
            
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        self.recon = Conv2DTranspose(1,
                                self.kernel_size,
                                activation='tanh',
                                name='G_image',
                                data_format='channels_last',padding='same')(x)
        if self.flatten_out:
            x = Flatten(name='G_image_flat')(self.recon)
        else:
            x = self.recon
        
        return x
    
    def __call__(self,x):
        self.input = x
        self.output = self.build(self.input)
        
        return self.output

class EDense():
    def __init__(self,enc_layers=[500,500],activations='relu',y_dim=10,z_dim=2,regularizer_weight=None):
        self.enc_layers = enc_layers
        self.activations='relu'
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.layers=[]
        self.regularizer_weight=regularizer_weight
        
    def build(self,x):
        # Build encoder layers
        if self.regularizer_weight is not None:
            reg = l1(self.regularizer_weight)
        else:
            reg = None
        x = Dense(self.enc_layers[0],activation=self.activations,kernel_regularizer=reg)(x)

        self.layers.append(x)
        for num_units in self.enc_layers[1:]:
            if self.regularizer_weight is not None:
                reg = l1(self.regularizer_weight)
            else:
                reg = None
            x = Dense(num_units,activation=self.activations,kernel_regularizer=reg)(x)
            self.layers.append(x)
        
        return x
    
    def __call__(self,x):
        self.input = x
        self.output = self.build(self.input)
        return self.output
    
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
