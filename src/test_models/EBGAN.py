import numpy as np

from keras.models import Model
from keras.layers import *
from ..losses import sse, mse
from ..models import ResBlock
import keras.backend as K
import tensorflow as tf


def split_tensor(x,pts=[],axis=-1):
    start_pts = [0]+pts[:-1]
    end_pts = pts
    split = [ x[:,s:e] for s,e in zip(start_pts,end_pts) ]
    split.append(x[:,pts[-1]:])
    
    return split

def resample(x):
    mean = K.mean(x,axis=0)
    std = K.std(x,axis=0)

    std_norm = K.random_normal(shape=K.shape(x),mean=0,stddev=1)
    
    return mean + std_norm*std

def gradient_penalty_loss(real_input,fake_input,D,lambda_g = 10):
    alpha = tf.random_uniform((K.shape(real_input)[0],1),0,1)

    interpolates = (alpha * real_input) + ((1-alpha)*fake_input)
#     print(K.int_shape(interpolates))
    D_interp = D(interpolates)

    interp_grad = K.gradients(D_interp,[interpolates])[0]

    grad_pen = lambda_g*K.mean(K.square(tf.norm(interp_grad,axis=-1)-1))
    
    return grad_pen

def generator_loss(y_true,y_pred):
    
    # WGAN-GP loss
    sse_loss = sse(y_true,y_pred)
    
    
    dg = D_fake

    D_L = -1*D_fake
    
    # EBGAN loss
#     D_L = sse(y_pred,G(Concatenate()(E(y_pred))))

#     D_pen = 0.001 * 
    return sse_loss

class Generator(object):
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

class Encoder(object):
    def __init__(self,input_shape,
                 y_dim=10,
                 z_dim=2,
                 layer_units=[3000,2000],
                 activations = 'relu',):
        
        if isinstance(activations,str):
            activations = [activations]*len(layer_units)
            
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.layer_units=layer_units
        self.activations = activations
        
        self.input = Input(shape=self.input_shape,name='Encoder_input')
    
    def build(self,x=None):
        if x is None:
            x = self.input
            
        net = Dense(self.layer_units[0], activation = self.activations[0])(x)
        
        for num_units,activ in zip(self.layer_units[1:],self.activations[1:]):
            net = Dense( num_units, activation=activ )(net)
            
        out_sz = self.y_dim + self.z_dim
        out_vec = Dense(out_sz,name='encoder_output')(net)
        out_tensors = Lambda(split_tensor,arguments={'pts':[self.y_dim]})(out_vec)
        
        self.model = Model(inputs=x,outputs=out_tensors)
        
        return out_tensors
    
    def __call__(self,x=None):
        if x is None:
            x = self.input
        else:
            self.input = x
        
        self.output = self.build(x)
        
        return self.output

class EBGAN(object):
    model_name="EBGAN"
    
    def __init__(self,input_shape,config):
        self.input_shape = input_shape
        self.config = config
       
    def decoder(self):
    
        
        self.D_fake = Activation(linear,name='D_fake')(self.E(self.G_output)[2])
        self.G = Model(
            inputs=G_input,
            outputs=self.G_output,
            name='G'
        )
    
    def build_model(self,input_shape):
        
        """ Hyper parameters"""
        bs = self.config.batch_size
        
        """ Graph Input """
        encoder = Encoder(input_shape=input_shape)
        self.input = encoder.input
        
        encoder_outputs = encoder.build(layer_units=self.config.enc_layers,activations='relu')
        
        self.y = Activation('softmax',name='y')(net_out[0])
        self.z = Activation('linear',name='z')(net_out[1])
        self.yz = Concatenate()([self.y,self.z])

        generator = Generator(y_dim = self.config.y_dim,
                              z_dim = self.config.z_dim,
                              dec_blocks = self.config.dec_blocks)
        
        G_input = Input(shape=(self.config.y_dim+self.config.z_dim,),name='decoder_input')
        decoder_output = (G_input)
        
        
        self.z_sampled = Lambda(resample,name='z_resampled')(self.z)
        
        self.shuffled_latent = Concatenate()([self.y,self.z_sampled])
        
        