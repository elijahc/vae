from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda,Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam,Nadam
from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf

from .partials import GResNet
from .encoders import build_conv_encoder
from .ops import lrelu,bn,linear,conv2d,center_of_mass,center_of_mass_crop

from ..keras_callbacks import PrintHistory
    
def sse(y_true,y_pred):
    y_shape = K.shape(y_pred)
    y_pred = K.reshape(y_pred,(y_shape[0],K.prod(y_shape[1:])))
    y_true = K.reshape(y_true,(y_shape[0],K.prod(y_shape[1:])))
        
    return K.sum(K.square(y_pred-y_true),axis=-1)
    

class CRDuplex(GResNet):
    def __init__(self,
                 img_shape=(28,28,1),
                 y_dim=35,z_dim=35,num_classes=10,
                 recon=1,xent=1,kernel_regularization=None,n_residual_blocks=3):
        
        self.num_classes = num_classes
        self.kernel_regularization = kernel_regularization
        self.n_residual_blocks = n_residual_blocks
        self.recon = recon
        self.xent = xent

        GResNet.__init__(self,img_shape,y_dim,z_dim)
        
        optimizer = Adam(0.00005, 0.5)
        losses = {
            'Generator':windowed_sse,
            'Classifier':'categorical_crossentropy'
        }
        loss_weights = {
            'Generator':self.recon,
            'Classifier':self.xent,
        }
        
        # Build and the discriminator and recognition network
        self.E, self.Q = self.build_enc_w_qnet()
        
        # Build and compile the recognition network Q
        self.Q.compile(loss=['categorical_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.G = self.build_generator()
        
        enc_input = Input(shape=self.img_shape,name='model_input')
        latent = self.E(enc_input)
        recon = self.G(latent)
        
        # The recognition network produces the predicted label
        pred_label = self.Q(enc_input)
        
        self.combined = Model(enc_input, [recon, pred_label])
        self.combined.compile(
            loss=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics={'Generator':'mse','Classifier':'accuracy'}
        )
        
    def build_enc_w_qnet(self):
        img = Input(shape=self.img_shape)
        embed_mod = build_conv_encoder(input_shape=self.img_shape,layers=[16,32,64,128],drop_rate=0.25)
        
        # z_lat_encoding
        x = embed_mod.layers[0](img)
        for l in embed_mod.layers[1:]:
            x = l(x)
        
        # z_lat_encoding
        z_lat = Dense(self.z_dim, activation='linear',name='z_dim')(x)

        # y_lat_encoding
        y_lat = Dense(self.y_dim, activation='linear',name='y_dim')(x)

        # Q net classifier
        q_net = Dense(128, activation='linear')(y_lat)
        label = Dense(self.num_classes, activation='softmax',name='label')(q_net)

        # Combined Latent Representation
        latent = Concatenate(name='latent')([y_lat,z_lat])

        # Return encoder (Encoder) and recognition network (Q)
        return Model(img, latent,name='Encoder'), Model(img, label,name='Classifier')
    
class CR_DAE(GResNet):
    def __init__(self,
        input_shape=(28,28,1),output_shape=(56,56,1),
        y_dim=500,z_dim=0,num_classes=10,
        recon=1,xent=1,kernel_regularization=None,n_residual_blocks=1):
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.kernel_regularization = kernel_regularization
        self.n_residual_blocks = n_residual_blocks

        GResNet.__init__(self,output_shape,y_dim,z_dim,n_residual_blocks)

        optimizer = Adam(0.00005, 0.5)
        loss_weights = {
                'Generator':recon,
                'Classifier':xent,        }

        # Build and the discriminator and recognition network
        self.E, self.Q = self.build_enc_w_qnet()

        # Build and compile the recognition network Q
        self.Q.compile(loss=['categorical_crossentropy'],
                optimizer=optimizer,
                metrics=['accuracy'])

        # Build the generator
        self.G = self.build_generator()
                        
        enc_input = Input(shape=self.input_shape,name='model_input')

        latent = self.E(enc_input)
        recon = self.G(latent)

        # The recognition network produces the predicted label
        pred_label = self.Q(enc_input)
        
        losses = {
                'Generator':sse,
                'Classifier':'categorical_crossentropy',
        }
        
        self.combined = Model(enc_input, [recon, pred_label])
        self.combined.compile(
                loss=losses,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics={'Generator':'mse','Classifier':'accuracy'}
            )
            
    def build_enc_w_qnet(self):

        img = Input(shape=self.input_shape)
        embed_mod = build_conv_encoder(input_shape=self.input_shape,layers=[16,32,64,128],drop_rate=0.25)
        
        # z_lat_encoding
        x = embed_mod.layers[0](img)
        for l in embed_mod.layers[1:]:
            x = l(x)
        
        # z_lat_encoding
        z_lat = Dense(self.z_dim, activation='linear',name='z_dim')(x)

        # y_lat_encoding
        y_lat = Dense(self.y_dim, activation='linear',name='y_dim')(x)

        # Q net classifier
        q_net = Dense(128, activation='linear')(y_lat)
        label = Dense(self.num_classes, activation='softmax',name='label')(q_net)

        # Combined Latent Representation
        latent = Concatenate(name='latent')([y_lat,z_lat])
        
        # Return encoder (Encoder) and recognition network (Q)
        return Model(img, latent,name='Encoder'), Model(img, label,name='Classifier')
    
    def g_loss(self,factor=4,win=28,):
        crop_size = tf.constant([win]*2,dtype=tf.int32)
        
        def loss_func(y_true,y_pred):
            y_shape = K.shape(y_true)
            
            y_true_mask = tf.where(y_true>-1,x=tf.ones_like(y_true),y=tf.zeros_like(y_true))
            centers = center_of_mass(y_true_mask)[:,:2]
            y_true_sm = tf.image.extract_glimpse(y_true,crop_size,offsets=centers,normalized=False,centered=False)
            y_pred_sm = tf.image.extract_glimpse(y_pred,crop_size,offsets=centers,normalized=False,centered=False)

            scale_difference = K.prod(y_shape[1:])/(win*win)

            error = sse(y_true,y_pred)/tf.cast(scale_difference,dtype=tf.float32)
            error += sse(y_true_sm,y_pred_sm)*factor

            return error / (factor+1)
    
        return loss_func