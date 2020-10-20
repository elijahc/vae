from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda,Activation,Add
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

class CRD(GResNet):
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
                'Classifier':xent,
        }

        # Build and the discriminator and recognition network
        self.E, self.Q = self.build_enc_w_qnet()

        # Build and compile the recognition network Q
        self.Q.compile(loss=['categorical_crossentropy'],
                optimizer=optimizer,
                metrics=['accuracy'])

        # Build the generator
        self.G = self.build_generator()
                
#         - d_loss_fake
        
        enc_input = Input(shape=self.input_shape,name='model_input')

        latent = self.E(enc_input)
        recon = self.G(latent)
        
        self.D = self._descriminator_closure()

        # The recognition network produces the predicted label
        pred_label = self.Q(enc_input)
        
        losses = {
                'Generator':sse,
                'Classifier':'categorical_crossentropy',
        }
        
        self.EG = Model(enc_input, [recon, pred_label])
        self.EG.compile(
                loss=losses,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics={'Generator':'mse','Classifier':'accuracy'}
            )
        
        real_input = Input(shape=self.input_shape,name='real_input')
        self.auto_crop = center_of_mass_crop(real_input)
        
        self.real_im = self.auto_crop(real_input)
        self.fake_im = self.auto_crop(recon)  
        
        self.D_fake_logits = self.D(self.fake_im)
        self.D_real_logits = self.D(self.real_im)
#         self.d_loss_fake = K.mean(D_fake_logits)
        
#         _, D_real_logits = self._discriminator(self.real_im)        
        
#         self.d_loss = Add()([self.d_loss_fake,self.d_real_loss])

        
        
        self.EGD = Model([enc_input,real_input],[self.D_fake_logits,self.D_real_logits, pred_label],name='EGD')
        self.EGD.compile(
            loss={'Discriminator':self.d_loss,'Classifier':'categorical_crossentropy'},
            optimizer='adam'

        )
        
    def d_loss(self,y_true,y_pred):
        d_loss = - K.mean(self.D_real_logits)
        d_loss += K.mean(self.D_fake_logits)

        return d_loss
        
        
            
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
    
#     def g_loss(self,factor=1,win=28,):
#         crop_size = tf.constant([win]*2,dtype=tf.int32)
        
#         def loss_func(y_true,y_pred):
#             y_shape = K.shape(y_true)
            
#             y_true_mask = tf.where(y_true>-1,x=tf.ones_like(y_true),y=tf.zeros_like(y_true))
#             centers = center_of_mass(y_true_mask)
#             y_true_sm = tf.image.extract_glimpse(y_true,crop_size,offsets=centers,normalized=False,centered=False)
#             y_pred_sm = tf.image.extract_glimpse(y_pred,crop_size,offsets=centers,normalized=False,centered=False)

#             scale_difference = K.prod(y_shape[1:])/(win*win)

#             error = sse(y_true,y_pred)/tf.cast(scale_difference,dtype=tf.float32)
#             error += sse(y_true_sm,y_pred_sm)*factor

#             return error / (factor+1)
    
#         return loss_func
    def _descriminator_closure(self):
        layers = [
            Conv2D(64,kernel_size=4,strides=(2,2),activation=LeakyReLU()),
            Conv2D(128,kernel_size=4,strides=(2,2)),
            BatchNormalization(),
            LeakyReLU(),
            Flatten(),
            Dense(1024),
            BatchNormalization(),
            LeakyReLU(),
        ]
        def desc(x):
            net = layers[0](x)
            for l in layers[1:]:
                net = l(net)
                
            out_logit = Dense(1)(net)
            out = Activation('sigmoid')(out_logit)
            
            return out_logit
        
        return Lambda(desc,name='Discriminator')
    
    def _discriminator(self, x, is_training=True):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
        net = Reshape((-1,))(net)
        net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
        out_logit = Dense(1)(net)
            
        out = Activation('sigmoid')(out_logit)
            
        return out, out_logit, net
        
    def build_discriminator(self, crop_shape=(28,28,1)):
        pass
#         D_input = Input(shape=crop_shape)

#         verdict, D_logits,_ = self._discriminator(D_input, is_training=True)

#         return Model(D_input, [verdict, D_logits], name='Discriminator')
    
    def grad_penalty(self,y_true,y_pred):
        """ Gradient Penalty """
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb

        alpha = tf.random_uniform(shape=K.shape(y_true), minval=0.,maxval=1.)
        differences = y_pred - y_true # This is different from MAGAN
        interpolates = y_true + (alpha * differences)
        _,D_inter=self.D(interpolates)
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        
        return gradient_penalty