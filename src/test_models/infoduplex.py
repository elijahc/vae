from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K

from .partials import GConvNet
from .encoders import build_dense_encoder
    
class INFODCDuplex(GConvNet):
    def __init__(self,img_shape=(28,28,1),y_dim=35,z_dim=35,num_classes=10):
        self.num_classes = num_classes

        GConvNet.__init__(self,img_shape,y_dim,z_dim)
        
        optimizer = Adam(0.0002, 0.5)
        losses = {
            'Generator':'mse',
            'Classifier':self.mutual_info_loss,
        }

        # Build and the discriminator and recognition network
        self.E, self.Q = self.build_enc_w_qnet()

        # Build and compile the recognition network Q
        self.Q.compile(loss=[self.mutual_info_loss],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.G = self.build_generator()

        # The encoder takes images as input
        # and generates a latent embedding
        enc_input = Input(shape=self.img_shape,name='image_input')
        latent = self.E(enc_input)
        
        # The classifier (Q) attempts to label the input image using the y_dim of latent
        target_label = self.Q(enc_input)

        # The generator uses the entire latent representation (y and z) to reconstruct the image
        recon = self.G(latent)
        
        # For the combined model we will only train the generator
#         self.E.trainable = False


        # The combined model  (stacked encoder and generator)
        self.combined = Model(enc_input, [recon, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)
    
    def build_enc_w_qnet(self):
        img = Input(shape=self.img_shape)
        img_embedding = build_dense_encoder(input_shape=self.img_shape,layers=[3000,2000,500])(img)
        # z_lat_encoding
        z_lat = Dense(self.z_dim, activation='linear',name='z_dim')(img_embedding)

        # y_lat_encoding
        y_lat = Dense(self.y_dim, activation='linear',name='y_dim')(img_embedding)
        
        # Q net classifier
        q_net = Dense(128, activation='relu')(y_lat)
        label = Dense(self.num_classes, activation='softmax',name='label')(q_net)
        
        # Combined Latent Representation
        latent = Concatenate(name='latent')([y_lat,z_lat])

        # Return encoder (Encoder) and recognition network (Q)
        return Model(img, latent,name='Encoder'), Model(img, label,name='Classifier')


    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy