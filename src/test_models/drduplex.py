from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam,Nadam
from keras.utils import to_categorical
import keras.backend as K

from .partials import GResNet
from .encoders import build_dense_encoder

from ..keras_callbacks import PrintHistory
    
    
def sse(y_true,y_pred):
    y_shape = K.shape(y_pred)
    y_pred = K.reshape(y_pred,(y_shape[0],K.prod(y_shape[1:])))
    y_true = K.reshape(y_true,(y_shape[0],K.prod(y_shape[1:])))
        
    return K.sum(K.square(y_pred-y_true),axis=-1)

class DRDuplex(GResNet):
    def __init__(self,
                 img_shape=(28,28,1),
                 y_dim=35,z_dim=35,num_classes=10,
                 recon=1,xent=1,kernel_regularization=None,n_residual_blocks=3):
        self.num_classes = num_classes
        self.kernel_regularization = kernel_regularization
        self.n_residual_blocks = n_residual_blocks

        GResNet.__init__(self,img_shape,y_dim,z_dim)
        
        optimizer = Adam(0.00005, 0.5)
        losses = {
            'Generator':sse,
            'Classifier':'categorical_crossentropy'
        }
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

        # The encoder takes images as input
        # and generates an embedding and the corresponding digit of that label
        enc_input = Input(shape=self.img_shape,name='model_input')
        latent = self.E(enc_input)
        recon = self.G(latent)

        # For the combined model we will only train the generator
#         self.E.trainable = False

        # The discriminator takes generated image as input and determines validity
#         valid = self.discriminator(img)
        # The recognition network produces the label
        target_label = self.Q(enc_input)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(enc_input, [recon, target_label])
        self.combined.compile(
            loss=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics={'Generator':'mse','Classifier':'accuracy'}
        )
    
    def build_enc_w_qnet(self):
        img = Input(shape=self.img_shape)
        img_embedding = build_dense_encoder(input_shape=self.img_shape,
                                            layers=[3000,2000,500],
                                            kernel_regularization=self.kernel_regularization)(img)
        
         # z_lat_encoding
        z_lat = Dense(self.z_dim, activation='linear',name='z_dim')(img_embedding)

        # y_lat_encoding
        y_lat = Dense(self.y_dim, activation='linear',name='y_dim')(img_embedding)
        
        # Q net classifier
        q_net = Dense(128, activation='linear')(y_lat)
        label = Dense(self.num_classes, activation='softmax',name='label')(q_net)
        
        # Combined Latent Representation
        latent = Concatenate(name='latent')([y_lat,z_lat])

        # Return encoder (Encoder) and recognition network (Q)
        return Model(img, latent,name='Encoder'), Model(img, label,name='Classifier')
    
    def train(self,epochs,data_loader,batch_size,**kwargs):
        
        X_tr,y_tr = data_loader.training_data()
        X = X_tr
        y = {
            'Generator':data_loader.fg_train,
            'Classifier':y_tr,
        }
        
        print_props = ['loss','Generator_loss','Classifier_loss','Classifier_acc','val_Classifier_acc']
        print_labels = {
            'loss':'loss',
            'Generator_loss':'G_loss',
            'Classifier_loss':'C_loss',
            'Classifier_acc':'C_acc',
            'val_Classifier_acc':'val_C_acc'}
        print_history = PrintHistory(print_props,labels=print_labels)
        callbacks=[
            print_history
        ]
        
        history = self.combined.fit(X,y,epochs=epochs,batch_size=batch_size,
                                    callbacks=callbacks,
                                    **kwargs)
        
        