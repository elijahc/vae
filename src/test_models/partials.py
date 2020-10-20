from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, multiply, concatenate,Add
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
import keras.backend as K

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(momentum=0.8)(input)
    return Activation("relu")(norm)

def _bn_relu_dconv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def build_conv_generator(latent_dim,img_shape):
    img_rows,img_cols,img_channels = img_shape
    
    model = Sequential()
        
    rows,cols = int(img_rows/4),int(img_cols/4)

    model.add(Dense(128 * rows * cols, activation="relu", input_dim=latent_dim))
    model.add(Reshape((rows, cols, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
    model.add(Activation("tanh"))

    gen_input = Input(shape=(latent_dim,))
    img = model(gen_input)

    return Model(gen_input, img,name='Generator')
    
class GConvNet():
    def __init__(self,img_shape=(28,28,1),y_dim=35,z_dim=35):
        self.img_shape = img_shape
        self.img_rows,self.img_cols,self.channels = img_shape

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.latent_dim = y_dim+z_dim
        
    def build_generator(self):

        model = Sequential()
        
        rows,cols = int(self.img_rows/4),int(self.img_cols/4)

        model.add(Dense(128 * rows * cols, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((rows, cols, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        gen_input = Input(shape=(self.latent_dim,),name='GConv_input')
        img = model(gen_input)

        return Model(gen_input, img,name='Generator')
    
class GResNet():
    def __init__(self,output_shape=(28,28,1),y_dim=35,z_dim=35,n_residual_blocks=3):
        self.output_shape = output_shape
        self.img_rows,self.img_cols,self.channels = output_shape

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.latent_dim = y_dim+z_dim
        
        self.gf = 16
        self.n_residual_blocks = n_residual_blocks
        
    def build_generator(self,):
        def residual_block(layer_input, filters,block_reps=2):
            for i,r in enumerate(range(block_reps)):
                if i == 0:
                    layer_input = UpSampling2D()(layer_input)
                
                d = _bn_relu_dconv(filters=filters, kernel_size=3, strides=1, padding='same')(layer_input)
                d = _bn_relu_dconv(filters=filters, kernel_size=3, strides=1, padding='same')(d)
                d = Add()([d, layer_input])
            
            return d
        
#         model = Sequential()
        gen_input = Input(shape=(self.latent_dim,),name='GConv_input')
        
        rows,cols = int(self.img_rows/(2**self.n_residual_blocks)),int(self.img_cols/(2**self.n_residual_blocks))
        c1 = Dense(self.gf * rows * cols)(gen_input)
        c1 = Reshape((rows, cols, self.gf))(c1)
        r = _bn_relu(c1)
#         c1 = UpSampling2D()(c1)
        
        for i in range(self.n_residual_blocks):
            r = residual_block(r, self.gf)
            
        # Post-residual block
        c2 = BatchNormalization(momentum=0.8)(r)
        c2 = Activation('relu')(c2)

        c2 = Conv2DTranspose(1, kernel_size=3, strides=1, padding='same',activation='tanh')(c2)
#         c2 = Add()([c2, c1])
        
        return Model(gen_input,c2,name='Generator')