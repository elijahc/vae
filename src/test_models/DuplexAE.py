from keras.layers import Lambda, Dense, Activation, Input, Concatenate
from keras.models import Model
import keras.backend as K
from ..models import GResNet

class SliceAE(object):
    model_name="DuplexAE"

    def __init__(self,input_shape,
                 layer_units=[3000,2000],
                 y_dim=10,
                 z_dim=5,
                 activations='relu'):

        self.input_shape = input_shape
        self.layer_units = layer_units
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.activations = activations

    def encoder(self,x):
        layer_units = self.layer_units
        if isinstance(self.activations,str):
            activations = [self.activations]*len(layer_units)

        net = Dense(layer_units[0], activation=activations[0],name='dense_1')(x)
        i = 2
        for units,act in zip(layer_units[1:],activations[1:]):
            net = Dense( units, activation=act, name='dense_{}'.format(i) )(net)
            i+=1
        return net

    def decoder(self,):
        pass

    def build_model(self):
        """ Hyper parameters"""
        # bs = self.config.batch_size
        
        """ Graph Input """
        self.input = Input(shape=self.input_shape,name='encoder_input')

        """ Build Encoder """
        net = self.encoder(self.input)

        """ Build Latent Representations """
        self.z_rep = Dense(self.z_dim, activation='linear',name='z_enc')(net)
        self.y_rep = Dense(self.y_dim, activation='softmax',name='class')(net)

        latent_vec = Concatenate()([self.y_rep,self.z_rep])
        """ Build Decoder """


        """ Build Partials """
        self.E = Model(
            inputs  = self.input,
            outputs = [self.z_rep,self.y_rep,self.loc_rep])

        self.G = Model(
            inputs=self.G_input,
            outputs=self.G_output,
            name='G'
        )

        self.GS = Model(
            inputs=[self.GS_dx_input,self.GS_dy_input,self.GS_im_input],
            outputs=self.GS_output,
            name='GSliced'
        )

        """ Build Model """
        self.model = Model(
            inputs=self.input,
            outputs=[self.G(latent_vec),self.y_rep],
        )  

        return self.model