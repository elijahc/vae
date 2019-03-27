from keras.layers import Lambda, Dense, Activation, Input, Concatenate
from keras.models import Model
import keras.backend as K
from ..models import GResNet

class SliceAE(object):
    model_name="SliceAE"

    def __init__(self,input_shape,
                 layer_units=[3000,2000],
                 y_dim=10,
                 z_dim=15,
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

    def masker(self,tensors):
        pass

    def slicer(self,tensors):
        dx = K.round(tensors[0],axis=-1)
        dy = K.round(tensors[1],axis=-1)
        im = tensors[2]

        sliced_im = K.slice(im,(dx,dy),(28,28))
        return sliced_im

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
        
        self.dx_rep = Dense(28, activation='softmax',name='loc_dx',)(self.z_rep)
        self.dy_rep = Dense(28, activation='softmax',name='loc_dy',)(self.z_rep)

        """ Build Decoder """
        self.G_input = Input(shape=(self.y_dim+self.z_dim,),name='G_input')
        self.G_output = GResNet(z_dim=15,flatten_out=False)(self.G_input)

        self.GS_dx_input = Input(shape=(28,),name='GS_dx_input')
        self.GS_dy_input = Input(shape=(28,),name='GS_dy_input')
        self.GS_im_input = Input(shape=(56,56,1,),name='GS_im_input')

        self.GS_output = Lambda(self.slicer)([self.GS_dx_input,self.GS_dy_input,self.GS_im_input])

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