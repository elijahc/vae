from .models import TandemVAEBuilder
from .losses import *
import keras.backend as K
from keras.losses import categorical_crossentropy

class Trainer(object):
    def __init__(self,config,data_loader):
        self.config = config
        self.data_loader = data_loader
        self.dataset = config.dataset

        self.optimizer=config.optimizer
        self.batch_size = config.batch_size

        builder = TandemVAEBuilder(config.enc_layers,config.y_dim,config.z_dim)
        self.layer_outputs = builder.layers
        self.model = builder.build(input_shape=self.data_loader.input_shape)

        y_class = self.model.get_layer('y_lat').output
        z_lat = self.model.get_layer('z_lat').output

        self.loss_fns = []

        def acc(y_true,y_pred):
            return categorical_accuracy(y_true,y_class)

        def xentropy(y_true,y_pred,weight):
            return weight*categorical_crossentropy(y_true,y_class)

        def recon_mse(y_true,y_pred):
            return K.mean(K.sum(K.square(y_pred-self.model.input),axis=-1),axis=0)

        self.metrics = []

        if self.config.recon:
            self.recon_loss = ReconstructionLoss(self.model.input,self.model.output,weight=self.config.recon)

        if self.config.xcov:
            self.xcov = XCov(y_lat,z_lat)

        self.model.compile(optimizer=config.optimizer,loss=self.loss,metrics=[acc,recon_mse])

    def loss(self,y_true,y_pred):
        total_loss=0
        if self.config.recon:
            self.loss_fns.append(K.sum(self.recon_loss(y_true,y_pred)))
        if self.config.xcov:
            self.loss_fns.append( self.config.xcov*self.xcov(y_true,y_pred) )
        if self.config.xent:
            self.xent = K.sum( self.config.xent*categorical_crossentropy(y_true,y_class) )
            self.loss_fns.append( self.xent )

        for L in loss_fns:
            total_loss += L

        return total_loss

    def go(self,x,y,**kwargs):
        self.model.fit(x,y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            **kwargs)

