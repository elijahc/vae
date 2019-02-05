import numpy as np
import keras.backend as K
from keras.losses import mse, categorical_crossentropy

def sse(y_true,y_pred):
    return K.sum(K.square(y_pred-y_true),axis=-1)

def mse(y_true,y_pred):
            return K.mean(K.sum(K.square(y_pred-y_true),axis=-1),axis=0)

class KLDivergenceLoss():
    def __init__(self,log_sigma,mean,weight=1,name=None):
        self.log_sigma = log_sigma
        self.mean = mean
        self.weight = weight
#         self.units = log_
        self.__name__ = name
        if name is None:
            self.__name__ = 'DKL'
        
    def __call__(self,y_true,y_pred):

        kl_loss = -0.5*K.sum(1+(2*self.log_sigma)-K.square(self.mean)- K.square(K.exp(self.log_sigma)),axis=-1) 
        
        return self.weight*kl_loss

class ReconstructionLoss():
    def __init__(self,inputs,outputs,weight=1,batch_size=64,agg_type='sse'):
        self.inputs = inputs
        self.outputs = outputs
        self.weight = weight
        self.agg_type = agg_type
#         self.input_shape = K.int_shape(inputs)[1:]
        self.units = np.prod(K.int_shape(inputs)[1:])
        self.batch_size = batch_size
        self.__name__ = 'recon'
    
    def pixel_var(self,inp):
        return K.square(K.std(inp,axis=-1))
    
    def recon_loss(self,inp,outp):
#         if len(self.input_shape)>1:
#             inp = K.reshape(inp,(self.batch_size,np.prod(self.input_shape)))
#             outp = K.reshape(outp,(self.batch_size,np.prod(self.input_shape)))
            
        return K.sum(K.sum(K.sum(K.square(outp-inp),axis=-1),axis=-1),axis=-1)
    
    def __call__(self,y_true,y_pred):
        return mse(K.reshape(self.inputs,(-1,56*56)),K.reshape(y_pred,(-1,56*56)))

class XCov():
    def __init__(self,x,y,weight=1):
        self.weight=weight
        self.x = x
        self.y = y
#         self.batch_size = batch_size
#         self.units = K.int_shape(x)[-1]*K.int_shape(y)[-1]
        self.__name__ = 'xcov'
    
    def __call__(self,y_true,y_pred):
#         batch_size = K.int_shape(y_true)[0]
        norm_x = self.x - K.mean( self.x, axis=0 )
        norm_y = self.y - K.mean( self.y, axis=0 )
        
        # scaled_x = norm_x / K.sqrt( K.sum(K.square(norm_x),axis=0) )
        # scaled_y = norm_y / K.sqrt( K.sum(K.square(norm_y),axis=0) )

        xcov_mat = K.batch_dot(
            K.expand_dims( norm_x, axis=-1 ),
            K.expand_dims( norm_y, axis=-1 ),
            axes=2)
    
        xcov = K.sum(K.square(K.mean(xcov_mat,axis=0)))/2
        
        return self.weight*xcov

class XEnt():
    def __init__(self,y_class,weight=1):
        self.weight=weight
        self.y_class = y_class
#         self.batch_size = batch_size
#         self.units = K.int_shape(x)[-1]*K.int_shape(y)[-1]
        self.__name__ = 'xcov'
    
    def __call__(self,y_true,y_pred):
        return K.sum(self.weight*categorical_crossentropy(y_true,self.y_class))
    

def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1.
        return K.mean(y_true * K.square(y_pred) +
                      (1. - y_true) * K.square(K.maximum(margin - y_pred, 0.)))