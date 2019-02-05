from keras.callbacks import Callback,EarlyStopping
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import model_from_json,Model
from keras.layers import Concatenate,Dense,Input
import keras.backend as K
import tensorflow as tf

class PrintHistory(Callback):
    def __init__(self,print_keys=None):
        if print_keys is None:
            print_keys = ['G_loss','val_G_loss','val_D_real_loss']
        
        self.print_keys = print_keys

    def on_train_begin(self,logs={}):
        self.col_width = 5+7
        print('Epoch'.ljust(self.col_width),"".join(w.ljust(self.col_width) for w in self.print_keys))
        pass

    def on_epoch_end(self,epoch,logs={}):
        # col_width = max(len(word) for row in data for word in row) + 2  # padding
#         print(logs.keys())

        out = "".join(str(round(logs[k],4)).ljust(self.col_width) for k in self.print_keys)
        print((str(epoch)+':').ljust(self.col_width),out)
        

class Update_k(Callback):
    def __init__(self,k_var,k_lr=0.001,gamma_k=0.5):
        self.k_var = k_var
        self.k_lr = k_lr
        self.gamma_k = 0.5
    
    def on_train_begin(self,logs={}):
        pass
#         self.fake_inp = self.model.get_layer('fake_inp').output
#         self.D_real = self.model.get_layer('D_real').output
#         self.D_fake = self.model.get_layer('D_fake')
        
        
        
    def on_batch_end(self,batch,logs={}):
        
        self.G_loss = logs['G_loss']
        self.D_loss = logs['D_real_loss']
        
#         self.gamma_k = self.G_loss.mean()/self.D_loss.mean()
        imbalance = (self.gamma_k*self.D_loss.mean())+self.G_loss.mean()
        
        self.k_var = K.update_add(self.k_var, self.k_lr*imbalance)
        
    def on_epoch_end(self,epoch,logs={}):
        print('updated k to: ', K.get_value(self.k_var))
        
class FVEHistory(Callback):
    def __init__(self,enc_inp,enc_outp):
        self.enc_inp = enc_inp
        self.enc_outp = enc_outp
        
    def on_train_begin(self,logs={}):
        self.z_encoder = Model(self.enc_inp,self.enc_outp)

    def on_epoch_end(self,epoch,logs={}):
        pass