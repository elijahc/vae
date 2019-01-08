from keras.callbacks import Callback,EarlyStopping
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import model_from_json,Model
from keras.layers import Concatenate,Dense,Input

class PrintHistory(Callback):
    def on_train_begin(self,logs={}):
        self.col_width = 5+7
        self.print_keys = ['G_loss','val_G_loss','val_class_acc']
        print('Epoch'.ljust(self.col_width),"".join(w.ljust(self.col_width) for w in self.print_keys))
        pass

    def on_epoch_end(self,epoch,logs={}):
        # col_width = max(len(word) for row in data for word in row) + 2  # padding
        
        out = "".join(str(round(logs[k],4)).ljust(self.col_width) for k in self.print_keys)
        print((str(epoch)+':').ljust(self.col_width),out)
        
class FVEHistory(Callback):
    def __init__(self,enc_inp,enc_outp):
        self.enc_inp = enc_inp
        self.enc_outp = enc_outp
        
    def on_train_begin(self,logs={}):
        self.z_encoder = Model(self.enc_inp,self.enc_outp)

    def on_epoch_end(self,epoch,logs={}):
        pass