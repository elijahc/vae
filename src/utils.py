import numpy as np

import keras.utils as kutils
from keras.datasets import mnist
import keras.backend as K
from keras.models import Model
from keras.callbacks import RemoteMonitor
from sklearn.manifold import Isomap

import requests
import datetime
import hashlib
import base64
import logging
import os

def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)

    for path in [config.log_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def process_mnist(normalize=True,verbose=False,y_onehot=True,flat=True,subset=None):
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28,28
    if flat:
        x_train = x_train.reshape(x_train.shape[0], img_rows*img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows*img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if normalize:
        x_train /= 255
        x_test /= 255

    if verbose:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    if y_onehot:
        y_train = kutils.to_categorical(y_train, 10)
        y_test = kutils.to_categorical(y_test, 10)

    # If subset, return subset of values
    if subset is not None:
        x_train = x_train[subset]
        y_train = y_train[subset]

    return (x_train,y_train), (x_test,y_test)

def get_encoder(model,encoder_slc=None):
    if encoder_slc is None:
        layer_sizes = [K.get_variable_shape(lay.output)[1] for lay in model.layers[1:]]
        encoded_layer_idx = np.argmax(np.array(layer_sizes))
        encoder_slc = slice(0,encoded_layer_idx-1)
        
    encoder_layers = model.layers[encoder_slc]
    return Model(inputs=encoder_layers[0].input,outputs=encoder_layers[-1].output)
    
def get_decoder(model,enc_input,decoder_slc=None):
    if decoder_slc is None:
        layer_sizes = [K.get_variable_shape(lay.output)[1] for lay in model.layers[1:]]
        encoded_layer_idx = np.argmax(np.array(layer_sizes))
        decoder_slc = slice(encoded_layer_idx-1,len(layer_sizes)+1)
    
    decoder_layers = model.layers[decoder_slc]
    x = decoder_layers[0](enc_input)
    for decoder_lay in decoder_layers[1:]:
        x = decoder_lay(x)
    dec_outputs = x 
    return Model(enc_input,dec_outputs)

def get_transcoder(enc,dec):
    return lambda x: dec.predict(enc.predict(x))

def gen_trajectory(x0,x1,delta=0.01,n=100):
    if isinstance(delta,float):
        delta = np.arange(0,1+delta,delta)
    # helper for linearly interpolating between two vectors
    x = lambda t: ((t*x1) + ((1-t)*x0))
    x_t = []
    for t in delta:
        x_t.append(x(t).tolist())

    return np.array(x_t)

def gen_sorted_isomap(X,C,rescale=False,**kwargs):
    X_iso = np.squeeze(Isomap(**kwargs).fit_transform(X))
    
    x_coord = zip(X,X_iso,C)
    x_coord_sort = sorted(x_coord,key=lambda x:x[1])
    x_t_out = []
    x_iso = []
    x_class = []
    for vec,iso,x_C in x_coord_sort:
        x_t_out.append(vec)
        x_iso.append(iso)
        x_class.append(x_C)
    
    return np.array(x_t_out),np.array(x_iso),np.array(x_class)

def gen_activation_functors(model,layer_idxs=None):
    if layer_idxs is None:
        layer_idxs = np.arange(len(model.layers)-1).tolist()
    inp = model.input
    lays = [model.layers[i] for i in layer_idxs]
    outputs = [lay.output for lay in lays]
    functors = [K.function([inp]+[K.learning_phase()],[out]) for out in outputs]

    return functors

class ElasticSearchMonitor(RemoteMonitor):
    def generate_session_id(self,ts):
        ts = str(ts).encode('utf-8')
        h = hashlib.md5(ts).digest()
        sid = base64.urlsafe_b64encode(h)[:6].decode('utf-8')
        return sid
    
    def on_train_begin(self,logs={}):
        self.losses = []
        # Log results every 20 batches
        self.interval = 50
        bad_sess_id = True
        while bad_sess_id:
            timest = round(datetime.datetime.now().timestamp())
            sess_id = self.generate_session_id(timest)
            if '-' not in sess_id:
                bad_sess_id=False
        
        self.session = sess_id
        print('session ID: ',self.session)
        
    def on_batch_end(self,batch,logs={}):
        if (batch-1)%self.interval==0:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")            
            logs['@timestamp']=ts
            logs['session']=self.session
            logs['user']='elijahc'
            
            payload = {k:str(v) for k,v in logs.items()}
#             print("{}{}/batch/".format(self.root,self.path))
#             print(payload)
            r = requests.post("{}{}-batch/doc".format(self.root,self.path),json=payload)
#             print(r.text)

    def on_epoch_end(self,epoch,logs={}):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")            
        logs['@timestamp']=ts
        logs['session']=self.session
        
        payload = {k:str(v) for k,v in logs.items()}
        
        r = requests.post("{}{}-epoch/doc".format(self.root,self.path),json=payload)
        
        print('posted epoch results!')
        print(payload)
