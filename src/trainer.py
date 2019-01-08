import os
from .models import TandemVAEBuilder
from .losses import *
from .keras_callbacks import *

import keras.backend as K
from keras.callbacks import Callback,EarlyStopping
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import model_from_json,Model
from keras.layers import Concatenate,Dense,Input,Lambda,Activation
from keras.activations import softmax,linear


tf_split_enc= lambda merge,y_dim: [merge[:,:y_dim],merge[:,y_dim:]]

def sse(y_true,y_pred):
    return K.sum(K.square(y_pred-y_true),axis=-1)

    
class Trainer(object):
    def __init__(self,config,data_loader,Ebuilder,Gbuilder,load_model=None):
        self.config = config
        self.optimizer = self.config.optimizer
        self.batch_size = self.config.batch_size
        if load_model is not None:
            self.load_model(mod_name=str(load_model))
        else:
            self.data_loader = data_loader
            self.dataset = config.dataset
            self.Ebuilder = Ebuilder
            self.Gbuilder = Gbuilder

#             self.layer_outputs = self.builder.layers

            self.build_model(input_shape=self.data_loader.input_shape)
        
    def build_encoder(self,input_shape):
        print('building encoder...')
        self.input = Input(shape=input_shape,name='input_image')
        E_output = self.Ebuilder(self.input)
#         self.y_lat = Dense(self.config.y_dim,activation='relu',name='y_lat')(E_output)
        self.enc_merge = Dense(self.config.y_dim+self.config.z_dim,name='enc_merge')(E_output)
        split_enc = Lambda(tf_split_enc,arguments={'y_dim':self.config.y_dim})(self.enc_merge)
        self.y_class = Activation(softmax,name='class')(split_enc[0])
        self.z_lat = Activation(linear,name='z_lat')(split_enc[1])
    
#         self.y_class = Dense(10,activation='softmax',name='class')(E_output)
#         self.z_lat = Dense(self.config.z_dim,name='z_lat')(E_output)
        self.real = Dense(2,activation='softmax',name='D')(E_output)
        self.E = Model(
            inputs=self.input,
            outputs=[self.y_class,self.z_lat,self.real],
            name='encoder'
        )
        
    def build_model(self,input_shape):
        self.build_encoder(input_shape)
        latent_vec = Concatenate()([self.y_class,self.z_lat])

        print('building decoder/generator...')
        G_input = Input(shape=(self.config.y_dim+self.config.z_dim,),name='G_input')
        self.G_output = self.Gbuilder(G_input)
        self.G = Model(
            inputs=G_input,
            outputs=self.G_output,
            name='G'
        )
        self.C = Model(
            inputs=self.input,
            outputs=self.y_class,
            name='C'
        )
        self.D = Model(
            inputs=self.input,
            outputs=self.real,
            name='D'
        )
        
        self.model = Model(
            inputs=self.input,
            outputs=[self.G(latent_vec),self.y_class,self.real],
        )  
        
    def compile_model(self):

        def mse(y_true,y_pred):
            return K.mean(K.sum(K.square(y_pred-y_true),axis=-1),axis=0)
        
        self.losses = {
            'class': 'categorical_crossentropy',
            'D': 'binary_crossentropy',
            'G': sse
        }
        lossWeights = {
            "class": self.config.xent,
            "D": 0.0,
            'G': self.config.recon
        }
        metrics = {
            'class': 'accuracy',
            'G': mse,
            
        }
        self.model.add_loss(self.config.xcov*XCov(self.y_class,self.z_lat)(self.y_class,self.z_lat)/self.config.batch_size)     
        self.model.compile(
            optimizer=self.config.optimizer,
            loss=self.losses,
            loss_weights=lossWeights,
            metrics=metrics,
        )
        
    
    def loss_func_generator(self,y_true,y_pred):
        term_weights = [self.config.recon, self.config.xcov,self.config.xent]
        
        y_class = self.y_lat
        z_lat = self.z_lat
        
        loss_funcs = [
            ReconstructionLoss(self.model.input,self.model.output,weight=1,batch_size=self.config.batch_size),
            XCov(y_class,z_lat),
            XEnt(y_class,1),
        ]
        
        for w,func in zip(term_weights,loss_funcs):
            if w is not None:
                yield K.sum(w*func(y_true,y_pred))
        
    def loss(self,y_true,y_pred):
        total_loss = 0
        
        for l in self.loss_func_generator(y_true,y_pred):
            total_loss += l
        
        return total_loss

    def go(self,x,y,**kwargs):
        print_history = PrintHistory()
        callbacks=[
            print_history
        ]
        if self.config.monitor is not None:
            early_stop = EarlyStopping(monitor=self.config.monitor,min_delta=self.config.min_delta,patience=10,restore_best_weights=True)
            callbacks.append(early_stop)
        
        history = self.model.fit(x,y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            **kwargs,
        )
        
        
        
        return history
    
    def go_gen(self,gen,**kwargs):
        print_history = PrintHistory()
        callbacks=[
            print_history
        ]
        if self.config.monitor is not None:
            early_stop = EarlyStopping(monitor=self.config.monitor,min_delta=self.config.min_delta,patience=10,restore_best_weights=True)
            callbacks.append(early_stop)
        
        self.history = self.model.fit_generator(gen,
            epochs=self.config.epochs,
            steps_per_epoch=int(60000.0/self.config.batch_size),
            callbacks=callbacks,
            **kwargs,
        )
        
        return self.history
    
    def get_encoder(self,layer_name):
        layer_lkup = self.model.get_layer(layer_name).output
        enc_mod = Model(self.model.input,layer_lkup)
        return enc_mod
    
    def save_model(self,model_json=True,overwrite=True):
        
        if model_json:
            # serialize model to JSON
            model_json = self.model.to_json()
            with open(os.path.join(self.config.model_dir,"model.json"), "w") as json_file:
                json_file.write(model_json)
                
        weight_fp = os.path.join(self.config.model_dir,'weights.h5')
        self.model.save_weights(weight_fp,overwrite=overwrite)
        
    def load_model(self,mod_name):
        setattr(self.config, 'model_name', str(mod_name))
        print('loading model ', self.config.model_name)
            
        if not hasattr(self.config, 'model_dir'):
            setattr(self.config, 'model_dir', os.path.join(self.config.log_dir,self.config.model_name))
            print('Set model_dir to ',self.config.model_dir)
            
        # load json and create model
        with open(os.path.join(self.config.model_dir,'model.json'), 'r') as json_file:
            model_json = json_file.read()
            self.model = model_from_json(model_json)

        self.model.load_weights(os.path.join(self.config.model_dir,'weights.h5'))
        
def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)    
    
    
class ContrastiveTrainer(Trainer):
    def __init__(self,config,data_loader,Ebuilder,Gbuilder,load_model=None):
        self.config = config
        self.optimizer = self.config.optimizer
        self.batch_size = self.config.batch_size
        if load_model is not None:
            self.load_model(mod_name=str(load_model))
        else:
            self.data_loader = data_loader
            self.dataset = config.dataset
            self.Ebuilder = Ebuilder
            self.Gbuilder = Gbuilder

#             self.layer_outputs = self.builder.layers

#             self.build_model(input_shape=self.data_loader.input_shape)

    def build_encoder(self,input_shape,outputs=[]):
        print('building encoder...')
        self.E_input = Input(shape=input_shape,name='input_image')
        E_output = self.Ebuilder(self.E_input)
#         self.y_lat = Dense(self.config.y_dim,activation='relu',name='y_lat')(E_output)
        self.y_class = Dense(self.config.y_dim,name='class')(E_output)
        self.z_lat = Dense(self.config.z_dim,name='z_lat')(E_output)
#         self.real = Dense(2,activation='softmax',name='D')(E_output)
        
        self.E = Model(
            inputs=self.E_input,
            outputs=[self.y_class,
                     self.z_lat,
#                      self.real,
                    ],
            name='encoder'
        )

    def build_decoder(self):
        print('building decoder/generator...')
        G_input = Input(shape=(self.config.y_dim+self.config.z_dim,),name='G_input')
        self.G_output = self.Gbuilder(G_input)
        self.G = Model(
            inputs=G_input,
            outputs=self.G_output,
            name='G'
        )
#         self.C = Model(
#             inputs=self.input,
#             outputs=self.y_class,
#             name='C'
#         )
#         self.D = Model(
#             inputs=self.input,
#             outputs=self.real,
#             name='D'
#         )

    def build_model(self,input_shape):
        self.build_encoder(input_shape)
        self.build_decoder()
        
        input_a = Input(shape=input_shape,name='A')
        input_b = Input(shape=input_shape,name='B')
        self.input = [input_a,input_b]
        
        self.A_enc = self.E(input_a)
        self.B_enc = self.E(input_b)
        
        self.Dw = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape,name='euclidean_distance')([self.A_enc[0],self.B_enc[0]]) 
        
        A_lat = Concatenate()(self.A_enc)
        B_lat = Concatenate()(self.B_enc)
        
        self.model = Model(
            inputs=self.input,
            outputs=[self.G(A_lat),self.G(B_lat),self.Dw],
        )
        
    def compile_model(self):
        self.losses = {
            'euclidean_distance': contrastive_loss,
#             'D': 'binary_crossentropy',
            'G': sse
        }
        
        lossWeights = {
            "euclidean_distance": self.config.xent,
#             "D": 0.0,
            'G': self.config.recon
        }
        metrics = {
#             'class': 'accuracy',
            'G': mse,
            
        }
        self.model.add_loss(self.config.xcov*XCov(self.A_enc[0],self.A_enc[1])(self.A_enc[0],self.A_enc[1])/self.config.batch_size)     

        self.model.compile(
            optimizer=self.config.optimizer,
            loss=self.losses,
            loss_weights=lossWeights,
#             metrics=metrics,
        )