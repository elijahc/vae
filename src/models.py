from datetime import date

from keras.layers import Dense
from keras.models import Model,load_model
import os


DEFAULT_DIR = '/home/elijahc/projects/vae'
def build_dense(inputs,layers,activations=None):
    if isinstance(activations,list):
        if len(activations) is not len(layers):
            raise Exception('activation lists and layers list must be same len')
        acts = activations
    elif isinstance(activations,str):
        acts = [activations]*len(layers)
    else:
        raise Exception('activations must be either list or str')

    x = Dense(layers[0],activation=acts[0])(inputs)
    for num_units,act in zip(layers[1:],acts[1:]):
        x = Dense(num_units,activation=act)(x)
    outputs = x
    return outputs

class CachedModel(Model):
    def __init__(self,inputs,outputs,name,verbose,version_date=None,proj_dir=DEFAULT_DIR,**model_params):
        
        if version_date is None:
            version = str(date.today())
        else:
            version = str(version_date)
            
        self.dir = model_dir = os.path.join(proj_dir,'models',version,name)
        model_file = os.path.join(model_dir,'model.h5')
        
        # Check I'm already supposed to exist
        if os.path.exists(model_file):
            if verbose:
                print('Loading Cached Model: models/'+version+'/'+name)
            
            # Load myself
            self.model = load_model(model_file)
        else:
            # I don't exist yet
            if verbose:
                print('Creating new model...')
            
            if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

            self.model = Model(inputs=inputs,outputs=outputs,name=name,**model_params)
            
            # Save myself
            if verbose:
                print('Saving to models/'+version+'/'+name)
            self.model.save(model_file)
        model_attrs = list(self.model.__dict__.keys())
        for attr in model_attrs:
            self.__setattr__(attr,self.model.__getattribute__(attr))    
    
    def compile(self,verbose=1,**kwargs):
        self.model.compile(**kwargs)
        model_file = os.path.join(self.dir,'model.h5')
        if verbose:
            self.model.save(model_file)
            print('Cached to: \n %s'%model_file)
            
    def fit(self,*args,**kwargs):
        verbose=1
        force=False
        if 'force' in list(kwargs.keys()):
            force = kwargs.pop('force')
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']
        weights_dir = os.path.join(self.dir,'trained_weights')
        weights_file = os.path.join(weights_dir,'weights.h5')
        
        if force:
            if verbose:
                print('Forced retraining...')
            self.model.fit(*args,**kwargs)
            
        elif os.path.exists(weights_file):
            if verbose:
                print('loading cached weights from '+weights_file)
            self.model.load_weights(weights_file)
            
        else:
            self.model.fit(*args,**kwargs)
        
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            
        if verbose:
            print('caching weights to: \n '+weights_file)
        
        self.model.save_weights(weights_file)