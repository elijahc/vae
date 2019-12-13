import os
import random
import json
import hashlib
import neptune
import numpy as np
import pandas as pd
import xarray

from keras.models import Model
from keras.optimizers import Adam,Nadam

from .utils import raw_to_xr, dprime, get_layer_encoders, sample_layer

from keras.models import model_from_json
from ..data_loader import Shifted_Data_Loader
from ..test_models.drduplex import sse

def get_exp_dir(proj_root,experiments):
    for e in exps:
        props = e.get_properties()
        yield os.path.join(proj_root,props['dir'])

def get_model_files(proj_root,experiments):
    exps = experiments
    for e in exps:
        props = e.get_properties()
        mod_dir = os.path.join(proj_root,props['dir'])
        weights_fp = os.path.join(mod_dir,'weights.h5')
        model_fp = os.path.join(mod_dir,'model.json')
        if os.path.exists(weights_fp) and os.path.exists(model_fp):
            yield (weights_fp,model_fp)

def load_models(proj_root,experiments,load_weights=True,compile_model=True):
    configs = [None]*len(experiments)
    if compile_model:
        configs = load_configs(experiments,proj_root)
        
    for files,conf in zip(get_model_files(proj_root,experiments),configs):
        w_fp,m_fp = files
#         conf = configs[i]
        
        with open(m_fp, 'r') as json_file:
            loaded_model_json = json_file.read()
        mod = model_from_json(loaded_model_json)
        
        if load_weights:
            mod.load_weights(w_fp)
            
        if compile_model:
            optimizer = Adam(0.00005, 0.5)
            losses = {
                'Generator':sse,
                'Classifier':'categorical_crossentropy'
            }
            loss_weights = {
                'Generator':conf['recon_weight'],
                'Classifier':conf['xent_weight'],
            }
            metrics={'Generator':'mse','Classifier':'accuracy'}
            
            print('Compiling model')
            mod.compile(
                loss=losses,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics={'Generator':'mse','Classifier':'accuracy'}
            )

            
        yield mod
        
def load_properties(experiments):
    for e in experiments:
        yield e.get_properties()
        
def load_params(experiments):
    for e in experiments:
        yield e.get_parameters()
        
def load_configs(exps,proj_root=None):
    props = load_properties(exps)
    params = load_params(exps)
    ids = [e.id for e in exps]
    for eid,pr,pa in zip(ids,props,params):
        out = {'id':eid}
        out.update(pr)
        out.update(pa)
        if proj_root is not None:
            out['exp_dir']=os.path.join(proj_root,pr['dir'])
        yield out

def generate_dense_encoders(mod):
    E = mod.layers[1]
    y_encoder = Model(E.layers[0].input, E.get_layer(name='y_dim').output)
    z_encoder = Model(E.layers[0].input, E.get_layer(name='z_dim').output)

    l3_encoder = Model(mod.input,  E.get_layer(name='embedder')(mod.input))
        
    l2_layers = E.layers[1].layers[:-2]
    x2 = l2_layers[0](mod.input)
    for l in l2_layers[1:]:
        x2 = l(x2)
    l2_encoder = Model(mod.input,x2)
        
#         l1_layers = E.layers[1].layers[:-4]
#         x1 = l1_layers[0](mod.input)
#         for l in l1_layers[1:]:
#             x1 = l(x1)
#         l1_encoder = Model(mod.input,x1)
    for l in [y_encoder,z_encoder,l3_encoder,l2_encoder]:
        yield l
        
def generate_convnet_encoders(mod):
    latent_encoders = list(get_layer_encoders(mod,['y_dim','z_dim']))
    mod_layers = [
                'flatten_1',
                'batch_normalization_3',
                'batch_normalization_2',
                'batch_normalization_1'
            ]
    
    other_encoders = get_layer_encoders(mod,mod_layers)
    encoders = []
    encoders.extend(latent_encoders)
    encoders.extend(other_encoders)
    for l in encoders:
        yield l
    
    
def prep_assemblies(proj_root, experiments,test_data=None,slug=None):  
    props = load_properties(experiments)
    params = load_params(experiments)
    mods = load_models(proj_root, experiments)
    
    if test_data is None:
        DL = Shifted_Data_Loader('fashion_mnist',rotation=None,translation=0.75,bg='natural',flatten=False)
        slug = [(dx,dy,float(lab),float(rxy)) for dx,dy,rxy,lab in zip(DL.dx[1],DL.dy[1],DL.dtheta[1],DL.y_test)]
        stim_set = pd.DataFrame({'dx':DL.dx[1]-14,'dy':DL.dy[1]-14,'numeric_label':DL.y_test,'rxy':DL.dtheta[1],'image_id':image_id})
        test_data = DL.sx_test
        
        
    image_id = [hashlib.md5(json.dumps(list(p),sort_keys=True).encode('utf-8')).digest().hex() for p in slug]
    stim_set = pd.DataFrame([{'dx':dx,'dy':dy,'numeric_label':int(yn),'rxy':rxy} for dx,dy,yn,rxy in slug])
    stim_set['image_id'] = image_id
    pix_data = test_data.reshape(test_data.shape[0],np.prod(test_data.shape[1:]))
    
    for pr, pa, mod in zip(props,params,mods):
        batch_sz = int(pa['batch_sz'])

        if pr['encoder_arch'] == 'dense':
            E = mod.layers[1]
            print('Generating encoders...')
            y_encoder,z_encoder,l3_encoder,l2_encoder = generate_dense_encoders(E)
            l3_enc = l3_encoder.predict(test_data,batch_size=batch_sz)
            l2_enc = l2_encoder.predict(test_data,batch_size=batch_sz)
            other_encoders = [l3_encoder, l2_encoder]
            other_names = ['dense_3','dense_2']
            other_encs = [l.predict(test_data, batch_size=batch_sz)]
            other_depths = [3,2]

            
        elif pr['encoder_arch'].startswith('conv'):
            E = mod.layers[1]
            print('Generating encoders...')
            layer_encoders = list(generate_convnet_encoders(E))
            y_encoder = layer_encoders[0]
            z_encoder = layer_encoders[1]
            other_encoders = layer_encoders[2:]
            other_names = ['conv_4','conv_3','conv_2','conv_1']
            other_depths = [4,3,2,1]
            
            print('Fetching activations...')
            other_encs = [sample_layer(e,test_data,batch_sz) for e in other_encoders]
        
        z_enc = z_encoder.predict(test_data,batch_size=batch_sz)
        y_enc = y_encoder.predict(test_data,batch_size=batch_sz)
        
        l_names = ['pixel','y_enc','z_enc']
        l_names.extend(other_names)
        
        l_encs = [pix_data,y_enc,z_enc]
        l_encs.extend(other_encs)
        
        l_depths = [0,np.max(other_depths)+1,np.max(other_depths)+1]
        l_depths.extend(other_depths)

        encodings = {k:v for k,v in zip(l_names,l_encs)}
        depths = {k:v for k,v in zip(l_names,l_depths)}
        
        yield encodings,depths,stim_set
#         yield raw_to_xr(encodings,depths,stim_set)

def load_assemblies(proj_root,experiments):
    for e in experiments:
        props = e.get_properties()
        mod_dir = os.path.join(proj_root,props['dir'])
        assembly_fp = os.path.join(mod_dir,'dataset.nc')
        if os.path.exists(assembly_fp):
            da = xarray.open_dataarray(assembly_fp)
            presentation_idxs = ['image_id','dx','dy','rxy','numeric_label','object_name','tx','ty','s']
            n_idxs = ['neuroid_id','layer','region']
            
            yield da.set_index(presentation=presentation_idxs,neuroid=n_idxs)
        
class NeptuneExperiments(object):
    def __init__(self,root_fp,project_qualified_name):
        self.root_fp = root_fp
        self.project_qualified_name = project_qualified_name
        neptune.init(self.project_qualified_name)
        neptune.set_project(self.project_qualified_name)
        
        self.project = neptune.project
        self.leaderboard = self.project.get_leaderboard()
        
    def get_models(self,id=None,):
        pass
    
    def get_properties(self):
        prop_cols = filter(lambda c: c.startswith('property'), exp_df.columns)
        return self.leaderboard[list(prop_cols)]