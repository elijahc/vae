import os
import random
import json
import hashlib
import neptune
import numpy as np
import pandas as pd
import xarray

from keras.models import Model

from .utils import raw_to_xr, dprime

from keras.models import model_from_json
from ..data_loader import Shifted_Data_Loader

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

def load_models(proj_root,experiments):
    for w_fp, m_fp in get_model_files(proj_root,experiments):
        with open(m_fp, 'r') as json_file:
            loaded_model_json = json_file.read()
        mod = model_from_json(loaded_model_json)
        
        mod.load_weights(w_fp)
        
        yield mod
        
def load_properties(experiments):
    for e in experiments:
        yield e.get_properties()
        
def load_params(experiments):
    for e in experiments:
        yield e.get_parameters()

def create_assemblies(proj_root, experiments,test_data=None,slug=None):  
    props = load_properties(experiments)
    params = load_params(experiments)
    mods = load_models(proj_root, experiments)
    
    if test_data is None:
        DL = Shifted_Data_Loader('fashion_mnist',rotation=None,translation=0.75,bg='natural',flatten=False)
        slug = [(dx,dy,float(lab),float(rxy)) for dx,dy,rxy,lab in zip(DL.dx[1],DL.dy[1],DL.dtheta[1],DL.y_test)]
        stim_set = pd.DataFrame({'dx':DL.dx[1]-14,'dy':DL.dy[1]-14,'numeric_label':DL.y_test,'rxy':DL.dtheta[1],'image_id':image_id})
        test_data = DL.sx_test
        
    for pr, pa, mod in zip(props,params,mods):
        E = mod.layers[1]
        y_encoder = Model(E.layers[0].input, E.get_layer(name='y_dim').output)
        z_encoder = Model(E.layers[0].input, E.get_layer(name='z_dim').output)
#         lat_encoder = Model(E.layers[0].input, E.get_layer(name='latent').output)
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
        
        batch_sz = int(pa['batch_sz'])
        z_enc = z_encoder.predict(test_data,batch_size=batch_sz)
        y_enc = y_encoder.predict(test_data,batch_size=batch_sz)
        l3_enc = l3_encoder.predict(test_data,batch_size=batch_sz)
        l2_enc = l2_encoder.predict(test_data,batch_size=batch_sz)
        image_id = [hashlib.md5(json.dumps(list(p),sort_keys=True).encode('utf-8')).digest().hex() for p in slug]
        stim_set = pd.DataFrame([{'dx':dx,'dy':dy,'numeric_label':int(yn),'rxy':rxy} for dx,dy,yn,rxy in slug])
        stim_set['image_id'] = image_id

        encodings = {
            'pixel':test_data.reshape(test_data.shape[0],np.prod(test_data.shape[1:])),
#             'dense_1':l1_enc,
            'dense_2':l2_enc,
            'dense_3':l3_enc,
            'y_lat':y_enc,
            'z_lat':z_enc
        }
        depths = {
            'pixel':0,
        #     'dense_1':1,
            'dense_2':2,
            'dense_3':3,
            'y_lat':4,
            'z_lat':4
        }
        
        yield raw_to_xr(encodings,depths,stim_set)

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