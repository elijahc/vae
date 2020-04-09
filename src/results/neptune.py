import os
import random
import json
import hashlib
import neptune
import numpy as np
import pandas as pd
import xarray

from keras.models import Model,Sequential
from keras.optimizers import Adam,Nadam
from tqdm import tqdm as tqdm
from sklearn.decomposition import PCA

from .utils import raw_to_xr, dprime, get_layer_encoders, sample_layer, pca_layer
from .dicarlo import dicarlo_slug

from keras.models import model_from_json
from ..data_loader import Shifted_Data_Loader
from ..test_models.drduplex import sse

class NeptuneExperimentRun(object):
    def __init__(self, proj_root,neptune_exp,name=None):
        self.proj_root = proj_root
        self.experiment = neptune_exp

        self.model = None
        self._compiled = False
        
        self.properties = None
        self.parameters = None
        if name is None:
            name = self.experiment.id
        self.name = name
    
    def get_properties(self):
        return self.experiment.get_properties()
        
    def get_parameters(self):
        return self.experiment.get_parameters()
        
    def __repr__(self):
        return self.experiment.__repr__()
    
    def get_config(self):
        if not hasattr(self,'config'):
#             print('loading config for '+self.experiment.__repr__())
            self.config = next(load_configs([self.experiment]))
        
        return self.config
    
    def _build_model(self):
        conf = self.get_config()
        print('building model {}(arch={}, recon={})...'.format(self.name,conf['encoder_arch'],conf['recon_weight']))
        m = next(load_models(self.proj_root, [self.experiment], load_weights = True, compile_model = True))
        m.name = self.name
        return m
    
    def pca_assembly(self, test_data, n_units=None, n_components=5, metadata=None, pca_kws={}):
        if metadata is None or not isinstance(metadata,dict):
            raise ValueError('Please provide stim metadata as dict')  
        elif len(test_data.shape) == 3:
            test_data = np.expand_dims(test_data,-1)
            
        stim_set = pd.DataFrame.from_records(metadata)
            
        enc = {k:v for k,v in self.activations(test_data,n_units)}
        enc_pca = {}
        pca_objs = {}
        depths = self._layer_depths()
        
        with tqdm(enc.items(),total=len(enc.keys())) as l_iter:
            print('PCA(n_components={})...'.format(n_components))
            for k,v in l_iter:
                l_iter.set_description('PCA: {}({}, {})'.format(k, *v.shape))
                
                if k in ['pixel','y_enc','z_enc']:
                    enc_pca[k] = v
                elif isinstance(n_components, int) and n_components>np.max(v.shape):
                    enc_pca[k] = v
                
                else:
                    pca = PCA(np.min([n_components,v.shape[0],v.shape[1]]), **pca_kws)

                    enc_pca[k] = pca.fit_transform(v)
                    pca_objs[k] = pca
        
        print(pca_objs)
        
        xr = raw_to_xr(enc_pca,depths,stim_set)
        
        return xr,pca_objs

    def _gen_layers(self):
        mod = self._build_model()
        pr = self.get_properties()

        if pr['encoder_arch'] == 'dense':
#             E = mod.layers[1]
            print('Generating dense encoders...')

            # yields Keras models which each output a layer activation in the following order
            # y_enc, z_enc, other layers in reverse order
            layers = generate_dense_encoders(mod)
            other_names = ['dense_3','dense_2','dense_1']
            other_depths = [3,2,1]
            
        elif pr['encoder_arch'].startswith('conv'):
            print('generating convolutional activations...')

            E = mod.layers[1]
            
            # Yields (encoder,name)
            layers =  generate_convnet_encoders(E)
            other_depths = [4,3,2,1]
            other_names = ['conv_4','conv_3','conv_2','conv_1']
        
        n = ['y_enc','z_enc']+other_names
        d = [7,7]+other_depths
        
        for l,name,depth in zip(layers,n,d):
            yield l,name,depth
    
    def activations(self, test_data, n_units=192):
        if len(test_data.shape) == 3:
                test_data = np.expand_dims(test_data,-1)
        
        pa = self.get_parameters()
        batch_sz = int(pa['batch_sz'])
        
        if len(test_data.shape) == 3:
                test_data = np.expand_dims(test_data,-1)
        
        pix_data = test_data.reshape(test_data.shape[0],np.prod(test_data.shape[1:]))
        
        yield 'pixel', pix_data
        
        layer_generator = self._gen_layers()
        for l,n,d in self._gen_layers():
            yield n,sample_layer(l, test_data, batch_sz, n_units)
        
    def _layer_depths(self):
        pr = self.get_properties()
        l_depths = {
            'pixel':0,
            'y_enc':7,
            'z_enc':7,
        }
        if pr['encoder_arch'] == 'dense':
            l_depths.update({
                'dense_1':1,
                'dense_2':2,
                'dense_3':3,
            })

        elif pr['encoder_arch'].startswith('conv'):
            l_depths.update({
                'conv_1':1,
                'conv_2':2,
                'conv_3':3,
                'conv_4':4
            })
            
        return l_depths
    
    def gen_assembly(self, test_data, n_units=192,**metadata):
        l_depths = self._layer_depths()
        act_iter = self.activations(test_data,n_units)
        enc = {k:v for k,v in act_iter}
        
        stim_set = pd.DataFrame.from_records(metadata)
        
        return raw_to_xr(enc, l_depths, stim_set)
            
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
    for latent_names in [['y_dim','z_dim'],['y_enc','z_enc']]:
        try:
            latent_layers = get_layer_encoders(mod.layers[1],latent_names)
            break
                
        except ValueError as e:
#             print('bad latent layer names: ',latent_names)
            continue
        raise ValueError('Unable to load latent layers')
    
    mod_layers = [
        'leaky_re_lu_3',
        'leaky_re_lu_2',
        'leaky_re_lu_1'
    ]
    m = mod
    inp = m.layers[0]
    embedder = m.layers[1].layers[1]
    other_layers = [[inp]+embedder.layers]
    other_layers += [[inp]+embedder.layers[:idx] for idx in [-2,-4]]
    
    # Yield latent layers
    for l,name in zip(latent_layers,['y_enc','z_enc']):
        l.name = name
        yield l
        
    # Yield the other layers
    for l_arr,name in zip(other_layers,['dense_3','dense_2','dense_1']):
        # l_arr references a list of layers, not the model so we need to create it
        l = Sequential(l_arr, name=name)
        yield l
        
def generate_convnet_encoders(mod):
    encoders = []
    mod_layers = [
                'leaky_re_lu_4',
                'leaky_re_lu_3',
                'leaky_re_lu_2',
                'leaky_re_lu_1'
            ]
    
    for latent_names in [['y_dim','z_dim'],['y_enc','z_enc']]:
        try:
            for layer in get_layer_encoders(mod,latent_names+mod_layers):
                yield layer
                
        except ValueError as e:
#             print('bad latent layer names: ',latent_names)
            continue
    
def build_stim_set(test_data=None, slug=None, image_id=None, object_name=None):
    if test_data is None:
        print('no images provided in test_data, generating images using exp params:')
        print("Shifted_Data_Loader('fashion_mnist',rotation=None,translation=0.75,bg='natural',flatten=False)")
        DL = Shifted_Data_Loader('fashion_mnist',rotation=None,translation=0.75,bg='natural',flatten=False)
        slug = [(dx,dy,float(lab),float(rxy)) for dx,dy,rxy,lab in zip(DL.dx[1],DL.dy[1],DL.dtheta[1],DL.y_test)]
        stim_set = pd.DataFrame({'dx':DL.dx[1]-14,'dy':DL.dy[1]-14,'numeric_label':DL.y_test,'rxy':DL.dtheta[1],'image_id':image_id})
        test_data = DL.sx_test
        
    if image_id is None:
        print('no image ids provided for test_data, generating ids...')
        image_id = [hashlib.md5(json.dumps(list(p),sort_keys=True).encode('utf-8')).digest().hex() for p in slug]
    
    recs = [{'dx':dx,'dy':dy,'category_name':yn,'rxy':rxy} for dx,dy,yn,rxy in slug]
    stim_set = pd.DataFrame(recs)
    
    stim_set['image_id'] = image_id
    
    if object_name is not None:
        stim_set['object_name']=object_name
        
    return stim_set
    
def prep_assemblies(proj_root, experiments,test_data=None,slug=None,image_id=None,object_name=None,n_units=192):  
    props = load_properties(experiments)
    params = load_params(experiments)
    mods = load_models(proj_root, experiments)
    
    stim_set = build_stim_set(test_data,slug,image_id,object_name)
    
    pix_data = test_data.reshape(test_data.shape[0],np.prod(test_data.shape[1:]))
    
    for pr, pa, mod in zip(props,params,mods):
        batch_sz = int(pa['batch_sz'])

        if pr['encoder_arch'] == 'dense':
            E = mod.layers[1]
            print('Generating dense encoders...')
            y_encoder,z_encoder,l3_encoder,l2_encoder = generate_dense_encoders(E)
            l3_enc = l3_encoder.predict(test_data,batch_size=batch_sz)
            l2_enc = l2_encoder.predict(test_data,batch_size=batch_sz)
            other_encoders = [l3_encoder, l2_encoder]
            other_names = ['dense_3','dense_2']
            other_encs = [l.predict(test_data, batch_size=batch_sz)]
            other_depths = [3,2]

            
        elif pr['encoder_arch'].startswith('conv'):
            E = mod.layers[1]
            
            # Yields (encoder,name)
            layer_encoders = list(generate_convnet_encoders(E))
                
            y_encoder = layer_encoders[0]
            z_encoder = layer_encoders[1]
            other_encoders = layer_encoders[2:]
            other_depths = [4,3,2,1]
                
            other_encs = [sample_layer(e,test_data,batch_sz,n_sample_units=n_units) for e in other_encoders]
            other_names = ['conv_4','conv_3','conv_2','conv_1']
        
        z_enc = z_encoder.predict(test_data,batch_size=batch_sz)
        y_enc = y_encoder.predict(test_data,batch_size=batch_sz)
        
        l_names = ['pixel','y_enc','z_enc']
        l_encs = [pix_data,y_enc,z_enc]


        l_names.extend(other_names)
        
        l_encs.extend(other_encs)
        
        for le,ln in zip(l_encs,l_names):
            le.name = ln
        
        l_depths = [0,np.max(other_depths)+1,np.max(other_depths)+1]
        l_depths.extend(other_depths)

        encodings = {k:v for k,v in zip(l_names,l_encs)}
        depths = {k:v for k,v in zip(l_names,l_depths)}
        
        yield encodings,depths,stim_set
#         yield raw_to_xr(encodings,depths,stim_set)

def load_assemblies(proj_root,experiments,fn='activations_1.nc'):
    for e in experiments:
        props = e.get_properties()
        mod_dir = os.path.join(proj_root,props['dir'])
        assembly_fp = os.path.join(mod_dir,fn)
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