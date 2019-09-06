import os
import json
import xarray
import numpy as np
import pandas as pd
import xarray
from tqdm import tqdm as tqdm

from .file_loaders import *

from numpy.random import RandomState

from brainio_base.assemblies import DataAssembly
from brainio_base.stimuli import StimulusSet
from keras.models import Model

def find_exp_runs(root):
    runs = []
    files = []
    for dp,dn,fn in os.walk(root):
        if len(fn) > 0:
            if 'config.json' in fn:
                runs.append(dp)
                files.append(fn)
    return runs,files
    
class _DateExperimentLoader(object):
    def __init__(self,date,project_root='/home/elijahc/projects/vae/',seed=7):
        self.experiment_root = os.path.join(project_root,'models',date)
        runs,run_files = find_exp_runs(self.experiment_root)
        self.runs = np.array(runs)
        self.run_files = np.array(run_files)
        self.random_state = RandomState(seed) 
        self.experiment_set = pd.DataFrame([load_config(rd) for rd in runs])
        self.history = None
        self.configs = None
    
    def load(self):
        self.load_assemblies()
        
    def load_assemblies(self,subset=None,fname='dataset.nc'):
        if subset is None:
            subset = np.arange(len(self.runs))
        assemblies = [xarray.open_dataarray(os.path.join(rd,fname)) for rd in self.runs[subset]]
        self.assemblies=[]
        loss_w = self.experiment_set[['xent','recon']]
        for da,xent,recon in zip(assemblies,loss_w.xent[subset],loss_w.recon[subset]):
            stim_df = da.presentation.to_dataframe(name='stimulus_set').reset_index().drop(columns=['presentation'])
            stim_set = StimulusSet(stim_df)
            stim_set.name = "lg.xent{}.recon{}".format(xent,recon)
            da = da.assign_attrs({'stimulus_set':stim_set})
            da = DataAssembly(da)
            da = da.assign_attrs({'xent':xent,'recon':recon})
            self.assemblies.append(da)
            
        return self.assemblies
    
    def load_configs(self,fname='config.json'):
        configs = np.array([load_config(run_dir=rd,filename=fname) for rd in self.runs])
        
        self.configs = configs
        
    def load_history(self,fname='train_history.parquet'):
        if self.configs is None:
            self.load_configs()
        
        train_history = [load_train_history(rd,conf,fname) for rd,conf in zip(self.runs,self.configs)]
        self.history = train_history
        return self.history
    
    def load_performance(self,fname='performance.json'):
        if self.configs is None:
            self.load_configs()
        
        performance = [load_performance(run_dir=rd,conf=conf,filename=fname) for rd,conf in zip(self.runs,self.configs)]
        
        self.performance = performance
        return self.performance
    
    def model_generator(self,subset=None,mod_json_fn='model.json',w_fn='weights.h5'):
        if self.configs is None:
            self.load_configs()
            
        if subset is None:
            subset = np.arange(len(self.runs))
        
        
        for rd,conf in tqdm(zip(self.runs[subset],self.configs[subset]),total=len(self.runs[subset])):
#             print('loading model {}'.format(rd))
            yield load_model(rd,mod_json_fn,w_fn)
            
    def generate_classifiers(self,subset=None,class_layer_name='class',mod_json_fn='model.json',w_fn='weights.h5'):
        if self.configs is None:
            self.load_configs()
                    
        if subset is None:
            subset = np.arange(len(self.runs))
        
        for mod in self.model_generator(subset,mod_json_fn,w_fn):
            classifier = Model(mod.input,mod.get_layer(class_layer_name).output)
            classifier.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
            yield classifier
