import os
import numpy as np
import neptune

from keras.models import model_from_json

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
            
class NeptuneResults(object):
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