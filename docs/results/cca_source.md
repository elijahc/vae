---

title: CCA Source
parent: Results
nav_exclude: true

---


```python
import os
import neptune
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from scipy.stats import ttest_ind as ttest,pearsonr
import scipy
import xarray as xr
from scipy.spatial.distance import pdist,squareform,cdist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold
from sklearn.cross_decomposition import CCA

# from tensorflow.python import keras as keras
from keras.models import Model

from src.results.experiments import _DateExperimentLoader
from src.results.utils import raw_to_xr, dprime
from src.results.neptune import get_model_files, load_models, load_assemblies, load_params, load_properties,prep_assemblies,NeptuneExperimentRun,generate_convnet_encoders
from src.results.dicarlo import get_dicarlo_su, process_dicarlo,err_neuroids
from src.data_loader import Shifted_Data_Loader
from src.data_generator import ShiftedDataBatcher
import src.rcca

import brainscore
from brainscore.assemblies import walk_coords,split_assembly
from brainscore.assemblies import split_assembly
# from brainscore.metrics import Score

from brainio_base.assemblies import DataAssembly

def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("talk")
    
    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Georgia","Times New Roman", "Palatino", "serif"]
    })
```

    Using TensorFlow backend.



```python
os.environ['NEPTUNE_API_TOKEN']="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiI3ZWExMTlmYS02ZTE2LTQ4ZTktOGMxMi0wMDJiZTljOWYyNDUifQ=="
neptune.init('elijahc/DuplexAE')
neptune.set_project('elijahc/DuplexAE')
proj_root = '/home/elijahc/projects/vae'
```


```python
from src.results.neptune import load_configs
```


```python
conv_eids = [
    'DPX-29',
    'DPX-30',
]
dense_eids = [
    'DPX-10',
    'DPX-16',
#     'DPX-27',
]
# eids = conv_eids+dense_eids
conv_exps = neptune.project.get_experiments(id=conv_eids)
dense_exps = neptune.project.get_experiments(id=dense_eids)
exps = np.array(conv_exps+dense_exps)
s_df = pd.DataFrame(list(load_configs(exps)))
s_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>assembly_fn</th>
      <th>augmentation</th>
      <th>batch_sz</th>
      <th>bg</th>
      <th>bg_contrast</th>
      <th>dataset</th>
      <th>dir</th>
      <th>encoder_arch</th>
      <th>generator_arch</th>
      <th>id</th>
      <th>im_translation</th>
      <th>n_epochs</th>
      <th>recon_weight</th>
      <th>su_selectivity_fn</th>
      <th>xent_weight</th>
      <th>y_dim</th>
      <th>z_dim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>dynamic</td>
      <td>512.0</td>
      <td>natural</td>
      <td>0.3</td>
      <td>fashion_mnist</td>
      <td>models/2019-11-04/DPX-29</td>
      <td>convnet</td>
      <td>resnet</td>
      <td>DPX-29</td>
      <td>0.75</td>
      <td>54000.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>dynamic</td>
      <td>512.0</td>
      <td>natural</td>
      <td>0.3</td>
      <td>fashion_mnist</td>
      <td>models/2019-11-04/DPX-30</td>
      <td>convnet</td>
      <td>resnet</td>
      <td>DPX-30</td>
      <td>0.75</td>
      <td>54000.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dataset.nc</td>
      <td>dynamic</td>
      <td>512.0</td>
      <td>natural</td>
      <td>0.3</td>
      <td>fashion_mnist</td>
      <td>models/2019-09-25/DPX-10</td>
      <td>dense</td>
      <td>resnet</td>
      <td>DPX-10</td>
      <td>0.75</td>
      <td>54000.0</td>
      <td>1.0</td>
      <td>selectivity.pqt</td>
      <td>15.0</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>dynamic</td>
      <td>512.0</td>
      <td>natural</td>
      <td>0.3</td>
      <td>fashion_mnist</td>
      <td>models/2019-09-25/DPX-16</td>
      <td>dense</td>
      <td>resnet</td>
      <td>DPX-16</td>
      <td>0.75</td>
      <td>54000.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>35</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import Lasso,LassoCV,MultiTaskLassoCV,MultiTaskLasso,SGDRegressor
from sklearn.svm import SVR,LinearSVR,NuSVR
```


```python
from sklearn.metrics import explained_variance_score
from sklearn.multioutput import MultiOutputRegressor,MultiOutputEstimator
```


```python
def lasso(x,neural_data,region=None,brain_region=['IT','V4'], cv=5, variation=[3],sortby='image_id',train_size=0.75):
    var_lookup = stimulus_set[stimulus_set.variation.isin(variation)].image_id.values
    x = x.where(x.image_id.isin(var_lookup),drop=True)
    nd = neural_data.where(neural_data.image_id.isin(var_lookup),drop=True)
    
    num_images = x.shape[0]
    out_recs = []
#     {'region':[],
#             'layer':[],
#             'fve':[],
#             'pearsonr':[],
#             'p-value':[],
#             'neuron':[],
#             'depth':[],
#             'iter':[],
#            }
    
    cv_tr = []
    cv_te = []
    
    for rand_delta in np.arange(cv):
        tr_idx, te_idx, _,_ = train_test_split(np.arange(num_images),np.arange(num_images),train_size=train_size,random_state=np.random.randint(0,50)+rand_delta)
        cv_tr.append(tr_idx)
        cv_te.append(te_idx)
    
    tr,te, _,_ = train_test_split(np.arange(num_images),np.arange(num_images),train_size=train_size,random_state=7)

    
    for br in brain_region:
        nd_reg = nd.sel(region=br)[:,:20]
        if region is None:
            region = np.unique(x.region.values)
        
        with tqdm(region,total=len(region)) as reg_iter:
            for reg in reg_iter:
                if reg == 'pixel':
                    continue
                else:
                    x_reg = x.sel(region=reg)
                    depth = np.unique(x_reg.layer.values)[0]
                reg_iter.set_description('{}{} x {}{}'.format(reg,x_reg.shape,br,nd_reg.shape))
                
#             with tqdm(np.arange(nd_reg.shape[-1])) as neurons:
#                 fve_mean = []
#                 r_mean = []
#                 neurons.set_description('{}{} x {}{}'.format(reg,x_reg.shape,br,nd_reg.shape))
                
#                 for n_idx in neurons:
#             estimator = MultiOutputRegressor(SGDRegressor(early_stopping=True),n_jobs=5)
                estimator=MultiOutputRegressor(LinearSVR(), n_jobs=5)
    #                     estimator = SGDRegressor()
    #                     estimator = SVR()
                estimator.fit(x_reg.values[tr],nd_reg.values[tr])
                y_pred = estimator.predict(x_reg.values[te])
#                 print(type(y_pred))
#                 print(type(nd_reg.values[te]),nd_reg.values[te].shape)
    #             r,pv = pearsonr(nd_reg.values[te],y_pred)
                fve = explained_variance_score(nd_reg.values[te],y_pred)

    #                     score = estimator.score(x_reg.values,nd_reg.values[:,n_idx])

    #                     r,pv = pearsonr(ab_vec[0],ab_vec[1])
    #             r_mean.append(np.nan_to_num(r))
    #             fve_mean.append(fve)
                reg_iter.set_postfix(fve=fve)
                out_recs.append({
                    'region':br,
                    'layer':reg,
                    'fve':fve,
    #                 'cc':r,
    #                 'neuron':n_idx,
                    'depth':depth,
                })
#                     out_dict['region'].append(br)
#                     out_dict['layer'].append(reg)
#                     out_dict['fve'].append(fve)
#                     out_dict['pearsonr'].append(r)
#                     out_dict['neuron'].append(n_idx)
# #                     out_dict['iter'].append(n)
#                     out_dict['depth'].append(depth)

#                     neurons.set_postfix(r_max=np.max(r_mean), r_mean=np.mean(r_mean), r_var=np.var(r_mean), fve_max=np.max(fve_mean), fve_mean=np.mean(fve_mean))
#     print({k:v.shape for k,v in out_dict.items()})

    return pd.DataFrame.from_records(out_recs)
```


```python
def cca(x,neural_data,region=None, brain_region=['IT','V4'], cv=5, n_components=5, variation=[3],sortby='image_id',train_size=0.75):
    var_lookup = stimulus_set[stimulus_set.variation.isin(variation)].image_id.values
    x = x.where(x.image_id.isin(var_lookup),drop=True)
    nd = neural_data.where(neural_data.image_id.isin(var_lookup),drop=True)
    
    x = x.sortby(sortby)
    nd = nd.sortby(sortby)
    
    assert list(getattr(x,sortby).values) == list(getattr(nd,sortby).values)
    num_images = x.shape[0]
    out_recs = []
#     out_dict = {'region':[],
#             'layer':[],
#             'pearsonr':[],
#             'fve':[],
# #             'p-value':[],
#             'iter':[],
#             'depth':[],
#            }
    
    cv_tr = []
    cv_te = []
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=cv)
    for tr,te in kf.split(np.arange(num_images)):
        cv_tr.append(tr)
        cv_te.append(te)
    
    for rand_delta in np.arange(cv):
        tr_idx, te_idx, _,_ = train_test_split(np.arange(num_images),np.arange(num_images),train_size=train_size,random_state=np.random.randint(0,50)+rand_delta)
        cv_tr.append(tr_idx)
        cv_te.append(te_idx)
    
    for br in brain_region:
        nd_reg = nd.sel(region=br)
        
        if region is None:
            region = np.unique(x.region.values)
            
        for reg in region:
            if reg == 'pixel':
                continue
            x_reg = x.sel(region=reg)
            
            depth = np.unique(x_reg.layer.values)[0]
            with tqdm(zip(np.arange(cv),cv_tr,cv_te), total=cv) as t:
                t.set_description('{}{} x {}{}'.format(reg,x_reg.shape,br,nd_reg.shape))
                
                r_mean = []
                fve_mean = []
                cca_mean = []
                for n,tr,te in t:
                    cca = CCA(n_components=n_components)
                    cca.fit(x_reg.values[tr],nd_reg.values[tr])

                    u,v = cca.transform(x_reg.values[te],nd_reg.values[te])
                    
                    y_pred = cca.predict(x_reg.values[te])
                    y_true = nd_reg.values[te]
                    
                    fve = explained_variance_score(y_true,y_pred,multioutput='uniform_average')
                    r_vals = [pearsonr(y_pred[:,i],y_true[:,i]) for i in range(y_pred.shape[-1])]
                    
                    cca_r = np.mean([pearsonr(u[:,i],v[:,i]) for i in np.arange(n_components)])

#                     r_vals = [pearsonr(ab_vec[0][:,i],ab_vec[1][:,i]) for i in range(ab_vec[0].shape[-1])]
                    
                    r_mean.append(np.mean([r for r,v in r_vals]))
                    cca_mean.append(cca_r)
                    fve_mean.append(fve)
                
                    out_recs.append({
                        'region':br,
                        'layer':reg,
                        'pearsonr': np.mean([r for r,v in r_vals]),
                        'cca_r':cca_r,
                        'fve':fve,
                        'iter':n,
                        'depth':depth,
                    })
                    
                    t.set_postfix(pearson=np.mean(r_mean), cca=np.mean(cca_mean), fve=np.mean(fve_mean))
                    
    return pd.DataFrame.from_records(out_recs)
```


```python
neural_data = brainscore.get_assembly(name="dicarlo.Majaj2015")
neural_data.load()
stimulus_set = neural_data.attrs['stimulus_set']
# # stimulus_set.to_csv('../data/dicarlo_images/stimulus_set.csv',index=False)
neural_data = process_dicarlo(neural_data)
```

    /home/elijahc/.pyenv/versions/fastai/lib/python3.6/site-packages/brainio_base/assemblies.py:213: FutureWarning: The inplace argument has been deprecated and will be removed in a future version of xarray.
      xr_data.set_index(append=True, inplace=True, **coords_d)
    /home/elijahc/.pyenv/versions/fastai/lib/python3.6/site-packages/brainio_base/assemblies.py:247: FutureWarning: The inplace argument has been deprecated and will be removed in a future version of xarray.
      result.reset_index(self.multi_group_name, drop=True, inplace=True)
    /home/elijahc/.pyenv/versions/fastai/lib/python3.6/site-packages/brainio_base/assemblies.py:248: FutureWarning: The inplace argument has been deprecated and will be removed in a future version of xarray.
      result.set_index(append=True, inplace=True, **{self.multi_group_name: self.group_coord_names})



```python
sm_imgs = np.load('../data/dicarlo_images/sm_imgs_56x56.npy')

ids3 = stimulus_set[stimulus_set.variation.values==3].image_id.values
sm_ims = list(zip(ids3,sm_imgs[stimulus_set.variation.values==3]))

Xm,Xs = (sm_imgs.mean(),sm_imgs.std())
scaled_sm_imgs = np.clip((sm_imgs-Xm)/Xs,-1,1)
```


```python
metadata = stimulus_set[['image_id','object_name','category_name','variation','dy_px','dx_px','rxy']].rename(columns={'dx_px':'dx','dy_px':'dy'})
metadata = {k:list(v.values()) for k,v in metadata.to_dict().items()}
# metadata
```


```python
# dfs = []
# for exp,name in zip(reversed(exps),['no-recon','w/ recon','w/ recon','no-recon']):
#     run = NeptuneExperimentRun(proj_root,neptune_exp=exp)
#     xrs = run.gen_assembly(scaled_sm_imgs, n_units=180, **metadata)
#     lasso_df = lasso(xrs,neural_data,region=None,variation=[0,3],cv=2)
#     lasso_df['model']= name
#     lasso_df['arch']=run.get_config()['encoder_arch']
#     dfs.append(lasso_df)

# lasso_3 = pd.concat(dfs)
```


```python
obj = ['no-recon','w/ recon','w/ recon','no-recon']
att = []
for exp,o in zip(exps,obj):
    run = NeptuneExperimentRun(proj_root,neptune_exp=exp)
    xr,pcas = run.pca_assembly(scaled_sm_imgs, n_units=None, n_components=0.8, metadata=metadata, pca_kws={'svd_solver':'full'})
    att.append(pcas)
```

    building model DPX-29(arch=convnet, recon=0.0)...
    WARNING:tensorflow:From /home/elijahc/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From /home/elijahc/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-13-22b1e1bd2c40> in <module>
          3 for exp,o in zip(exps,obj):
          4     run = NeptuneExperimentRun(proj_root,neptune_exp=exp)
    ----> 5     xr,pcas = run.pca_assembly(scaled_sm_imgs, n_units=None, n_components=0.8, metadata=metadata, pca_kws={'svd_solver':'full'})
          6     att.append(pcas)


    ~/projects/vae/notebooks/src/results/neptune.py in pca_assembly(self, test_data, n_units, n_components, metadata, pca_kws)
         65         stim_set = pd.DataFrame.from_records(metadata)
         66 
    ---> 67         enc = {k:v for k,v in self.activations(test_data,n_units)}
         68         enc_pca = {}
         69         pca_objs = {}


    ~/projects/vae/notebooks/src/results/neptune.py in <dictcomp>(.0)
         65         stim_set = pd.DataFrame.from_records(metadata)
         66 
    ---> 67         enc = {k:v for k,v in self.activations(test_data,n_units)}
         68         enc_pca = {}
         69         pca_objs = {}


    ~/projects/vae/notebooks/src/results/neptune.py in activations(self, test_data, n_units)
        135 
        136         layer_generator = self._gen_layers()
    --> 137         for l,n,d in self._gen_layers():
        138             yield n,sample_layer(l, test_data, batch_sz, n_units)
        139 


    ~/projects/vae/notebooks/src/results/neptune.py in _gen_layers(self)
         91 
         92     def _gen_layers(self):
    ---> 93         mod = self._build_model()
         94         pr = self.get_properties()
         95 


    ~/projects/vae/notebooks/src/results/neptune.py in _build_model(self)
         53         conf = self.get_config()
         54         print('building model {}(arch={}, recon={})...'.format(self.name,conf['encoder_arch'],conf['recon_weight']))
    ---> 55         m = next(load_models(self.proj_root, [self.experiment], load_weights = True, compile_model = True))
         56         m.name = self.name
         57         return m


    ~/projects/vae/notebooks/src/results/neptune.py in load_models(proj_root, experiments, load_weights, compile_model)
        197         with open(m_fp, 'r') as json_file:
        198             loaded_model_json = json_file.read()
    --> 199         mod = model_from_json(loaded_model_json)
        200 
        201         if load_weights:


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/engine/saving.py in model_from_json(json_string, custom_objects)
        490     config = json.loads(json_string)
        491     from ..layers import deserialize
    --> 492     return deserialize(config, custom_objects=custom_objects)
        493 
        494 


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/layers/__init__.py in deserialize(config, custom_objects)
         53                                     module_objects=globs,
         54                                     custom_objects=custom_objects,
    ---> 55                                     printable_module_name='layer')
    

    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/utils/generic_utils.py in deserialize_keras_object(identifier, module_objects, custom_objects, printable_module_name)
        143                     config['config'],
        144                     custom_objects=dict(list(_GLOBAL_CUSTOM_OBJECTS.items()) +
    --> 145                                         list(custom_objects.items())))
        146             with CustomObjectScope(custom_objects):
        147                 return cls.from_config(config['config'])


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/engine/network.py in from_config(cls, config, custom_objects)
       1020         # First, we create all layers and enqueue nodes to be processed
       1021         for layer_data in config['layers']:
    -> 1022             process_layer(layer_data)
       1023         # Then we process nodes in order of layer depth.
       1024         # Nodes that cannot yet be processed (if the inbound node


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/engine/network.py in process_layer(layer_data)
       1006 
       1007             layer = deserialize_layer(layer_data,
    -> 1008                                       custom_objects=custom_objects)
       1009             created_layers[layer_name] = layer
       1010 


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/layers/__init__.py in deserialize(config, custom_objects)
         53                                     module_objects=globs,
         54                                     custom_objects=custom_objects,
    ---> 55                                     printable_module_name='layer')
    

    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/utils/generic_utils.py in deserialize_keras_object(identifier, module_objects, custom_objects, printable_module_name)
        143                     config['config'],
        144                     custom_objects=dict(list(_GLOBAL_CUSTOM_OBJECTS.items()) +
    --> 145                                         list(custom_objects.items())))
        146             with CustomObjectScope(custom_objects):
        147                 return cls.from_config(config['config'])


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/engine/network.py in from_config(cls, config, custom_objects)
       1030                 if layer in unprocessed_nodes:
       1031                     for node_data in unprocessed_nodes.pop(layer):
    -> 1032                         process_node(layer, node_data)
       1033 
       1034         name = config.get('name')


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/engine/network.py in process_node(layer, node_data)
        989             # and building the layer if needed.
        990             if input_tensors:
    --> 991                 layer(unpack_singleton(input_tensors), **kwargs)
        992 
        993         def process_layer(layer_data):


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/engine/base_layer.py in __call__(self, inputs, **kwargs)
        455             # Actually call the layer,
        456             # collecting output(s), mask(s), and shape(s).
    --> 457             output = self.call(inputs, **kwargs)
        458             output_mask = self.compute_mask(inputs, previous_mask)
        459 


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/layers/core.py in call(self, inputs, training)
        124                                  seed=self.seed)
        125             return K.in_train_phase(dropped_inputs, inputs,
    --> 126                                     training=training)
        127         return inputs
        128 


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py in in_train_phase(x, alt, training)
       3121 
       3122     # else: assume learning phase is a placeholder tensor.
    -> 3123     x = switch(training, x, alt)
       3124     if uses_learning_phase:
       3125         x._uses_learning_phase = True


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py in switch(condition, then_expression, else_expression)
       3056         x = tf.cond(condition,
       3057                     then_expression_fn,
    -> 3058                     else_expression_fn)
       3059     else:
       3060         # tf.where needs its condition tensor


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
        505                 'in a future version' if date is None else ('after %s' % date),
        506                 instructions)
    --> 507       return func(*args, **kwargs)
        508 
        509     doc = _add_deprecated_arg_notice_to_docstring(


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/ops/control_flow_ops.py in cond(pred, true_fn, false_fn, strict, name, fn1, fn2)
       2147             (val_x.dtype.name, val_y.dtype.name))
       2148 
    -> 2149     merges = [merge(pair)[0] for pair in zip(res_f_flat, res_t_flat)]
       2150     merges = _convert_flows_to_tensorarrays(nest.flatten(orig_res_t), merges)
       2151 


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/ops/control_flow_ops.py in <listcomp>(.0)
       2147             (val_x.dtype.name, val_y.dtype.name))
       2148 
    -> 2149     merges = [merge(pair)[0] for pair in zip(res_f_flat, res_t_flat)]
       2150     merges = _convert_flows_to_tensorarrays(nest.flatten(orig_res_t), merges)
       2151 


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/ops/control_flow_ops.py in merge(inputs, name)
        464         return gen_control_flow_ops.ref_merge(inputs, name)
        465       else:
    --> 466         return gen_control_flow_ops.merge(inputs, name)
        467     elif all(isinstance(v, sparse_tensor.SparseTensor) for v in inputs):
        468       # Only handle the case when all inputs are SparseTensor.


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/ops/gen_control_flow_ops.py in merge(inputs, name)
        416   _attr_N = len(inputs)
        417   _, _, _op = _op_def_lib._apply_op_helper(
    --> 418         "Merge", inputs=inputs, name=name)
        419   _result = _op.outputs[:]
        420   _inputs_flat = _op.inputs


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(self, op_type_name, name, **keywords)
        786         op = g.create_op(op_type_name, inputs, output_types, name=scope,
        787                          input_types=input_types, attrs=attr_protos,
    --> 788                          op_def=op_def)
        789       return output_structure, op_def.is_stateful, op
        790 


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
        505                 'in a future version' if date is None else ('after %s' % date),
        506                 instructions)
    --> 507       return func(*args, **kwargs)
        508 
        509     doc = _add_deprecated_arg_notice_to_docstring(


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in create_op(***failed resolving arguments***)
       3298           input_types=input_types,
       3299           original_op=self._default_original_op,
    -> 3300           op_def=op_def)
       3301       self._create_op_helper(ret, compute_device=compute_device)
       3302     return ret


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in __init__(self, node_def, g, inputs, output_types, control_inputs, input_types, original_op, op_def)
       1821           op_def, inputs, node_def.attr)
       1822       self._c_op = _create_c_op(self._graph, node_def, grouped_inputs,
    -> 1823                                 control_input_ops)
       1824 
       1825     # Initialize self._outputs.


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in _create_c_op(graph, node_def, inputs, control_inputs)
       1640   for op_input in inputs:
       1641     if isinstance(op_input, (list, tuple)):
    -> 1642       c_api.TF_AddInputList(op_desc, [t._as_tf_output() for t in op_input])
       1643     else:
       1644       c_api.TF_AddInput(op_desc, op_input._as_tf_output())


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in <listcomp>(.0)
       1640   for op_input in inputs:
       1641     if isinstance(op_input, (list, tuple)):
    -> 1642       c_api.TF_AddInputList(op_desc, [t._as_tf_output() for t in op_input])
       1643     else:
       1644       c_api.TF_AddInput(op_desc, op_input._as_tf_output())


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in _as_tf_output(self)
        586     # cache of executor(s) stored for every session.
        587     if self._tf_output is None:
    --> 588       self._tf_output = c_api_util.tf_output(self.op._c_op, self.value_index)
        589     return self._tf_output
        590     # pylint: enable=protected-access


    ~/.pyenv/versions/fastai/lib/python3.6/site-packages/tensorflow/python/framework/c_api_util.py in tf_output(c_op, index)
        184   ret = c_api.TF_Output()
        185   ret.oper = c_op
    --> 186   ret.index = index
        187   return ret
        188 


    KeyboardInterrupt: 



```python
pca_dfs = []
recs = []
for mod,o in zip(att,obj):
    for k,v in mod.items():
        df = pd.DataFrame.from_records({'fve':v.explained_variance_,
                                        'fve_ratio':v.explained_variance_ratio_,
                                        'component':np.arange(len(v.explained_variance_))})
        df['arch'] = k[:4]
        df['depth'] = int(k[-1:])
        df['layer']=k
        df['objective']=o
        pca_dfs.append(df)
pca_80 = pd.concat(pca_dfs)
```


```python
pca_80['cum_fve'] = pca_80.groupby(['arch','objective','layer'])['fve_ratio'].transform('cumsum')

```


```python
pca_80.to_pickle('../data/cca/pca_80fve.pk')
```


```python
sns.set_context('talk')
g = sns.FacetGrid(data=pca_80,row='objective',col='arch',hue='depth',sharex='col',margin_titles=True,
                  ylim=(0,1),
                  height=4, palette='plasma',legend_out=True,
                 )
# plt.xscale('log')
g.map(sns.lineplot,'component','cum_fve').add_legend()
for a in g.axes.ravel():
    pass
#     a.set_xscale('log')
# g.map(plt.hlines,y=0.8,xmin=0,xmax=600,colors='k',linestyle='dashed')

# sns.lineplot(x='index'y='fve_ratio',hue='')
```


```python
count_pca_80 = pca_80.groupby(['arch','objective','layer'])['fve'].count().reset_index().rename(columns={'fve':'n_components'})
```


```python
g = sns.FacetGrid(data=count_pca_80,col='arch',row='objective',sharex='col',sharey=False,margin_titles=True,
#                   ylim=(0,1),
                  height=4, palette='plasma',legend_out=True,
                 )
# plt.xscale('log')
g.map(plt.bar,'layer','n_components')
g.fig.autofmt_xdate(rotation=45)
```

### Inspired by [BrainScore](https://www.biorxiv.org/content/10.1101/407007v1.full), we do PCA on whole layer activations and PLS/CCA on the PCA components


```python
dfs = []

pca_comps = [500, 250, 100, 100]
obj = ['no-recon','w/ recon','w/ recon','no-recon']

for exp,n_c,obj in zip(exps,pca_comps, obj):
    run = NeptuneExperimentRun(proj_root,neptune_exp=exp)
    xr,pca_objs = run.pca_assembly(scaled_sm_imgs, n_units=None, n_components=n_c, metadata=metadata)
    
    cca_df = cca(xr,neural_data[:,~neural_data.neuroid_id.isin(err_neuroids)],
                 variation=[0,3],cv=6, n_components=1,
                 region=None,brain_region=['IT','V4'],sortby='image_id')
    cca_df['objective']= obj
    cca_df['arch']=run.get_config()['encoder_arch']
    dfs.append(cca_df)

pca_cca_nc_1 = pd.concat(dfs)
```

    building model DPX-29(arch=convnet, recon=0.0)...
    Compiling model
    generating convolutional activations...


    PCA: conv_4(5760, 6272):   0%|          | 0/7 [00:00<?, ?it/s]

    PCA(n_components=500)...


    PCA: conv_1(5760, 12544): 100%|██████████| 7/7 [00:18<00:00,  3.97s/it]
    conv_1(3200, 500) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.99it/s, cca=0.444, fve=0.023, pearson=0.12]  
    conv_2(3200, 500) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.95it/s, cca=0.44, fve=0.0219, pearson=0.115]  
    conv_3(3200, 500) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.97it/s, cca=0.442, fve=0.0214, pearson=0.115] 
    conv_4(3200, 500) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.95it/s, cca=0.44, fve=0.021, pearson=0.115]   
    y_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00, 11.36it/s, cca=0.388, fve=0.0149, pearson=0.107] 
    z_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00, 10.32it/s, cca=0.323, fve=0.00889, pearson=0.0836]
    conv_1(3200, 500) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  3.18it/s, cca=0.456, fve=0.0599, pearson=0.22] 
    conv_2(3200, 500) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  3.43it/s, cca=0.448, fve=0.0562, pearson=0.207]
    conv_3(3200, 500) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  3.44it/s, cca=0.446, fve=0.0557, pearson=0.205]
    conv_4(3200, 500) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  3.37it/s, cca=0.445, fve=0.0546, pearson=0.203]
    y_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 20.09it/s, cca=0.392, fve=0.0386, pearson=0.173]
    z_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 19.72it/s, cca=0.316, fve=0.0127, pearson=0.102]  


    building model DPX-30(arch=convnet, recon=1.0)...
    Compiling model
    generating convolutional activations...


    PCA: conv_4(5760, 6272):   0%|          | 0/7 [00:00<?, ?it/s]

    PCA(n_components=250)...


    PCA: conv_1(5760, 12544): 100%|██████████| 7/7 [00:10<00:00,  2.20s/it]
    conv_1(3200, 250) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  5.53it/s, cca=0.451, fve=0.0231, pearson=0.12] 
    conv_2(3200, 250) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  5.89it/s, cca=0.451, fve=0.022, pearson=0.117] 
    conv_3(3200, 250) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  5.77it/s, cca=0.449, fve=0.0225, pearson=0.119] 
    conv_4(3200, 250) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  5.74it/s, cca=0.449, fve=0.0227, pearson=0.122]
    y_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00, 10.78it/s, cca=0.388, fve=0.0152, pearson=0.0944]
    z_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00, 10.97it/s, cca=0.418, fve=0.0189, pearson=0.106]
    conv_1(3200, 250) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00,  7.69it/s, cca=0.464, fve=0.0615, pearson=0.23] 
    conv_2(3200, 250) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00,  7.40it/s, cca=0.462, fve=0.0583, pearson=0.216]
    conv_3(3200, 250) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00,  7.44it/s, cca=0.457, fve=0.0559, pearson=0.208]
    conv_4(3200, 250) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00,  7.75it/s, cca=0.456, fve=0.055, pearson=0.205] 
    y_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 19.50it/s, cca=0.418, fve=0.0456, pearson=0.178]
    z_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 19.73it/s, cca=0.433, fve=0.0605, pearson=0.209]


    building model DPX-10(arch=dense, recon=1.0)...
    Compiling model
    Generating dense encoders...


    PCA: dense_2(5760, 2000):  67%|██████▋   | 4/6 [00:00<00:00, 20.35it/s]

    PCA(n_components=100)...


    PCA: dense_1(5760, 3000): 100%|██████████| 6/6 [00:01<00:00,  3.05it/s]
    dense_1(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00,  9.14it/s, cca=0.454, fve=0.0241, pearson=0.122]
    dense_2(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00,  8.96it/s, cca=0.454, fve=0.0238, pearson=0.121]
    dense_3(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00,  8.86it/s, cca=0.441, fve=0.0225, pearson=0.117]
    y_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00, 11.02it/s, cca=0.372, fve=0.0181, pearson=0.11] 
    z_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00, 10.78it/s, cca=0.365, fve=0.0172, pearson=0.101]
    dense_1(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 14.23it/s, cca=0.465, fve=0.0625, pearson=0.233]
    dense_2(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 15.21it/s, cca=0.464, fve=0.0622, pearson=0.234]
    dense_3(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 14.52it/s, cca=0.455, fve=0.0595, pearson=0.228]
    y_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 19.81it/s, cca=0.388, fve=0.0484, pearson=0.184]
    z_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 19.48it/s, cca=0.366, fve=0.0502, pearson=0.186]


    building model DPX-16(arch=dense, recon=0.0)...
    Compiling model
    Generating dense encoders...


    PCA: dense_2(5760, 2000):  67%|██████▋   | 4/6 [00:00<00:00, 21.01it/s]

    PCA(n_components=100)...


    PCA: dense_1(5760, 3000): 100%|██████████| 6/6 [00:01<00:00,  3.06it/s]
    dense_1(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00,  9.16it/s, cca=0.454, fve=0.0241, pearson=0.123]
    dense_2(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00,  9.00it/s, cca=0.453, fve=0.0232, pearson=0.12] 
    dense_3(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00,  8.73it/s, cca=0.442, fve=0.022, pearson=0.116] 
    y_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00, 10.20it/s, cca=0.382, fve=0.0141, pearson=0.0895]
    z_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:00<00:00, 10.58it/s, cca=0.38, fve=0.0141, pearson=0.106] 
    dense_1(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 14.35it/s, cca=0.465, fve=0.0626, pearson=0.233]
    dense_2(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 14.61it/s, cca=0.464, fve=0.0626, pearson=0.233]
    dense_3(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 14.50it/s, cca=0.457, fve=0.0623, pearson=0.232]
    y_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 19.18it/s, cca=0.393, fve=0.0599, pearson=0.208]
    z_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00, 19.15it/s, cca=0.388, fve=0.0583, pearson=0.209]



```python
pca_cca_nc_1.to_pickle('../data/cca/cca(1_component)_w_pca.pk')
```


```python
pca_cca_nc_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cca_r</th>
      <th>depth</th>
      <th>fve</th>
      <th>iter</th>
      <th>layer</th>
      <th>pearsonr</th>
      <th>region</th>
      <th>objective</th>
      <th>arch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.440154</td>
      <td>1</td>
      <td>0.018476</td>
      <td>0</td>
      <td>conv_1</td>
      <td>0.100639</td>
      <td>IT</td>
      <td>no-recon</td>
      <td>convnet</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.453577</td>
      <td>1</td>
      <td>0.024964</td>
      <td>1</td>
      <td>conv_1</td>
      <td>0.122054</td>
      <td>IT</td>
      <td>no-recon</td>
      <td>convnet</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.444036</td>
      <td>1</td>
      <td>0.025950</td>
      <td>2</td>
      <td>conv_1</td>
      <td>0.130113</td>
      <td>IT</td>
      <td>no-recon</td>
      <td>convnet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.438579</td>
      <td>1</td>
      <td>0.023511</td>
      <td>3</td>
      <td>conv_1</td>
      <td>0.129363</td>
      <td>IT</td>
      <td>no-recon</td>
      <td>convnet</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.448699</td>
      <td>1</td>
      <td>0.023180</td>
      <td>4</td>
      <td>conv_1</td>
      <td>0.121109</td>
      <td>IT</td>
      <td>no-recon</td>
      <td>convnet</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfs = []

pca_comps = [500, 250, 100, 100]
obj = ['no-recon','w/ recon','w/ recon','no-recon']

for exp,n_c,obj in zip(exps,pca_comps, obj):
    run = NeptuneExperimentRun(proj_root,neptune_exp=exp)
    xr,pca_objs = run.pca_assembly(scaled_sm_imgs, n_units=None, n_components=n_c, metadata=metadata)
    
    cca_df = cca(xr,neural_data[:,~neural_data.neuroid_id.isin(err_neuroids)],
                 variation=[0,3],cv=6, n_components=5,
                 region=None,brain_region=['IT','V4'],sortby='image_id')
    cca_df['objective']= obj
    cca_df['arch']=run.get_config()['encoder_arch']
    dfs.append(cca_df)

pca_cca_nc_5 = pd.concat(dfs)
```

    building model DPX-29(arch=convnet, recon=0.0)...
    Compiling model
    generating convolutional activations...


    PCA: conv_4(5760, 6272):   0%|          | 0/7 [00:00<?, ?it/s]

    PCA(n_components=500)...


    PCA: conv_1(5760, 12544): 100%|██████████| 7/7 [00:18<00:00,  3.94s/it]
    conv_1(3200, 500) x IT(3200, 168): 100%|██████████| 6/6 [00:10<00:00,  1.72s/it, cca=0.359, fve=0.0716, pearson=0.273]
    conv_2(3200, 500) x IT(3200, 168): 100%|██████████| 6/6 [00:10<00:00,  1.78s/it, cca=0.353, fve=0.0723, pearson=0.269]
    conv_3(3200, 500) x IT(3200, 168): 100%|██████████| 6/6 [00:10<00:00,  1.68s/it, cca=0.375, fve=0.0994, pearson=0.306]
    conv_4(3200, 500) x IT(3200, 168): 100%|██████████| 6/6 [00:10<00:00,  1.80s/it, cca=0.377, fve=0.113, pearson=0.323]
    y_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.74it/s, cca=0.309, fve=0.0277, pearson=0.239]
    z_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  3.12it/s, cca=0.278, fve=0.00255, pearson=0.223]
    conv_1(3200, 500) x V4(3200, 88): 100%|██████████| 6/6 [00:08<00:00,  1.42s/it, cca=0.369, fve=0.122, pearson=0.368]
    conv_2(3200, 500) x V4(3200, 88): 100%|██████████| 6/6 [00:08<00:00,  1.41s/it, cca=0.353, fve=0.124, pearson=0.385]
    conv_3(3200, 500) x V4(3200, 88): 100%|██████████| 6/6 [00:08<00:00,  1.46s/it, cca=0.355, fve=0.157, pearson=0.414]
    conv_4(3200, 500) x V4(3200, 88): 100%|██████████| 6/6 [00:09<00:00,  1.52s/it, cca=0.358, fve=0.17, pearson=0.421] 
    y_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  5.05it/s, cca=0.276, fve=0.0405, pearson=0.315]
    z_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  5.81it/s, cca=0.248, fve=-.0425, pearson=0.273]


    building model DPX-30(arch=convnet, recon=1.0)...
    Compiling model
    generating convolutional activations...


    PCA: conv_4(5760, 6272):   0%|          | 0/7 [00:00<?, ?it/s]

    PCA(n_components=250)...


    PCA: conv_1(5760, 12544): 100%|██████████| 7/7 [00:10<00:00,  2.12s/it]
    conv_1(3200, 250) x IT(3200, 168): 100%|██████████| 6/6 [00:04<00:00,  1.32it/s, cca=0.369, fve=0.0698, pearson=0.261]
    conv_2(3200, 250) x IT(3200, 168): 100%|██████████| 6/6 [00:04<00:00,  1.36it/s, cca=0.375, fve=0.081, pearson=0.274] 
    conv_3(3200, 250) x IT(3200, 168): 100%|██████████| 6/6 [00:04<00:00,  1.34it/s, cca=0.379, fve=0.0837, pearson=0.277]
    conv_4(3200, 250) x IT(3200, 168): 100%|██████████| 6/6 [00:04<00:00,  1.25it/s, cca=0.378, fve=0.0877, pearson=0.283]
    y_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  3.11it/s, cca=0.311, fve=0.0347, pearson=0.243]
    z_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  3.20it/s, cca=0.313, fve=0.0206, pearson=0.233]
    conv_1(3200, 250) x V4(3200, 88): 100%|██████████| 6/6 [00:03<00:00,  1.90it/s, cca=0.368, fve=0.0968, pearson=0.333]
    conv_2(3200, 250) x V4(3200, 88): 100%|██████████| 6/6 [00:03<00:00,  1.85it/s, cca=0.37, fve=0.127, pearson=0.38]  
    conv_3(3200, 250) x V4(3200, 88): 100%|██████████| 6/6 [00:03<00:00,  1.81it/s, cca=0.361, fve=0.129, pearson=0.383]
    conv_4(3200, 250) x V4(3200, 88): 100%|██████████| 6/6 [00:03<00:00,  1.82it/s, cca=0.361, fve=0.132, pearson=0.391]
    y_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00,  6.18it/s, cca=0.306, fve=0.0436, pearson=0.321]
    z_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  5.91it/s, cca=0.307, fve=0.0333, pearson=0.299]


    building model DPX-10(arch=dense, recon=1.0)...
    Compiling model
    Generating dense encoders...


    PCA: dense_2(5760, 2000):  67%|██████▋   | 4/6 [00:00<00:00, 21.95it/s]

    PCA(n_components=100)...


    PCA: dense_1(5760, 3000): 100%|██████████| 6/6 [00:01<00:00,  3.02it/s]
    dense_1(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.47it/s, cca=0.36, fve=0.0679, pearson=0.263] 
    dense_2(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.32it/s, cca=0.369, fve=0.0881, pearson=0.291]
    dense_3(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.39it/s, cca=0.367, fve=0.0907, pearson=0.294]
    y_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  3.21it/s, cca=0.303, fve=0.023, pearson=0.239] 
    z_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  3.19it/s, cca=0.289, fve=0.0227, pearson=0.209]
    dense_1(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  4.17it/s, cca=0.37, fve=0.0824, pearson=0.299] 
    dense_2(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  3.80it/s, cca=0.374, fve=0.0985, pearson=0.319]
    dense_3(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  3.76it/s, cca=0.365, fve=0.12, pearson=0.362]  
    y_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  4.72it/s, cca=0.289, fve=-.00133, pearson=0.276]
    z_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  5.22it/s, cca=0.279, fve=0.0287, pearson=0.29] 


    building model DPX-16(arch=dense, recon=0.0)...
    Compiling model
    Generating dense encoders...


    PCA: dense_2(5760, 2000):  67%|██████▋   | 4/6 [00:00<00:00, 21.92it/s]

    PCA(n_components=100)...


    PCA: dense_1(5760, 3000): 100%|██████████| 6/6 [00:01<00:00,  3.12it/s]
    dense_1(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.39it/s, cca=0.361, fve=0.0623, pearson=0.249]
    dense_2(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.41it/s, cca=0.362, fve=0.0745, pearson=0.272]
    dense_3(3200, 100) x IT(3200, 168): 100%|██████████| 6/6 [00:02<00:00,  2.36it/s, cca=0.358, fve=0.0761, pearson=0.277]
    y_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  3.20it/s, cca=0.306, fve=0.019, pearson=0.228] 
    z_enc(3200, 35) x IT(3200, 168): 100%|██████████| 6/6 [00:01<00:00,  3.07it/s, cca=0.294, fve=0.0309, pearson=0.219]
    dense_1(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  4.20it/s, cca=0.371, fve=0.0808, pearson=0.296]
    dense_2(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  4.03it/s, cca=0.369, fve=0.087, pearson=0.307] 
    dense_3(3200, 100) x V4(3200, 88): 100%|██████████| 6/6 [00:01<00:00,  4.12it/s, cca=0.364, fve=0.0902, pearson=0.312]
    y_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00,  6.16it/s, cca=0.312, fve=0.04, pearson=0.279]  
    z_enc(3200, 35) x V4(3200, 88): 100%|██████████| 6/6 [00:00<00:00,  6.17it/s, cca=0.311, fve=0.0603, pearson=0.292]



```python
pca_cca_nc_5.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cca_r</th>
      <th>depth</th>
      <th>fve</th>
      <th>iter</th>
      <th>layer</th>
      <th>pearsonr</th>
      <th>region</th>
      <th>objective</th>
      <th>arch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.354693</td>
      <td>1</td>
      <td>0.076837</td>
      <td>0</td>
      <td>conv_1</td>
      <td>0.275760</td>
      <td>IT</td>
      <td>no-recon</td>
      <td>convnet</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.367437</td>
      <td>1</td>
      <td>0.079543</td>
      <td>1</td>
      <td>conv_1</td>
      <td>0.282509</td>
      <td>IT</td>
      <td>no-recon</td>
      <td>convnet</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.359474</td>
      <td>1</td>
      <td>0.076044</td>
      <td>2</td>
      <td>conv_1</td>
      <td>0.277623</td>
      <td>IT</td>
      <td>no-recon</td>
      <td>convnet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.349500</td>
      <td>1</td>
      <td>0.058966</td>
      <td>3</td>
      <td>conv_1</td>
      <td>0.263896</td>
      <td>IT</td>
      <td>no-recon</td>
      <td>convnet</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.357205</td>
      <td>1</td>
      <td>0.071879</td>
      <td>4</td>
      <td>conv_1</td>
      <td>0.274738</td>
      <td>IT</td>
      <td>no-recon</td>
      <td>convnet</td>
    </tr>
  </tbody>
</table>
</div>




```python
pca_cca_nc_5.to_pickle('../data/cca/cca_5_component_w_pca.pk')
```


```python
sns.set_context('talk')
g = sns.FacetGrid(col='arch',row='region', hue='objective', data=pca_cca_nc_1,height=4,sharex=False,sharey='row',margin_titles=True)
g.map(sns.stripplot,'layer','fve').add_legend()
g.fig.autofmt_xdate(rotation=45)
```

    /home/elijahc/.pyenv/versions/fastai/lib/python3.6/site-packages/seaborn/axisgrid.py:715: UserWarning: Using the stripplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)



![png](output_26_1.png)



```python
sns.set_context('talk')
g = sns.FacetGrid(col='region',row='model', hue='arch', data=pca_cca_3,height=4,sharex=False,sharey='row',margin_titles=True)
g.map(sns.boxenplot,'depth','fve').add_legend()
g.fig.autofmt_xdate(rotation=45)
```


```python
sns.set_context('talk')
g = sns.FacetGrid(col='arch',row='region',hue='model', data=pca_cca_3,height=5,sharex=False)
g.map(sns.boxenplot,'layer','fve').add_legend()
g.fig.autofmt_xdate(rotation=45)
```


```python
sns.boxplot()
```


```python
cca_3.head()
```


```python
sns.set_context('talk')
g = sns.FacetGrid(col='arch',row='region',hue='model', data=cca_3,height=5,sharex=False,palette='viridis')
g.map(sns.boxenplot,'layer','pearsonr').add_legend()
g.fig.autofmt_xdate(rotation=45)
```


```python
sns.boxenplot(x='depth',y='fve',hue='model',data=cca_3.query('arch == "dense"'))
```


```python
sns.lineplot(x='layer',y='pearsonr',style='model',hue='region',
#                  data=conv_cca.query('{} == "{}"'.format(split_on,col)),
             data=cca_3.query('region == "IT"'),)
    
plt.xticks(rotation=45)
```


```python
cca_df_all = cca(xrs,neural_data,variation=[0,3,6],cv=35,region=['conv_1','conv_2','conv_3','conv_4','y_enc','z_enc'],sortby='image_id')
```


```python
sns.set_context('talk')
cca_df_all['model']='no-recon'
cca_df['model'] = 'no-recon'
g = sns.FacetGrid(col='region',row='model',data=cca_df,height=5)
g.map(sns.stripplot,'layer','pearsonr')
g.fig.autofmt_xdate(rotation=45)
```


```python
sns.lineplot(x='layer',y='pearsonr',style='model',hue='region',
#                  data=conv_cca.query('{} == "{}"'.format(split_on,col)),
             data=cca_df,)
    
plt.xticks(rotation=45)
```


```python
import src.rcca as rcca
from sklearn.cross_decomposition import CCA

def dicarlo_cca(data,stimulus_set,region,variation=[3],cv=10):
    
    
#     print(data.image_id.values)
#     print(nd.image_id.values)
    
    print('same order? \t',list(data.sortby('image_id').image_id.values) == list(nd.sortby('image_id').image_id.values))
    
    print('model.shape\t',data.shape)
    print('dicarlo.shape\t',nd.shape)
    out_dict = {'region':[],
#                 'variation':[],
#                 'rdm':[],
                'layer':[],
                'pearsonr':[],
                'p-value':[],
                'iter':[],
               }
    xrs = []
    ab_vectors = []
    ccas = []
        
    cv_tr = []
    cv_te = []
    
    num_images = data.shape[0]
    print(num_images)
    
    for rand_delta in np.arange(cv):
        tr_idx, te_idx, _,_ = train_test_split(np.arange(num_images),np.arange(num_images),train_size=0.75,random_state=np.random.randint(0,50)+rand_delta)
        cv_tr.append(tr_idx)
        cv_te.append(te_idx)
        
    for reg in region:
        sub_dat = data.sel(region=reg)
#         print(sub_dat)
        
        for brain_region in ['V4','IT']:
            
            pairing = '{} x {}'.format(reg,brain_region)
            for n, tr,te in tqdm(zip(np.arange(cv),cv_tr,cv_te),total=cv,desc=pairing):
                cca = CCA(n_components=1)
                cca.fit(sub_dat.values[tr],nd.sel(region=brain_region).values[tr])
            
                ab_vec = cca.transform(sub_dat.values[te],nd.sel(region=brain_region).values[te])
        
                r,pv = pearsonr(ab_vec[0],ab_vec[1])

                out_dict['region'].append(brain_region)
                out_dict['layer'].append(reg)
                out_dict['pearsonr'].append(r[0])
                out_dict['p-value'].append(pv[0])
                out_dict['iter'].append(n)
            
#             print(out_dict)
        
#         ccas.append(cca)
        
#         cca_score = r
        
#         cca_score = cca.score(sub_dat.values,nd.sel(region='IT').values)
        
#         cca = CCA(kernelcca = False, reg = 0.001, numCC = 2)
    
#         X_tr, X_te, y_tr, y_te = train_test_split(np.arange(2560),np.arange(2560))
        
#         data_vecs = [sub_dat.values,sub_dat.values,nd.sel(region='IT').values,nd.sel(region='IT').values]
        
#         idxs = [X_tr, X_te, y_tr, y_te]
        
#         X_tr,X_te, y_tr, y_te = tuple([d[idx] for d,idx in zip(data_vecs,idxs)])
        
# #         ,nd.sel(region='IT').values
        
#         print(X_tr.shape,y_tr.shape)
#         print(X_te.shape,y_te.shape)
        
#         cca.train([X_tr,y_tr])
        
#         cca_score = cca.validate([X_te,y_te])
#         print([t.shape for t in cca_score])

#         xrs.append(cca_score)
        
    return out_dict
        
```
