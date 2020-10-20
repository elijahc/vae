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

from sklearn.metrics import explained_variance_score
from sklearn.multioutput import MultiOutputRegressor,MultiOutputEstimator


# class Evaluator(object):
#     def __init__(self, n_splits=None,cv=None,train_size=0.75, x_regions=None, y_regions=None, variation=[3],sortby='image_id',random_state=None):
#         self.cv = 5
#         self.train_size=0.75
#         self.n_splits=10
#         self.random_state = random_state
        
#         self._iter_idx = 0
        
#         if cv is not None:
#             np.arange()
#             for i in range(self.cv):
#                 self._split()
    
#     def _split(self,X, random_state):
#         tr,te = train_test_split(X, train_size=self.train_size, random_state=self.random_state)
        
#         return tr,te
    
#     def __next__(self):
#         if 
#         i = 0
#         for rand_delta in np.arange(cv):
#             tr_idx, te_idx, _,_ = train_test_split(np.arange(num_images),np.arange(num_images),
#                                                    train_size=train_size,
#                                                    random_state=np.random.randint(0,50)+rand_delta)
#         cv_tr.append(tr_idx)
#         cv_te.append(te_idx)

def cca(x,neural_data,region=None, brain_region=['IT','V4'], cv=5, n_components=5, variation=[0,3,6],sortby='image_id',train_size=0.75):
#     var_lookup = stimulus_set[stimulus_set.variation.isin(variation)].image_id.values
#     x = x.where(x.image_id.isin(var_lookup),drop=True)
#     nd = neural_data.where(neural_data.image_id.isin(var_lookup),drop=True)
    
    x = x.sortby(sortby)
    nd = neural_data.sortby(sortby)
    
    assert list(getattr(x,sortby).values) == list(getattr(nd,sortby).values)
    num_images = x.shape[0]
    out_recs = []
    
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
                    
                    fve = explained_variance_score(y_true,y_pred,multioutput='raw_values')
                    r_vals = [pearsonr(y_pred[:,i],y_true[:,i]) for i in range(y_pred.shape[-1])]
                    
                    cca_r = np.mean([pearsonr(u[:,i],v[:,i]) for i in np.arange(n_components)])

#                     r_vals = [pearsonr(ab_vec[0][:,i],ab_vec[1][:,i]) for i in range(ab_vec[0].shape[-1])]
                    
                    r_mean.append(np.mean([r for r,v in r_vals]))
                    cca_mean.append(cca_r)
                    fve_mean.append(np.mean(fve))
                    
                    for rv,f,nid in zip(r_vals,fve,nd_reg[te].neuroid_id.values):                    
                        out_recs.append({
                            'region':br,
                            'layer':reg,
                            'pearsonr': rv[0],
                            'cca_r':cca_r,
                            'fve':f,
                            'iter':n,
                            'depth':depth,
                            'neuroid_id':nid,
                        })
                    
                    t.set_postfix(pearson=np.mean(r_mean), cca=np.mean(cca_mean), fve=np.mean(fve_mean))
                    
    return pd.DataFrame.from_records(out_recs)

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