import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy.stats import norm

Z = norm.ppf

def r(activations,prop):
    '''
    Expects activations in shape [n_trials, n_units]
    '''
    r = []
    num_units = activations.shape[1]

    for i in np.arange(num_units):
        r_,p_ = pearsonr(activations[:,i],prop)
        r.append(np.abs(r_))
        
    return r
    

def dprime(df,num_units=10,col='class',mask_missing=True):        
    uniq_cls = np.unique(df[col].values)
    o = []
    r = []
    F = []
    H = []
    d = []
    if mask_missing:
        mask = list(~np.any(df.groupby(col).var().values[:,:num_units]==0,axis=0))
        cols = df.columns[mask+([True]*3)]
    else:
        cols = df.columns
    mdf = df[cols]
    for i in uniq_cls:
        oi = mdf[mdf[col]==i].values[:,:len(mdf.columns)-3]
        o_mu = oi.mean(axis=0)
        o_kde = [gaussian_kde(oi[:,u]) for u in np.arange(oi.shape[1])]
        o.append(o_kde)

        ri = mdf[mdf[col]!=i].values[:,:len(mdf.columns)-3]
        r_kde = [gaussian_kde(ri[:,u]) for u in np.arange(ri.shape[1])]
        r.append(r_kde)
        F.append([k.integrate_box_1d(om,k.dataset.max()) for k,om in zip(r_kde,o_mu)])
        H.append([k.integrate_box_1d(low,om) for k,om,low in zip(o_kde,o_mu,oi.min(axis=0))])
        d.append(Z(H[-1])-Z(F[-1]))
    
    d = np.abs(d)
    class_d_max = np.argmax(d,axis=0)
    d_idxs = (class_d_max,np.arange(len(class_d_max)))
    d = d[d_idxs]
    return d,mask

def selectivity(df,num_units=10,col='class'):
    mu = df.groupby(col).mean().values[:,:num_units]
    var = df.groupby(col).var().values[:,:num_units]

    mu_max_idxs = np.argmax(mu,axis=0)
    mu_min_idxs = np.argmin(mu,axis=0)

    mu_max = np.array([mu[maxi,i] for i,maxi in zip(np.arange(len(mu_max_idxs)),mu_max_idxs)])
    mu_min = np.array([mu[maxi,i] for i,maxi in zip(np.arange(len(mu_min_idxs)),mu_min_idxs)])
    
    var_b = np.array([var[maxi,i] for i,maxi in zip(np.arange(len(mu_max_idxs)),mu_max_idxs)])
    var_w = np.array([var[mini,i] for i,mini in zip(np.arange(len(mu_min_idxs)),mu_min_idxs)])
#     neg_max = [(allv-mu_max)/9.0 for allv,mu_max in zip(all_vals.sum(axis=1),mu_max)]
    
    sel = [(mu_b-mu_w)/np.sqrt((vb+vw)/2) for mu_b,mu_w,vb,vw in zip(mu_max,mu_min,var_b,var_w)]
    return sel
