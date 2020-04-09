import os
import numpy as np
import pandas as pd
import xarray
import hashlib
import random
import json
from collections import OrderedDict
from keras.models import Model


def raw_to_xr(encodings,l_2_depth,stimulus_set):
    obj_names = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Dress Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    all_das = []
    for layer,activations in encodings.items():
        neuroid_n = activations.shape[1]
        n_idx = pd.MultiIndex.from_arrays([
            pd.Series(['{}_{}'.format(layer,i) for i in np.arange(neuroid_n)],name='neuroid_id'),
            pd.Series([l_2_depth[layer]]*neuroid_n,name='layer'),
            pd.Series([layer]*neuroid_n,name='region')
        ])
        
        p_vars = [stimulus_set[k] for k in stimulus_set.keys()]
        
        if 'tx' not in stimulus_set.keys():
            tx = pd.Series(stimulus_set.dx.values/28, name='tx')
            p_vars.append(tx)
            
        if 'ty' not in stimulus_set.keys():
            ty = pd.Series(stimulus_set.dy.values/28, name='ty')
            p_vars.append(ty)
        
        if 's' not in stimulus_set.keys():
            p_vars.append(pd.Series([1.0]*len(stimulus_set),name='s'))
            
        p_idx = pd.MultiIndex.from_arrays(p_vars)
        da = xarray.DataArray(activations.astype('float32'),
                         coords={'presentation':p_idx,'neuroid':n_idx},
                         dims=['presentation','neuroid'])
        all_das.append(da)
        
    return xarray.concat(all_das,dim='neuroid')

def save_assembly(da,run_dir,fname,**kwargs):
    da = da.reset_index(da.coords.dims)
    da.attrs = OrderedDict()
    with open(os.path.join(run_dir,fname), 'wb') as fp:
        da.to_netcdf(fp,**kwargs)

DEFAULT_DPRIME_MODE = 'binary'

def dprime(A, B=None, mode=DEFAULT_DPRIME_MODE,\
        max_value=np.inf, min_value=-np.inf,\
        max_ppf_value=np.inf, min_ppf_value=-np.inf,\
        **kwargs):
    """Computes the d-prime sensitivity index of predictions
    from various data formats.  Depending on the choice of
    `mode`, this function can take one of the following format:
    * Binary classification outputs (`mode='binary'`; default)
    * Positive and negative samples (`mode='sample'`)
    * True positive and false positive rate (`mode='rate'`)
    * Confusion matrix (`mode='confusionmat'`)
    Parameters
    ----------
    A, B:
        If `mode` is 'binary' (default):
            A: array, shape = [n_samples],
                True values, interpreted as strictly positive or not
                (i.e. converted to binary).
                Could be in {-1, +1} or {0, 1} or {False, True}.
            B: array, shape = [n_samples],
                Predicted values (real).
        If `mode` is 'sample':
            A: array-like,
                Positive sample values (e.g., raw projection values
                of the positive classifier).
            B: array-like,
                Negative sample values.
        If `mode` is 'rate':
            A: array-like, shape = [n_groupings]
                True positive rates
            B: array-like, shape = [n_groupings]
                False positive rates
        if `mode` is 'confusionmat':
            A: array-like, shape = [n_classes (true), n_classes (pred)]
                Confusion matrix, where the element M_{rc} means
                the number of times when the classifier or subject
                guesses that a test sample in the r-th class
                belongs to the c-th class.
            B: ignored
    mode: {'binary', 'sample', 'rate'}, optional, (default='binary')
        Directs the interpretation of A and B.
    max_value: float, optional (default=np.inf)
        Maximum possible d-prime value.
    min_value: float, optional (default=-np.inf)
        Minimum possible d-prime value.
    max_ppf_value: float, optional (default=np.inf)
        Maximum possible ppf value.
        Used only when mode is 'rate' or 'confusionmat'.
    min_ppf_value: float, optional (default=-np.inf).
        Minimum possible ppf value.
        Used only when mode is 'rate' or 'confusionmat'.
    kwargs: named arguments, optional
        Passed to ``confusion_matrix_stats()`` and used only when `mode`
        is 'confusionmat'.  By assigning ``collation``,
        ``fudge_mode``, ``fudge_factor``, etc. one can
        change the behavior of d-prime computation
        (see ``confusion_matrix_stats()`` for details).
    Returns
    -------
    dp: float or array of shape = [n_groupings]
        A d-prime value or array of d-primes, where each element
        corresponds to each grouping of positives and negatives
        (when `mode` is 'rate' or 'confusionmat')
    References
    ----------
    http://en.wikipedia.org/wiki/D'
    http://en.wikipedia.org/wiki/Confusion_matrix
    """

    # -- basic checks and conversion
    if mode == 'sample':
        pos, neg = np.array(A), np.array(B)

    elif mode == 'binary':
        y_true, y_pred = A, B

        assert len(y_true) == len(y_pred)
        assert np.isfinite(y_true).all()

        y_true = np.array(y_true)
        assert y_true.ndim == 1

        y_pred = np.array(y_pred)
        assert y_pred.ndim == 1

        i_pos = y_true > 0
        i_neg = ~i_pos

        pos = y_pred[i_pos]
        neg = y_pred[i_neg]

    elif mode == 'rate':
        TPR, FPR = np.array(A), np.array(B)
        assert TPR.shape == FPR.shape

    elif mode == 'confusionmat':
        # A: confusion mat
        # row means true classes, col means predicted classes
        P, N, TP, _, FP, _ = confusion_matrix_stats(A, **kwargs)

        TPR = TP / P
        FPR = FP / N

    else:
        raise ValueError('Invalid mode')

    # -- compute d'
    if mode in ['sample', 'binary']:
        assert np.isfinite(pos).all()
        assert np.isfinite(neg).all()

        if pos.size <= 1:
            raise ValueError('Not enough positive samples'\
                    'to estimate the variance')
        if neg.size <= 1:
            raise ValueError('Not enough negative samples'\
                    'to estimate the variance')

        pos_mean = pos.mean()
        neg_mean = neg.mean()
        pos_var = pos.var(ddof=1)
        neg_var = neg.var(ddof=1)

        num = pos_mean - neg_mean
        div = np.sqrt((pos_var + neg_var) / 2.)

        dp = num / div

    else:   # mode is rate or confusionmat
        ppfTPR = norm.ppf(TPR)
        ppfFPR = norm.ppf(FPR)
        ppfTPR = np.clip(ppfTPR, min_ppf_value, max_ppf_value)
        ppfFPR = np.clip(ppfFPR, min_ppf_value, max_ppf_value)
        dp = ppfTPR - ppfFPR

    # from Dan's suggestion about clipping d' values...
    dp = np.clip(dp, min_value, max_value)

    return dp

def get_layer_encoders(m,layer_names,input=None,new_names=None):
    if input is None:
        input = m.layers[0].input
    layer_outs = [m.get_layer(name=ln).output for ln in layer_names]
    
    if new_names is not None and len(new_names)==len(layer_names):
        layer_names = new_names
        
    for out,name in zip(layer_outs,layer_names):
        yield Model(input,out,name=name)
        
def sample_layer(l,test_data,batch_sz,n_sample_units=None):
    n_samples = test_data.shape[0]
    l_enc = l.predict(test_data,batch_size=batch_sz)
    n_units = np.prod(l_enc.shape[1:])
    
    if n_sample_units is None:
        n_sample_units = n_units
    
    l_enc = l_enc.reshape(n_samples,n_units)
    
    if n_units < n_sample_units:
        # Can't sample more than max units
        idxs = np.arange(n_units)
    else:
        idxs = np.random.choice(np.arange(n_units),size=n_sample_units,replace=False)
    
    return l_enc[:,idxs]

def pca_layer(l,test_data,batch_sz,n_components=5, n_sample_units=None, **pca_kws):
    
    
    return l_pca
    