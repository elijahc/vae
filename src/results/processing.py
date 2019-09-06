import xarray
import numpy as np
import pandas as pd

from .experiments import _DateExperimentLoader

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

def make_xr(df,layer_depth,conf=None,region=None,var_name='unit',value_name='activity',id_vars=['class','dx','dy']):
    if region is None:
        region = 'Layer_{}'.format(layer_depth)
    p_idx = pd.MultiIndex.from_arrays([
        pd.Series(df['class'].values,name='numeric_label'),
        pd.Series([obj_names[c] for c in df['class'].values],name='object_name'),
        df['dx'],
        df['dy'],
        pd.Series(df['dx'].values/28, name='tx'),
        pd.Series(df['dy'].values/28, name='ty'),
        pd.Series([1.0]*len(df),name='s'),
    ])
    unit_cols = df.columns[:-len(id_vars)]
    n_idx = pd.MultiIndex.from_arrays([
        pd.Series(unit_cols,name='neuroid_id'),
        pd.Series([layer_depth]*len(unit_cols),name='layer',dtype='int8'),
        pd.Series([region]*len(unit_cols),name='region'),
    ])

    da = xarray.DataArray(df.values[:,:-len(id_vars)].astype(np.float32),
                          coords={'presentation':p_idx,'neuroid':n_idx,},dims=['presentation','neuroid'],
                         )
    return da