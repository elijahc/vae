import numpy as np
from sklearn.manifold import Isomap
from keras.datasets import fashion_mnist
from src.metrics import var_expl

delta_test = np.load('data/delta_test.npy')
dxs = delta_test[:,0]-14
dys = delta_test[:,1]-14
y_class_enc = np.load('data/y_class_enc.npy')
y_test_oh = np.load('data/y_test_oh.npy')
z_enc = np.load('data/z_mean_enc.npy')

z_cond_dx_var = var_expl(features=z_enc,cond=dxs,bins=15)
z_cond_dy_var = var_expl(features=z_enc,cond=dys,bins=15)
fve_zdx = (dxs.var()-z_cond_dx_var)/dxs.var()
fve_zdy = (dys.var()-z_cond_dy_var)/dys.var()
