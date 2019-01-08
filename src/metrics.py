import numpy as np

def var_expl(features,cond,bins=11):
    Z = features
    num_features = Z.shape[-1]
    Z_mean = Z.mean(axis=0)
    Z_std = Z.std(axis=0)

    Z_bins = np.zeros((bins,num_features))
    Z_bin_idx = np.zeros_like(Z,dtype=np.int)
    Z_cond_var = np.zeros((bins,num_features))


    for i,z_mu,z_sigma in zip(np.arange(num_features),Z_mean,Z_std):
        bin_r = np.linspace(z_mu-(2*z_sigma),z_mu+(2*z_sigma),bins)
        Z_bins[:,i] = bin_r
        Z_bin_idx[:,i] = np.digitize(Z[:,i],bins=bin_r)

    for i in np.arange(num_features):
        for j in np.arange(bins):
            Z_cond_var[j,i] = cond[np.where(Z_bin_idx[:,i]==j)[0]].var()

    return Z_cond_var

def norm_var_expl(features,cond,bins=11):
    fve = var_expl(features,cond,bins)

    return np.nan_to_num((cond.var()-fve)/np.nan_to_num(cond.var()))

def eval_corr_var(features,cond,bins=11,):
    pass