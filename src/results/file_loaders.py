import os
import json
import numpy as np
import pandas as pd

from keras.models import model_from_json

def load_performance(run_dir, conf, th=None, filename='performance.json'):
    path = os.path.join(run_dir,filename)
#     dirname,fname = os.path.split(path)
#     arch = dirname.split('/')[-2]
    if conf is None:
        return None
    elif conf['recon'] > 0 and conf['xent'] > 0:
        arch = 'both'
    elif conf['recon'] > 0:
        arch = 'only_recon'
    elif conf['xent'] > 0:
        arch = 'only_xent'

    if os.path.exists(path):
        with open(path, 'rb') as json:
            
            perf = pd.read_json(json)
            perf['architecture'] = arch

            perf['test_err'] = 1-perf['test_acc']
        if th is not None:
            perf['test_acc_max'] = th['val_class_acc'].values.max()
            perf['tt_overfit'] = np.argmax(th['val_class_acc'].values)
            tt_mem_bin_vec = (th['class_acc'].values<= 1-conf['label_corruption']).astype(int)
            perf['tt_memorization'] = tt_mem_bin_vec.sum()*3
            perf['test_err_min'] = 1-perf['test_acc_max']
            perf['test_acc_auc'] = np.trapz(th['val_class_acc'])
            perf['test_loss_auc'] = np.trapz(th['val_class_loss'],th['epoch'])
            perf['train_loss_auc'] = np.trapz(th['class_loss'])
        return perf

def load_model_spec(run_dir,filename='model.json'):
    path = os.path.join(run_dir,filename)

    if os.path.exists(path):
        with open(path,'r') as model_json:
            mod_spec = json.load(model_json)
            
        return mod_spec
    
def load_model(run_dir,mod_json_fn='model.json',w_fn='weights.h5'):
    mod_spec = load_model_spec(run_dir,mod_json_fn)
    
    model = model_from_json(json.dumps(mod_spec))
    model.load_weights(os.path.join(run_dir,w_fn))
    
    return model

def load_config(run_dir,filename='config.json'):
    path = os.path.join(run_dir,filename)
    if os.path.exists(path):
        with open(path,'r') as config_json:
            conf = json.load(config_json)
        return conf
    
def load_train_history(run_dir,conf,filename='train_history.parquet'):
    path = os.path.join(run_dir,filename)
    dirname,fname = os.path.split(path)
#     lab_corruption = np.round(float(dirname.split('/')[-1].split('_')[-1]),decimals=1)
#     arch = dirname.split('/')[-2]
    if conf is None:
        return None
    elif conf['recon'] > 0 and conf['xent'] > 0:
        obj = 'both'
    elif conf['recon'] == 0:
        obj = 'only_xent'
    else:
        obj = 'only_recon'

    if os.path.exists(path):
        hist = pd.read_parquet(path)
        hist['enc_arch'] = conf['enc_arch']
        hist['objective_type'] = obj
        hist['label_corruption'] = conf['label_corruption']
        hist['ecc_max'] = conf['ecc_max']
        hist['xent'] = conf['xent']
        hist['bg_noise'] = conf['bg_noise']
        hist['recon'] = conf['recon']
        hist['epoch'] = list(hist.index.values*3)
#         hist['val_loss'] = sma(hist['val_loss'].values,win_size=3)
#         hist['loss'] = sma(hist['loss'].values,win_size=3)
        hist['val_dL'] = np.gradient(hist['val_loss'])
        hist['test_err'] = 1-hist['val_class_acc']
        hist['train_err'] = 1-hist['class_acc']
        hist['recon_gen_err'] = hist.G_loss - hist.val_G_loss
        hist['gen_err'] = hist.loss - hist.val_loss
        hist['class_gen_err'] = hist.class_loss - hist.val_class_loss
        hist['class_gen_acc'] = hist.class_acc - hist.val_class_acc

        return hist