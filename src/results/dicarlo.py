import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


MED_IMAGES = 'med_imgs_112x112.npy'

SM_IMAGES = 'sm_imgs_56x56.npy'

def get_dicarlo_su(proj_dir,fn='su_selectivity_dicarlo_hi_var.pqt'):
    return pd.read_parquet(os.path.join(proj_dir,'data',fn))

err_neuroids = ['Tito_L_P_8_5', 'Tito_L_P_7_3', 'Tito_L_P_7_5', 'Tito_L_P_5_1', 'Tito_L_P_9_3',
                        'Tito_L_P_6_3', 'Tito_L_P_7_4', 'Tito_L_P_5_0', 'Tito_L_P_5_4', 'Tito_L_P_9_6',
                        'Tito_L_P_0_4', 'Tito_L_P_4_6', 'Tito_L_P_5_6', 'Tito_L_P_7_6', 'Tito_L_P_9_8',
                        'Tito_L_P_4_1', 'Tito_L_P_0_5', 'Tito_L_P_9_9', 'Tito_L_P_3_0', 'Tito_L_P_0_3',
                        'Tito_L_P_6_6', 'Tito_L_P_5_7', 'Tito_L_P_1_1', 'Tito_L_P_3_8', 'Tito_L_P_1_6',
                        'Tito_L_P_3_5', 'Tito_L_P_6_8', 'Tito_L_P_2_8', 'Tito_L_P_9_7', 'Tito_L_P_6_7',
                        'Tito_L_P_1_0', 'Tito_L_P_4_5', 'Tito_L_P_4_9', 'Tito_L_P_7_8', 'Tito_L_P_4_7',
                        'Tito_L_P_4_0', 'Tito_L_P_3_9', 'Tito_L_P_7_7', 'Tito_L_P_4_3', 'Tito_L_P_9_5']

def process_dicarlo(assembly,avg_repetition=True, variation=[0, 3, 6], tasks=['ty','tz','rxy']):
    stimulus_set = assembly.attrs['stimulus_set']
    stimulus_set['dy_deg'] = stimulus_set.tz*stimulus_set.degrees
    stimulus_set['dx_deg'] = stimulus_set.ty*stimulus_set.degrees
    stimulus_set['dy_px'] = stimulus_set.dy_deg*32
    stimulus_set['dx_px'] = stimulus_set.dx_deg*32
    
    assembly.attrs['stimulus_set'] = stimulus_set
    
    groups = ['category_name', 'object_name', 'image_id']+tasks
    if not avg_repetition:
        groups.append('repetition')
        
    data = assembly.multi_groupby(groups)     # (2)
    data = data.mean(dim='presentation')
    data = data.squeeze('time_bin')    #   (3)
#     data.attrs['stimulus_set'] = stimulus_set.query('variation == {}'.format(variation))
    data = data.T
    data = data[stimulus_set.variation.isin(variation),:]
    
    return data

def dicarlo_slug(stimulus_set):
    stim = stimulus_set
    slug = [(dx,dy,lab,float(rxy)) for dx,dy,rxy,lab in zip(stim.dx_px.values,stim.dy_px.values,stim.rxy.values,stim.category_name.values)]
    
    return slug


def load_dicarlo_images(proj_root,filename,normalize=True):
    fp = os.path.join(proj_root,'data','dicarlo_images',filename)
    
    images = np.load(fp)
    
    if normalize:
        scaler = MinMaxScaler(feature_range=(-1,1))
#         Xm,Xs = (images.mean(),images.std())
        images = np.clip(scaler.fit_transform(images.reshape(images.shape[0],-1)), -1,1).reshape(*images.shape)
    return images

load_md_images = lambda proj_root, normalize=True:load_dicarlo_images(proj_root,MED_IMAGES,normalize)
load_sm_images = lambda proj_root, normalize=True:load_dicarlo_images(proj_root,SM_IMAGES,normalize)