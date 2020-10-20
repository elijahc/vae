import numpy as np
import skimage as skim
from edcutils.datasets import bsds500
from edcutils.image import get_patch
from scipy.ndimage import rotate
from keras.utils import to_categorical
from .data_loader import norm_im, rescale_contrast
from .plot import plot_img_row
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .results.utils import FMNIST_CATEGORIES
from .augmented_stimuli import MNISTAugmented,FashionMNIST

class MNISTM(MNISTAugmented):
    # http://jmlr.org/papers/volume17/15-239/15-239.pdf
    
    def __init__(self, bg_scale=2, rotation=0.75, translation=0.75,flatten=False,seed=None,bg_contrast=0.8,batch_size=32,blend=None,**kwargs):
        self.bg_contrast = bg_contrast
        self.bg_imgs = self._load_natural_backgrounds(self.bg_contrast)
        self.blend = blend
        
        super().__init__(scale=bg_scale,
                 rotation=rotation,
                 flatten=flatten,
                 seed=seed,
                 batch_size=batch_size,
                 **kwargs)
        
        self._load_dataset()
        self._fg_batch = super().batch
    
    def _load_natural_backgrounds(self,contrast=0.8):
        bg_imgs = None
        print('loading bsds500...')
        bg_imgs,_ = bsds500.load_data()
        bg_imgs = [np.expand_dims(skim.color.rgb2gray(im),-1) for im in bg_imgs]
        if contrast < 1.0:
            bg_imgs = [rescale_contrast(im,contrast) for im in bg_imgs]
        return bg_imgs
                
    def gen_backgrounds(self,X,bg_imgs,out=None,random_seed=None,with_idxs=False):
        if random_seed is None:
            rand = np.random.RandomState(7)

        if len(X.shape) == 3:
            n,w,h = X.shape
        elif len(X.shape) == 4:
            n,w,h,ch = X.shape

        dtype = bg_imgs[0].dtype
        if out is None:
            out = np.zeros([n,w,h,ch], dtype)

#         import pdb; pdb.set_trace()
        for i in np.arange(n):
            n_bg_imgs = len(bg_imgs)
            bg_idxs = rand.choice(np.arange(n_bg_imgs))
            bg_im = bg_imgs[bg_idxs]
            bg = get_patch(image=bg_im,shape=(w,h))
            if len(bg.shape)==2:
                bg = np.stack([bg.reshape(w,h,1)]*ch,-1)
            out[i] = bg
        if with_idxs:
            return out, bg_idxs
        else:
            return out
    
    def rasterize(self,image_volumes,blend=None):
        bg = image_volumes[0]
        for v in image_volumes[1:]:
            if blend == 'difference':
                bg = np.abs(bg - v)
            else:
                border_mask = v<0.2

                bg[~border_mask] = v[~border_mask]

                # alpha blend the rest
                bg[border_mask] = (v[border_mask]*v[border_mask]) + (1-v[border_mask])*bg[border_mask]
            
        return bg.clip(0.0,1.0)
    
    def batch(self, data, labels, batch_size=None, bg_imgs=None):
        bg_imgs = bg_imgs or self.bg_imgs
        batch_size = batch_size or self.batch_size
                
        X_fg,X_sm,y,metadata = self._fg_batch(data=data,labels=labels,batch_size=batch_size)
        X_bg = self.gen_backgrounds(X_fg,bg_imgs)        
        
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(-1,1))        
            self.scaler.fit(X_bg.reshape(X_bg.shape[0],-1))
        
        X = self.rasterize([X_bg,X_fg],blend=self.blend)
        X = self.scaler.transform(X.reshape(X.shape[0],-1)).reshape(*X.shape)
        X = np.clip(X,-1,1)
            
        X_bg = self.scaler.transform(X_bg.reshape(X_bg.shape[0],-1)).reshape(*X_bg.shape)
        X_bg = np.clip(X_bg,-1,1)
            
        X_fg = self.scaler.transform(X_fg.reshape(X_fg.shape[0],-1)).reshape(*X_fg.shape)
        X_fg = np.clip(X_fg,-1,1)
        
        X_sm = self.sm_scaler.transform(X_sm.reshape(X_sm.shape[0],-1)).reshape(*X_sm.shape)
        X_sm = np.clip(X_sm,-1,1)
        
        return X,X_fg,X_bg,X_sm,y,metadata
    
    def gen_eval_batches(self, num_batches, batch_size=None, bg_imgs=None,n_objects=100):
        eval_idxs = np.random.choice(np.arange(self.num_test),size=n_objects,replace=False)
        for i in range(num_batches):
            X,X_fg,X_bg,X_sm,y,metadata =  self.batch(data=self.x_te[eval_idxs],labels=self.y_test_oh[eval_idxs], batch_size=batch_size,bg_imgs=bg_imgs)
            
            metadata['object_index'] = [eval_idxs[i] for i in metadata['object_index']]
            
            images = {
                'whole':X,
                'background':X_bg,
                'foreground':X_fg,
                'object':X_sm
            }
            
            yield images,y,metadata
    
    def gen_test_batches(self, num_batches, batch_size=None, bg_imgs=None):
        for i in range(num_batches):
            X,X_fg,X_bg,X_sm,y,metadata =  self.batch(data=self.x_te,labels=self.y_test_oh, batch_size=batch_size,bg_imgs=bg_imgs)
            
            images = {
                'whole':X,
                'background':X_bg,
                'foreground':X_fg,
                'object':X_sm
            }
            
            yield images,y,metadata
            
    def gen_train_batches(self, num_batches, batch_size=None, bg_imgs=None):
        for i in range(num_batches):
            X,X_fg,X_bg,X_sm,y,metadata =  self.batch(data=self.x_tr,labels=self.y_train_oh, batch_size=batch_size)

            images = {
                'whole':X,
                'background':X_bg,
                'foreground':X_fg,
                'object':X_sm
            }
            
            yield images,y,metadata
    
class FashionMNISTM(MNISTM):
    def _load_dataset(self):
        from keras.datasets import fashion_mnist
        (self.x_tr, y_train),(self.x_te, y_test) = fashion_mnist.load_data()
        
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_oh = to_categorical(y_train)
        self.y_test_oh = to_categorical(y_test)
        self.num_classes = self.y_train_oh.shape[-1]
        
        
    def batch(self, data, labels, batch_size,bg_imgs=None):
        
        X,X_fg,X_bg,X_sm,y,metadata = super().batch(data=data, labels=labels, batch_size=batch_size, bg_imgs=bg_imgs)
        
        y_cat = [FMNIST_CATEGORIES[i] for i in metadata['category_id']]
        
        metadata['category_name'] = y_cat
        obj_names = []
        for cat in np.unique(y_cat):
            o_idxs = [oi for oi,c in zip(metadata['object_index'],metadata['category_name']) if c==cat]
            seen_idxs = []
            obj_idx_names = {}
            counter = 1
            for idx in o_idxs:
                if str(idx) not in obj_idx_names.keys():
                    obj_idx_names[str(idx)] = '{}_{}'.format(cat,counter)
                    counter+=1
                obj_names.append(obj_idx_names[str(idx)])
                
        metadata['object_name'] = obj_names
        
        return X,X_fg,X_bg,X_sm,y,metadata
    
ShiftedDataBatcher = MNISTAugmented