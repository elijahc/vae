import numpy as np
import skimage as skim
from hashlib import md5
from edcutils.datasets import bsds500
from edcutils.image import get_patch
from scipy.ndimage import rotate
from keras.utils import to_categorical
from .data_loader import norm_im, rescale_contrast
from .plot import plot_img_row
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .results.utils import FMNIST_CATEGORIES


class MNISTAugmented(object):
    def __init__(self,dataset='mnist',scale=2,
                 rotation=0.75,
                 translation=0.75,
                 scaling_range=0.4,
                 flatten=False,
                 seed=None,
                 bg_noise=None,
                 contrast_level=1,
                 noise_mode=None,
                 noise_kws=None,
                 batch_size=32,
                 autoload=True,
                ):
        
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.input_shape = (28*self.scale,28*self.scale,1)
        self.batch_size=batch_size
        self.scaler = None
        
        if autoload:
            self._load_dataset()
        
        self.num_train = len(self.y_train)
        self.num_test =  len(self.y_test)
        
    def __size__(self):
        return int(self.num_train/self.batch_size)
        
    def _load_dataset(self):
        from keras.datasets import mnist
        (self.x_tr, y_train),(self.x_te, y_test) = mnist.load_data()
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_oh = to_categorical(y_train)
        self.y_test_oh = to_categorical(y_test)
        self.num_classes = self.y_train_oh.shape[-1]
        
    def gen_pan_deltas(self,step=2):
        px_max = int(28/2*(self.scale-1)*self.translation)
        x_span = y_span = np.linspace(-px_max,px_max,num=(2*px_max/step)+1,dtype=int)
        
        t2b = [(dy,dx) for dx,dy in zip(x_span,y_span)]
        r2l = [(px_max,dx) for dx in np.flip(x_span)]

        b2t = [(dy,dx) for dx,dy in zip(x_span,np.flip(y_span))]
        r2l2 = [(-px_max,dx) for dx in np.flip(x_span)]
        out = []
        out.extend(t2b)
        out.extend(r2l[1:])
        out.extend(b2t[1:])
        out.extend(r2l2[1:])
        return out
    
    def gen_pan_X(self, pX_fg, bg_imgs):
        if bg_imgs is None and self.bg_imgs is not None:
            bg_imgs = self.bg_imgs
            
        pX_bg = self.gen_backgrounds(X_fg,bg_imgs)
        pX = self.rasterize([pX_bg.copy(),pX_fg],blend=self.blend)
        pX = self.scaler.transform(pX.reshape(pX.shape[0],-1)).reshape(*pX.shape)

        return pX
        
    def gen_pan_test(self,im_stack,output,dxs,dys,scale=2):
        num_im = len(im_stack)
        for i in range(num_im):
            dx = dxs[i]
            dy = dys[i]
            letter = im_stack[i]
            
            new_im = self.regen_shift_image(letter,dx,dy,scale)
                
            output[i] = np.reshape(new_im,(1,)+self.input_shape)
            
        output = norm_im(output,self.flatten)

    def batch(self, data, labels, batch_size):
        n_samples,x_sz,y_sz = data.shape
        
        if batch_size > n_samples:
            batch_idxs = np.random.choice(np.arange(n_samples),size=batch_size,replace=True)
        else:
            batch_idxs = np.random.choice(np.arange(n_samples),size=batch_size,replace=False)
        X_ = data[batch_idxs].reshape(batch_size,x_sz,y_sz,1)
        
        X = np.zeros((batch_size,)+self.input_shape)

        y = labels[batch_idxs]
        y_int = np.argmax(y,axis=1).tolist()
        metadata = {
            'object_index':batch_idxs,
            'category_id':y_int,
            'size': [self.scale * 28.0] * batch_size,
            's':[1.0]*batch_size,
            'degrees': [self.scale * 28.0 / 256 * 8.0]*batch_size,
            'image_id':[],
        }
        
        rotations = []
        dx = []
        dy = []

        for j in np.arange(batch_size):
            new_im, rot = self.rotate_image(X_[j],self.rotation,reshape=False)
            rotations.append(rot)
            new_im, offsets = self.shift_image(new_im,self.translation)
            dx.append(offsets[0])
            dy.append(offsets[1])
            h = md5()
            h.update('{}_{}_{}_{}'.format(batch_idxs[j],dx[j],dy[j],rot).encode())
            metadata['image_id'].append(h.hexdigest())
            X[j]=new_im
        
        metadata['rxy'] = np.array(rotations)
        metadata['dx'] = np.array(dx)
        metadata['tx'] = np.array(dx)/(self.scale*28.0)
        metadata['dy'] = np.array(dy)
        metadata['ty'] = np.array(dy)/(self.scale*28.0)
                    
        X_fg = norm_im(X,flatten=False)
        X_sm = norm_im(X_,flatten=False)
        
        if not hasattr(self,'sm_scaler'):
            self.sm_scaler = MinMaxScaler(feature_range=(-1,1))
            self.sm_scaler.fit(X_sm.reshape(X_sm.shape[0],-1))
        
        X_bg = np.zeros_like(X)

        
        return X_fg,X_sm,y,metadata
    
    def gen_test_batches(self, num_batches, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        for i in range(num_batches):
            X_fg,X_sm,y,metadata = self.batch(self.x_te,self.y_test_oh,
                                batch_size=batch_size,
                                bg_imgs=bg_imgs,)
            
                    
            if not hasattr(self,'scaler') or self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(-1,1))
                self.scaler.fit(X_fg.reshape(X_fg.shape[0],-1))
            
            X_fg = self.scaler.transform(X_fg.reshape(X_fg.shape[0],-1)).reshape(*X_fg.shape)
            X_fg = np.clip(X_fg,-1,1)
        
            X_sm = self.sm_scaler.transform(X_sm.reshape(X_sm.shape[0],-1)).reshape(*X_sm.shape)
            X_sm = np.clip(X_sm,-1,1)
            
            images = {
                'whole':X_fg,
                'foreground':X_fg,
                'object':X_sm
            }
            
            yield images,y,metadata
        
    def gen_train_batches(self,num_batches, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        for i in range(num_batches):
            X_fg,X_sm,y,metadata = self.batch(self.x_tr,self.y_train_oh,
                             batch_size=batch_size)
        
            X_fg = self.scaler.transform(X_fg.reshape(X_fg.shape[0],-1)).reshape(*X_fg.shape)
            X_fg = np.clip(X_fg,-1,1)
        
            X_sm = self.sm_scaler.transform(X_sm.reshape(X_sm.shape[0],-1)).reshape(*X_sm.shape)
            X_sm = np.clip(X_sm,-1,1)
            
            images = {
                'whole':X_fg,
                'foreground':X_fg,
                'object':X_sm
            }
            
            yield images,y,metadata
    
    def gen_bg_noise(self,im_stack,mode='uniform',**noise_kws):
        # Generate the noise
        if mode is 'uniform':
            if 'width' not in noise_kws.keys():
                noise_kws['width'] = 0.8
            
            if 'amount' not in noise_kws.keys():
                noise_kws['amount'] = 1
            
            print('creating noise uniform({})...'.format(noise_kws))
            noise_im = self.gen_uniform_noise(im_stack,**noise_kws)
        else:
            print('creating noise {}({})...'.format(mode,noise_kws))
            noise_im = self.gen_skimage_noise(im_stack,mode=mode,**noise_kws)
        
        return noise_im
    
    def _translate_image(self,X,dx,dy):
        # Expects images in channel-last format
        # [x,y,c]
        
        x_sz,y_sz,c = X.shape
        assert x_sz==y_sz
        
        bg_size = (x_sz*self.scale,y_sz*self.scale,c)
        
        new_im = np.zeros(bg_size)
        try:
            new_im[dx:dx+x_sz,dy:dy+y_sz] = X
        except ValueError as e:
            print('dx: {} dy: {}'.format(dx,dy))
            raise e
        
        return new_im
    
    def shift_image(self,X,max_translation):
        if X.ndim != 3:
            raise ValueError('X must be ndim=3 but is {}'.format(X.ndim))
        x_sz,y_sz,c = X.shape
        assert x_sz==y_sz
        bg_size = (x_sz*self.scale,y_sz*self.scale,c)

        if max_translation is not None:
            center = int(bg_size[0]-x_sz)/2
            delta = int(center*max_translation)

            dx = int(np.random.randint(center-delta,center+delta))
            dy = int(np.random.randint(center-delta,center+delta))

        else:
            dx = int(center)
            dy = int(center)
        
        return self._translate_image(X,dx,dy),[dx,dy]
    
    def rotate_image(self,X,rot_max=0.5,reshape=True):
        if X.ndim != 3:
            raise ValueError('X must be ndim=3 but is {}'.format(X.ndim))
            
        if rot_max is not None:
            angle_range = [rot_max*-180,rot_max*180]
            rot = int( np.random.randint(angle_range[0],angle_range[1]) )

            rot_im = rotate(X,angle=rot,reshape=reshape)
        else:
            rot_im = X
            rot=0
        
        return rot_im,rot
        
    def gen_uniform_noise(self,im_stack,width=1,amount=1):
        bg_size = im_stack.shape
        hw = width/2.0
        noise_bg = np.random.uniform(low=-hw,high=hw,size=bg_size)
        if amount < 1:
            revert_idxs = np.random.uniform(size=bg_size)>amount
            noise_bg[revert_idxs]=im_stack[revert_idxs]
            
#         noise_bg = norm_im(new_bg,flatten=False)
        
        return noise_bg
    
    def _load_uniform_backgrounds(self,contrast=1.0):
        bg_stack = self.gen_bg_noise(im_stack=np.zeros(shape=(50,512,512,1)))
        bg_imgs = [bg_stack[i].reshape(512,512,1) for i in np.arange(50)]
        if bg_contrast < 1.0:
            bg_imgs = [rescale_contrast(im,bg_contrast) for im in bg_imgs]
                
        return bg_imgs
    
    def gen_skimage_noise(self,im_stack,mode='gaussian',**noise_kws):
        noise_ims = skim.util.random_noise(im_stack,mode=mode,**noise_kws)        
        return noise_ims
    
    def plot_example(self,bg=None,bg_contrast=1.0):
        batch = next(self.gen_train_batches(1,bg=bg,bg_contrast=bg_contrast))
        idx = np.random.randint(self.batch_size)
        
        plot_img_row([im[idx] for im in batch[:-1]])
        
        return batch


class FashionMNIST(MNISTAugmented):
    def __init__(self,scale=2,
                 rotation=0.75,
                 translation=0.75,
                 scaling_range=0.4,
                 flatten=False,
                 seed=None,
                 bg_noise=None,
                 contrast_level=1,
                 noise_mode=None,
                 noise_kws=None,
                 batch_size=32,
                 blend=None,
                ):
        
        super().__init__(scale=2,
                 rotation=0.75,
                 translation=0.75,
                 scaling_range=0.4,
                 flatten=False,
                 seed=None,
                 bg_noise=None,
                 contrast_level=1,
                 noise_mode=None,
                 noise_kws=None,
                 batch_size=32)
    
    def batch(self, data, labels, batch_size):
        X,Xsm,y,metadata = super().batch(data, labels, batch_size)
        
        y_cat = [FMNIST_CATEGORIES[i] for i in metadata['category_id']]
        metadata['category_name'] = y_cat
        
        return X,Xsm,y,metadata
        
    def _load_dataset(self):
        from keras.datasets import fashion_mnist
        (self.x_tr, y_train),(self.x_te, y_test) = fashion_mnist.load_data()
        
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_oh = to_categorical(y_train)
        self.y_test_oh = to_categorical(y_test)
        self.num_classes = self.y_train_oh.shape[-1]  