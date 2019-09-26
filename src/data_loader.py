import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tqdm import tqdm as tqdm
from tqdm import trange
import skimage as skim
from scipy.ndimage import rotate
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from edcutils.datasets import bsds500
from edcutils.image import get_patch


from tqdm import tqdm as tqdm

def get_loader(config):
    if config.dataset is 'mnist':
        from keras.datasets import mnist
        return prepare_keras_dataset(mnist)
    elif config.dataset is 'fashion_mnist':
        from keras.datasets import fashion_mnist
        return prepare_keras_dataset(fashion_mnist)

def norm_to_8bit(X,flatten=True):
    X = (X*255).astype('int8')
    if flatten:
        X = X.reshape( (len(X), np.prod(X.shape[1:])) )
    
    return X

def norm_im(X,flatten=True):
    X = X.astype('float32') / 255.
    if flatten:
        X = X.reshape( (len(X), np.prod(X.shape[1:])) )
    
    return X

def rescale_contrast(im_stack,c_level):
    out_im = im_stack * c_level
    out_im += (1-c_level)/2
    
    return out_im

def prepare_keras_dataset(x_train,y_train,x_test,y_test):
    # (x_train, y_train), (x_test, y_test) = k_data.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # print(x_train.shape)
    # print(x_test.shape)
    return (x_train,y_train),(x_test,y_test)

def upsample_dataset(dataset,n,axis=0,scale_idxs=False,dataset_sz=None):
    if not scale_idxs and dataset_sz is None:
        dataset_sz = dataset.shape[axis]
    out = [dataset]
    
    # Extend by whole integers of the dataset
    while n > dataset_sz:
        out.append(dataset)
        n += -dataset_sz
    
    # Extend by the last partial fragment
    ext_idxs = np.arange(n)
    out.append(dataset[ext_idxs])

    return np.concatenate(out,axis=axis)

def _fast_tx(max_translation,size=1,scale=2,seed=7,im_shape=(56,56)):
    np.random.seed(seed)
    (x_sz,y_sz) = im_shape
    bg_size = (28*scale,28*scale)
    dx_max = dy_max = int(28/2*(self.scale-1)*max_translation)
    
    tx={
        'dx':[],
        'dy':[],
    }
    for _ in np.arange(size):
        dx = int(np.random.randint(-dx_max,dx_max)+x_sz/2)
        dy = int(np.random.randint(-dy_max,dy_max)+y_sz/2)

        dx = max(dx,0)
        dx = min(dx,bg_size[0]-x_sz)
        tx['dx'].append(dx)
        
        dy = max(dy,0)
        dy = min(dy,bg_size[0]-y_sz)
        tx['dy'].append(dy)
        
    return tx

def _shift_image(X,dx,dy,scale=2):
    if len(X.shape) == 2:
        (x_sz,y_sz) = X.shape
        n=1

    elif len(X.shape) == 3:
        (n,x_sz,y_sz) = X.shape
        
    bg_size = (n,x_sz*scale,y_sz*scale)
    dx += int(x_sz/2)
    dy += int(y_sz/2)
    new_im = np.zeros(bg_size)

    for i in np.arange(n):  
        new_im[i,dx:dx+x_sz,dy:dy+y_sz] = X[i]

    return new_im
        
class Shifted_Data_Loader():
    def gen_uniform_noise(self,im_stack,width=1,amount=1):
        bg_size = im_stack.shape
        hw = width/2.0
        noise_bg = np.random.uniform(low=-hw,high=hw,size=bg_size)
        if amount < 1:
            revert_idxs = np.random.uniform(size=bg_size)>amount
            noise_bg[revert_idxs]=im_stack[revert_idxs]
            
#         noise_bg = norm_im(new_bg,flatten=False)
        
        return noise_bg
        
    def gen_skimage_noise(self,im_stack,mode='gaussian',**noise_kws):
        noise_ims = skim.util.random_noise(im_stack,mode=mode,**noise_kws)
#         noise_ims = norm_to_8bit(noise_ims,flatten=self.flatten)
        
        return noise_ims
    
    def gen_bg_noise(self,im_stack,mode='uniform',**noise_kws):
        # Generate the noise
        if mode is 'uniform':
            if 'width' not in noise_kws.keys():
                noise_kws['width'] = self.bg_noise
            
            if 'amount' not in noise_kws.keys():
                noise_kws['amount'] = 1
            
            print('creating noise uniform({})...'.format(noise_kws))
            noise_im = self.gen_uniform_noise(im_stack,**noise_kws)
        else:
            print('creating noise {}({})...'.format(mode,noise_kws))
            noise_im = self.gen_skimage_noise(im_stack,mode=mode,**noise_kws)
        
        return noise_im
    
    def __init__(self,dataset,
                 scale=2,
                 rotation=0.75,
                 translation=0.75,
                 scaling_range=0.4,
                 flatten=True,
                 num_train=60000,
                 autoload=True,
                 seed=None,
                 bg_noise=None,
                 contrast_level=1,
                 bg=None,
                 noise_mode=None,
                 bg_only=True,
                 noise_kws=None,
                 blend=None,
                ):
        self.scale=scale
        self.dataset=dataset
        self.num_train=num_train
        self.seed = seed
        
        # Image Manipulations
        self.rotation = rotation
        self.translation = translation
        self.scaling_range = scaling_range
        self.flatten = flatten
        self.contrast_level = contrast_level

        # Background
        self.bg = bg
        self.bg_noise = bg_noise
        self.bg_only = bg_only
        self.blend = blend

        # Noise
        self.noise_mode = noise_mode
        if noise_kws is None:
            self.noise_kws={}
        elif noise_kws is not None:
            self.noise_kws=noise_kws
        elif bg_noise is not None and self.noise_mode is None:
            self.noise_mode = 'uniform'
            self.noise_kws = {
                'width': bg_noise,
                'amount': 1.0,
            }
        
        self.flatten_arr = lambda X: X.reshape( (len(X), np.prod(X.shape[1:])) )
                
        if self.rotation is not None:
            self.rotation=float(self.rotation)
            assert self.rotation >= 0.0
            assert self.rotation <= 1.0
        
        if self.translation is not None:
            self.translation=float(self.translation)
            assert self.translation >= 0.
            assert self.translation <= 1.
            
        if flatten:
            self.input_shape = (784*self.scale*self.scale,)
        else:
            self.input_shape = (28*self.scale,28*self.scale,1)
            
        if seed is None:
            np.random.seed(7)
            
        print('input_shape: ',self.input_shape)
        print('dataset: ',self.dataset)
        print('background: ', self.bg)
        print('blend mode: ', self.blend)
        print('scale: ',self.scale)
        print('tx_max: ', self.translation)
        print('rot_max: ', self.rotation)
        print('contrast_level: ', self.contrast_level)
        print('noise_mode: ', self.noise_mode)
        
        noise_kws_strs = ['  {}: {}'.format(k,self.noise_kws[k]) for k in self.noise_kws.keys()]
        for s in noise_kws_strs:
            print(s)
#         print('bg_noise:', self.bg_noise)

        
#         print('loading {}...'.format(self.dataset))
        if dataset=='mnist':
            from keras.datasets import mnist
            (x_train, y_train),(x_test, y_test) = mnist.load_data()

        elif dataset=='fashion_mnist':
            from keras.datasets import fashion_mnist
            (x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
        
        if self.num_train > 60000:
            num_add = int(self.num_train-60000)
            x_train = upsample_dataset(x_train,n=num_add)
            y_train = upsample_dataset(y_train,n=num_add)
        
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_oh = to_categorical(y_train)
        self.y_test_oh = to_categorical(y_test)
        self.num_classes = self.y_train_oh.shape[-1]
        
        num_train = len(self.y_train)
        num_test =  len(self.y_test)

        self.sx_train = np.zeros((num_train,)+self.input_shape)
        self.sx_test =  np.zeros((num_test,)+self.input_shape)
        
        self.bg_train = np.zeros_like(self.sx_train)
        self.bg_test = np.zeros_like(self.sx_test)
        
        
        if self.bg == 'natural':
            print('building background images...')
            bg_imgs,_ = bsds500.load_data()
            bg_imgs = [np.expand_dims(skim.color.rgb2gray(im),-1) for im in bg_imgs]
            self.bg_train = self.gen_backgrounds(self.sx_train,bg_imgs)
#             self.bg_train = np.expand_dims(skim.color.rgb2gray(self.bg_train),-1)
            
            self.bg_test = self.gen_backgrounds(self.sx_test,bg_imgs)
#             self.bg_test = np.expand_dims(skim.color.rgb2gray(self.bg_test),-1)
            
        self.bg_combined = np.concatenate([self.bg_train,self.bg_test],axis=0)
        
        if self.noise_mode is not None:
            self.regen_bg_noise(mode=self.noise_mode,**noise_kws)
            
            
        self.delta_train = np.empty((num_train,3))
        self.delta_test = np.empty((num_test,3))
#         print('sx_train: ',self.sx_train.shape)
        
        self.x_train_orig = x_train.copy()
        self.x_test_orig = x_test.copy()
        self.x_train = norm_im(x_train,flatten)
        self.x_test = norm_im(x_test,flatten)

        self.training_data = lambda :(self.sx_train,self.y_train_oh)
        if autoload:
            self.gen_new_shifted(x_train,x_test,flatten)        
        
    
    def add_natural_bg(self,im_stack,bg_imgs,blend='difference',bg_contrast=1):
        n, w, h, _ = im_stack.shape
        
        X_ = np.zeros([n,w,h,3],np.uint8)
        for i in np.arange(n):
            d = im_stack[i]
            d = d.reshape([w, h, 1]) * 255
            d = d.astype(np.int)
            d = np.concatenate([d, d, d], 2)
            
            bg_img = np.random.choice(bg_imgs)
            d = compose_image(d,bg_img,blend=blend)
            
            X_[i] = d
            
        return X_
        
    def gen_backgrounds(self,X,bg_imgs,rand=None,out=None):
        if rand is None:
            rand = np.random.RandomState(7)

        if len(X.shape) == 3:
            n,w,h = X.shape
        elif len(X.shape) == 4:
            n,w,h,ch = X.shape

        dtype = bg_imgs[0].dtype
        if out is None:
            out = np.zeros([n,w,h,ch], dtype)

        for i in trange(n,desc='loading background'):
            bg = get_patch(image=rand.choice(bg_imgs),shape=(w,h))
            if len(bg.shape)==2:
                bg = np.stack([bg.reshape(w,h,1)]*ch,-1)
            out[i] = bg
        
        out = rescale_contrast(out,c_level=0.5)
        
        return out

    def add_noise(self,im_stack,noise_bg,fg_mask=None):
        # Add background noise to whole image
        im_stack += noise_bg
        if fg_mask is not None:
            # Subtract noise that occurs on the fg
            im_stack -= noise_bg*fg_mask
            
    def rasterize(self,image_volumes,blend=None):
        bg = image_volumes[0]
        for v in image_volumes[1:]:
            if blend == 'difference':
                bg = np.abs(bg - v)
            else:
                mask = v>0.05
                v[mask] = v[mask]*0.80
                bg[mask] = bg[mask]*0.20
                bg += v
            
        return bg.clip(0.0,1.0)
    
    def regen_bg_noise(self,mode=None,**noise_kws):
        if mode is None:
            mode = self.noise_mode
        
        bg_combined_blank = np.zeros_like(self.bg_combined)
        self.bg_combined = self.gen_bg_noise(bg_combined_blank,mode=mode,**noise_kws)
        self.bg_train = self.bg_combined[:len(self.sx_train)]
        self.bg_test = self.bg_combined[len(self.sx_train):]
        if self.flatten:
            self.bg_train = self.flatten_arr(self.bg_train)
            self.bg_test = self.flatten_arr(self.bg_test)   
        
    def gen_new_shifted(self,x_train,x_test,flatten=True):
        
#         print('transforming: ')
        self.transform_im(x_train,self.sx_train,self.delta_train,msg='train images')

#         print('making testing data...')
        self.transform_im(x_test,self.sx_test,self.delta_test,msg='test_images')
        
#         (self.sx_train,_),(self.sx_test,_) = prepare_keras_dataset(self.sx_train,y_train,self.sx_test,y_test)
        self.dx = (self.delta_train[:,0],self.delta_test[:,0])
        self.dy = (self.delta_train[:,1],self.delta_test[:,1])
        self.dtheta = (self.delta_train[:,2],self.delta_test[:,2])
        
        self.meta_train = pd.DataFrame.from_records(
            {
                'dx':self.dx[0],
                'dy':self.dy[0],
                'rotation':self.dtheta[0]
            })
        
        # Normalize and flatten data
        self.sx_train = norm_im(self.sx_train,flatten,)
        self.sx_test = norm_im(self.sx_test,flatten,)
        self.fg_mask_train = self.sx_train>0.05
        self.fg_mask_test = self.sx_test>0.05
        
        # Rescale Contrast
        if self.contrast_level < 1.0:
            print('Rescaling contrast to {}'.format(self.contrast_level))
            self.sx_train = rescale_contrast(self.sx_train,c_level=self.contrast_level)
            self.sx_test = rescale_contrast(self.sx_test,c_level=self.contrast_level)
        
        # Save foreground copies
        self.fg_train = self.sx_train.copy()
        self.fg_test = self.sx_test.copy()
        
        # 
        if self.bg is not None:
            self.sx_train = self.rasterize([self.bg_train.copy(), self.sx_train],blend=self.blend)
            self.sx_test = self.rasterize([self.bg_test.copy(), self.sx_test],blend=self.blend)
        
        # Check if background should have added noise
        if self.noise_mode is not None:
            # Update bg_tr/te by creating noise and adding it to the images
            print('adding noise to training set')
            self.add_noise(self.sx_train, self.bg_train, fg_mask=self.fg_mask_train, bg_only=self.bg_only)
            self.sx_train = np.clip(self.sx_train,0,1)
            
            print('adding noise to test set')
            self.add_noise(self.sx_test, self.bg_test, fg_mask=self.fg_mask_test, bg_only=self.bg_only)
            self.sx_test = np.clip(self.sx_test,0,1)
    
#         self.sx_test = norm_im(self.sx_test,flatten)
        
    def transform_im(self,im_stack,output,delta,msg='transforming'):
        num_im = len(im_stack)
        for i in trange(num_im,desc=msg):
            letter = im_stack[i]
            if self.rotation is not None:
                letter,rot = self.rotate_image(letter,rot_max=self.rotation)
            else:
                new_im = letter
                rot = [0]
                
            new_im,offsets = self.shift_image(letter,max_translation=self.translation)
                
            output[i] = np.reshape(new_im,(1,)+self.input_shape)
            offsets.extend(rot)
            delta[i] = np.array(offsets)
                    
    def shift_image(self,X,max_translation):
        (x_sz,y_sz) = X.shape
        bg_size = (28*self.scale,28*self.scale)

        if max_translation is not None:
            dx_max = dy_max = int(28/2*(self.scale-1)*max_translation)

            dx = int(np.random.randint(-dx_max,dx_max)+x_sz/2)
            dy = int(np.random.randint(-dy_max,dy_max)+y_sz/2)

            dx = max(dx,0)
            dx = min(dx,bg_size[0]-x_sz)

            dy = max(dy,0)
            dy = min(dy,bg_size[0]-y_sz)
        else:
            dx = int(x_sz/2)
            dy = int(y_sz/2)
            
        
        new_im = np.zeros(bg_size)
        new_im[dx:dx+x_sz,dy:dy+y_sz] = X
        
        return new_im,[dx,dy]


    def rotate_image(self,X,rot_max=0.5,reshape=True):
        if rot_max is not None:
            angle_range = [rot_max*-180,rot_max*180]
            rot = int( np.random.randint(angle_range[0],angle_range[1]) )

            rot_im = rotate(X,angle=rot,reshape=reshape)
            # Calculate the added margins from rotation
            # crop_margin = np.floor(np.array(rot_im.shape)-np.array([28,28])).astype(np.int)
            # if crop_margin[0]>0:
                # Crop to center
                # new_im = rot_im[crop_margin[0]:crop_margin[0]+28,crop_margin[1]:crop_margin[1]+28]
            # else:
            new_im = rot_im
        else:
            new_im = X
            rot=0
        
        return new_im,[rot]
    
    def from_deltas(self,im_stack,output,dxs,dys,scale=2):
        num_im = len(im_stack)
        for i in range(num_im):
            dx = dxs[i]
            dy = dys[i]
            letter = im_stack[i]
            
            new_im = self.regen_shift_image(letter,dx,dy,scale)
                
            output[i] = np.reshape(new_im,(1,)+self.input_shape)
            
        output = norm_im(output,self.flatten)
    
    def regen_shift_image(self,X,dx,dy,scale=2):
        (x_sz,y_sz) = X.shape
        bg_size = (28*scale,28*scale)
        new_im = np.zeros(bg_size)

        new_im[dx:dx+x_sz,dy:dy+y_sz] = X

        return new_im
    
    def gen_corrupted_shift_image(self,corr_idxs,corr_labels):
        print('generating train_sx_corr...')
        self.sx_train_corrupted = self.sx_train.copy()
#         self.sx_train_corrupted = np.squeeze(self.sx_train_corrupted)
        masks_by_label = [self.y_train==n for n in np.arange(10)]
        idxs_by_label = [np.arange(len(self.x_train))[m] for m in masks_by_label]
        false_options = [list(filter(lambda x: x!= i,np.arange(10))) for i in np.arange(10)]
        for idx in tqdm(corr_idxs):
            new_lab = np.random.choice(false_options[self.y_train[idx]])
            (dx,dy,dr) = self.delta_train[idx]
            new_X = self.x_train_orig[np.random.choice(idxs_by_label[new_lab])]
            new_im = self.regen_shift_image(new_X,int(dx),int(dy),scale=self.scale)
            new_im = norm_im(np.reshape(new_im,(1,)+self.input_shape),self.flatten)
            self.sx_train_corrupted[idx] = new_im
    
    def train_generator(self,batch_size):
        while True:
            x_batch = np.empty((batch_size,)+self.input_shape)
            train_delta = np.empty((batch_size,3))
            y_lat = None
            for idx in np.arange(int(60000.0/batch_size)):
                x_stack = self.x_train_orig[idx * batch_size:(idx + 1) * batch_size]
                self.transform_im(x_stack,x_batch,train_delta)
                
                y_lat = self.y_train_oh[idx * batch_size:(idx + 1) * batch_size]
                
#                 if self.rotation is not None:
#                     letter,rot = self.rotate_image(letter,rot_max=self.rotation)
#                 else:
#                     new_im = letter
#                     rot = [0]
                    
#                 new_im,offsets = self.shift_image(letter,max_translation=self.translation)
                
#                 output = np.reshape(new_im,self.input_shape)
#                 offsets.extend(rot)
#                 deltas = np.array(offsets)
                
#                 x_batch.append(output)
#                 generator.append(output)
                
            yield x_batch,{'y_lat':y_lat,'real_fake':to_categorical(np.ones(batch_size)),'generator':x_batch.copy()}