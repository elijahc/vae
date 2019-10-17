import numpy as np
import skimage as skim
from edcutils.datasets import bsds500
from edcutils.image import get_patch
from keras.utils import to_categorical
from .data_loader import norm_im, rescale_contrast

class ShiftedDataBatcher():
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
                 batch_size=32,
                 blend=None,
                ):
        
        self.translation = translation
        self.scale = scale
        self.bg = bg
        self.input_shape = (28*self.scale,28*self.scale,1)
        self.blend = blend
        self.batch_size=batch_size
        self.bg_imgs,_ = bsds500.load_data()
        self.bg_imgs = [np.expand_dims(skim.color.rgb2gray(im),-1) for im in self.bg_imgs]
        
        if dataset=='mnist':
            from keras.datasets import mnist
            (self.x_tr, y_train),(self.x_te, y_test) = mnist.load_data()


        elif dataset=='fashion_mnist':
            from keras.datasets import fashion_mnist
            (self.x_tr, y_train),(self.x_te, y_test) = fashion_mnist.load_data()
            
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_oh = to_categorical(y_train)
        self.y_test_oh = to_categorical(y_test)
        self.num_classes = self.y_train_oh.shape[-1]
        
        self.num_train = len(self.y_train)
        self.num_test =  len(self.y_test)
        
    def __size__(self):
        return int(self.num_train/self.batch_size)
    
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
        
        return pX
        
    
    def gen_batch(self, data, labels, batch_size, bg_imgs=None):
        n_samples = data.shape[0]
        batch_idxs = np.random.choice(np.arange(n_samples),size=batch_size)
        X_ = data[batch_idxs]
        X = np.zeros((batch_size,)+self.input_shape)
#         X = np.ones((batch_size,)+self.input_shape)
        y = labels[batch_idxs]
            
        for j in np.arange(batch_size):
            new_im, offsets = self.shift_image(X_[j],self.translation)
            X[j]=np.expand_dims(new_im,-1)
            
        X_fg = norm_im(X,flatten=False)
        if bg_imgs is not None:
            X_bg = self.gen_backgrounds(X_fg,bg_imgs)
                
        else:
            X_bg = np.zeros_like(X)
            
        X = self.rasterize([X_bg.copy(),X_fg],blend=self.blend)
        Xm,Xs = (X.mean(),X.std())
        X = np.clip((X-Xm)/Xs,-1,1)
        X_fg = np.clip((X_fg-Xm)/Xs,-1,1)
            
        return X,X_fg,y
        
    def gen_test_batches(self, num_batches, batch_size=None, bg=None, bg_contrast=1.0):
        if batch_size is None:
            batch_size = self.batch_size
            
        bg_imgs = None
        if bg is 'natural':
            bg_imgs,_ = bsds500.load_data()
            bg_imgs = [np.expand_dims(skim.color.rgb2gray(im),-1) for im in bg_imgs]
            if bg_contrast < 1.0:
                bg_imgs = [rescale_contrast(im,bg_contrast) for im in bg_imgs]
        elif bg is 'uniform':
            bg_stack = self.gen_bg_noise(im_stack=np.zeros(shape=(50,512,512,1)))
            bg_imgs = [bg_stack[i].reshape(512,512,1) for i in np.arange(50)]
            
        
            
        
        for i in range(num_batches):
            
            test_batch = self.gen_batch(self.x_te,
                                         self.y_test_oh,
                                         batch_size=batch_size,
                                         bg_imgs=bg_imgs,
                                        )
                
            yield test_batch
            
            
    def gen_train_batches(self,num_batches, bg=None, bg_contrast = 1.0, image_range=(0,1)):
        bg_imgs = None
        if bg is 'natural':
            bg_imgs,_ = bsds500.load_data()
            bg_imgs = [np.expand_dims(skim.color.rgb2gray(im),-1) for im in bg_imgs]
        elif bg is 'uniform':
            bg_stack = self.gen_bg_noise(im_stack=np.zeros(shape=(50,512,512,1)))
            bg_imgs = [bg_stack[i].reshape(512,512,1) for i in np.arange(50)]
            if bg_contrast < 1.0:
                bg_imgs = [rescale_contrast(im,bg_contrast) for im in bg_imgs]
            
        for i in range(num_batches):
            
            (X, X_fg, y) = self.gen_batch(self.x_tr,
                                         self.y_train_oh,
                                         batch_size=self.batch_size,
                                         bg_imgs=bg_imgs,
                                        )                
            
            
            
            yield (X,X_fg,y)
            
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

#         import pdb; pdb.set_trace()
        for i in np.arange(n):
            n_bg_imgs = len(bg_imgs)
            bg_im = bg_imgs[rand.choice(np.arange(n_bg_imgs))]
            bg = get_patch(image=bg_im,shape=(w,h))
            if len(bg.shape)==2:
                bg = np.stack([bg.reshape(w,h,1)]*ch,-1)
            out[i] = bg        
        return out
    
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
            
    def shift_image(self,X,max_translation,):
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
    
    def gen_pan_test(self,im_stack,output,dxs,dys,scale=2):
        num_im = len(im_stack)
        for i in range(num_im):
            dx = dxs[i]
            dy = dys[i]
            letter = im_stack[i]
            
            new_im = self.regen_shift_image(letter,dx,dy,scale)
                
            output[i] = np.reshape(new_im,(1,)+self.input_shape)
            
        output = norm_im(output,self.flatten)
    
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
    
    