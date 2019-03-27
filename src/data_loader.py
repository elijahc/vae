import os
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm as tqdm
from tqdm import trange
from scipy.ndimage import rotate

def get_loader(config):
    if config.dataset is 'mnist':
        from keras.datasets import mnist
        return prepare_keras_dataset(mnist)
    elif config.dataset is 'fashion_mnist':
        from keras.datasets import fashion_mnist
        return prepare_keras_dataset(fashion_mnist)


def norm_im(X,flatten=True):
    X = X.astype('float32') / 255.
    if flatten:
        X = X.reshape( (len(X), np.prod(X.shape[1:])) )
    
    return X

def prepare_keras_dataset(x_train,y_train,x_test,y_test):
    # (x_train, y_train), (x_test, y_test) = k_data.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # print(x_train.shape)
    # print(x_test.shape)
    return (x_train,y_train),(x_test,y_test)

class Shifted_Data_Loader():
    def __init__(self,dataset,scale=2,rotation=0.75,translation=0.75,flatten=True,train_downsample=None,autoload=True):
        self.scale=scale
        self.dataset=dataset
        self.rotation = rotation
        self.translation = translation
        self.train_downsample=train_downsample
        self.flatten = flatten
        
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
            
        print('input_shape: ',self.input_shape)
        print('dataset: ',self.dataset)
        print('scale: ',self.scale)
        print('tx_max: ', self.translation)
        print('rot_max: ', self.rotation)
        
        print('loading {}...'.format(self.dataset))
        if dataset=='mnist':
            from keras.datasets import mnist
            (x_train, y_train),(x_test, y_test) = mnist.load_data()

        elif dataset=='fashion_mnist':
            from keras.datasets import fashion_mnist
            (x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
        
        if self.train_downsample is not None:
            pass
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_oh = to_categorical(y_train)
        self.y_test_oh = to_categorical(y_test)
        
        num_train = len(self.y_train)
        num_test =  len(self.y_test)

        self.sx_train = np.empty((num_train,)+self.input_shape)
        self.sx_test =  np.empty((num_test,)+self.input_shape)

        self.delta_train = np.empty((num_train,3))
        self.delta_test = np.empty((num_test,3))
        print('sx_train: ',self.sx_train.shape)
        
        self.x_train_orig = x_train
        self.x_test_orig = x_test
        self.x_train = norm_im(x_train,flatten)
        self.x_test = norm_im(x_test,flatten)

        if autoload:
            self.gen_new_shifted(x_train,x_test,flatten)
        
    def gen_new_shifted(self,x_train,x_test,flatten=True):
        
        print('making training data...')
        self.transform_im(x_train,self.sx_train,self.delta_train)

        print('making testing data...')
        self.transform_im(x_test,self.sx_test,self.delta_test)
        
#         (self.sx_train,_),(self.sx_test,_) = prepare_keras_dataset(self.sx_train,y_train,self.sx_test,y_test)
        self.dx = (self.delta_train[:,0],self.delta_test[:,0])
        self.dy = (self.delta_train[:,1],self.delta_test[:,1])
        self.dtheta = (self.delta_train[:,2],self.delta_test[:,2])
        
        # (x_train_pp,y_train_pp),(x_test_pp,y_test_pp) = prepare_keras_dataset(x_train,y_train,x_test,y_test)
#         self.input_shape = self.input_shape+(1,)
        
        self.sx_train = norm_im(self.sx_train,flatten,)
        self.sx_test = norm_im(self.sx_test,flatten)
        
    def transform_im(self,im_stack,output,delta):
        num_im = len(im_stack)
        for i in range(num_im):
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