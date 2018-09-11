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
    def __init__(self,dataset,scale=2,rotation=0.75,translation=0.75):
        self.scale=scale
        self.dataset=dataset
        if rotation is not None:
            self.rotation=float(rotation)
            assert self.rotation >= 0.0
            assert self.rotation <= 1.0
        else:
            self.rotation = None
            
        if translation is not None:
            self.translation=float(translation)
            assert self.translation >= 0.
            assert self.translation <= 1.
        else:
            self.translation = None
            
        self.input_shape = (784*self.scale*self.scale,)
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
    
        (x_train_pp,y_train_pp),(x_test_pp,y_test_pp) = prepare_keras_dataset(x_train,y_train,x_test,y_test)
        self.x_train = x_train_pp
        self.y_train = y_train_pp
        self.x_test = x_test_pp
        self.y_test = y_test_pp

        self.y_train_oh = to_categorical(y_train_pp)
        self.y_test_oh = to_categorical(y_test_pp)
        num_train = len(self.y_train)
        num_test =  len(self.y_test)

        self.sx_train = np.empty((num_train, self.input_shape[0]))
        self.sx_test =  np.empty((num_test, self.input_shape[0]))

        self.delta_train = np.empty((num_train,3))
        self.delta_test = np.empty((num_test,3))
        
        print('making training data...')
        self.transform_im(x_train,self.delta_train)

        print('making testing data...')
        self.transform_im(x_test,self.delta_test)
        
        (self.sx_train,_),(self.sx_test,_) = prepare_keras_dataset(self.sx_train,y_train,self.sx_test,y_test)
        self.dx = (self.delta_train[:,0],self.delta_test[:,0])
        self.dy = (self.delta_train[:,1],self.delta_test[:,1])
        self.dtheta = (self.delta_train[:,2],self.delta_test[:,2])
        
    def transform_im(self,im_stack,delta):
        num_im = len(im_stack)
        for i in trange(num_im):
            letter = im_stack[i]
            if self.rotation is not None:
                letter,rot = self.rotate_image(letter,rot_max=self.rotation)
            else:
                rot = [0]
            
            if self.translation is not None:
                new_im,offsets = self.shift_image(letter,max_translation=self.translation)
            else:
                offsets = [0,0]
                
            self.sx_train[i] = new_im.reshape(1,self.input_shape[0])
            offsets.extend(rot)
            delta[i] = np.array(offsets)
                    
    def shift_image(self,X,max_translation):
        (x_sz,y_sz) = X.shape
        bg_size = (28*self.scale,28*self.scale)
        dx_max = dy_max = int(28/2*(self.scale-1)*max_translation)
        
        dx = int(np.random.randint(-dx_max,dx_max)+x_sz/2)
        dy = int(np.random.randint(-dy_max,dy_max)+y_sz/2)
        
        dx = max(dx,0)
        dx = min(dx,bg_size[0]-x_sz)
        
        dy = max(dy,0)
        dy = min(dy,bg_size[0]-y_sz)
        new_im = np.zeros(bg_size)

        new_im[dx:dx+x_sz,dy:dy+y_sz] = X
        
        return new_im,[dx,dy]

    def rotate_image(self,X,rot_max=0.5,reshape=True):
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
        
        return new_im,[rot]