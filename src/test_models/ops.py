import tensorflow as tf
import keras.backend as K
from keras.layers import Lambda,Conv2D,LeakyReLU,Input,BatchNormalization,Reshape,Dense
from keras.models import Model

def center_of_mass(volumes):
    # From: https://stackoverflow.com/questions/51724450/finding-centre-of-mass-of-tensor-tensorflow
    # Input volumes
    # volumes = tf.placeholder(tf.float32, [None, 64, 64, 64])
    # Make array of coordinates (each row contains three coordinates)
    v_shape = K.shape(volumes)
    ii, jj, kk = tf.meshgrid(tf.range(v_shape[1]), tf.range(v_shape[2]), tf.range(v_shape[3]), indexing='ij')
    coords = tf.stack([tf.reshape(ii, (-1,)), tf.reshape(jj, (-1,)), tf.reshape(kk, (-1,))], axis=-1)
    coords = tf.cast(coords, tf.float32)
    # Rearrange input into one vector per volume
    volumes_flat = tf.reshape(volumes, [-1, v_shape[1] * v_shape[2] * v_shape[3], 1])
    # Compute total mass for each volume
    total_mass = tf.reduce_sum(volumes_flat, axis=1)
    # Compute centre of mass
    centre_of_mass = tf.reduce_sum(volumes_flat * coords, axis=1) / total_mass
    
    return centre_of_mass

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    conv_l = Conv2D(output_dim,kernel_size=(k_h,k_w),strides=(d_h,d_w),padding='SAME',name=name)
    
    
#         w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#               initializer=tf.truncated_normal_initializer(stddev=stddev))
#         conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

#         biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
#         conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv_l(input_)

def lrelu(input_,leak=0.2,name='lrelu'):
    layer = LeakyReLU(leak)
    return layer(input_)

def bn(x, is_training, scope):
    with tf.variable_scope(scope):
        layer = BatchNormalization(momentum=0.9,epsilon=1e-5)
        return layer(x)

def linear(input_, output_size, scope=None, k_init='glorot_uniform', bias_start='zeros', with_w=False):
    with tf.variable_scope(scope or "Linear"):
        layer = Dense(output_size,kernel_initializer=k_init,bias_initializer=bias_start)
        if with_w:
            return layer(input_), layer.get_weights()
        else:
            return layer(input_)
        
def center_of_mass_crop(source,crop_size=[28,28]):
    source_mask = tf.where(source>-1,x=tf.ones_like(source),y=tf.zeros_like(source))
    centers = center_of_mass(source_mask)[:,:2]
    def func(x):
        cropped_im = tf.image.extract_glimpse(x,tf.constant(crop_size,tf.int32),offsets=centers,normalized=False,centered=False)
        return cropped_im
    
    return Lambda(func)
