from src.models import EResNet
from keras.layers import Input, Flatten, Dense
from keras.models import Model

x_in = Input(shape=(56,56,1))

EResNet()

Model(inputs=x_in,outputs=x).summary()
