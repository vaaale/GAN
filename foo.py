import tensorflow as tf
from tensorflow.contrib.keras.python.keras import Input
from tensorflow.contrib.keras.python.keras.datasets import cifar10
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation, concatenate, Convolution2D, UpSampling2D, \
    Flatten, Reshape, AveragePooling2D, add, initializers, Conv2DTranspose
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization


image_size = 64
gf_dim = 64
s16 = int(image_size/16)
c_dim = 3

input = Input(shape=(100,))
h = Dense(gf_dim*8*s16*s16)(input)
h = Activation('relu')(BatchNormalization()(Reshape(target_shape=[s16, s16, gf_dim*8])(h)))
h = Activation('relu')(BatchNormalization()(Conv2DTranspose(gf_dim*4, kernel_size=(5, 5), strides=(2, 2), padding='same')(h)))
h = Activation('relu')(BatchNormalization()(Conv2DTranspose(gf_dim*2, kernel_size=(5, 5), strides=(2, 2), padding='same')(h)))
h = Activation('relu')(BatchNormalization()(Conv2DTranspose(gf_dim, kernel_size=(5, 5), strides=(2, 2), padding='same')(h)))
h = Conv2DTranspose(c_dim, kernel_size=(5, 5), strides=(2, 2), padding='same')(h)

model = Model(inputs=input, outputs=h)
model.summary()
