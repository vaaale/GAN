import matplotlib
# Force matplotlib to not use any Xwindows backend.
from datasets import load_data

matplotlib.use('Agg')

import tensorflow.contrib.keras.api.keras.backend as K
import tensorflow as tf
from tensorflow.contrib.keras.python.keras import Input
from tensorflow.contrib.keras.python.keras.datasets import cifar10
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation, concatenate, Convolution2D, Flatten, Reshape, \
    Conv2DTranspose, LeakyReLU
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.utils import to_categorical
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, y_train = load_data()
X_dim = (28, 28, 1)

mb_size = 64
Z_dim = 64
y_dim = 10
h_dim = 128

gamma = 0.75
lamda = 0.001
learning_rate = 0.0002
beta1 = 0.5

# x_train = x_train / 255.
# x_test = x_test / 255.

logs_path = 'logs'
if not os.path.isdir(logs_path):
    os.makedirs(logs_path)

if not os.path.exists('out/'):
    os.makedirs('out/')

K.set_learning_phase(True)


# Discriminator Net model
def discriminator(X_dim):
    # It must be Auto-Encoder style architecture
    # Architecture : (64)4c2s-FC32_BR-FC64*14*14_BR-(1)4dc2s_S
    x_in = Input(shape=X_dim, name='X_input')
    D_h = Activation('relu')(
        Convolution2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', name='d_conv1')(x_in))
    D_h = Flatten()(D_h)
    code = Activation('relu')(BatchNormalization()(Dense(32)(D_h)))
    D_h = Activation('relu')(BatchNormalization()(Dense(64 * 14 * 14)(code)))
    D_h = Reshape(target_shape=[14, 14, 64])(D_h)
    out = Conv2DTranspose(1, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='sigmoid')(D_h)
    D_model = Model(inputs=x_in, outputs=[out, code])
    # recon loss
    # recon_error = tf.sqrt(2 * tf.nn.l2_loss(out - x_in)) / self.batch_size
    # return out, recon_error, code
    D_model.summary()
    return D_model


def generator(Z_dim):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    z = Input(shape=(Z_dim,), name='Z_input')
    net = Activation('relu')(BatchNormalization()(Dense(1024)(z)))
    net = Activation('relu')(BatchNormalization()(Dense(128 * 7 * 7)(net)))
    net = Reshape(target_shape=[7, 7, 128])(net)
    net = Activation('relu')(
        BatchNormalization()(Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(net)))
    out = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation='sigmoid', padding='same')(net)
    G_model = Model(inputs=z, outputs=out)
    G_model.summary()
    return G_model


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # plt.imshow(sample.reshape(X_dim[0], X_dim[1], X_dim[2]), cmap='Greys_r')
        plt.imshow(sample.reshape(X_dim[0], X_dim[1]), cmap='Greys_r')

    return fig


""" Input placeholders """
X = tf.placeholder(tf.float32, shape=[None, X_dim[0], X_dim[1], X_dim[2]])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
k = tf.Variable(0., trainable=False)

D_m = discriminator(X_dim)
theta_D = D_m.trainable_variables

G_m = generator(Z_dim)
theta_G = G_m.trainable_variables

G_sample = G_m(Z)
D_real_img, D_real_code = D_m(X)
D_real_err = tf.sqrt(2 * tf.nn.l2_loss(D_real_img - X)) / mb_size

D_fake_img, D_fake_code = D_m(G_sample)
D_fake_err = tf.sqrt(2 * tf.nn.l2_loss(D_fake_img - G_sample)) / mb_size


# Losses
D_loss = D_real_err - k*D_fake_err
# get loss for generator
G_loss = D_fake_err

# convergence metric
M = D_real_err + tf.abs(gamma * D_real_err - D_fake_err)

# operation for updating k
update_k = k.assign(k + lamda * (gamma * D_real_err - D_fake_err))

D_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(D_loss,  var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(G_loss, var_list=theta_G)


log_var = tf.Variable(0.0)
tf.summary.scalar("loss", log_var)
write_op = tf.summary.merge_all()

with tf.device('/gpu:0'):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

'''
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images  # Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

gen = datagen.flow(x_train, y_train, batch_size=mb_size)
'''

i = 0
for it in range(100000):
    if it % 1000 == 0:
        n_sample = 16
        Z_sample = sample_Z(n_sample, Z_dim)

        samples = sess.run(G_sample, feed_dict={Z: Z_sample})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    # X_mb, y_mb = next(gen)
    # y_mb = to_categorical(y_mb, num_classes=y_dim) * 0.9
    X_mb, y_mb = mnist.train.next_batch(mb_size)
    X_mb = X_mb.reshape(X_mb.shape[0], 28, 28, 1)

    Z_sample = sample_Z(X_mb.shape[0], Z_dim)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample})
    _, M_value, k_value = sess.run([update_k, M, k], feed_dict={X: X_mb, Z: Z_sample})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print('M: {:.4}'.format(M_value))
        print('k: {:.4}'.format(k_value))
        print()
