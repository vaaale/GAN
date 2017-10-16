import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import tensorflow.contrib.keras.api.keras.backend as K
import tensorflow as tf
from tensorflow.contrib.keras.python.keras import Input
from tensorflow.contrib.keras.python.keras.datasets import cifar10
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation, concatenate, Convolution2D, UpSampling2D, \
    Flatten, Reshape, AveragePooling2D
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.utils import to_categorical
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
mb_size = 64
Z_dim = 200
# X_dim = mnist.train.images.shape[1]
X_dim = (32, 32, 3)
# y_dim = mnist.train.labels.shape[1]
y_dim = 10
h_dim = 128

logs_path = 'logs'
if not os.path.isdir(logs_path):
    os.makedirs(logs_path)

K.set_learning_phase(True)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Discriminator Net model
def discriminator(X_dim, y_dim):
    x_in = Input(shape=(X_dim), name='X_input')
    y_in = Input(shape=(y_dim,), name='Y_input')
    D_h = Activation('relu')(BatchNormalization()(Convolution2D(64, kernel_size=(5, 5), padding='same')(x_in)))
    D_h = Activation('relu')(BatchNormalization()(Convolution2D(128, kernel_size=(5, 5), padding='same')(D_h)))
    D_h = Activation('relu')(BatchNormalization()(Convolution2D(256, kernel_size=(5, 5), padding='same')(D_h)))
    D_h = AveragePooling2D(pool_size=(2, 2), padding='valid')(D_h)
    D_h = Flatten()(D_h)
    D_h = concatenate([D_h, y_in])
    D_logit = Dense(1, name='Discriminator_Output')(D_h)
    D = Model(inputs=[x_in, y_in], outputs=D_logit, name='Discriminator')
    print('=== Discriminator ===')
    D.summary()
    print('\n\n')
    return D


# Generator Net model
def generator(Z_dim, y_dim):
    z_in = Input(shape=(Z_dim,), name='Z_input')
    y_in = Input(shape=(y_dim,), name='y_input')
    inputs = concatenate([z_in, y_in])
    G_h = Dense(100 * 16 * 16, activation='relu')(inputs)
    G_h = Reshape(target_shape=(16, 16, 100))(G_h)
    G_h = UpSampling2D(size=(2, 2))(G_h)
    G_h = Activation('relu')(BatchNormalization()(Convolution2D(256, kernel_size=(5, 5), padding='same')(G_h)))
    G_h = Activation('relu')(BatchNormalization()(Convolution2D(128, kernel_size=(5, 5), padding='same')(G_h)))
    G_h = Activation('relu')(BatchNormalization()(Convolution2D(64, kernel_size=(5, 5), padding='same')(G_h)))
    # G_h = Activation('relu')(BatchNormalization()(Convolution2D(64, kernel_size=(5, 5), padding='same')(G_h)))
    G_prob = Convolution2D(3, kernel_size=(1, 1), padding='same', activation='sigmoid')(G_h)

    G = Model(inputs=[z_in, y_in], outputs=G_prob, name='Generator')
    print('=== Generator ===')
    G.summary()
    print('\n\n')

    return G


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
        plt.imshow(sample.reshape(32, 32, 3), cmap='Greys_r')

    return fig


""" Input placeholders """
X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

D_m = discriminator(X_dim, y_dim)
theta_D = D_m.trainable_variables

G_m = generator(Z_dim, y_dim)
theta_G = G_m.trainable_variables

G_sample = G_m([Z, y])
D_logit_real = D_m([X, y])
D_logit_fake = D_m([G_sample, y])

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)
write_op = tf.summary.merge_all()

with tf.device('/gpu:0'):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')


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

D_loss_writer = tf.summary.FileWriter(logs_path+'/D_loss', graph=tf.get_default_graph())
# G_loss_writer = tf.summary.FileWriter(logs_path+'/G_loss', graph=tf.get_default_graph())


i = 0
for it in range(100000):
    if it % 100 == 0:
        n_sample = 16
        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[range(n_sample), np.random.randint(0, 10, n_sample)] = 1

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    # X_mb, y_mb = mnist.train.next_batch(mb_size)
    # X_mb = X_mb.reshape(-1, 32, 32, 3)
    X_mb, y_mb = next(gen)
    # print('X_mb: {}'.format(X_mb.shape))
    # print('y_mb: {}'.format(y_mb.shape))
    y_mb = to_categorical(y_mb, num_classes=y_dim)

    Z_sample = sample_Z(X_mb.shape[0], Z_dim)
    _, D_loss_curr, D_summary = sess.run([D_solver, D_loss, write_op], feed_dict={X: X_mb, Z: Z_sample, y: y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y: y_mb})
    D_loss_writer.add_summary(D_summary)
    # G_loss_writer.add_summary(G_summary)
    D_loss_writer.flush()
    # G_loss.flush()

    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
