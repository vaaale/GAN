import tensorflow as tf
from tensorflow.contrib.keras.python.keras import Input
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation, concatenate, Convolution2D, UpSampling2D, \
    Flatten, Reshape
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 100
# X_dim = mnist.train.images.shape[1]
X_dim = (28, 28, 1)
y_dim = mnist.train.labels.shape[1]
h_dim = 128


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """


def discriminator(X_dim, y_dim):
    x_in = Input(shape=(X_dim), name='X_input')
    y_in = Input(shape=(y_dim,), name='Y_input')
    D_h1 = Convolution2D(50, kernel_size=(2, 2), padding='same', activation='relu')(x_in)
    D_h2 = Convolution2D(50, kernel_size=(2, 2), padding='same', activation='relu')(D_h1)
    D_f = Flatten()(D_h2)
    merge = concatenate([D_f, y_in])
    D_d1 = Dense(100, activation='relu')(merge)
    D_logit = Dense(1)(D_d1)
    D = Model(inputs=[x_in, y_in], outputs=D_logit)

    return D

# def discriminator(X_dim, y_dim):
#     x_in = Input(shape=(X_dim,), name='X_input')
#     y_in = Input(shape=(y_dim,), name='Y_input')
#     inputs = concatenate([x_in, y_in])
#     D_h1 = Dense(h_dim, activation='relu')(inputs)
#     D_logit = Dense(1)(D_h1)
#
#     D = Model(inputs=[x_in, y_in], outputs=D_logit)
#
#     return D
#

""" Generator Net model """


def dense_generator(Z_dim, y_dim):
    z_in = Input(shape=(Z_dim,), name='Z_input')
    y_in = Input(shape=(y_dim,), name='y_input')
    inputs = concatenate([z_in, y_in])
    G_h1 = Dense(h_dim, activation='relu')(inputs)
    G_prob = Dense(X_dim, activation='sigmoid')(G_h1)

    G = Model(inputs=[z_in, y_in], outputs=G_prob)

    return G


def generator(Z_dim, y_dim):
    z_in = Input(shape=(Z_dim,), name='Z_input')
    y_in = Input(shape=(y_dim,), name='y_input')
    inputs = concatenate([z_in, y_in])
    G_h1 = Dense(100*14*14, activation='relu')(inputs)
    G_reshaped = Reshape(target_shape=(14, 14, 100))(G_h1)
    G_up = UpSampling2D(size=(2, 2))(G_reshaped)
    G_c1 = Convolution2D(50, kernel_size=(2, 2), padding='same', activation='relu')(G_up)
    G_prob = Convolution2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid')(G_c1)
    # G_prob = Flatten()(G_c2)

    G = Model(inputs=[z_in, y_in], outputs=G_prob)
    G.summary()

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
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

""" Input placeholders """
X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])


D_m = discriminator(X_dim, y_dim)
theta_D = D_m.trainable_variables

G_m = generator(Z_dim, y_dim)
theta_G = G_m.trainable_variables

G_sample = G_m([Z, y])
D_logit_real = D_m([X, y])
D_logit_fake = D_m([G_sample, y])

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


with tf.device('/gpu:0'):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
n_sample = 16
for it in range(1000000):
    if it % 1000 == 0:

        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[range(n_sample), np.random.randint(0, 10, n_sample)] = 1

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, y_mb = mnist.train.next_batch(mb_size)
    X_mb = X_mb.reshape(-1, 28, 28, 1)

    Z_sample = sample_Z(mb_size, Z_dim)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()