#!/usr/bin/python3
"""
The file contains a Variational Autoencoder in tensorflow.

Copyright (c) 2017
Licensed under the MIT License (see LICENSE for details)
Written by Arash Tehrani
"""
#   ---------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflearn


from tensorflow.examples.tutorials.mnist import input_data

#   ----------------------------------------------
class VAE(object):

    def __init__(self, #session, 
        reduced_dim=10, keepprob=0.8, dec_in_channels=1,
        LR=1e-3, optimizer='adam', tb_verbose=3):
        tf.reset_default_graph()
        #self.sess = session
        self.reduced_dim = reduced_dim
        self.keepprob= keepprob
        self.dec_in_channels = dec_in_channels
        self.LR = LR
        self.optimizer = optimizer
        self.tb_verbose = tb_verbose
        self._setup()
        self._build_model()

    def _setup(self):
        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, 28 * 28])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
        self.reshaped_dim = [-1, 7, 7, self.dec_in_channels]
        self.inputs_decoder = 49 * self.dec_in_channels

    def encoder(self):
        activation = self.lrelu
        with tf.variable_scope("encoder", reuse=None):
            #input_shape = [None, 28, 28, 1]
            #X = tflearn.layers.core.input_data(shape=input_shape, name='input')
            X = tf.reshape(self.X_in, shape=[-1, 28, 28, 1])
            x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.contrib.layers.flatten(x)
            mn = tf.layers.dense(x, units=self.reduced_dim)
            sd = 0.5 * tf.layers.dense(x, units=self.reduced_dim)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.reduced_dim]))
            z = mn + tf.multiply(epsilon, tf.exp(sd))

            return z, mn, sd

    def decoder(self, sampled_z):
        with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(sampled_z, units=self.inputs_decoder, activation=self.lrelu)
            x = tf.layers.dense(x, units=self.inputs_decoder * 2 + 1, activation=self.lrelu)
            x = tf.reshape(x, self.reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same',
                                           activation=tf.nn.relu)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=tf.nn.relu)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=tf.nn.relu)

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
            img = tf.reshape(x, shape=[-1, 28, 28])

            return img

    def _build_model(self):
        sampled, mn, sd = self.encoder()
        dec = self.decoder(sampled)
        unreshaped = tf.reshape(dec, [-1, 28 * 28])
        self.img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, self.Y_flat), 1)
        self.latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
        self.loss = tf.reduce_mean(self.img_loss + self.latent_loss)
        self.acc = tf.reduce_mean(self.loss)

    def train(self, trainX, testX, batch_size=128):   
        optimizer = tflearn.optimizers.Adam(learning_rate=self.LR, name='Adam')
        step = tflearn.variable("step", initializer='zeros', shape=[])
        optimizer.build(step_tensor=step)
        optim_tensor = optimizer.get_tensor()

        trainop = tflearn.TrainOp(loss=self.loss, optimizer=optim_tensor,
                                metric=self.acc, batch_size=128,
                                step_tensor=step)
        trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=self.tb_verbose)
        trainer.fit({self.X_in: trainX, self.Y: trainX}, val_feed_dicts={self.X_in: testX, self.Y: testY},
                    n_epoch=50, show_metric=True)

            
    @staticmethod
    def lrelu(x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))

#   ----------------------------------------------
if __name__ == '__main__':

    import tflearn.datasets.mnist as mnist

    trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

    #sess = tf.Session()
    vae = VAE()
    vae.train(trainX, testX)
    #sess.close()














"""
# Generating random data
randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

for img in imgs:
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img, cmap='gray')
"""