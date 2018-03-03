#!/usr/bin/python3
"""
The file contains a Variational Autoencoder in tflearn.

Copyright (c) 2017
Licensed under the MIT License (see LICENSE for details)
Written by Arash Tehrani
"""
#   ---------------------------------------------------

from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

import tflearn

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
#   ---------------------------------------------------
class VAE(object):
    def __init__(self, name='ConvolutionalVAE'):
        self.name = name
    def encoder(self, input_shape, reduced_dim, keep_prob=0.8,
        activation=tflearn.activations.leaky_relu,
        weights_init=tflearn.initializations.xavier(uniform=False),
        bias_init=tflearn.initializations.xavier(uniform=False)
        ):
        #with tf.variable_scope("encoder", reuse=None):
        inputs = tflearn.layers.core.input_data(shape=input_shape, name='input')
        net = tf.reshape(inputs, shape=[-1, 28, 28, 1])
        net = tflearn.layers.conv.conv_2d(net, 64, 4,
                                            activation=activation, 
                                            scope='L1_conv1',
                                            strides=2,
                                            padding='same',
                                            bias=True,
                                            weights_init=weights_init,
                                            bias_init=bias_init)
        net = tflearn.layers.core.dropout(net, keep_prob, noise_shape=None, name='dropout1')
        net = tflearn.layers.conv.conv_2d(net, 64, 4,
                                            activation=activation, 
                                            scope='L2_conv2',
                                            strides=2,
                                            padding='same',
                                            bias=True,
                                            weights_init=weights_init,
                                            bias_init=bias_init)
        net = tflearn.layers.core.dropout(net, keep_prob, noise_shape=None, name='dropout2')
        net = tflearn.layers.conv.conv_2d(net, 64, 4,
                                            activation=activation, 
                                            scope='L3_conv3',
                                            strides=1,
                                            padding='same',
                                            bias=True,
                                            weights_init=weights_init,
                                            bias_init=bias_init)
        net = tflearn.layers.core.dropout(net, keep_prob, noise_shape=None, name='dropout3')
        net = tflearn.layers.core.flatten(net)
        z_mean = tflearn.layers.core.fully_connected(net, self.reduced_dim,
                                            activation=activation, 
                                            scope='L41_fc1',
                                            bias=True,
                                            weights_init=weights_init,
                                            bias_init=bias_init)
        z_std = tflearn.layers.core.fully_connected(net, self.reduced_dim,
                                            activation=activation, 
                                            scope='L42_fc2',
                                            bias=True,
                                            weights_init=weights_init,
                                            bias_init=bias_init)
        eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, 
                                mean=0., stddev=1.0,
                                name='epsilon')
        
        z = z_mean + tf.exp(z_std / 2) * eps            
        
        return z, z_mean, z_std


    def decoder(self, sampled_z,
        starting_dim=7*7,
        keep_prob=0.8,
        activation=tflearn.activations.leaky_relu,
        weights_init=tflearn.initializations.xavier(uniform=False),
        bias_init=tflearn.initializations.xavier(uniform=False) 
        ):
        dummy_dim = int(np.sqrt(starting_dim))
        #with tf.variable_scope("decoder", reuse=None):
        net = tflearn.layers.core.fully_connected(sampled_z, starting_dim,
                                            activation=activation, 
                                            scope='L5_fc3',
                                            bias=True,
                                            weights_init=weights_init,
                                            bias_init=bias_init)
        net = tf.reshape(net, [-1, dummy_dim, dummy_dim, 1])
        net = tf.layers.conv2d_transpose(net, filters=64, kernel_size=4, strides=2, 
                                        padding='same',
                                        activation=tflearn.activations.leaky_relu, 
                                        kernel_initializer=weights_init,
                                        bias_initializer=bias_init,
                                        name ='L6_convt1')
        net = tflearn.layers.core.dropout(net, keep_prob, noise_shape=None, name='dropout4')
        net = tf.layers.conv2d_transpose(net, filters=64, kernel_size=4, strides=2, 
                                        padding='same',
                                        activation=tflearn.activations.leaky_relu, 
                                        kernel_initializer=weights_init,
                                        bias_initializer=bias_init,
                                        name ='L7_convt2')
        net = tflearn.layers.core.dropout(net, keep_prob, noise_shape=None, name='dropout4')
        net = tf.layers.conv2d_transpose(net, filters=64, kernel_size=4, strides=2, 
                                        padding='same',
                                        activation=tflearn.activations.leaky_relu, 
                                        kernel_initializer=weights_init,
                                        bias_initializer=bias_init,
                                        name ='L8_convt3')
        net = tflearn.layers.core.flatten(net)
        output = tflearn.layers.core.fully_connected(net, 28*28,
                                            activation=tf.nn.sigmoid, 
                                            scope='L9_fc4',
                                            bias=True,
                                            weights_init=weights_init,
                                            bias_init=bias_init)
        #output = tf.reshape(net, shape=[-1, 28, 28])

        return output

    def build_model(self, 
        input_shape, reduced_dim, 
        starting_dim=7*7,
        LR=1e-3, optimizer='adam',
        keep_prob=0.8,
        activation=tflearn.activations.leaky_relu,
        weights_init=tflearn.initializations.xavier(uniform=False),
        bias_init=tflearn.initializations.xavier(uniform=False),
        tb_dir='./tflearn_logs/',
        tb_verbose = 3
        ):
        """
        Build the VAE network
        Arguments:
        input_shape: 1Darray, shape of the input
        
        LR: scalar, learning rate
        optimizer: string, optimizer to use
        tb_verbose: int
        Output:
        tflearn dnn object
        """
        self.reduced_dim = reduced_dim
        z, z_mean, z_std = self.encoder(input_shape, 
                                        reduced_dim=self.reduced_dim, 
                                        keep_prob=keep_prob,
                                        activation=activation,
                                        weights_init=weights_init,
                                        bias_init=bias_init
                                        )
        self.Lowdimensional_rep = z
        dec = self.decoder(z,
                        starting_dim=starting_dim,
                        keep_prob=keep_prob,
                        activation=activation,
                        weights_init=weights_init,
                        bias_init=bias_init 
                        )
                        # Define VAE Loss
        def _vae_loss(x_reconstructed, x_true):
            # Reconstruction loss
            encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                                + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
            encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
            # KL Divergence loss
            kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
            kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
            return tf.reduce_mean(encode_decode_loss + kl_div_loss)
        net = tflearn.layers.estimator.regression(dec,
                                                optimizer=optimizer,
                                                learning_rate=LR,
                                                loss=_vae_loss,
                                                metric=None,
                                                name='output')
        self.model = tflearn.DNN(net, tensorboard_dir=tb_dir, tensorboard_verbose=tb_verbose)
        #self.generator = tflearn.DNN(dec, session=self.model.session)

    def train(self, x, val_x, n_epochs=10,
              batch_size=128, snapshot_step=5000, show_metric=True):
        """
        Train the convolutional VAE
        :param x: input data to feed the network
        :param val_x: validation data
        :param n_epochs: int, number of epochs
        :param batch_size: int
        :param snapshot_step: int
        :param show_metric: boolean
        """
        #self.sess.run(tf.global_variables_initializer())
        self.model.fit({'input': x}, {'output': x}, #{'keep_prob':keep_prob}, 
                       n_epoch=n_epochs,
                       batch_size=batch_size,
                       validation_set=({'input': val_x}, {'output': val_x}),
                       snapshot_step=snapshot_step,
                       show_metric=show_metric)
                       #run_id='ConvolutionalVAE1')
    '''
    def generator(self, n_images):
        samples = np.random.randn(n_images, self.reduced_dim)
        reconstructed = self.generator.predict({'input_noise': samples})
        return reconstructed
    '''
    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)
#   ---------------------------------------------------
if __name__ == '__main__':
    import os
    os.chdir('/home/arash/Desktop/python/nnexplore/VAE')
    import tflearn.datasets.mnist as mnist

    trainX, trainY, testX, testY = mnist.load_data(one_hot=True)
    #trainX = trainX.reshape([-1, 28, 28])
    #testX = testX.reshape([-1, 28, 28])
    vae = VAE()
    vae.build_model(input_shape=[None, 784], reduced_dim=10)
    vae.train(trainX, testX, n_epochs=10)
    

