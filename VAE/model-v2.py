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
        activation='LeakyReLU',
        weights_init=tflearn.initializations.xavier(uniform=False),
        bias_init=tflearn.initializations.xavier(uniform=False)),
        ):
        with tf.variable_scope("encoder", reuse=None):
            input = tflearn.layers.core.input_data(shape=input_shape, name='input')
            net = tflearn.layers.conv.conv_2d(input, 64, 4,
                                             activation=activation, 
                                             scope='L1_conv1',
                                             strides=2,
                                             padding='same',
                                             bias=True,
                                             weights_init=weights_init,
                                             bias_init=bias_init)
            net = tflearn.layers.core.dropout (net, keep_prob, noise_shape=None, name='dropout1')
            net = tflearn.layers.conv.conv_2d(net, 64, 4,
                                             activation=activation, 
                                             scope='L2_conv2',
                                             strides=2,
                                             padding='same',
                                             bias=True,
                                             weights_init=weights_init,
                                             bias_init=bias_init)
            net = tflearn.layers.core.dropout (net, keep_prob, noise_shape=None, name='dropout2')
            net = tflearn.layers.conv.conv_2d(net, 64, 4,
                                             activation=activation, 
                                             scope='L3_conv3',
                                             strides=1,
                                             padding='same',
                                             bias=True,
                                             weights_init=weights_init,
                                             bias_init=bias_init)
            net = tflearn.layers.core.dropout (ney, keep_prob, noise_shape=None, name='dropout3')
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
        keep_prob=0.8,
        activation='LeakyReLU',
        weights_init=tflearn.initializations.xavier(uniform=False),
        bias_init=tflearn.initializations.xavier(uniform=False) 
        ):
        with tf.variable_scope("decoder", reuse=None):
            net = tflearn.layers.core.fully_connected(sampled_z, self.decoder_dim,
                                             activation=activation, 
                                             scope='L5_fc3',
                                             bias=True,
                                             weights_init=weights_init,
                                             bias_init=bias_init)
            net = tf.reshape(net, self.reshaped_dim)
            net = tflearn.layers.conv.conv_2d_transpose(
                                            net, 64, 4,
                                            activation=activation, 
                                            scope='L6_convt1',
                                            strides=2,
                                            padding='same',
                                            bias=True,
                                            weights_init=weights_init,
                                            bias_init=bias_init)
            net = tflearn.layers.core.dropout (ney, keep_prob, noise_shape=None, name='dropout4')
            net = tflearn.layers.conv.conv_2d_transpose(
                                            net, 64, 4,
                                            activation=activation, 
                                            scope='L7_convt2',
                                            strides=1,
                                            padding='same',
                                            bias=True,
                                            weights_init=weights_init,
                                            bias_init=bias_init)
            net = tflearn.layers.core.dropout (ney, keep_prob, noise_shape=None, name='dropout4')
            net = tflearn.layers.conv.conv_2d_transpose(
                                            net, 64, 4,
                                            activation=activation, 
                                            scope='L8_convt3',
                                            strides=1,
                                            padding='same',
                                            bias=True,
                                            weights_init=weights_init,
                                            bias_init=bias_init)
            net = tflearn.layers.core.flatten(net)
            net = tflearn.layers.core.fully_connected(sampled_z, self.decoder_dim,
                                             activation=tf.nn.sigmoid, 
                                             scope='L9_fc4',
                                             bias=True,
                                             weights_init=weights_init,
                                             bias_init=bias_init)
            output = tf.reshape(x, shape=[-1, 28, 28])
        return output


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


    def build_model(self, 
        input_shape, reduced_dim, 
        LR=1e-3, optimizer='adam', tb_verbose=3,
        keep_prob=0.8,
        activation='LeakyReLU',
        weights_init=tflearn.initializations.xavier(uniform=False),
        bias_init=tflearn.initializations.xavier(uniform=False)
        ):
        """
        Build the VAE netwprl
        Arguments:
        input_shape: 1Darray, shape of the input
        
        LR: scalar, learning rate
        optimizer: string, optimizer to use
        tb_verbose: int
        Output:
        tflearn dnn object
        """
        z, z_mean, z_std = self.encoder(input_shape)
        self.Lowdimensional_rep = z
        net = self.decoder(z)
        net = tflearn.layers.estimator.regression(net,
                                                optimizer=optimizer,
                                                learning_rate=LR,
                                                loss=_vae_loss,
                                                name='output')
        self.model = tflearn.DNN(net, tensorboard_dir='logs', tensorboard_verbose=tb_verbose)
        self.generator = tflearn.DNN(decoder, session=self.model.session)

    def train(self, x, val_x, n_epochs=10,
              batch_size=128, snapshot_step=5000, show_metric=True):
        """
        Train the sparseAE
        :param x: input data to feed the network
        :param val_x: validation data
        :param n_epochs: int, number of epochs
        :param batch_size: int
        :param snapshot_step: int
        :param show_metric: boolean
        """
        #self.sess.run(tf.global_variables_initializer())
        self.model.fit({'input': x}, {'output': x}, {'keep_prob':keep_prob}, 
                       n_epoch=n_epochs,
                       batch_size=batch_size,
                       validation_set=({'input': val_x}, {'targets': val_x}),
                       snapshot_step=snapshot_step,
                       show_metric=show_metric
                       run_id='ConvolutionalVAE1')

    def generate(self, n_images):
        samples = np.randn(self.reduced_dim, n_images)
        reconstructed = self.generator.predict({'input_noise': samples})
        return reconstructed


    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)
#   ---------------------------------------------------
if __name__ == __main__:


