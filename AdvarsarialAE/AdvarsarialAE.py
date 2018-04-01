#!/usr/bin/python3
"""
The file contains a Advarsarial Autoencoder in tensorflow.

Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by Arash Tehrani
"""
#   ---------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflearn
from tflearn.helpers.evaluator import Evaluator
#   ----------------------------------------------
class AAE(object):

    def __init__(self,
        input_shape=[28, 28],
        reduced_dim=10, 
        batch_size=128, channel_size=128,
        activation=tflearn.activations.leaky_relu,
        lr=1e-3, optimizer='adam', tb_verbose=3,
        weight_init=tflearn.initializations.xavier(uniform=False),
        bias_init=tflearn.initializations.xavier(uniform=False),
        tensorboar_dir='./AAE/tflearn_logs/'):
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        self.input_shape = input_shape
        self.reduced_dim = reduced_dim
        self.activation=activation
        self.lr = lr
        self.optimizer = optimizer
        self.tb_verbose = tb_verbose
        self.small_size_img = int(input_shape[0]*input_shape[1]/16)
        self.batch_size = batch_size
        self.channel_size = channel_size
        dummy_dim = int(np.sqrt(self.small_size_img))
        self.reshaped_dim = [-1, dummy_dim, dummy_dim, self.channel_size] #[-1, 7, 7, 1]
        self.tensorboar_dir = tensorboar_dir
        self.weight_init = weight_init
        self.bias_init = bias_init
        self._setup()
        self._build_model()


    def _setup(self):
        """
        setup the placeholders and dimensions
        """
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.input_shape[0], 
                                self.input_shape[1]], name='X')
        self.Z = tf.placeholder(dtype=tf.float32, shape=[None, self.reduced_dim], name='Z')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.input_shape[0], 
                                self.input_shape[1]], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, self.input_shape[0] * self.input_shape[1]])
        self.Z_prior = tf.placeholder(dtype=tf.float32, shape=[None,  self.reduced_dim], 
                                name='Z_prior')


    def encoder(self, x, reuse=None):
        """
        Encoder network
        """
        with tf.variable_scope("encoder", reuse=None):
            x = tf.reshape(x, shape=[-1, self.input_shape[0], self.input_shape[1], 1])
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, 
                            padding='same', 
                            activation=self.activation, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='enc_L1_conv')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, 
                            padding='same', 
                            activation=None, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='enc_L2_conv')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.batch_normalization(x, name='enc_L3_bn')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
            x = tf.layers.dense(x, units=1024, 
                            activation=None, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init,
                            name ='enc_L4_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.batch_normalization(x, name='enc_L5_bn')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
            x = tf.layers.dense(x, units=self.reduced_dim, 
                            activation=None, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init,
                            name ='enc_L6_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)

            return x


    def decoder(self, z, reuse=False):
        """
        Decoder network
        """
        dim = int(self.input_shape[0]*self.input_shape[1]*self.channel_size/16)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("decoder", reuse=reuse):
            x = tf.layers.dense(z, units=1024, 
                            activation= None,
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='dec_L1_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.batch_normalization(x, name='dec_L2_bn')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
            x = tf.layers.dense(self.z, units=dim, 
                            activation= None,
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='dec_L3_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.batch_normalization(x, name='dec_L4_bn')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
            x = tf.reshape(x, shape=self.reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, 
                                        padding='same',
                                        activation= None,
                                        kernel_initializer=self.weight_init,
                                        bias_initializer=self.bias_init, 
                                        name ='dec_L5_convt')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.batch_normalization(x, name='dec_L6_bn')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
            x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=4, strides=2, 
                                        padding='same',
                                        activation= None,
                                        kernel_initializer=self.weight_init,
                                        bias_initializer=self.bias_init, 
                                        name ='dec_L7_convt')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            
            return x#tf.reshape(x, shape=[-1, self.input_shape[0], self.input_shape[1]])


    def discriminator(self, z, reuse=False):
        """
        Discriminator network
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope('discriminator', reuse=reuse):
            x = tf.layers.dense(z, units=1000, 
                            activation= self.activation,
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='dis_L1_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.dense(self.z, units=1000, 
                            activation= None,
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='dis_L2_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.batch_normalization(x, name='dis_L3_bn')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
            x = tf.layers.dense(self.z, units=1, 
                            activation= tf.nn.sigmoid,
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='dis_L4_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            return x
            
    def _build_model(self):
        """
        Building the model and loss function
        """
        with tf.variable_scope(tf.get_variable_scope()):
            self.z = self.encoder(self.X)
            self.y = self.decoder(self.z)
            y_flat = tf.reshape(self.y, [-1, self.input_shape[0] * self.input_shape[1]])

        with tf.variable_scope(tf.get_variable_scope()):
            d_real = self.discriminator(self.Z_prior)
            d_fake = self.discriminator(self.z, reuse=True)

        # loss
        self.rec_loss = tf.reduce_mean(tf.square(self.Y_flat - y_flat))

        self.gen_loss = -tf.reduce_mean(tf.log(d_fake + 1e-10))
        self.dis_loss = -tf.reduce_mean(tf.log(d_real + 1e-10) \
                        + tf.log(1. - d_fake + 1e-10))

        # step of the optimization
        # playing with this variable allows number of training iterations 
        # for discriminator and generator
        step = tflearn.variable('step', initializer='zeros', shape=[])
        
        # building the optimizers
        #optimizer = tflearn.optimizers.Adam(learning_rate=self.lr)
        #optimizer.build(step_tensor=step)
        #optim_tensor = optimizer.get_tensor()

        rec_opt = tflearn.Adam(learning_rate=self.lr).get_tensor()
        gen_opt = tflearn.Adam(learning_rate=self.lr).get_tensor()
        dis_opt = tflearn.Adam(learning_rate=self.lr).get_tensor()
        
        # collecting trainable variables
        all_variables = tf.trainable_variables()
        enc_vars = [var for var in all_variables if 'enc' in var.name]
        dec_vars = [var for var in all_variables if 'dec' in var.name]
        dis_vars = [var for var in all_variables if 'dis' in var.name]
        rec_vars = enc_vars + dec_vars

        # Defining trainOps
        rec_trainop = tflearn.TrainOp(loss=self.rec_loss, optimizer=rec_opt,
                                metric=None, batch_size=self.batch_size,
                                trainable_vars=rec_vars,
                                step_tensor=step,
                                name='rec_train')
        gen_trainop = tflearn.TrainOp(loss=self.gen_loss, optimizer=gen_opt,
                                metric=None, batch_size=self.batch_size,
                                trainable_vars=enc_vars,
                                step_tensor=step,
                                name='gen_train')
        dis_trainop = tflearn.TrainOp(loss=self.dis_loss, optimizer=dis_opt,
                                metric=None, batch_size=self.batch_size,
                                trainable_vars=dis_vars,
                                step_tensor=step,
                                name='dis_train')
        
        self.trainer = tflearn.Trainer(train_ops=[rec_trainop, dis_trainop, gen_trainop], 
                                graph=self.graph,
                                tensorboard_dir=self.tensorboar_dir, 
                                tensorboard_verbose=self.tb_verbose)
                                

    def train(self, trainX, testX, Z, n_epoch=50): 
        """
        train the neural net
        Arguments:
            trainX: numpy or python array, training data
            testX: numpy or python array, testing data
            n_epoch: int, number of epochs
        """
        self.trainer.fit(
            [{self.X: trainX, self.Y: trainX}, {self.X: trainX, self.Z_prior: Z}, {self.X: trainX, self.Z_prior: Z}], 
            #val_feed_dicts=[{self.X: testX, self.Y: testX}, {}, {}],
            n_epoch=n_epoch, show_metric=True)
    
    
    def save(self, model_file):
        """
        save model weights
        Arguments:
            model_file: string, address of the saved file
        """
        self.trainer.save(model_file)


    def load(self, model_file, trainable_variable_only=False):
        """
        Restore model weights.
        Arguments:
            model_file: string, address of the saved file.
            trainable_variable_only: boolean, set to True if you only want to load the weights
        """
        self.trainer.restore(model_file,
                            trainable_variable_only=trainable_variable_only, 
                            verbose=True)


#   ----------------------------------------------
if __name__ == '__main__':
    import tflearn.datasets.mnist as mnist

    # get the data
    trainX, trainY, testX, testY = mnist.load_data(one_hot=True)
    trainX = trainX.reshape([-1, 28, 28])
    testX = testX.reshape([-1, 28, 28])
    # ----------------------------------------
    # build the model
    aae = AAE()

    Z = tf.convert_to_tensor(
                            np.random.uniform(-1, 1, [trainX.shape[0], aae.reduced_dim]).astype(np.float32))
            
    # train and save the model
    aae.train(trainX, testX, Z, n_epoch=5)
    aae.save('./AdvarsarialAE/saved_models/model.tfl')

    # load the model
    #aae.load('./VAE/saved_models/model.tfl')
