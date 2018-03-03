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
from tflearn.helpers.evaluator import Evaluator
#   ----------------------------------------------
class VAE(object):

    def __init__(self, #session,
        reduced_dim=10, keep_prob=0.8,
        activation=tflearn.activations.leaky_relu,
        lr=1e-3, optimizer='adam', tb_verbose=3,
        small_size_img = 7*7,
        weight_init=tflearn.initializations.xavier(uniform=False),
        bias_init=tflearn.initializations.xavier(uniform=False),
        batch_size=128, tensorboar_dir='./tflearn_logs/'):
        #tf.reset_default_graph()
        #self.input_shape = input_shape 
        self.reduced_dim = reduced_dim
        self.keep_prob= keep_prob
        self.activation=activation
        self.lr = lr
        self.optimizer = optimizer
        self.tb_verbose = tb_verbose
        self.small_size_img = small_size_img
        self.batch_size = batch_size
        self.tensorboar_dir = tensorboar_dir
        self.weight_init = weight_init
        self.bias_init = bias_init
        self._setup()
        self._build_model()

    def _setup(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, 28 * 28])
        #self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
        self.reshaped_dim = [-1, 7, 7, 1]
        self.Z_in = tf.placeholder(dtype=tf.float32, shape=[None, self.reduced_dim], name='Z_in')

    def encoder(self):
        with tf.variable_scope("encoder", reuse=None):
            x = tf.reshape(self.X, shape=[-1, 28, 28, 1])
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, 
                            padding='same', 
                            activation=self.activation, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='L1_conv1')
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, 
                            padding='same', 
                            activation=self.activation, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='L2_conv2')
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, 
                            padding='same', 
                            activation=self.activation, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init,
                            name ='L3_conv3')
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.contrib.layers.flatten(x)
            z_mean = tf.layers.dense(x, units=self.reduced_dim, 
                            activation=None, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init,
                            name ='L41_fc1')
            z_std = tf.layers.dense(x, units=self.reduced_dim, 
                            activation=None, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init,
                            name ='L42_fc2')
            eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, 
                            mean=0., stddev=1.0,
                            name='epsilon')
            z = z_mean + tf.exp(z_std / 2) * eps            
            return z, z_mean, z_std

    def decoder(self, sampled_z):
        
        with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(sampled_z, units=self.small_size_img, 
                            activation= self.activation,
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='L5_fc3')
            #x = tf.layers.dense(x, units=self.inputs_decoder * 2 + 1, activation=self.lrelu)
            x = tf.reshape(x, self.reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, 
                                        padding='same',
                                        activation= self.activation,
                                        kernel_initializer=self.weight_init,
                                        bias_initializer=self.bias_init, 
                                        name ='L6_convt1')
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, 
                                        padding='same',
                                        activation= self.activation,
                                        kernel_initializer=self.weight_init,
                                        bias_initializer=self.bias_init,
                                        name ='L7_convt2')
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, 
                                        padding='same',
                                        activation= self.activation,
                                        kernel_initializer=self.weight_init,
                                        bias_initializer=self.bias_init,
                                        name ='L8_convt3')

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid, name ='L9_fc4')
            img = tf.reshape(x, shape=[-1, 28, 28])

            return img

    def _build_model(self):
        sampled, z_mean, z_std = self.encoder()
        self.dec = self.decoder(sampled)
        dec_flat = tf.reshape(self.dec, [-1, 28 * 28])

        # Reconstruction loss
        self.encode_decode_loss = self.Y_flat * tf.log(1e-10 + dec_flat) \
                            + (1 - self.Y_flat) * tf.log(1e-10 + 1 - dec_flat)
        self.encode_decode_loss = -tf.reduce_sum(self.encode_decode_loss, 1)
        # KL Divergence loss
        self.kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
        self.kl_div_loss = -0.5 * tf.reduce_sum(self.kl_div_loss, 1)
        self.loss = tf.reduce_mean(self.encode_decode_loss + self.kl_div_loss)
        optimizer = tflearn.optimizers.Adam(learning_rate=self.lr, name='Adam')
        step = tflearn.variable("step", initializer='zeros', shape=[])
        optimizer.build(step_tensor=step)
        optim_tensor = optimizer.get_tensor()

        trainop = tflearn.TrainOp(loss=self.loss, optimizer=optim_tensor,
                                metric=None, batch_size=self.batch_size,
                                step_tensor=step)
        self.trainer = tflearn.Trainer(train_ops=trainop, tensorboard_dir=self.tensorboar_dir, tensorboard_verbose=self.tb_verbose)
    #def _build_generator(self):
    #    self.reconstructed_x = self.decoder(self.Z_in)

    def generate(self, z=None):
        if z is None:
            z = np.random.randn(size=self.reduced_dim)
        return self.trainer.session.run(self.decoder(self.Z_in), 
                             feed_dict={self.Z_in: z})


    def train(self, trainX, testX, n_epoch=50):   
        self.trainer.fit({self.X: trainX, self.Y: trainX}, val_feed_dicts={self.X: testX, self.Y: testX},
                    n_epoch=n_epoch, show_metric=True)
    
    
    
    # save and load are copied directly from tflearn source code
    def save(self, model_file):
        """
        save model weights
        Arguments:
            model_file: string, address of the saved file
        """
        self.trainer.save(model_file)

    def load(self, model_file, trainable_variable_only=False):
        """ Load.
        Restore model weights.
        Arguments:
            model_file: string, Model path.
            weights_only: `bool`. If True, only weights will be restored (
                and not intermediate variable, such as step counter, moving
                averages...). Note that if you are using batch normalization,
                averages will not be restored as well.
            optargs: optional extra arguments for trainer.restore (see helpers/trainer.py)
                     These optional arguments may be used to limit the scope of
                     variables restored, and to control whether a new session is 
                     created for the restored variables.
        """
        self.trainer.restore(model_file,
                            trainable_variable_only=trainable_variable_only, 
                            verbose=True)


#   ----------------------------------------------
if __name__ == '__main__':
    import os
    os.chdir('/home/arash/Desktop/python/nnexplore/VAE')
    import tflearn.datasets.mnist as mnist

    trainX, trainY, testX, testY = mnist.load_data(one_hot=True)
    trainX = trainX.reshape([-1, 28, 28])
    testX = testX.reshape([-1, 28, 28])
    #sess = tf.Session()
    vae = VAE()
    vae.train(trainX, testX, n_epoch=1)
    #sess.close()
    plt.imshow(vae.generate())
    plt.show()












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