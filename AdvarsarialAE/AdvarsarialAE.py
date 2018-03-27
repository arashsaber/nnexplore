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
        tensorboar_dir='./VAE/tflearn_logs/'):
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
        self.Z_prior = tf.placeholder(dtype=tf.float32, 
                                                shape=[self.batch_size,  self.reduced_dim], 
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
            x = tf.nn.dropout(x, self.keep_prob)
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
            x = tf.layers.batch_normalization(x, name='dec_L4_bn')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
            x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=4, strides=2, 
                                        padding='same',
                                        activation= None,
                                        kernel_initializer=self.weight_init,
                                        bias_initializer=self.bias_init, 
                                        name ='dec_L6_convt')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            
            return tf.reshape(x, shape=[-1, 28, 28])


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
            x = tf.layers.batch_normalization(x, name='dec_L3_bn')
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
        self.dis_loss = -tf.reduce_mean(tf.log(d_real + 1e-10) + tf.log(1. - d_fake + 1e-10))



        # Reconstruction loss
        encode_decode_loss = self.Y_flat * tf.log(1e-10 + dec_flat) \
                            + (1 - self.Y_flat) * tf.log(1e-10 + 1 - dec_flat)
        encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
        # KL Divergence loss
        kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
        kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
        self.loss = tf.reduce_mean(encode_decode_loss + kl_div_loss)
        optimizer = tflearn.optimizers.Adam(learning_rate=self.lr, name='Adam')
        step = tflearn.variable("step", initializer='zeros', shape=[])
        optimizer.build(step_tensor=step)
        optim_tensor = optimizer.get_tensor()

        trainop = tflearn.TrainOp(loss=self.loss, optimizer=optim_tensor,
                                metric=None, batch_size=self.batch_size,
                                step_tensor=step)
        self.trainer = tflearn.Trainer(train_ops=trainop, graph=self.graph,
                                    tensorboard_dir=self.tensorboar_dir, 
                                    tensorboard_verbose=self.tb_verbose)


    def reconstruct(self,x):
        """
        Produce the reconstruction for input images x
        Arguments:
            x: 3d array [num_images,h,w], input images
        """
        return self.trainer.session.run(self.dec, feed_dict={self.X: x.reshape((-1,28,28))})


    def reconstructor_viewer(self, x):
        """
        produce an image to view reconstructed data together with the original data
        Arguments:
            x: 3d array [num_images,h,w], input images
        """  
        reconstructed = self.reconstruct(x)
        
        num_images = x.shape[0]
        h, w = x.shape[1], x.shape[2]

        n = np.sqrt(num_images).astype(np.int32)
        img = np.ones((h*n, 2*w*n+2*n))
        for j in range(n):
            for i in range(n):
                img[i*h:(i+1)*h, j*2*w+2*j:(j+1)*2*w+2*j] = \
                            np.hstack((
                                x[n*j+i, :, :].reshape(h,w),
                                reconstructed[n*j+i, :, :].reshape(h,w), 
                            ))             

        return img


    def reduce_dimension(self, x):
        """
        Produce the z vectors for the given inputs
        Arguments:
            x: 3d array [num_images,h,w], input images
        """
        return self.trainer.session.run(self.z, feed_dict={self.X: x.reshape((-1,28,28))})

    
    def visualization_2d(self, x, y):
        """
        2d visualization for the case reduced_dim =2
        Arguments:
            x: 3d array [num_images,h,w], input images
            y: 2d array [num_images, label], labels of the classes
        """
        z = self.trainer.session.run(self.z, feed_dict={self.X: x.reshape((-1,28,28))})
        assert z.shape[1] == 2, 'reduced_dim, i.e., dimension of z, must be 2 for this display to work'
        plt.figure(figsize=(10, 8)) 
        plt.scatter(z[:, 0], z[:, 1], c=np.argmax(y, axis=1))
        plt.colorbar()
        plt.grid()


    def generate(self, num_images=None, z=None):
        """
        generate data from noise input
        Arguments:
            num_images: int, number of images to be generated.
            z: numpy 2d array, noise input
        Note: it will use either the num_images or the given z
        """
        if z is None:
            z = np.random.randn(num_images, self.reduced_dim)
        else:
            assert z.shape[1] == self.reduced_dim, 'z.shape[1] should be equal to {}.'.format(self.reduced_dim)

        return self.trainer.session.run(self.dec, feed_dict={self.z: z})


    def generator_viewer(self, num_images):
        """
        produce an image to view generated data
        Arguments:
            num_images: int, number of images to be generated.
        """  
        generated = self.generate(num_images)

        n = np.sqrt(num_images/2).astype(np.int32)
        h = 28
        w = 28
        img = np.zeros((h*(n+1), 2*w*n))
        for i in range(n+1):
            for j in range(2*n):
                if 2*n*i+j < num_images:
                    img[i*h:(i+1)*h, j*w:(j+1)*w] = \
                            generated[2*n*i+j, :, :].reshape(w,h)

        return img


    def spectum_2d(self, num_imgs_row):
        """
        spectrum of the changes of generator
        Arguments:
            num_imgs_row: int, number of images in each row
        """
        h = w = 28
        
        x = np.linspace(-2.5, 2.5, num_imgs_row)
        img = np.empty((h*num_imgs_row, w*num_imgs_row))
        for i, xi in enumerate(x):
            z = np.vstack(([xi]*num_imgs_row, x)).transpose()
            generated = self.generate(z=z)
            dummy = np.empty((h, 0), dtype=float)
            for j in range(generated.shape[0]):
                dummy = np.hstack((
                    dummy, generated[j,:,:].reshape(w,h)
                    ))

            img[i*h:(i+1)*h, :] = dummy 

        plt.figure(figsize=(8, 8))        
        plt.imshow(img, cmap="gray")


    def train(self, trainX, testX, n_epoch=50): 
        """
        train the neural net
        Arguments:
            trainX: numpy or python array, training data
            testX: numpy or python array, testing data
            n_epoch: int, number of epochs
        """ 
        self.trainer.fit({self.X: trainX, self.Y: trainX}, 
                    val_feed_dicts={self.X: testX, self.Y: testX},
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
    vae = VAE()

    # train and save the model
    #vae.train(trainX, testX, n_epoch=50)
    #vae.save('./VAE/saved_models/model.tfl')

    # load the model
    vae.load('./VAE/saved_models/model.tfl')

    # test the generator
    plt.imshow(vae.generator_viewer(128), cmap='gray')

    # test the dimensionality reduction
    z = vae.reduce_dimension(trainX[10:15,:,:])

    # test the reconstruction
    plt.figure()
    plt.imshow(np.hstack((trainX[10,:,:].reshape(28,28), 
                        vae.reconstruct(trainX[10,:,:]).reshape(28,28)
                        )), cmap='gray')
    plt.figure()
    plt.imshow(vae.reconstructor_viewer(trainX[:128,:,:]), cmap='gray')
    
    plt.show()
    
    # ----------------------------------------
    # Visualization through VAEs
    
    # build the model
    vae2d = VAE(reduced_dim=2)

    #vae2d.train(trainX, testX, n_epoch=50)
    #vae2d.save('./VAE/saved_models/model2d.tfl')
    
    # load the model
    vae2d.load('./VAE/saved_models/model2d.tfl')
    
    # the scatter plot of 2d latent features
    vae2d.visualization_2d(testX[:1000,:,:], testY[:1000,:])

    # the spectrum of the generated images
    vae2d.spectum_2d(25)

    plt.show()