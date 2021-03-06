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

    def __init__(self,
        input_shape=[28, 28],
        reduced_dim=10, keep_prob=0.8,
        activation=tflearn.activations.leaky_relu,
        lr=1e-3, optimizer='adam', tb_verbose=3,
        weight_init=tflearn.initializations.xavier(uniform=False),
        bias_init=tflearn.initializations.xavier(uniform=False),
        batch_size=128, tensorboar_dir='./VAE/tflearn_logs/'):
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        self.input_shape = input_shape
        self.reduced_dim = reduced_dim
        self.keep_prob= keep_prob
        self.activation=activation
        self.lr = lr
        self.optimizer = optimizer
        self.tb_verbose = tb_verbose
        self.small_size_img = int(input_shape[0]*input_shape[1]/16)
        self.batch_size = batch_size
        self.tensorboar_dir = tensorboar_dir
        self.weight_init = weight_init
        self.bias_init = bias_init
        self._setup()
        self._build_model()


    def _setup(self):
        """
        setup the placeholders and dimensions
        """
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.input_shape[0], self.input_shape[1]], name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.input_shape[0], self.input_shape[1]], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, self.input_shape[0] * self.input_shape[1]])
        #self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
        dummy_dim = int(np.sqrt(self.small_size_img))
        self.reshaped_dim = [-1, dummy_dim, dummy_dim, 1] #[-1, 7, 7, 1]
        

    def encoder(self):
        """
        Encoder network
        """
        with tf.variable_scope("encoder", reuse=None):
            x = tf.reshape(self.X, shape=[-1, self.input_shape[0], self.input_shape[1], 1])
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, 
                            padding='same', 
                            activation=self.activation, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='L1_conv1')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, 
                            padding='same', 
                            activation=self.activation, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='L2_conv2')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, 
                            padding='same', 
                            activation=self.activation, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init,
                            name ='L3_conv3')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.contrib.layers.flatten(x)
            z_mean = tf.layers.dense(x, units=self.reduced_dim, 
                            activation=None, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init,
                            name ='L41_fc1')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, z_mean)
            z_std = tf.layers.dense(x, units=self.reduced_dim, 
                            activation=None, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init,
                            name ='L42_fc2')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, z_std)
            eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, 
                            mean=0., stddev=1.0,
                            name='epsilon')
            self.z = z_mean + tf.exp(z_std / 2) * eps           
            return z_mean, z_std


    def decoder(self):
        """
        Decoder network
        """
        with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(self.z, units=self.small_size_img, 
                            activation= self.activation,
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='L5_fc3')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            
            #x = tf.layers.dense(x, units=self.inputs_decoder * 2 + 1, activation=self.lrelu)
            x = tf.reshape(x, self.reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, 
                                        padding='same',
                                        activation= self.activation,
                                        kernel_initializer=self.weight_init,
                                        bias_initializer=self.bias_init, 
                                        name ='L6_convt1')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, 
                                        padding='same',
                                        activation= self.activation,
                                        kernel_initializer=self.weight_init,
                                        bias_initializer=self.bias_init,
                                        name ='L7_convt2')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, 
                                        padding='same',
                                        activation= self.activation,
                                        kernel_initializer=self.weight_init,
                                        bias_initializer=self.bias_init,
                                        name ='L8_convt3')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=self.input_shape[0]*self.input_shape[1], activation=tf.nn.sigmoid, name ='L9_fc4')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            self.dec = tf.reshape(x, shape=[-1, self.input_shape[0], self.input_shape[1]])


    def _build_model(self):
        """
        Building the model and loss function
        """
        z_mean, z_std = self.encoder()
        self.decoder()
        dec_flat = tf.reshape(self.dec, [-1, self.input_shape[0] * self.input_shape[1]])

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
        return self.trainer.session.run(self.dec, feed_dict={self.X: x.reshape((-1,self.input_shape[0], self.input_shape[1]))})


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
        return self.trainer.session.run(self.z, feed_dict={self.X: x.reshape((-1,self.input_shape[0], self.input_shape[1]))})

    
    def visualization_2d(self, x, y):
        """
        2d visualization for the case reduced_dim =2
        Arguments:
            x: 3d array [num_images,h,w], input images
            y: 2d array [num_images, label], labels of the classes
        """
        z = self.trainer.session.run(self.z, feed_dict={self.X: x.reshape((-1, self.input_shape[0], self.input_shape[1]))})
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
        h = self.input_shape[0]
        w = self.input_shape[1]
        img = np.zeros((h*(n+1), 2*w*n))
        for i in range(n+1):
            for j in range(2*n):
                if 2*n*i+j < num_images:
                    img[i*h:(i+1)*h, j*w:(j+1)*w] = \
                            generated[2*n*i+j, :, :].reshape(w,h)

        return img


    def manifold_2d(self, num_imgs_row):
        """
        spectrum of the changes of generator
        Arguments:
            num_imgs_row: int, number of images in each row
        """
        h, w = self.input_shape[0], self.input_shape[1]

        
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
    vae.train(trainX, testX, n_epoch=5)
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

    # the spectrum of the manifold of the generated images
    vae2d.manifold_2d(25)

    plt.show()