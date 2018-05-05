#!/usr/bin/python3
"""
The file contains a Adversarial Autoencoder in tensorflow.

Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by Arash Tehrani
"""
#   ---------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflearn
import os
import datetime
from progressbar import ProgressBar, ETA, Bar, Percentage, Timer
#   ----------------------------------------------
class AAE(object):

    def __init__(self,
        input_shape=[28, 28],
        reduced_dim=10, 
        batch_size=128, channel_size=128,
        activation=tf.nn.relu, #learn.activations.leaky_relu,
        lr=1e-3, beta1=0.5,
        weight_init=tflearn.initializations.xavier(uniform=False),
        bias_init=tflearn.initializations.xavier(uniform=False),
        tensorboar_dir='./AdversarialAE/tf_logs'):
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        self.input_shape = input_shape
        self.reduced_dim = reduced_dim
        self.activation=activation
        self.lr = lr
        self.beta1 = beta1
        self.small_size_img = int(input_shape[0]*input_shape[1]/16)
        self.batch_size = batch_size
        self.channel_size = channel_size
        dummy_dim = int(np.sqrt(self.small_size_img))
        self.reshaped_dim = [-1, dummy_dim, dummy_dim, self.channel_size] #[-1, 7, 7, .self.channel_size]
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.tensorboar_dir = tensorboar_dir
        self._setup()
        self._build_model()


    def _setup(self):
        """
        setup the placeholders and dimensions
        """
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.input_shape[0], 
                                self.input_shape[1]], name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.input_shape[0], 
                                self.input_shape[1]], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, self.input_shape[0] * self.input_shape[1]])
        self.Z = tf.placeholder(dtype=tf.float32, shape=[None, self.reduced_dim], name='Z')
        self.Z_prior = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,  self.reduced_dim], 
                                name='Z_prior')


    def encoder(self, x, reuse=None):
        """
        Encoder network
        """
        with tf.variable_scope('encoder', reuse=None):
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
            #tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=1024, 
                            activation=None, 
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init,
                            name ='enc_L4_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.batch_normalization(x, name='enc_L5_bn')
            #tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
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
        with tf.variable_scope('decoder', reuse=reuse):
            x = tf.layers.dense(z, units=1024, 
                            activation= None,
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='dec_L1_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.batch_normalization(x, name='dec_L2_bn')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
            x = tf.layers.dense(x, units=dim, 
                            activation= None,
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='dec_L3_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.reshape(x, shape=self.reshaped_dim)
            x = tf.layers.batch_normalization(x, name='dec_L4_bn')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
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
            x = tf.reshape(x, shape=[-1, self.input_shape[0], self.input_shape[1]])

            return x 


    def discriminator(self, z, reuse=False):
        """
        Discriminator network
        Note: if uncomment the monitor of activations, then two discriminator in the 
        tensorboard will be created one for each real or fake distribution.
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
            x = tf.layers.dense(x, units=1000, 
                            activation= None,
                            kernel_initializer=self.weight_init,
                            bias_initializer=self.bias_init, 
                            name ='dis_L2_fc')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = tf.layers.batch_normalization(x, name='dis_L3_bn')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            x = self.activation(x)
            x = tf.layers.dense(x, units=1, 
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

        with tf.variable_scope(tf.get_variable_scope()):
            self.gen = self.decoder(self.Z_prior, reuse=True)
        

        # loss
        self.rec_loss = tf.reduce_mean(tf.square(self.Y_flat - y_flat))

        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=tf.ones_like(d_fake), logits=d_fake))
        
        self.dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones_like(d_real), logits=d_real)) \
                        + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.zeros_like(d_fake), logits=d_fake))

        # collecting trainable variables
        all_variables = tf.trainable_variables()
        enc_vars = [var for var in all_variables if 'encoder' in var.name]
        dec_vars = [var for var in all_variables if 'decoder' in var.name]
        dis_vars = [var for var in all_variables if 'discriminator' in var.name]
        rec_vars = enc_vars + dec_vars
        
        # building the optimizers
        opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1)
        self.rec_opt = opt.minimize(self.rec_loss, var_list=rec_vars)
        self.dis_opt = opt.minimize(self.dis_loss, var_list=dis_vars)
        self.gen_opt = opt.minimize(self.gen_loss, var_list=enc_vars)

        # getting the gradients (I could also tf.gradients)
        rec_grads = opt.compute_gradients(self.rec_loss, rec_vars)
        rec_grads = list(zip(rec_grads, rec_vars))
    
        dis_grads = opt.compute_gradients(self.dis_loss, dis_vars)
        dis_grads = list(zip(dis_grads, dis_vars))
        
        gen_grads = opt.compute_gradients(self.gen_loss, enc_vars)
        gen_grads = list(zip(gen_grads, enc_vars))

        self.saver = tf.train.Saver()
        
        self._summary_setup(enc_vars, dec_vars, dis_vars,
                            rec_grads, dis_grads, gen_grads)

        self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=self.graph)        


    def _trainer(self, dataset, n_epoch=100, z_std=5, checkpoint_interval=50, report_flag=True):
        """
        train the adversarial autoencoder
        Arguments:
            dataset: tensorflow dataset object
            n_epoch: int, number of epochs
            z_std: float, standard deviation of z_prior
            checkpoint_interval: int, interval in number of batches to store the network parameters
            report_flag: bool, a flag to print the log values
        """
        step = 0
        self.n_epoch = n_epoch
        self.sess.run(tf.global_variables_initializer())
        path = os.path.join(self.tensorboar_dir, self.run_id)
        if not os.path.exists(path):
            os.mkdir(path)
        writer = tf.summary.FileWriter(logdir=path, graph=self.graph)
        self._log_setup(path)
        
        with self.sess.as_default() as sess:
            for epoch_num in range(self.n_epoch):

                n_batches = int(dataset.train.num_examples / self.batch_size)
                widgets = ['epoch {}|'.format(epoch_num), Percentage(), Bar(), ETA(), Timer()]
                pbar = ProgressBar(maxval=n_batches, widgets=widgets)
                pbar.start()
                
                for batch_num in range(1, n_batches + 1):
                    pbar.update(batch_num)
                    # getting the batch data
                    z_prior = np.random.randn(self.batch_size, self.reduced_dim) * z_std
                    batch_x, _ = dataset.train.next_batch(self.batch_size)
                    batch_x = batch_x.reshape((-1, 28,28))

                    sess.run(self.rec_opt, feed_dict={self.X: batch_x, self.Y: batch_x})
                    sess.run(self.dis_opt,
                            feed_dict={self.X: batch_x, self.Y: batch_x, self.Z_prior: z_prior})
                    sess.run(self.gen_opt, feed_dict={self.X: batch_x, self.Y: batch_x})
                    
                    if batch_num % checkpoint_interval == 0:
                        a_loss, d_loss, g_loss, summary = sess.run(
                                [self.rec_loss, self.dis_loss, self.gen_loss, self.summary_op],
                                feed_dict={self.X: batch_x, self.Y: batch_x, self.Z_prior: z_prior})
                        writer.add_summary(summary, global_step=step)
                        if report_flag:
                            print('Epoch: {}, iteration: {}'.format(epoch_num, batch_num))
                            print('Autoencoder Loss: {}'.format(a_loss))
                            print('Discriminator Loss: {}'.format(d_loss))
                            print('Generator Loss: {}'.format(g_loss))
                        with open(path+'/log.txt', 'a') as log:
                            log.write('Epoch: {}, batch number: {}\n'.format(epoch_num, batch_num))
                            log.write('Autoencoder Loss: {}\n'.format(a_loss))
                            log.write('Discriminator Loss: {}\n'.format(d_loss))
                            log.write('Generator Loss: {}\n'.format(g_loss))
                        
                    step += 1
        writer.flush()
        writer.close()


    def train(self, dataset, n_epoch=100, z_std=5, checkpoint_interval=50, report_flag=True, run_id=None): 
        """
        train the neural net
        Arguments:
            dataset: tensorflow binary object, dataset
            n_epoch: int, number of epochs
            z_std: float, standard deviation of z-prior
            checkpoint_interval: int, interval in number of batches to write the log file or print the output
            report_flag: bool, a flag to print the log values
        """
        if run_id is None:
            run_id = 'Adversarial_AE_{}'.format(datetime.datetime.now())
        
        self.run_id = run_id
        self._trainer(dataset, n_epoch=n_epoch, z_std=z_std, checkpoint_interval=checkpoint_interval, report_flag=report_flag)
    
    
    def _summary_setup(self, enc_vars, dec_vars, dis_vars,
                        rec_grads, dis_grads, gen_grads
                        ):
        """
        Build the sumary of trainable variables, activations, and gradiesnts as suggested by 
        the name of the arguments
        """
        # Summarize all activations
        for actv in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
            tf.summary.histogram(actv.name, actv)
        
        # Summarize the weights and biases
        for var in enc_vars:
            tf.summary.histogram(var.name, var)
        
        for var in dec_vars:
            tf.summary.histogram(var.name, var)
        
        for var in dis_vars:
            tf.summary.histogram(var.name, var)
        
        # Summarize all gradients
        for grad, var in rec_grads:
            tf.summary.histogram(var.name + '/rec_gradient', grad)
            
        for grad, var in dis_grads:
            tf.summary.histogram(var.name + '/dis_gradient', grad)
        
        for grad, var in gen_grads:
            tf.summary.histogram(var.name + '/gen_gradient', grad)

        # summarize the losses
        tf.summary.scalar(name='Reconstruction Loss', tensor=self.rec_loss)
        tf.summary.scalar(name='Discriminator Loss', tensor=self.dis_loss)
        tf.summary.scalar(name='Generator Loss', tensor=self.gen_loss)
        tf.summary.histogram(name='Latent Distribution', values=self.z)
        tf.summary.histogram(name='Real Distribution', values=self.Z_prior)
        tf.summary.image(name='Input Images', 
                        tensor=tf.reshape(self.X, shape=[-1, self.input_shape[0], self.input_shape[1], 1]), 
                        max_outputs=10)      
        tf.summary.image(name='Reconstructed Images', 
                        tensor=tf.reshape(self.y, shape=[-1, self.input_shape[0], self.input_shape[1], 1]),  
                        max_outputs=10)
        tf.summary.image(name='Generated Images', 
                        tensor=tf.reshape(self.gen, shape=[-1, self.input_shape[0], self.input_shape[1], 1]),  
                        max_outputs=10)

        self.summary_op = tf.summary.merge_all()


    def save(self, model_file):
        """
        save model weights
        Arguments:
            model_file: string, address of the saved file
        """
        self.saver.save(self.sess, save_path=model_file)


    def load(self, model_file):
        """
        Restore model weights.
        Arguments:
            model_file: string, address of the saved file.
            trainable_variable_only: boolean, set to True if you only want to load the weights
        """
        self.saver.restore(self.sess, model_file)

    def _log_setup(self, path):
        """
        Store the network parameters in the log file
        Arguments:
            path: str, path to the corresponding folder
        """
        with open(path+'/log.txt', 'w') as log:
            log.write('Adversarial AutoEncoder\n')
            log.write('-------------------------------------------\n')
            log.write('trained on: {}\n'.format(datetime.datetime.now()))
            log.write('Parameters:\n')
            log.write('dimension of z: {}\n'.format(self.reduced_dim))
            log.write('learning rate: {}\n'.format(self.lr))
            log.write('batch size: {}\n'.format(self.batch_size))
            log.write('number of epochs: {}\n'.format(self.n_epoch))
            log.write('beta1: {}\n'.format(self.beta1))
            log.write('channel size: {}\n'.format(self.channel_size))      
            log.write('-------------------------------------------\n')

    def reconstruct(self, x):
        """
        Produce the reconstruction for input images x
        Arguments:
            x: 3d array [num_images,h,w], input images
        """
        return self.sess.run(self.y, feed_dict={self.X: x.reshape((-1,self.input_shape[0], self.input_shape[1]))})


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
        return self.sess.run(self.z, feed_dict={self.X: x.reshape((-1,self.input_shape[0], self.input_shape[1]))})

    
    def visualization_2d(self, x, y):
        """
        2d visualization for the case reduced_dim =2
        Arguments:
            x: 3d array [num_images,h,w], input images
            y: 2d array [num_images, label], labels of the classes
        """
        z = self.sess.run(self.z, feed_dict={self.X: x.reshape((-1, self.input_shape[0], self.input_shape[1]))})
        assert z.shape[1] == 2, 'reduced_dim, i.e., dimension of z, must be 2 for this display to work'
        plt.figure(figsize=(10, 8)) 
        plt.scatter(z[:, 0], z[:, 1], c=np.argmax(y, axis=1))
        plt.colorbar()
        plt.grid()


    def generate(self, num_images=None, z=None, z_std=5):
        """
        generate data from noise input
        Arguments:
            num_images: int, number of images to be generated.
            z: numpy 2d array, noise input
        Note: it will use either the num_images or the given z
        """
        if z is None:
            if num_images is None:
                z = np.random.randn(self.batch_size, self.reduced_dim) * z_std
            else:
                z = np.random.randn(num_images, self.reduced_dim) * z_std
        else:
            assert z.shape[1] == self.reduced_dim, 'z.shape[1] should be equal to {}.'.format(self.reduced_dim)

        return self.sess.run(self.decoder(self.Z_prior, reuse=True), feed_dict={self.Z_prior: z})


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


    def spectum_2d(self, num_imgs_row):
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

   

#   ----------------------------------------------
if __name__ == '__main__':

    # get the data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)

    # ----------------------------------------
    '''
    # build the model
    aae = AAE()

    # train and save the model
    #aae.train(dataset=mnist, n_epoch=20, report_flag=False)
    #aae.save('./AdversarialAE/saved_models/model.ckpt')

    # load the model
    aae.load('./AdversarialAE/saved_models/model.ckpt')
    
    # test the generator
    plt.imshow(aae.generator_viewer(128), cmap='gray')
    
    # get the images
    images, labels = mnist.test.next_batch(128)
    images = images.reshape((-1,28,28))
    image = images[0,:,:]
                    
    # test the dimensionality reduction
    z = aae.reduce_dimension(images)

    # test the reconstruction
    plt.figure()
    plt.imshow(np.hstack((image.reshape(28,28), 
                        aae.reconstruct(image).reshape(28,28)
                        )), cmap='gray')
    plt.figure()
    plt.imshow(aae.reconstructor_viewer(images), cmap='gray')
    
    plt.show()
    '''
    # ----------------------------------------
    # build the 2 dimensional model
    aae2d = AAE(reduced_dim=2) 

    # train and save the model
    aae2d.train(dataset=mnist, n_epoch=20, report_flag=False)
    aae2d.save('./AdversarialAE/saved_models/model2d.ckpt')

    # load the model
    aae2d.load('./AdversarialAE/saved_models/model2d.ckpt')
    

        
