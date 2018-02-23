import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflearn


from tensorflow.examples.tutorials.mnist import input_data

#   ----------------------------------------------
class VAE(object):

    def __init__(self, session, reduced_dim=10, keepprob=0.8, dec_in_channels=1,
                 LR=1e-3, optimizer='adam', tb_verbose=3):
        tf.reset_default_graph()
        self.sess = session
        self.reduced_dim = reduced_dim
        self.keepprob= keepprob
        self.dec_in_channels = dec_in_channels
        self.LR = LR
        self.optimizer = optimizer
        self.tb_verbose = tb_verbose
        self.setup()
        self._build_model()

    def setup(self):
        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, 28 * 28])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
        self.reshaped_dim = [-1, 7, 7, self.dec_in_channels]
        self.inputs_decoder = 49 * self.dec_in_channels

    def encoder(self):
        activation = self.lrelu
        with tf.variable_scope("encoder", reuse=None):
            input_shape = [None, 28, 28, 1]
            X = tflearn.layers.core.input_data(shape=input_shape, name='input')
            #X = tf.reshape(self.X_in, shape=[-1, 28, 28, 1])
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
        loss = tf.reduce_mean(self.img_loss + self.latent_loss)
        net = tflearn.layers.estimator.regression(unreshaped,
                                                  optimizer=self.optimizer,
                                                  learning_rate=self.LR,
                                                  loss=loss,
                                                  name='targets')

        self.model = tflearn.DNN(net, tensorboard_dir='logs', tensorboard_verbose=self.tb_verbose, session=self.sess)

    def train(self, x, val_x, n_epochs=410,
              batch_size=128, snapshot_step=200, show_metric=True):

        """
        Train the sparseAE
        :param x: input data to feed the network
        :param val_x: validation data
        :param n_epochs: int, number of epochs
        :param batch_size: int
        :param snapshot_step: int
        :param show_metric: boolean
        """
        self.sess.run(tf.global_variables_initializer())
        self.model.fit({'input': x}, {'targets': x}, n_epoch=n_epochs,
                       batch_size=batch_size,
                       validation_set=({'input': val_x}, {'targets': val_x}),
                       snapshot_step=snapshot_step,
                       show_metric=show_metric,
                       run_id='VAE')
        '''
        sess.run(tf.global_variables_initializer())

        for i in range(num_epochs):
            print(i)
            batch = [np.reshape(b, [28, 28]) for b in data.train.next_batch(batch_size=self.batch_size)[0]]
            sess.run(optimizer, feed_dict={self.X_in: batch, self.Y: batch, self.keep_prob: self.keepprob})

            if not i % 200:
                ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd],
                                                       feed_dict={self.X_in: batch, self.Y: batch, self.keep_prob: 1.0})
                plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
                plt.show()
                plt.imshow(d[0], cmap='gray')
                plt.show()
                print(i, ls, np.mean(i_ls), np.mean(d_ls))
        '''
    @staticmethod
    def lrelu(x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))





#   ----------------------------------------------
if __name__ == '__main__':

    import tflearn.datasets.mnist as mnist

    trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

    sess = tf.Session()
    vae = VAE(sess)
    vae.train(trainX, testX)
    sess.close()














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