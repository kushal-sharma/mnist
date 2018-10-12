import tensorflow as tf
import numpy as np
from lib import mnist_to_png, MnistGenerator
import os
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


class DenseGenerator:
    def __init__(self):
        self.W1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[100, 128], stddev=0.1))
        self.b1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[128]))

        self.W2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[128, 784], stddev=0.1))
        self.b2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[784], stddev=0.1))

        # For LeakyRelu
        # Tf doesn't have an inbuilt implementation
        self.alpha = 0.01

    def eval(self, z):
        z1 = tf.matmul(z, self.W1) + self.b1
        h1 = tf.maximum(z1, self.alpha * z1)

        z2 = tf.matmul(h1, self.W2) + self.b2
        h2 = tf.nn.tanh(z2)

        return h2

    def get_vars(self):
        return [self.W1, self.b1, self.W2, self.b2]


class DenseDiscriminator:
    def __init__(self, squash=False):
        self.W1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[784, 128], stddev=0.1))
        self.b1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[128], stddev=0.1))

        self.W2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[128, 1], stddev=0.1))
        self.b2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[1], stddev=0.1))
        self.squash = squash
        self.alpha = 0.01

    def eval(self, xin):
        z1 = tf.matmul(xin, self.W1) + self.b1
        h1 = tf.maximum(z1, self.alpha * z1)

        z2 = tf.matmul(h1, self.W2) + self.b2
        h2 = tf.maximum(z2, self.alpha * z2)
        if self.squash:
            return tf.nn.sigmoid(h2)
        else:
            return h2

    def get_vars(self):
        return [self.W1, self.b1, self.W2, self.b2]


def train(path_mnist_train, path_img_save, write_dir, label):
    k = 2
    batch_size = 50
    nepochs = 30
    latent_variance = 0.5
    z = tf.placeholder(dtype=tf.float32, shape=[None, 100])
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    noise = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    datagen = MnistGenerator(path_mnist_train)
    generator = DenseGenerator()
    discriminator = DenseDiscriminator(squash=False)
    xout = generator.eval(z)
    loss_discriminator = -tf.reduce_mean(discriminator.eval(x + noise) -
                                         discriminator.eval(xout + noise))

    loss_generator = -tf.reduce_mean(discriminator.eval(xout + noise))

    optim_disc = tf.train.AdamOptimizer(1e-4).minimize(loss_discriminator, var_list=discriminator.get_vars())
    optim_gen = tf.train.AdamOptimizer(1e-4).minimize(loss_generator, var_list=generator.get_vars())

    ntrain_step = 0
    tf.summary.scalar('loss_gen', loss_generator)
    tf.summary.scalar('loss_disc', loss_discriminator)
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.join(write_dir, 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(0, nepochs):
            for img, label in datagen.get_train_batches(path_mnist_train, batch_size, label=label):
                ntrain_step = ntrain_step + 1
                zin = np.random.normal(0, latent_variance, size=[batch_size, 100])
                xin = img.reshape([-1, 28 * 28])
                noise_in = np.random.normal(0, 0.001, xin.shape)
                if ntrain_step % k == 0:
                    summary_, optim_gen_, loss_generator_ = sess.run(feed_dict={
                        z: zin,
                        x: xin,
                        noise: noise_in
                    }, fetches=[merged, optim_gen, loss_generator])
                else:
                    summary_, optim_disc_, loss_discriminator_ = sess.run(feed_dict={
                        z: zin,
                        x: xin,
                        noise: noise_in
                    }, fetches=[merged, optim_disc, loss_discriminator])
                if ntrain_step % 100 == 0:
                    writer.add_summary(summary_, ntrain_step)
                    print("losses_generator = ", loss_generator_)
                    print("losses_discriminator = ", loss_discriminator_)
            # See how good is the generator
            zin = np.random.normal(0, latent_variance, size=[1, 100])
            xout_, = sess.run(feed_dict={
                z: zin
            }, fetches=[xout])
            mnist_to_png(path_img_save.format(i), xout_.reshape((28, 28)))


if __name__ == "__main__":
    # train('/Users/kushal/Development/data/mnist')
    path_mnist = '/Users/kushal/Development/data/mnist'
    write_path = '/Users/kushal/Development/mnist/tf_logs'
    path_img_save = os.path.join(path_mnist, 'generated', '{0}.png')
    train(path_mnist, path_img_save, write_path, 9)
