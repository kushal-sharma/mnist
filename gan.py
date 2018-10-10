import tensorflow as tf
import numpy as np
from lib import mnist_to_png, MnistGenerator
import os


class DenseGenerator:
    def __init__(self):
        self.W1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[100, 128], stddev=0.1))
        self.b1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[128]))

        self.W2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[128, 784], stddev=0.1))
        self.b2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[784], stddev=0.1))

        self.alpha = 0.01

    def eval(self, z):
        h1 = tf.layers.batch_normalization(tf.matmul(z, self.W1) + self.b1, center=True, scale=True)
        h2 = tf.maximum(h1, 0.01 * h1)
        dense2 = tf.layers.batch_normalization(tf.nn.tanh(tf.matmul(h2, self.W2) + self.b2), center=True, scale=True)
        return dense2

    def get_vars(self):
        return [self.W1, self.b1, self.W2, self.b2]


class DenseDiscriminator:
    def __init__(self):
        self.W1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[784, 128], stddev=0.1))
        self.b1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[128], stddev=0.1))

        self.W2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[128, 1], stddev=0.1))
        self.b2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[1], stddev=0.1))
        self.alpha = 0.01

    def eval(self, xin):
        h1 = tf.layers.batch_normalization(tf.matmul(xin, self.W1) + self.b1, center=True, scale=True)
        h2 = tf.maximum(h1, self.alpha * h1)

        h3 = tf.layers.batch_normalization(tf.matmul(h2, self.W2) + self.b2, center=True, scale=True)
        h4 = tf.maximum(h3, self.alpha * h3)

        sf = tf.nn.sigmoid(h4)
        return sf

    def get_vars(self):
        return [self.W1, self.b1, self.W2, self.b2]


def train(path_mnist_train, path_img_save):
    k = 2
    batch_size = 50
    nepochs = 100
    z = tf.placeholder(dtype=tf.float32, shape=[None, 100])
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    datagen = MnistGenerator(path_mnist_train)
    generator = DenseGenerator()
    discriminator = DenseDiscriminator()
    xout = generator.eval(z)
    loss_discriminator = -tf.reduce_mean(tf.log(discriminator.eval(x)) + tf.log(1 - discriminator.eval(xout)))

    loss_generator = -tf.reduce_mean(tf.log(discriminator.eval(xout)))

    optim_disc = tf.train.AdamOptimizer(1e-4).minimize(loss_discriminator, var_list=discriminator.get_vars())
    optim_gen = tf.train.AdamOptimizer(1e-4).minimize(loss_generator, var_list=generator.get_vars())

    ntrain_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(0, nepochs):
            for img, label in datagen.get_train_batches(path_mnist_train, batch_size):
                ntrain_step = ntrain_step + 1
                zin = np.random.normal(0, 0.5, size=[batch_size, 100])
                xin = img.reshape([-1, 28 * 28])
                if ntrain_step % k == 0:
                    optim_gen_, loss_generator_ = sess.run(feed_dict={
                        z: zin,
                        x: xin
                    }, fetches=[optim_gen, loss_generator])
                    print("losses_generator = ", loss_generator_)
                else:
                    optim_disc_, loss_discriminator_ = sess.run(feed_dict={
                        z: zin,
                        x: xin
                    }, fetches=[optim_disc, loss_discriminator])
                    print("losses_discriminator = ", loss_discriminator_)

            # See how good is the generator
            zin = np.random.normal(0, 0.5, size=[1, 100])
            xout_, = sess.run(feed_dict={
                z: zin
            }, fetches=[xout])
            mnist_to_png(path_img_save.format(i), xout_.reshape((28, 28)))


if __name__ == "__main__":
    # train('/Users/kushal/Development/data/mnist')
    path_mnist = '/Users/kushal/Development/data/mnist'
    path_img_save = os.path.join(path_mnist, 'generated', '{0}.png')
    train(path_mnist, path_img_save)
