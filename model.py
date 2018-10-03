from lib import read, get_train_batches, get_test
import numpy as np
import tensorflow as tf

path_mnist_train = '/Users/kushal/Development/data/mnist/'
img = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.int64)

conv1Filter = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[32]))
conv1 = tf.nn.relu(tf.nn.conv2d(img, conv1Filter, padding='SAME', strides=[1, 1, 1, 1]) + b1)
mpool1 = tf.nn.max_pool(conv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

conv2Filter = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.conv2d(mpool1, conv2Filter, padding='SAME', strides=[1, 1, 1, 1]) + b2)
mpool2 = tf.nn.max_pool(conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

mpool2_reshape = tf.reshape(mpool2, [-1, 7 * 7 * 64])
W = tf.Variable(tf.truncated_normal(shape=(7 * 7 * 64, 1024), stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[1024]))
dense1 = tf.nn.relu(tf.matmul(mpool2_reshape, W) + b3)

W2 = tf.Variable(tf.truncated_normal(shape=(1024, 10), stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[10]))
dense2 = tf.matmul(dense1, W2)

# Loss blows up when using tf.nn.softmax_cross_entropy_with_logits
loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=dense2))
accuracy = tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(dense2, 1), y), tf.float32
))
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

test_img, test_labels = get_test(path_mnist_train)
with tf.Session() as sess:
    i = 0
    sess.run(tf.global_variables_initializer())
    for i in range(0, 20000):
        for img_batch, label_batch in get_train_batches(path_mnist_train):
            train_op_, loss_ = sess.run(
                feed_dict={
                    img: img_batch,
                    y: label_batch
                },
                fetches=[train_op, loss])
            if i % 10 == 0:
                print("Loss = ", loss_)
        if i % 10 == 0:
            accuracy_, = sess.run(
                feed_dict={
                    img: test_img,
                    y: test_labels
                },
                fetches=[accuracy]
            )
            print("accuracy = ", accuracy_)
