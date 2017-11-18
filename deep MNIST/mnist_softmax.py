from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# to do efficient numerical computing in Python, we use libraries like NumPy
# that do expensive operations outside of Python (i.e. in another language)
# there is a lot of overhead from switching back to Python every operation
# Tensorflow lets us describe a graph of interacting operations that run outside
# of Python, so we avoid the ovehead by doing all the operations 'at once'
# So our Python code is for building this compuation graph, and to dictate which
# parts of this graph should be run, and when

def main(_):
    # download and read in data
    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

    # Start building a computation graph by creating nodes for input images
    # and target output classes
    # x and y_ aren't specific values, they are placeholders - a value that
    # we'll input when we ask Tensorflow to run a computation
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    # None indicates it can be of any size (i.e. any batch size)
    # shape is optional argument, but allows Tensorflow to automatically catch
    # bugs relating to wrong dimensions

    # define weights W, biases b
    # Variable is a value that lives in Tensorflow's computation graph
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # class prediction
    y = tf.matmul(x, W) + b

    # loss
    # tf.nn.cross_entropy_with_logits takes sum, tf.reduce_mean averages
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable., so instead use:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = y_,
        logits = y)
    )

    # because of computation graph, Tensorflow can automatically differentiate,
    # also has built-in optimization algorithms, so training model is simple

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # add several operations to computation graph:
    # compute gradients, compute parameter update steps, and appyl update steps
    # to parameters

    # Tensorflow relies on efficient C++ background for computations
    # connection to this backend is called a session
    # usually first create a computation graph, then launch in a session
    # InteractiveSession allows you to build computation graph and run it at the
    # same time
    sess = tf.InteractiveSession()

    # initialize variables, so that they can be used in this session
    sess.run(tf.global_variables_initializer())

    # train the model
    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict = {x: batch[0], y_: batch[1]})
        # feed_dict replaces placeholders x, y_ with training examples

    # see if predicted and label are the same (list of booleans)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # cast array of boooleans to 0 and 1, compute mean (i.e. accuracy)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict = {x: mnist.test.images, y_:mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
