from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # Reshapes features["x"] into tensor of
    # [batch_size, image_width, image_height, channels].
    # Here the -1 parameter for batch_size is a placeholder for however many
    # examples features["x"] contains

    # Convolutional Layer # 1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5, 5],
        padding = "same", # adds 0 padding so that output is same dimension as
        # input, i.e. still 28x28
        activation = tf.nn.relu
    )
    # output is [batch_size, 28, 28, 32], since 32 channels corresponding to 32
    # filters

    # Pooling Layer # 1
    pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = [2, 2],
        strides = 2
    )
    # output is [batch_size, 14, 14, 32], i.e. reduces by 1/2

    # Convolutional Layer # 2
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu
    )

    # Pooling Layer # 2
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = [2, 2],
        strides = 2
    )
    # output is [batch_size, 7, 7, 64], i.e. max pooling reduces by 1/2,
    # and convolution makes 64 channels

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # Flatten into matrix. Once again -1 is a placeholder for batch_size.
    dense = tf.layers.dense(
        inputs = pool2_flat,
        units = 1024,
        activation = tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs = dense,
        rate = 0.4,
        training = mode == tf.estimator.ModeKeys.TRAIN
        # only use dropout when training
    )

    # Logits Layer
    logits = tf.layers.dense(
        inputs = dropout,
        units = 10
    )

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input = logits, axis = 1),
        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the
        # 'logging_hook'.
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
        # we add an explicit name so that we can reference this operation later
    }
    # "classes" is class prediction, while "probabilities" is a vector of
    # probabilities for each class

    # compile results in dictionary, return as EsimatorSpec object
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions
        )

    # Calculate Loss (for both TRAIN AND EVAL modes)
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 10)
    # converts from label to boolean vector, i.e.
    # 1 -> [0, 1, 0, 0, ..., 0]
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels = onehot_labels,
            logits = logits
    )
    # use cross entropy loss function for multi-class classification

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # pass in loss function to optimize, minimize it using stochastic
        # gradient descent w/ learning rate 0.001
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss, # loss function as defined above
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(
            mode = mode,
            loss = loss,
            train_op = train_op
        )

    # Add evaluation metrics (for EVAL mode)
    # i.e. a metric to measure accuracy of our model
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels = labels,
            predictions = predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode = mode,
        loss = loss,
        eval_metric_ops = eval_metric_ops
    )

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype = np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype = np.int32)

    # Create the Estimator
    # i.e. a Tensorflow object used for performing high-level model manipulation
    mnist_classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn, # the model function to use
        model_dir = "temp" # where to save model data, i.e. checkpoints
        )

    # Set up logging for predictions
    # training can take a while, so need to track progress
    tensors_to_log = {"probabilities": "softmax_tensor"} # dictionary of labels,
    # as named above
    logging_hook = tf.train.LoggingTensorHook(
        tensors = tensors_to_log, # log tensors_to_log, i.e. probabilities
        every_n_iter = 50  # log every 50 iterations
        )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data}, # as named above, dictionary so can index into it
        # as in cnn_model_fn
        y = train_labels, # as named above
        batch_size = 100, # mini batch size
        num_epochs = None, # i.e. as long as it takes to do iterations
        shuffle = True # shuffle the data
        )
    mnist_classifier.train(
        input_fn = train_input_fn,
        steps = 20000, # train for this many steps (i.e. don't worry about
        # epochs)
        hooks = [logging_hook] # pass logging_hook as named above so that it's
        # triggered during training
        )

    # Evaluate the model and print results
    # i.e. when model done training, want to determine how accurate it is
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x", eval_data},
        y = eval_labels,
        num_epochs = 1, # go through training set once
        shuffle = False
    )
    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)

# Our application logic will be added here
# if __name__ == 'main':
#     tf.app.run()

tf.app.run()
