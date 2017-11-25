from __future__ import print_function
import tensorflow as tf

# Central unit of data is Tensor
# Tensor is an array w/ rank corresponding to number of dimensions

# TensorFlow programs are composed of two sections:
#   - building the computational graph
#   - running the computational graph

# computational graph is a series of TensorFlow operations arranged into a graph
# of nodes
# each node takes zero or more tensors as inputs and produces a tensor as output

# Simple computational graph
node1 = tf.constant(3.0, dtype = tf.float32) # constant node takes no inputs
node2 = tf.constant(4.0) # implicitly dtype = tf.float32

# Printing nodes doesn't print their values
print(node1, node2)

# To print their values, we need to evaluate them, and we do this by running the
# computational graph within a session
# a session encapsulates the control and state of the TensorFlow runtime

sess = tf.Session()
print(sess.run([node1, node2]))


node3 = tf.add(node1, node2) # combine nodes with operations, also a node
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# parametrize graph to accept inputs w/ placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides shortcut for tf.add()

# use feed_dict argument to run method to feed concrete values to placeholders
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# add another layer of computation
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# we want to create a trainable model, so we need to be able to modify the graph
# to get new outputs w/ the same input
# Variables allow us to add trainable parameters to a graph
# constructed with type and initial value
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# need to initialize variables
init = tf.global_variables_initializer()
sess.run(init) # init is a handle to the TensorFlow sub-graph that initializes
                # all the global variables. until we call sess.run, the
                # variables are unitialized

# since x is a placeholder, we can evaluate linear model for several values of
# x simultaneously
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# define loss function on <y> labels
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# assign perfect values to variables
fixW = tf.assign(W, [-1.])
fixB = tf.assign(b, [1.])
sess.run([fixW, fixB])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
