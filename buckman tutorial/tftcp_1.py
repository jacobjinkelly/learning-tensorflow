import tensorflow as tf


#### TensorFlow: The Confusing Parts (1)


### The Computation Graph


# tf.constant returns a pointer to the node we just created
# this code will create four nodes
two_node = tf.constant(2)
another_two_node = tf.constant(2)
two_node = tf.constant(2)
tf.constant(3)

# here we have one node, and two pointers pointing to it
two_node = tf.constant(2)
another_pointer_at_two_node = two_node
two_node = None
print(two_node)
print(another_pointer_at_two_node)

# the sum node does not contain 5, but instead the operation to add two_node and
# three_node, whatever their values may be
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node  # equivalent to tf.add(two_node, three_node)
print(sum_node)


### The Session


# the computation graph is a "template" for the computations we want to do:
# it lays out all the steps

# we need to create a session which goes through every graph of the node and
# allocates memory to store its output

# the session contains a pointer to the global graph, thus it is updated
# whenever new nodes are added, so it doesn't matter where we create the session

# we call sess.run(node) to return the value of <node>, and Tensorflow performs
# all computations necessary to determine that value
sess = tf.Session()
print(sess.run(sum_node))

# we can pass a list of nodes to sess.run(), and get a list of the corresponding
# outputs. sess.run is a huge bottleneck in TensorFlow programs, so it is better
# to make one call specifying multiple nodes instead of several calls
print(sess.run([three_node, sum_node]))


## Placeholders & feed_dict


# we'd like to provide inputs to our graph so that the result isn't the same
# every time we run it

# we use a placeholder, which is a node which requires a value to be fed to it

# notice the keys in the dict passed to feed_dict are pointers to the
# placeholder nodes we'd like to assign to
input_placeholder = tf.placeholder(tf.int32)
print(sess.run(input_placeholder, feed_dict={input_placeholder: 2}))


### Computation Paths


# calling sess.run on sum_node will result in an error saying we haven't fed
# a value to the placeholder input_placeholder even though we didn't call
# sess.run on input_placeholder because sum_node depends on it

# we can still call sess.run(three_node) as it doesn't depend on
# input_placeholder

# in other words, sess.run will only compute the nodes it has
# to for the current evaluation
# this can result in huge savings in runtime on very big graphs, and allows
# us to construct multi-purpose graphs which use a single, shared set of core
# nodes to do different things depending on the computation path
input_placeholder = tf.placeholder(tf.int32)
three_node = tf.constant(3)
sum_node = input_placeholder + three_node
sess.run(three_node)


## Variables & Side Effects

# so far we've seen two types of nodes which never have other nodes as
# ancestors: tf.constant, which is the same for every run, and tf.placeholder,
# which is specified by the programmer via feed_dict

# a third node with no ancestor is a variable, which we create with
# tf.get_variable(name, shape), where <name> must be unique to the global graph
# (we can also use scoping)

# if we try to call sess.run on this, we will get an error as count_variable
# is essentially null at this point
count_variable = tf.get_variable("count", [])


## tf.assign


zero_node = tf.constant(0.)
assign_node = tf.assign(count_variable, zero_node)
sess.run(assign_node)
print(sess.run(count_variable))

# tf.assign(target, value) has some unique properties:
#   - identity operation.
#   - side effects. when computation "flows" through assign_node, side effects
#   happen to other things in the graph
#   - non-dependent edges. even though the count_variable and the assign_node
#   are connected in the graph, neither is dependent on the other. this means
#   computation will not flow back through that edge when evaluating either node
#   (i.e. we need to call sess.run(assign_node) before we call
#   sess.run(count_variable))

# as computation flows through any node in the graph, it will enact any side
# effects controlled by that node

## initializers


# calling sess.run at this point will cause an error
# we've set the initializer property of count_variable, meaning we've added
# a node to the graph, but we need to tell our session to initialize
const_init_node = tf.constant_initializer(0.)
count_variable = tf.get_variable("count_w_init", [], initializer=const_init_node)

# we add a global_variables_initializer node to our graph, which, like assign,
# has side effects, but unlike it, initializes all variables in the graph
# (so we don't have to specify which variables to initialize as input)

# global_variables_initializer will look at the global graph at the moment of
# its creation and automatically add dependencies to every tf.initializer in the
# graph
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(count_variable))


## Variable Sharing

# you may encounter TensorFlow code with Variable Sharing (resue=True); it is
# far better to instead simply maintain a pointer to that node programmatically


## Debugging with tf.Print

# we'd like to inspect the intermediate values of a computation

# it's possible to do this with sess.run:
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
answer, inspection = sess.run([sum_node, [two_node, three_node]])
print(answer, inspection)

# but as code becomes more complex, it can be a bit awkward, so it's better to
# use tf.Print

# tf.Print requires a node to copy, and a list of nodes to print
# tf.Print is an identity wrt the node to copy, and prints the values of all the
# nodes in the list of things to print as a side effect
print_sum_node = tf.Print(sum_node, [two_node, three_node])
print(sess.run(print_sum_node))

# tf.Print is a side effect, thus printing only occurs if printing flows through
# the tf.Print node
# in particular, even if the node tf.Print copies is on the computation path,
# tf.Print may not be, and so there may be no printing, as below
print_two_node = tf.Print(two_node, [two_node, three_node, sum_node])
print(sess.run(sum_node))

# it makes sense to have tf.Print as a side effect, as this way we don't print
# if we just want to do some computations using the node tf.Print happens to copy
