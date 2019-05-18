import tensorflow as tf

#### TensorFlow: The Confusing Parts (2)

### Naming Variables and Tensors

# when crash, error trace will refer to a specific op
# if you have many ops of the same type, it can be tough to figure out which
# one is problematic
a = tf.constant(0.)
b = tf.constant(1.)
c = tf.constant(2., name="cool_const")
d = tf.constant(3., name="cool_const")
print(a.name, b.name, c.name, d.name)

### Using Scopes

# scopes subdivide graphs into smaller chunks
# by prefixing (recursively) the name of the scope
a = tf.constant(0.)
b = tf.constant(1.)
with tf.variable_scope("first_scope"):
    c = a + b
    d = tf.constant(2., name="cool_const")
    coef1 = tf.get_variable("coef", [], initializer=tf.constant_initializer(2.))
    with tf.variable_scope("second_scope"):
        e = coef1 * d
        coef2 = tf.get_variable("coef", [], initializer=tf.constant_initializer(3.))
        f = tf.constant(1.)
        g = coef2 * f

print(a.name, b.name)
print(c.name, d.name)
print(e.name, f.name, g.name)
print(coef1.name)
print(coef2.name)

### Saving a Model

# neural network consists of weights and computation graph
# tensorflow separates these two components
a = tf.get_variable("a", [])
b = tf.get_variable("b", [])
init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.save(sess, "./tftcp.model")

# 4 output files:

# tftcp.model.data-00000-of-00001 contains weights
# tftcp.model.meta contains network
# tftcp.model.index indexing structure linking weights with network
# checkpoint keeps track of multiple versions of model

# what is the need for session and variable initializer?
# the computations are in the graph, but the values of the computation are in the session
# tf.train.Saver saves the values of the variables through the session
# the initializer ensures all the values we save() are initialized

### Loading a Model

a = tf.get_variable("a", [])
b = tf.get_variable("b", [])
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "./tftcp.model")
print(sess.run([a, b]))

### Choosing Your Variables

a = tf.get_variable("a", [])
b = tf.get_variable("b", [])
saver = tf.train.Saver()
c = tf.get_variable("c", [])
print(saver._var_list)

# saver will only record variables created before it was initialized

# save only a subset of variables
a = tf.get_variable('a', [])
b = tf.get_variable('b', [])
c = tf.get_variable('c', [])
saver = tf.train.Saver(var_list=[a,b])
print(saver._var_list)

### Loading Modified Models

## Saving a whole model and loading only part of it
a = tf.get_variable('a', [])
b = tf.get_variable('b', [])
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.save(sess, './tftcp.model')

a = tf.get_variable('a', [])
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, './tftcp.model')
print(sess.run(a))

## Loading a model as part of a larger one
a = tf.get_variable('a', [])
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.save(sess, './tftcp.model')

a = tf.get_variable('a', [])
d = tf.get_variable('d', [])
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, './tftcp.model')

# this will fail, since it will try to load d
# a similar scenario occurs when we want to load one model's parameters
# into a different model's computation graph
# globabally unique variable names makes saving much easier!
# we can use var_list as a dict mapping names to variables:

a = tf.get_variable('a', [])
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.save(sess, './tftcp.model')

d = tf.get_variable('d', [])
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list={'a': d})
sess = tf.Session()
sess.run(init)
saver.restore(sess, './tftcp.model')
print(sess.run(d))

# we can use this to use pre-trained word embeddings, or if you changed the
# parameterization of your model between training runs and want to checkpoint

# "A word of caution: it’s very important to know exactly how the parameters
# you are loading are meant to be used. If possible, you should use the exact
# code the original authors used to build their model, to ensure that that
# component of your computational graph is identical to how it looked during
# training. If you need to re-implement, keep in mind that basically any change,
#  no matter how minor, is likely to severely damage the performance of your
# pre-trained net. Always benchmark your reimplementation against the original!

# Inspecting saved models
a = tf.get_variable('a', [])
b = tf.get_variable('b', [10,20])
c = tf.get_variable('c', [])
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.save(sess, './tftcp.model')
print(tf.contrib.framework.list_variables('./tftcp.model'))
