import tensorflow as tf

### build the graph

## set up the parameters
m = tf.get_variable("m", [], initializer=tf.constant_initializer(0.))
b = tf.get_variable("b", [], initializer=tf.constant_initializer(0.))
# create the initialize once all of our variables have been added to the graph
init = tf.global_variables_initializer()

## set up computations
input_placeholder = tf.placeholder(tf.float32)
output_placeholder = tf.placeholder(tf.float32)

x = input_placeholder
y = output_placeholder
y_guess = m * x + b

loss = tf.square(y - y_guess)

## set up the optimizer, and minimization node
optimizer = tf.train.GradientDescentOptimizer(1e-3)
train_op = optimizer.minimize(loss)

### start the session
sess = tf.Session()
sess.run(init)

### perform training loop
import random

## set up problem
true_m = random.random()
true_b = random.random()

for update_i in range(100000):
    ## get input and output
    input_data = random.random()
    output_data = true_m * input_data + true_b

    _loss, _ = sess.run([loss, train_op], feed_dict={
                                                input_placeholder: input_data,
                                                output_placeholder: output_data
                                                })
    print(update_i, _loss)

### print out the values we learned for our two variables
print("True parameters:     m=%.4f, b=%.4f" % (true_m, true_b))
print("Learned parameters:  m=%.4f, b=%.4f" % tuple(sess.run([m, b]))


# the line <optimizer = tf.train.GradientDescentOptimizer(1e-3)> simply returns
# a python object with useful functions and doesn't add any nodes to the graph

# the line <train_op = optimizer.minimize(loss)> is adding a node to the graph
# which has no output, but a complicated side effect
# train_op finds all variable nodes in the computation path of <loss> and, via
# backpropagation, computes the gradient of the loss wrt to that variable, then
# performs an update of the variable determined by passing the gradient to the
# optimizer
