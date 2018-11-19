#### More on Graph Inspection


import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.get_variable('b', [], dtype=tf.int32)
c = a + b

g = tf.get_default_graph()
ops_list = g.get_operations()
print()

print("tf.Variable objects are really a bundle of four operations (and their corresponding tensors):")
print(b)
print(ops_list)
print()

print("Two of these are accessed via their tf.Operations,"),
print("the core", b.op.__repr__(), "and the initializer", b.initializer.__repr__())
print("The other two are accessed via their tf.Tensors,"),
print("the initial-value", b.initial_value, "and the current-value", b.value())
print()

print("A tf.Variable core-op takes no inputs, and outputs a tensor of type *_ref:")
print(b.op.__repr__())
print(list(b.op.inputs), b.op.outputs)
print()

print("A tf.Variable current-value is the output of a \"/read\" operation, which converts from *_ref to a tensor with a concrete data-type.")
print("Other ops use the concrete node as their input:")
print(b.value())
print(b.value().op.__repr__())
print(list(c.op.inputs))
