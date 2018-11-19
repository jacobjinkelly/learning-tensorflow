#### More on Graph Inspection


# nodes in the computational graph are operations, edges are tensors

# creating a new node (operation) requires:
#   - gather all incoming tf.Tensor objects, corresponding to incoming edges
#   - create the tf.Operation object, a node
#   - create outgoing tf.Tensor objects (edges), return pointers to them

# there are three ways we can inspect the graph:
#   - list all nodes: tf.Graph.get_operations()
#   - inspecting nodes: tf.Operation.inputs, tf.Operation.outputs
#   - inspecting edges: tf.Tensor.op return the operation for which this tensor
#     is the output, tf.Tensor.consumers() returns a list of all ops for which
#     this tensor is used as input

import tensorflow as tf

a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
c = a + b

print("Our tf.Tensor objects:")
print(a)
print(b)
print(c)
print()

a_op = a.op
b_op = b.op
c_op = c.op

print("Our tf.Operation objects, printed in compressed form:")
print(a_op.__repr__())
print(b_op.__repr__())
print(c_op.__repr__())
print()

print("The default behaviour of printing a tf.Operation object is to pretty-print:")
print(c_op)

print("Inspect consumers for each tensor:")
print(a.consumers())
print(b.consumers())
print(c.consumers())
print()

print("Inspect input tensors for each op:")
# it's in a weird format, tensorflow.python.framework.ops._InputList, so we need to convert to list() to inspect
print(list(a_op.inputs))
print(list(b_op.inputs))
print(list(c_op.inputs))
print()

print("Inspect output tensors for each op:")
print(a_op.outputs)
print(b_op.outputs)
print(c_op.outputs)
print()

print("The list of all nodes (tf.Operations) in the graph:")
g = tf.get_default_graph()
ops_list = g.get_operations()
print(ops_list)
print()

print("The list of all edges (tf.Tensors) in the graph, by way of list comprehension:")
tensors_list = [tensor for op in ops_list for tensor in op.outputs]
print(tensors_list)
print()

print("Note that these are the same pointers we can find by referring to our various graph elements directly:")
print(ops_list[0] == a_op, tensors_list[0] == a)
