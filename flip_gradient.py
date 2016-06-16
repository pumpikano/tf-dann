import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

flip_gradient_module = tf.load_op_library('../tensorflow-fork/bazel-bin/tensorflow/core/user_ops/flip_gradient.so')
flip_gradient = flip_gradient_module.flip_gradient

@ops.RegisterGradient("FlipGradient")
def _flip_gradient_grad(op, grad):
    """The gradients for `flip_gradient`.

    Args:
        op: The `flip_gradient` `Operation` that we are differentiating, which we can use
            to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the `flip_gradient` op.

    Returns:
        Gradients with respect to the input of `flip_gradient`.
    """
    s = op.inputs[1]
    return [math_ops.neg(grad) * s, None]
