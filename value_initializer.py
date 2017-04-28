import numpy as np
import tensorflow as tf

def glorot_uniform(shape, dtype, partition_info=None):
	if not dtype.is_floating:
		raise ValueError('Expected a floating point type, got {}'.format(dtype))

	n_in = np.prod(shape[:-1])
	n_out = shape[-1]

	r = tf.cast(tf.sqrt(6.0 /  (n_int + n_out)), tf.float32)
	return tf.random_uniform(shape, -r, r, dtype=dtype)


def reference_initializer(ref):
	def initializer(shape, dtype, partition_info=None):
		return tf.cast(tf.constant(np.reshape(ref, shape)),dtype)
	return initializer


