
import numpy as np
import tensorflow as tf

from value_initializer import glorot_uniform
from utils import myshape

def dense(x, n_out,
        non_linear=None,
        use_bias=True,
        weight_init=glorot_uniform,
        bias_init=tf.constant_initializer(0.0),
        trainable=True,
        scope=None,
        reuse=False,
        name='Dense'):
    '''
    non_linear: function like rele, sigmod .etc

    '''
    x_shape = myshape(x)
    n_dims = len(x_shape)
    n_in = x_shape[-1]
    # n_in , n_out is the shape of weight
    with tf.variable_scope(scope, default_name=name, values=[x], reuse=reuse) as scope:
        # reshape for broadcasting
        x_reshape = tf.reshape(x, (-1, n_in))

        W = tf.get_variable('W', shape=(n_in,n_out),
                                 initializer=weight_init,
                                 trainable=trainable)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)
        tf.summary.histogram('W', W)

        y = tf.matmul(x_reshape, W)

        if use_bias:
            b = tf.get_variable('b',
                                shape=(n_out,)),
                                initializer=bias_init,
                                trainable=trainable)
            tf.add_to_collection(tf.GraphKeys.BIASES, b)
            tf.summary.histogram('b', b)
            y += b
        if non_linear:
            y = non_linear(y)

        new_shape = tf.concat([tf.shape(x)[:n_dims-1], [n_out]], axis=0)
        y = tf.reshape(y, new_shape)

        new_dims = x_shape[:-1] + [n_out]
        y.set_shape(new_dims)
        tf.summary.histogram('activations', y)
    return y

def embedding(indices, n_vocabulary, n_out,
               weight_init=glorot_uniform,
               reference=None,
               trainable=True,
               scope=None,
               reuse=False, name='Embedding'):
    if isinstance(n_out, int):
        n_out = (n_out,)
    with tf.variable_scope(scope, default_name=name, reuse=reuse) as scope:
        if reference is None:
            W = tf.get_variable('W', shape=(n_vocabulary,) + n_out,
                                     initializer=weight_init,
                                     trainable=trainable)
        else:
            W = tf.get_variable('W', shape=(n_vocabulary,) + n_out,
                                initializer=reference_initializer(reference),
                                trainable=trainable)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)

        y = tf.nn.embedding_lookup(W, indices)

    return y



def masked_reduce(x, mask=None, axes=None,
                  reduce_op=tf.reduce_sum,
                  keep_dims=False,
                  scope=None, name='masked_reduce'):
    scope_vars = [x]
    if mask is not None:
        scope_vars.append(mask)

    with tf.variable_scope(scope, default_name=name,
                           values=scope_vars) as scope:
        if mask is not None:
            mask = tf.cast(mask > 0, tf.float32)
            x *= mask

        y = reduce_op(x, axes, keep_dims)

    return y


def masked_sum(x, mask=None, axes=None,
               keep_dims=False,
               scope=None, name='masked_sum'):
    return masked_reduce(x, mask, axes, tf.reduce_sum,
                         keep_dims, scope, name)


def masked_mean(x, mask=None, axes=None,
                keep_dims=False,
                scope=None, name='masked_mean'):
    if mask is None:
        mred = masked_reduce(x, mask, axes, tf.reduce_mean,
                             keep_dims, scope, name)
    else:
        msum = masked_reduce(x, mask, axes, tf.reduce_sum,
                             keep_dims, scope, name)
        mask = tf.cast(mask > 0, tf.float32)
        N = tf.reduce_sum(mask, axes, keep_dims)
        N = tf.maximum(N, 1)
        mred = msum / N
    return mred






