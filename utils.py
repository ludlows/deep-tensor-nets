import tensorflow as tf
import numpy as np


def myshape(x):
    '''
    make it easy to get shapes
    even if it is a tensot or numpy array
    Return: shape in a list 
    '''
    if isinstance(x, tf.Tensor):
        return x.get_shape().as_list()
    return list(np.shape(x))