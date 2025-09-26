import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import layers

def scaled_sigmoid_activation(x):
    sigmoid = tf.math.sigmoid(x)
    return (sigmoid * 0.9) +0.1

def weighted_bce_loss(y_true, y_pred, gamma):
    """
    Implements the specialized BCE loss function.
    L = -γ * t * log(o) - (1-γ) * (1-t) * log(1-o)
    """
    # Making sure all are of type float32
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    gamma_t = tf.cast(gamma, tf.float32)
    
    y_pred = (y_pred - 0.1) / 0.9
    y_true = (y_true + 1.0) / 3.0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    loss = -gamma * y_true * K.log(y_pred) - (1 - gamma) * (1 - y_true) * K.log(1 - y_pred)
    return loss

class Sampling(layers.Layer):
    """
    This class uses the z_mean and z_log_var in order to sample z from the distribution
    """
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape = (batch,dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon