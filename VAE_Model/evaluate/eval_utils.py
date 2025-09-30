import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def calculate_reconstruction_similarity(originals, reconstructions, threshold=0.6):
    """
    Calculates the average Intersection over Union (IoU) similarity percentage.
    """
    # Binarize the voxel grids based on the threshold
    original_binary = tf.cast(originals > threshold, dtype=tf.float32)
    reconstructed_binary = tf.cast(reconstructions > threshold, dtype=tf.float32)
    
    original_flat = tf.reshape(original_binary, [original_binary.shape[0], -1])
    reconstructed_flat = tf.reshape(reconstructed_binary, [reconstructed_binary.shape[0], -1])
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(original_flat * reconstructed_flat, axis=1)
    union = tf.reduce_sum(original_flat, axis=1) + tf.reduce_sum(reconstructed_flat, axis=1) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return tf.reduce_mean(iou) * 100



