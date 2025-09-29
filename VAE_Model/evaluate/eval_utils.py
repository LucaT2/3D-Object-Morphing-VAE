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

def plot_voxels(voxel_grid, title="", threshold = 0.8):
    """
    Plots a 3D voxel grid from a NumPy array.
    """
    voxel_grid = np.squeeze(voxel_grid)
    
    # Use a threshold to convert the model's probabilistic output (0.0 to 1.0)
    # into a definite binary grid for clear plotting.
    binary_grid = voxel_grid > threshold
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Use the ax.voxels() function to draw the grid
    ax.voxels(binary_grid, edgecolor='k')
    
    ax.set_title(title)
    plt.show()

