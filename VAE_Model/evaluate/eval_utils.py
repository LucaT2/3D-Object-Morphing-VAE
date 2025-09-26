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

def plot_voxels(voxel_grid, title=""):
    """
    Plots a 3D voxel grid from a NumPy array.
    """
    voxel_grid = np.squeeze(voxel_grid)
    
    # Use a threshold to convert the model's probabilistic output (0.0 to 1.0)
    # into a definite binary grid for clear plotting.
    binary_grid = voxel_grid > 0.6
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Use the ax.voxels() function to draw the grid
    ax.voxels(binary_grid, edgecolor='k')
    
    ax.set_title(title)
    plt.show()

# def reconstruct_one_object():
#     random_number = np.rand() %100
#     random_model = np.expand_dims(np.load(val_full_paths[random_number]), axis=0) 
#     random_model = np.expand_dims(random_model, axis=-1)
#     plot_voxels(np.squeeze(random_model), title="Original Random Model")

#     #z_random_mean = recreated_vae.encoder.predict(random_model)[0]
#     z_random = recreated_vae.encoder.predict(random_model)[2]

#     #plot_voxels(np.squeeze(z_random_mean), title="Mean of random object")
#     #plot_voxels(np.squeeze(z_random), title = "Sample of random object")

#     reconstructed_voxel_grid = recreated_vae.decoder.predict(z_random)
#     plot_voxels(np.squeeze(reconstructed_voxel_grid), title="Reconstruction from random latent sample")