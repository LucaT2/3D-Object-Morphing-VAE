import matplotlib.pyplot as plt
import numpy as np

def plot_voxels(voxel_grid, title="", threshold = 0.5):
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
    #plt.show()
    return fig