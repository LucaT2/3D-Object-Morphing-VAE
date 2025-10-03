import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import imageio
import os

def plot_voxels_numpy(voxel_grid, title="", threshold = 0.6):
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
    
    ax.voxels(binary_grid, edgecolor='k')
    
    ax.set_title(title)
    #plt.show()
    return fig



def plot_voxels_pyvista(voxel_grid, title= "", threshold=0.6):
    """
    Plots a 3D voxel grid from a NumPy array using Pyvista, so that makes it much faster because it uses
    gpu to plot the grid.
    """
    voxel_grid = np.squeeze(voxel_grid)
    plotter = pv.Plotter(off_screen=True)
    grid = pv.wrap(voxel_grid)
    thresholded_grid = grid.threshold(threshold)
    plotter.add_mesh(thresholded_grid,
                      show_edges=False,
                        color='deepskyblue',
                        smooth_shading = False,
                        )
    plotter.add_title(title, color ='white', font_size = 16)
    plotter.view_isometric()

    plotter.background_color = 'black'

    screenshot = plotter.screenshot(transparent_background=False, return_img=True)
    plotter.close()
    return screenshot



def create_morphing_gif(voxel_grids, output_filename="voxel_morph.gif", duration=10, threshold=0.6):
    """
    Creates a GIF from a sequence of voxel grids.

    Args:
        voxel_grids (list[np.ndarray]): A list of voxel grids to include in the GIF.
        output_filename (str): The filename for the output GIF.
        duration (int): The duration (in ms) for each frame in the GIF.
        threshold (float): The plotting threshold for the voxel grids.
    """
    print("Generating frames for GIF...")
    frames = []
    for i, grid in enumerate(voxel_grids):
        title = f"Morph Step {i}/{len(voxel_grids) - 1}"
        frame_image = plot_voxels_pyvista(grid, title=title, threshold=threshold)
        frames.append(frame_image)
    
    # Add a pause at the beginning and end
    frames = [frames[0]] * 5 + frames + [frames[-1]] * 5

    print(f"Creating GIF: {output_filename}...")
    with imageio.get_writer(output_filename, mode='I', duration=duration) as writer:
        for frame in frames:
            writer.append_data(frame)
    print("GIF created successfully!")