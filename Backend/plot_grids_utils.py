import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import imageio
import os
from PIL import Image

from Backend.prepare_model import initialize_model, warm_up_model



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

def reconstruct_object(uploaded_file, threshold):
    if uploaded_file is None:
        return None, None
    model = initialize_model()

    original_voxel_data = np.expand_dims(np.load(uploaded_file.name), axis = 0)
    original_voxel_data = np.expand_dims(original_voxel_data, axis = -1)
    
    original_plot = plot_voxels_pyvista(original_voxel_data, title="Original")

    reconstructed_voxels = model.reconstruct_given_object(original_voxel_data)
    reconstructed_plot = plot_voxels_pyvista(reconstructed_voxels, title="Reconstructed", threshold=threshold)

    return original_plot, reconstructed_plot

def show_morphing_gif(file_1, file_2, steps, threshold):
    if file_1 is None or file_2 is None:
        return None # Return nothing if files are missing
    model = initialize_model()

    # Get the list of voxel grids for each frame of the animation
    voxel_grids = model.get_interpolation(file_1.name, file_2.name, num_steps=int(steps))

    # Create the GIF from the voxel grids
    frames = []
    for i, grid in enumerate(voxel_grids):
        if i == 0:
            title = "Original Object A"
        elif i == len(voxel_grids) - 1:
            title = "Original Object B"
        else:
            title = f"Step {i}/{len(voxel_grids)-1}"
        frame_image = plot_voxels_pyvista(grid, title=title, threshold=threshold)
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame_image)
        frames.append(pil_image)

    frames = [frames[0]] * 5 + frames[1:-1] + [frames[-1]] * 5

    # Create a temporary file path for the GIF
    gif_path = "Frontend/interpolation.gif"
    # Save GIF using Pillow, duration in milliseconds per frame
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # 500 ms per frame (adjust as needed)
        loop=0
    )

    return gif_path