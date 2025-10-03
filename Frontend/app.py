# Library imports
import gradio as gr
import tensorflow as tf
import numpy as np
import imageio

# Project files imports
from Backend.plot_grids_utils import plot_voxels_numpy, plot_voxels_pyvista, create_morphing_gif
from VAE_Model.evaluate.eval import Eval_VAE
from Backend.plot_grids import trained_model
model_instance = None

def echo(text):
    return text

def initialize_model():
    global model_instance
    if model_instance is None:
        model_instance = trained_model(weights_file = r'VAE_Model/Weights/best_vae_model.weights.h5')
        model_instance.initialize_model()
    return model_instance

def warm_up_model(model):
    """
    Runs a dummy prediction with the maximum possible batch size to
    trigger JIT compilation and autotuning for all interpolation tasks.
    """
    print("🔥 Warming up the model for interpolation... This may take a minute.")
    
    # --- WARM UP THE DECODER ---
    # Set this to the maximum value of your Gradio slider.
    warmup_batch_size = 15 
    latent_dim = 128 # Make sure this matches your model's latent dimension
    
    # Create a fake batch of latent vectors
    dummy_latent_vectors = np.zeros((warmup_batch_size, latent_dim), dtype=np.float32)
    reconstruction_dummy = np.zeros((1,latent_dim), dtype=np.float32)
    # Run a dummy prediction through the decoder.
    # This will compile and tune the model for the largest expected batch size.
    _ = model.vae_test_model.decoder.predict(dummy_latent_vectors, verbose=0)
    _ = model.vae_test_model.decoder.predict(reconstruction_dummy, verbose=0)
    
    print("✅ Model is warmed up and ready!")


def reconstruct_object(uploaded_file):
    if uploaded_file is None:
        return None, None
    model = initialize_model()

    original_voxel_data = np.expand_dims(np.load(uploaded_file.name), axis = 0)
    original_voxel_data = np.expand_dims(original_voxel_data, axis = -1)
    
    original_plot = plot_voxels_pyvista(original_voxel_data, title="Original")

    reconstructed_voxels = model.reconstruct_given_object(original_voxel_data)
    reconstructed_plot = plot_voxels_pyvista(reconstructed_voxels, title="Reconstructed")

    return original_plot, reconstructed_plot

def show_morphing_gif(file_1, file_2, steps):
    if file_1 is None or file_2 is None:
        return None # Return nothing if files are missing
    model = initialize_model()

    # Get the list of voxel grids for each frame of the animation
    voxel_grids = model.get_interpolation(file_1.name, file_2.name, num_steps=int(steps))
    
    # Create the GIF from the voxel grids
    frames = []
    for i, grid in enumerate(voxel_grids):
        title = f"Step {i}/{len(voxel_grids)-1}"
        frame_image = plot_voxels_pyvista(grid, title=title, threshold=0.6)
        frames.append(frame_image)
        
    # Create a temporary file path for the GIF
    gif_path = "interpolation.gif"
    imageio.mimsave(gif_path, frames, duration=5, loop=0) # loop=0 means infinite loop
    
    return gif_path

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 3D Variational Autoencoder (VAE) Playground")
    with gr.Tabs():
        # Tab 1: Reconstruction
        with gr.TabItem("🔄 Object Reconstruction"):
            gr.Markdown("Upload a single `.npy` voxel object to see how well the VAE can reconstruct it.")
            with gr.Row():
                file_input = gr.File(label="Upload Your 3D Voxel Object (.npy)")
            with gr.Row():
                original_plot_output = gr.Image(type="numpy", label="Original Object", interactive=False)
                reconstructed_plot_output = gr.Image(type="numpy", label="VAE Reconstruction", interactive=False)
            
            file_input.upload(fn=reconstruct_object, inputs=file_input, outputs=[original_plot_output, reconstructed_plot_output])

        # Tab 2: Interpolation
        with gr.TabItem("✨ Latent Space Interpolation"):
            gr.Markdown("Upload two `.npy` objects to see the VAE morph one into the other. This visualizes a smooth path through the model's 'understanding' of 3D shapes.")
            with gr.Row():
                file_input_1 = gr.File(label="Upload Object A (.npy)")
                file_input_2 = gr.File(label="Upload Object B (.npy)")
            
            steps_slider = gr.Slider(minimum=5, maximum=25, value=15, step=1, label="Number of Morphing Steps")
            
            generate_button = gr.Button("Generate Interpolation GIF", variant="primary")
            
            gif_output = gr.Image(label="Interpolation Animation", interactive=False)
            
            generate_button.click(
                fn=show_morphing_gif,
                inputs=[file_input_1, file_input_2, steps_slider],
                outputs=gif_output
            )

if __name__ == "__main__":
    model = initialize_model()
    warm_up_model(model)
    demo.launch()