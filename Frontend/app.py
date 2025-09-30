# Library imports
import gradio as gr
import tensorflow as tf
import numpy as np

# Project files imports
from Backend.plot_grids_utils import plot_voxels
from VAE_Model.evaluate.eval import Eval_VAE
from Backend.plot_grids import trained_model
model_instance = None

def echo(text):
    return text

def initialize_model():
    global model_instance
    if model_instance is None:
        model_instance = trained_model(weights_file = r'VAE_Model\Weights\best_vae_model.weights.h5')
        model_instance.initialize_model()
    return model_instance
def reconstruct_object(uploaded_file):
    if uploaded_file is None:
        return None, None
    model = initialize_model()
    original_voxel_data = np.expand_dims(np.load(uploaded_file.name), axis = 0)
    original_voxel_data = np.expand_dims(original_voxel_data, axis = -1)
    
    original_plot = plot_voxels(original_voxel_data, title="Original")

    reconstructed_voxels = model.reconstruct_given_object(original_voxel_data)
    reconstructed_plot = plot_voxels(reconstructed_voxels, title="Reconstructed")

    return original_plot, reconstructed_plot

with gr.Blocks() as demo:
    gr.Markdown("# 🔄 3D VAE Object Reconstructor")
    gr.Markdown("Upload a 3D object file to see its reconstruction by the VAE.")
    with gr.Row():
        file_input = gr.File(label="Upload Your 3D Object")
    with gr.Row():
        original_plot_output = gr.Plot(label="Original Object")
        # Output component: a plot
        reconstructed_plot_output = gr.Plot(label="Voxel Reconstruction")
    

    outputs = [original_plot_output, reconstructed_plot_output]

    # Connect the "change" event of the file uploader to our main function
    file_input.upload(fn=reconstruct_object, inputs=file_input, outputs=outputs)

if __name__ == "__main__":
    demo.launch()