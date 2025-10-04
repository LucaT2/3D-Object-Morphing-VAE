# Library imports
import gradio as gr
import numpy as np
import imageio
import os
import json

from PIL import Image



# Project files imports
from Backend.plot_grids_utils import plot_voxels_numpy, plot_voxels_pyvista
from VAE_Model.evaluate.eval import Eval_VAE
from Backend.app_model import trained_model
from Backend.prepare_model import initialize_model, warm_up_model, preload_objects
from Backend.plot_grids_utils import reconstruct_object, show_morphing_gif

category_prews = {}

def load_previews():
    print("Loading previews from cache...")
    global category_prews
    CACHE_DIR = r"Frontend/previews_cache"

    METADATA_FILE = os.path.join(CACHE_DIR, "previews_metadata.json")
    print(f"Attempting to open metadata file: {METADATA_FILE}")

    # Load the metadata from the JSON file
    with open(METADATA_FILE, 'r') as f:
        cached_metadata = json.load(f)

    for category, items in cached_metadata.items():
        previews = []
        for item in items:
            # Load the pre-generated PNG image from disk
            image_data = imageio.imread(item["image_cache_path"])
            previews.append({
                "name": item["name"],
                "path": item["path"],
                "image": image_data ,
                "image_cache_path": item["image_cache_path"]
            })
        category_prews[category] = previews
        print("Previews loaded successfully!")



load_previews()

def create_reconstruct_callback(previews_list):
    """Factory function to create a unique and correctly scoped callback for each category tab."""
    def reconstruct_callback(threshold: float, evt: gr.SelectData):
        # evt.index gives us the exact index of the clicked item
        selected_index = evt.index
        
        # Get the corresponding object from our list
        found_item = previews_list[selected_index]

        # Get the .npy path for reconstruction
        file_path = found_item["path"]
        file_obj = type('File', (), {'name': file_path})()
        return reconstruct_object(file_obj, threshold)
        
    return reconstruct_callback

def create_interpolation_callback(previews_list):
    """Factory function that now works with indices from gr.State."""
    def interpolation_callback(index_a, index_b, steps, threshold):
        if index_a is None or index_b is None:
            raise gr.Error("Please select one object from each gallery (A and B).")
        
        # We now have the correct indices directly
        found_item_a = previews_list[index_a]
        found_item_b = previews_list[index_b]

        path_a = found_item_a["path"]
        path_b = found_item_b["path"]

        file_obj_a = type('File', (), {'name': path_a})()
        file_obj_b = type('File', (), {'name': path_b})()
        
        return show_morphing_gif(file_obj_a, file_obj_b, steps, threshold)
    return interpolation_callback



with gr.Blocks() as demo:
    
    gr.Markdown("# 🤖 3D Variational Autoencoder (VAE) Playground")
    with gr.Tabs():
        # Tab 1: Reconstruction
        with gr.TabItem("🔄 Object Reconstruction"):
            with gr.Tabs() as category_tabs:
                for category, previews in category_prews.items():
                    with gr.TabItem(category):
                        gr.Markdown(f"### {category.capitalize()} Objects")
                        gallery = gr.Gallery(
                            value=[p["image"] for p in previews],
                            label="Preview",
                            show_label=False,
                            columns=5,
                            height=360
                        )
                        with gr.Row():
                            #reconstruct_button = gr.Button("Generate Reconstruction", variant="primary")
                            threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.01, label="Threshold for the Voxel Grid")
                        with gr.Row():
                            original_plot_output = gr.Image(type="numpy", label="Original Object", interactive=False)
                            reconstructed_plot_output = gr.Image(type="numpy", label="VAE Reconstruction", interactive=False)
                        callback_for_this_tab = create_reconstruct_callback(previews)

                    
                        gallery.select(
                            fn=callback_for_this_tab,
                            inputs=[threshold_slider],
                            outputs=[original_plot_output, reconstructed_plot_output]
                        )

        with gr.TabItem("✨ Latent Space Interpolation"):

            with gr.Tabs() as category_tabs_interp:
                for category, previews in category_prews.items():
                    with gr.TabItem(category):
                        gr.Markdown(f"### Select Two {category.capitalize()} Objects to Morph")
                        
                        selected_index_A = gr.State(value=None)
                        selected_index_B = gr.State(value=None)
                        
                        with gr.Row():
                            interpolation_gallery_A = gr.Gallery(
                                value=[p["image"] for p in previews],
                                label="Select object A",
                                columns=4,
                                height=360,
                                allow_preview=False
                            )
                            interpolation_gallery_B = gr.Gallery(
                                value=[p["image"] for p in previews],
                                label="Select object B",
                                columns=4,
                                height=360,
                                allow_preview=False
                            )
                        def store_selection_A(evt: gr.SelectData):
                            return evt.index
                        def store_selection_B(evt: gr.SelectData):
                            return evt.index
                        
                        interpolation_gallery_A.select(fn=store_selection_A, inputs=None, outputs=selected_index_A)
                        interpolation_gallery_B.select(fn=store_selection_B, inputs=None, outputs=selected_index_B)

                        with gr.Row():
                            threshold_slider_interp = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.01, label="Voxel Threshold")
                            steps_slider = gr.Slider(minimum=5, maximum=30, value=15, step=1, label="Number of Morphing Steps")
                        
                        generate_button = gr.Button("Generate Interpolation GIF", variant="primary")
                        gif_output = gr.Image(label="Interpolation Animation", interactive=False)
                        
                        callback_for_this_interp_tab = create_interpolation_callback(previews)
                        
                        generate_button.click(
                            fn=callback_for_this_interp_tab,
                            inputs=[
                                selected_index_A, # Pass the stored index
                                selected_index_B, # Pass the stored index
                                steps_slider,
                                threshold_slider_interp
                            ],
                            outputs=gif_output
                        )

if __name__ == "__main__":
    model = initialize_model()
    #warm_up_model(model)
    demo.launch()