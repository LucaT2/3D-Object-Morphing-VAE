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
    """Factory function to create a unique and correctly scoped callback for each category's interpolation."""
    def interpolation_callback(selected_items: list, steps: int, threshold: float):
        if not selected_items or len(selected_items) != 2:
            raise gr.Error("Please select exactly two objects from the gallery.")
        
        temp_path_a = selected_items[0][0]
        temp_path_b = selected_items[1][0]
        
        filename_a = os.path.basename(temp_path_a)
        filename_b = os.path.basename(temp_path_b)

        # Match filenames against the specific previews_list for this category
        found_item_a = next((p for p in previews_list if os.path.basename(p["image_cache_path"]) == filename_a), None)
        found_item_b = next((p for p in previews_list if os.path.basename(p["image_cache_path"]) == filename_b), None)

        if not found_item_a or not found_item_b:
            raise gr.Error("Could not match the selected images. Please try again.")

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
                            height=500
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

        # Tab 2: Interpolation
        # with gr.TabItem("✨ Latent Space Interpolation"):
        #     gr.Markdown("Upload two `.npy` objects to see the VAE morph one into the other. This visualizes a smooth path through the model's 'understanding' of 3D shapes.")
        #     object_dropdown_1 = gr.Dropdown(choices=object_choices, label="Choose Object A")
        #     object_dropdown_2 = gr.Dropdown(choices=object_choices, label="Choose Object B")
        #     threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.01, label="Threshold for the Voxel Grid")
        #     steps_slider = gr.Slider(minimum=5, maximum=25, value=15, step=1, label="Number of Morphing Steps")
            
        #     generate_button = gr.Button("Generate Interpolation GIF", variant="primary")
            
        #     gif_output = gr.Image(label="Interpolation Animation", interactive=False)
            
        #     generate_button.click(
        #         fn=show_morphing_gif,
        #         inputs=[object_dropdown_1, object_dropdown_2, steps_slider, threshold_slider],
        #         outputs=gif_output
        #     )

if __name__ == "__main__":
    model = initialize_model()
    #warm_up_model(model)
    demo.launch()