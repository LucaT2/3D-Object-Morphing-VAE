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
all_previews_list = []
all_preview_images = []

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
            image_data = imageio.imread(item["image_cache_path"])
            preview_item = {
                "name": item["name"],
                "path": item["path"],
                "image": image_data ,
                "image_cache_path": item["image_cache_path"]
            }
            previews.append(preview_item)
            all_previews_list.append(preview_item)
            
        category_prews[category] = previews
        print("Previews loaded successfully!")



load_previews()

def create_reconstruct_callback(previews_list):
    """function to create a unique and correct callback for each category tab."""
    def reconstruct_callback(threshold: float, evt: gr.SelectData):
        # evt.index gives us the exact index of the clicked item
        selected_index = evt.index
        
        # Get the corresponding object from our list
        found_item = previews_list[selected_index]

        # Get the .npy path for reconstruction
        file_path = found_item["path"]
        file_obj = type('File', (), {'name': file_path})()
        return reconstruct_object(model, file_obj, threshold)
        
    return reconstruct_callback


def interpolation_callback(item_a, item_b, steps, threshold):
    """Callback that receives the full item data from gr.State."""
    if item_a is None or item_b is None:
        raise gr.Error("Please select one object from each gallery (A and B).")
    
    path_a = item_a["path"]
    path_b = item_b["path"]

    file_obj_a = type('File', (), {'name': path_a})()
    file_obj_b = type('File', (), {'name': path_b})()
    
    return show_morphing_gif(model, file_obj_a, file_obj_b, steps, threshold)



with gr.Blocks() as demo:
    
    gr.Markdown("# 🤖 3D Variational Autoencoder (VAE) Playground")
    with gr.Tabs():
        # Tab 1: Latent Space Interpolation
        with gr.TabItem("✨ Latent Space Interpolation"):
            gr.Markdown("### Select an Object A and an Object B to Morph")
            gr.Markdown("Use the dropdown above each gallery to filter its contents.")

            # State variables to hold the full data of the selected objects
            selected_item_A = gr.State(value=None)
            selected_item_B = gr.State(value=None)
            
            category_choices = ["All"] + sorted(list(category_prews.keys()))
            all_preview_images = [p['image'] for p in all_previews_list]

            with gr.Row():
                with gr.Column():
                    category_filter_A = gr.Dropdown(choices=category_choices, value="All", label="Filter Gallery A by Category")
                    interpolation_gallery_A = gr.Gallery(value=all_preview_images, 
                                                         label="Select Object A", 
                                                         columns=4, height=360, 
                                                         allow_preview=False)
                
                with gr.Column():
                    category_filter_B = gr.Dropdown(choices=category_choices, value="All", label="Filter Gallery B by Category")
                    interpolation_gallery_B = gr.Gallery(value=all_preview_images, 
                                                         label="Select Object B", 
                                                         columns=4, height=360, 
                                                         allow_preview=False)
            
            def update_gallery(category):
                """Updates a gallery's content based on the selected category."""
                if category == "All":
                    return gr.Gallery.update(value=all_preview_images)
                else:
                    images_to_show = [p['image'] for p in category_prews[category]]
                return gr.Gallery(value=images_to_show)

            def store_selection(category, evt: gr.SelectData):
                """Finds the full data for the selected item and returns it to be stored in State."""
                if category == "All":
                    return all_previews_list[evt.index]
                else:
                    return category_prews[category][evt.index]
            
            # When a filter dropdown changes, update its corresponding gallery
            category_filter_A.change(fn=update_gallery, inputs=[category_filter_A], outputs=[interpolation_gallery_A])
            category_filter_B.change(fn=update_gallery, inputs=[category_filter_B], outputs=[interpolation_gallery_B])
            
            # When an image is selected, store its full data in the appropriate state variable
            interpolation_gallery_A.select(fn=store_selection, inputs=[category_filter_A], outputs=[selected_item_A])
            interpolation_gallery_B.select(fn=store_selection, inputs=[category_filter_B], outputs=[selected_item_B])

            with gr.Row():
                threshold_slider_interp = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.01, label="Voxel Threshold")
                steps_slider = gr.Slider(minimum=5, maximum=30, value=15, step=1, label="Number of Morphing Steps")
            
            generate_button = gr.Button("Generate Interpolation GIF", variant="primary")
            gif_output = gr.Image(label="Interpolation Animation", interactive=False)
            
            generate_button.click(
                fn=interpolation_callback,
                inputs=[
                    selected_item_A,
                    selected_item_B,
                    steps_slider,
                    threshold_slider_interp
                ],
                outputs=gif_output
            )
        # Tab 2: Reconstruction
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



if __name__ == "__main__":
    model = initialize_model()
    #warm_up_model(model)
    demo.launch()